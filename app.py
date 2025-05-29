import cv2
import mediapipe as mp
import xgboost as xgb
import numpy as np
import json
import base64 # Base64 인코딩/디코딩용
import uuid # 고유 ID 생성을 위해 추가
import os # 파일 시스템 작업을 위해 추가
from datetime import datetime # 타임스탬프를 위해 추가
from flask import Flask, render_template, send_from_directory, request, jsonify, Response, session, redirect, url_for
from flask import Response, after_this_request

# Firebase 관련 임포트 (이전에 성공적으로 설정했으므로 그대로 사용)
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore # Firestore 사용 시 추가

app = Flask(__name__, static_url_path='/static', static_folder='static') # static_url_path 및 static_folder 명확히 설정
app.secret_key = 'your_strong_secret_key_here' # 세션 사용을 위한 secret key 설정 (보안상 강력한 키로 변경 필요)

# --- Firebase 초기화 (이전에 올바른 버킷 이름으로 수정했다고 가정) ---
# serviceAccountKey.json 파일 경로를 확인하세요. (app.py와 같은 폴더에 있어야 함)
service_account_key_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'smilefit-350ea.firebasestorage.app' # <-- 여기에 실제 버킷 이름을 넣으세요. (gs:// 제외)
        })
        print("[Firebase] Firebase 앱이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"[Firebase] Firebase 앱 초기화 실패: {e}")
        sys.exit(1) # 앱 초기화 실패 시 강제 종료

db = firestore.client() # Firestore 클라이언트 초기화
bucket = storage.bucket() # Storage 버킷 가져오기

# 임시 파일 저장 디렉토리 설정
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- MediaPipe 및 XGBoost 모델 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# XGBoost 모델 및 특징 로드 - 파일 경로 확인 필요 (app.py와 같은 폴더에 있다고 가정)
model = None
feature_cols = []
try:
    model = xgb.XGBRegressor()
    model.load_model('expression_similarity_model.json') # <-- 이 파일 인코딩 확인
    with open('feature_cols.json', 'r', encoding='utf-8') as f: # <-- 인코딩 명시
        feature_cols = json.load(f)
    print("XGBoost 모델 및 특징 파일 로드 완료.")
except Exception as e:
    print(f"XGBoost 모델 또는 특징 파일 로드 실패: {e}")
    # 모델 로드 실패 시, 웹캠 기능은 작동하더라도 예측은 안될 수 있습니다.


# AU 계산을 위한 랜드마크 인덱스 (FACS 기준, MediaPipe 랜드마크 기반)
AU_LANDMARKS = {
    'AU01': [336, 296], # 이마 안쪽 (inner brow raiser)
    'AU02': [334, 298], # 이마 바깥쪽 (outer brow raiser)
    'AU04': [9, 8],     # 눈썹 내림 (brow furrower)
    'AU06': [205, 206],  # 광대 (cheek raiser)
    'AU12': [308, 78],  # 입꼬리 (lip corner puller)
    'AU25': [13, 14] # 입술 개방 (lips part)
}

def calculate_au(landmarks):
    """MediaPipe 랜드마크 기반 AU 계산 (거리 기반)"""
    au_dict = {}

    if not landmarks.any(): # 랜드마크가 없을 경우 빈 딕셔너리 반환
        for col in feature_cols:
            au_dict[col] = 0.0
        return au_dict

    # 기본 AU 계산 (랜드마크 간 유클리드 거리)
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else:
            au_dict[au] = 0.0 # 랜드마크 인덱스가 유효하지 않으면 0

    # 가중치 컬럼 생성 (임시 0.8 곱함) - 이 부분은 모델 학습 방식에 따라 조정 필요
    for au_key in AU_LANDMARKS.keys():
        au_dict[f"{au_key}_w"] = au_dict.get(au_key, 0.0) * 0.8

    final_au_features = {}
    for col in feature_cols:
        final_au_features[col] = au_dict.get(col, 0.0)

    return final_au_features


# 웹캠 객체를 전역 변수로 관리하여 중복 초기화 방지 및 재사용
global_cap = None
last_calculated_au_values = {} # 가장 최근 계산된 AU 값 저장
last_calculated_score = 0.0    # 가장 최근 계산된 Score 저장


def generate_frames():
    global global_cap, last_calculated_au_values, last_calculated_score
    if global_cap is None or not global_cap.isOpened():
        global_cap = cv2.VideoCapture(0) # 0은 기본 웹캠
        if not global_cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

    frame_count = 0
    
    while global_cap.isOpened():
        success, frame = global_cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1) # 좌우 반전 (거울 효과)

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = np.array([
                    (lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                    for lm in results.multi_face_landmarks[0].landmark
                ], dtype=np.float32)
                
                if frame_count % 5 == 0: # 5프레임마다 AU 계산 및 예측
                    current_au_features = calculate_au(landmarks)
                    last_calculated_au_values = {k: v for k, v in current_au_features.items() if k.replace('_w','') in AU_LANDMARKS.keys()}
                    
                    if model is not None and feature_cols:
                        user_X = [[current_au_features.get(col, 0.0) for col in feature_cols]]
                        last_calculated_score = model.predict(user_X)[0]
                    else:
                        last_calculated_score = 0.0
            else:
                last_calculated_au_values = {au: 0.0 for au in AU_LANDMARKS.keys()}
                last_calculated_score = 0.0
            
            # --- 텍스트 오버레이 그리기 및 반전 처리 ---
            # 텍스트를 그릴 임시 프레임을 생성합니다.
            temp_text_overlay = np.zeros_like(frame)

            y_offset = 30
            for i, (au_key, au_value) in enumerate(last_calculated_au_values.items()):
                display_au_key = au_key.replace('_w', '')
                cv2.putText(temp_text_overlay, f"{display_au_key}: {au_value:.2f}", (10, y_offset + i * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA) # 초록색 텍스트
            
            cv2.putText(temp_text_overlay, f"Score: {last_calculated_score:.2f}", (10, y_offset + len(last_calculated_au_values) * 30 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA) # 파란색 텍스트

            flipped_text_overlay = cv2.flip(temp_text_overlay, 1) # 텍스트도 반전

            # 텍스트 오버레이를 원본 프레임에 합성
            gray_flipped_text = cv2.cvtColor(flipped_text_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_flipped_text, 1, 255, cv2.THRESH_BINARY)
            
            mask_inv = cv2.bitwise_not(mask)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_inv_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

            frame = cv2.bitwise_and(frame, mask_inv_rgb)
            frame = cv2.add(frame, cv2.bitwise_and(flipped_text_overlay, mask_rgb))
            # --- 텍스트 오버레이 그리기 및 반전 처리 끝 ---

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            frame_count += 1
            
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {str(e)}")
            continue
    
    if global_cap is not None:
        # global_cap.release()
        pass

# --- Flask 라우팅 (기존 흐름 유지) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_mode')
def game_mode():
    return render_template('game_mode.html') # game_mode.html 필요

@app.route('/rehab_mode')
def rehab_mode():
    return render_template('rehab_mode.html') # rehab_mode.html 필요

@app.route('/focus')
def focus():
    return render_template('focus.html') # focus.html 필요

@app.route('/complex')
def complex():
    return render_template('complex.html') # complex.html (선생님 선택 페이지) 필요

@app.route('/complex_fit')
def complex_fit():
    teacher_id = request.args.get('teacher')
    if not teacher_id:
        return redirect(url_for('complex')) # 선생님 ID 없으면 다시 complex 페이지로
    return render_template('complex_fit.html', teacher=teacher_id) # complex_fit.html 필요

@app.route('/feedback')
def feedback():
    return render_template('feedback.html') # feedback.html 필요

@app.route('/video_feed')
def video_feed():
    @after_this_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*' # 모든 출처 허용 (개발용)
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- AU 데이터 제출을 위한 라우트 (수정 완료) ---
@app.route('/submit_au_data_with_image', methods=['POST'])
def submit_au_data_with_image():
    temp_photo_path = None
    teacher_id_from_form = "unspecified_teacher" # 초기화 및 기본값 설정
    photo_storage_url = "upload_failed" # Storage URL 기본값 설정

    try:
        # 클라이언트에서 전송된 데이터 (FormData)
        # AU 값과 Score는 이 라우트에서 직접 계산하는 대신, generate_frames()에서 계산된 최신 값을 사용
        # 혹은, 클라이언트에서 hidden input 등으로 받아올 수도 있습니다.
        # 여기서는 서버에서 가장 최근 계산된 global 변수 값을 사용하도록 하겠습니다.
        
        # 클라이언트(complex_fit.html)에서 'teacher' 파라미터가 아닌 'teacher_id'로 보낸다고 가정
        teacher_id_from_form = request.form.get('teacher_id')
        if not teacher_id_from_form:
            print("[app.py] 경고: 폼에서 선생님 ID가 제공되지 않아 'unspecified_teacher'로 처리됩니다.")
            teacher_id_from_form = "unspecified_teacher" # 기본값 재확인

        photo_file = request.files.get('photo')

        if photo_file and photo_file.filename != '':
            # 1. 이미지 파일 임시 저장
            unique_filename = f"au_capture_{uuid.uuid4()}{os.path.splitext(photo_file.filename)[1]}"
            temp_upload_dir = os.path.join(app.root_path, UPLOAD_FOLDER) # UPLOAD_FOLDER 사용
            temp_photo_path = os.path.join(temp_upload_dir, unique_filename)
            
            os.makedirs(temp_upload_dir, exist_ok=True)
            photo_file.save(temp_photo_path)
            print(f"[app.py] 사진 파일 임시 저장: {temp_photo_path}")

            # 2. Firebase Storage에 업로드
            try:
                blob = bucket.blob(f"au_captures/{teacher_id_from_form}/{unique_filename}")
                blob.upload_from_filename(temp_photo_path)
                blob.make_public() # 공개적으로 접근 가능하게 설정 (보안 규칙도 중요)
                photo_storage_url = blob.public_url
                print(f"[Firebase Storage] 이미지 업로드 성공: {photo_storage_url}")
            except Exception as e:
                print(f"[Firebase Storage] 이미지 업로드 실패: {e}")
                photo_storage_url = "upload_failed" 
        else:
            print("[app.py] 이미지 파일이 전달되지 않았습니다.")
            photo_storage_url = "no_image_provided"

        # 3. AU 값과 Storage URL을 Firestore에 저장
        # generate_frames()에서 계산된 최신 AU 값과 Score를 사용
        # last_calculated_au_values는 딕셔너리이므로 필요에 따라 직렬화
        # 현재는 AU_LANDMARKS.keys()에 해당하는 값들만 저장
        au_values_for_db = {k: float(f"{v:.4f}") for k, v in last_calculated_au_values.items()}
        au_value_overall_score = float(f"{last_calculated_score:.4f}")

        if save_au_data_to_firestore(teacher_id_from_form, au_value_overall_score, photo_storage_url, au_values_for_db):
            return jsonify({
                "status": "success", 
                "message": f"AU 값 {au_value_overall_score}가 성공적으로 Firebase에 저장되었습니다.", 
                "photo_url": photo_storage_url,
                "au_detail": au_values_for_db
            }), 200
        else:
            return jsonify({"status": "error", "message": "AU 값 Firebase 저장 실패"}), 500

    except Exception as e:
        print(f"[app.py] 요청 처리 중 예상치 못한 오류 발생: {e}")
        return jsonify({"status": "error", "message": f"서버 처리 중 오류 발생: {e}"}), 500
    finally:
        if temp_photo_path and os.path.exists(temp_photo_path):
            os.remove(temp_photo_path)
            print(f"[app.py] 임시 사진 파일 삭제됨: {temp_photo_path}")

# --- Firebase 저장 함수 (AU 세부 값 저장을 위해 수정) ---
def save_au_data_to_firestore(teacher_id, overall_score, photo_storage_url, au_detail_values):
    """주어진 AU 값, 선생님 정보, 사진 Storage URL, AU 세부 값을 Firestore 'au_results_by_teacher' 컬렉션에 저장합니다."""
    try:
        doc_ref = db.collection('au_results_by_teacher').add({
            'teacher_id': teacher_id,
            'overall_score': overall_score, # 전체 Score 또는 주요 AU 값
            'au_detail_values': au_detail_values, # 각 AU별 세부 값 딕셔너리
            'photo_url': photo_storage_url, 
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print(f"[Firebase] 데이터 저장 성공: {overall_score} (선생님: {teacher_id}, 사진 URL: {photo_storage_url}) 문서 ID: {doc_ref[1].id}")
        return True
    except Exception as e:
        print(f"[Firebase] AU 데이터 저장 중 오류 발생: {e}")
        return False

# --- 앱 실행 ---
if __name__ == '__main__':
    # 웹캠이 열려있다면 닫기 (서버 재시작 시 필요)
    if global_cap is not None and global_cap.isOpened():
        # global_cap.release()
        pass
    app.run(debug=True, host='0.0.0.0', port=5000)