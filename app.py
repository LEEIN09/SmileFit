import cv2
import mediapipe as mp
import xgboost as xgb
import numpy as np
import json
import base64
import uuid
import os
import sys
from datetime import datetime
from flask import Flask, render_template, send_from_directory, request, jsonify, Response, session, redirect, url_for
from flask import Response, after_this_request

# Firebase 관련 임포트
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_strong_secret_key_here' # 실제 배포 시에는 반드시 강력한 키로 변경하세요.

# --- Firebase 초기화 ---
service_account_key_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'smilefit-350ea.firebasestorage.app'
        })
        print("[Firebase] Firebase 앱이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"[Firebase] Firebase 앱 초기화 실패: {e}")
        sys.exit(1)

db = firestore.client()
bucket = storage.bucket()

# 임시 파일 저장 디렉토리 설정
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- MediaPipe 및 XGBoost 모델 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
# 선생님 이미지 처리용 FaceMesh (static_image_mode=True)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 웹캠 스트리밍용 FaceMesh (static_image_mode=False) - generate_frames에서 사용
stream_face_mesh_instance = None # 전역 변수로 선언하여 재사용

model = None
feature_cols = []
try:
    model = xgb.XGBRegressor()
    model.load_model('expression_similarity_model.json')
    with open('feature_cols.json', 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    print("XGBoost 모델 및 특징 파일 로드 완료.")
except Exception as e:
    print(f"XGBoost 모델 또는 특징 파일 로드 실패: {e}")

# --- AU 계산을 위한 랜드마크 인덱스 (FACS 기준, MediaPipe 랜드마크 기반) ---
# feature_cols.json에 있는 모든 AU에 맞춰 확장했습니다.
AU_LANDMARKS = {
    'AU01': [336, 296], # Inner Brow Raiser (이마 안쪽 올림) - 눈썹 안쪽 끝 랜드마크
    'AU02': [334, 298], # Outer Brow Raiser (이마 바깥쪽 올림) - 눈썹 바깥쪽 끝 랜드마크
    'AU04': [9, 8],     # Brow Lowerer (눈썹 내림) - 미간 랜드마크
    'AU05': [159, 144], # Upper Lid Raiser (윗눈꺼풀 올림) - 윗눈꺼풀 중앙과 아랫눈꺼풀 중앙 랜드마크 (눈 크게 뜨기)
    'AU06': [205, 206], # Cheek Raiser (광대 올림) - 눈 밑 광대 랜드마크
    'AU07': [159, 145], # Lid Tightener (눈꺼풀 조이기) - 눈꺼풀 사이의 랜드마크 (실제로는 눈꼬리 움직임)
    'AU09': [19, 219],  # Nose Wrinkler (코 찡그림) - 콧방울 랜드마크
    'AU10': [11, 10],   # Upper Lip Raiser (윗입술 올림) - 윗입술 중앙 랜드마크
    'AU11': [61, 291],  # Nasolabial Furrow Deepener (코입술 주름 깊게 하기) - 콧망울 옆 주름 랜드마크
    'AU12': [308, 78],  # Lip Corner Puller (입꼬리 당기기) - 입꼬리 랜드마크
    'AU14': [41, 13],   # Dimpler (보조개 만들기) - 입꼬리 옆 보조개 랜드마크 (얼굴 중앙선 기준)
    'AU15': [84, 17],   # Lip Corner Depressor (입꼬리 내리기) - 입꼬리 아래 랜드마크
    'AU17': [13, 15],   # Chin Raiser (턱 올리기) - 윗입술 중앙과 턱 중앙 랜드마크
    'AU20': [32, 262],  # Lip Stretcher (입술 늘리기) - 입술 좌우 끝 랜드마크
    'AU23': [13, 14],   # Lip Tightener (입술 조이기) - 입술 중앙 상하 (입술 누르기)
    'AU24': [13, 14],   # Lip Pressor (입술 누르기) - AU23과 유사 (입술 중앙 상하)
    'AU25': [13, 14],   # Lips Part (입술 벌리기) - 입술 중앙 상하 랜드마크
    'AU26': [10, 152],  # Jaw Drop (턱 내리기) - 윗입술 중앙과 턱 끝 랜드마크
    'AU28': [13, 14],   # Lip Suck (입술 빨기) - AU23/24/25와 유사 (입술 내부로 당기기, 랜드마크 거리로 측정 어려움)
    'AU43': [145, 374]  # Eye Closure (눈 감기) - 눈꺼풀 상하 랜드마크 (눈 감는 정도)
}

# 보충 설명: AU23, AU24, AU25, AU28은 MediaPipe 랜드마크 거리만으로 정확히 측정하기 어려울 수 있습니다.
# 특히 AU28 (Lip Suck)은 입술의 부피 변화에 가깝습니다.
# AU23 (Lip Tightener)과 AU24 (Lip Pressor)는 AU25 (Lips Part)와 유사한 랜드마크를 사용하므로,
# 실제로는 이들의 차이를 미세하게 구분하기 어렵습니다. 모델이 AU_LANDMARKS를 통해 측정된
# 거리 값들 사이의 복잡한 관계를 학습해야 합니다.

def calculate_au(landmarks):
    """MediaPipe 랜드마크 기반 AU 계산 (거리 기반)"""
    au_dict = {}

    if not landmarks.any():
        for col in feature_cols:
            au_dict[col] = 0.0
        return au_dict

    # AU_LANDMARKS에 정의된 AU만 계산
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else:
            # 랜드마크 인덱스가 유효하지 않거나 얼굴이 감지되지 않으면 0으로 처리 (이는 정상적인 상황이 아님)
            # 보통 랜드마크가 없으면 landmarks.any()에서 걸러지므로, 여기에 도달하면 인덱스 오류일 가능성 높음
            au_dict[au] = 0.0 

    # 가중치 컬럼 생성 (임시 0.8 곱함) - 이 부분은 모델 학습 방식에 따라 조정 필요
    for au_key in AU_LANDMARKS.keys():
        au_dict[f"{au_key}_w"] = au_dict.get(au_key, 0.0) * 0.8

    # feature_cols에 정의된 모든 AU 특징을 최종 딕셔너리에 포함 (AU_LANDMARKS에 없으면 0.0)
    final_au_features = {}
    for col in feature_cols:
        final_au_features[col] = au_dict.get(col, 0.0)

    return final_au_features


# 웹캠 객체를 전역 변수로 관리
global_cap = None

def generate_frames():
    global global_cap, stream_face_mesh_instance
    if global_cap is None or not global_cap.isOpened():
        global_cap = cv2.VideoCapture(0)
        if not global_cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

    if stream_face_mesh_instance is None:
        stream_face_mesh_instance = mp_face_mesh.FaceMesh(
            static_image_mode=False, # 라이브 스트리밍용
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    while global_cap.isOpened():
        success, frame = global_cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1) # 좌우 반전 (거울 효과)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    # 스트리밍 종료 시 리소스 해제 (선택 사항, 웹 앱에서는 계속 켜두는 경우 많음)
    # if stream_face_mesh_instance:
    #     stream_face_mesh_instance.close()


# --- Flask 라우팅 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_mode')
def game_mode():
    return render_template('game_mode.html')

@app.route('/rehab_mode')
def rehab_mode():
    return render_template('rehab_mode.html')

@app.route('/focus')
def focus():
    return render_template('focus.html')

@app.route('/complex')
def complex():
    return render_template('complex.html')

@app.route('/complex_fit')
def complex_fit():
    teacher_id = request.args.get('teacher')
    if not teacher_id:
        return redirect(url_for('complex'))
    return render_template('complex_fit.html', teacher=teacher_id)

@app.route('/feedback')
def feedback():
    # complex_fit에서 넘어올 때 teacher_id를 파라미터로 받을 것임
    teacher_id = request.args.get('teacher_id')
    # 이 teacher_id를 feedback.html 템플릿으로 전달합니다.
    return render_template('feedback.html', teacher_id=teacher_id)

@app.route('/video_feed')
def video_feed():
    @after_this_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- AU 데이터 제출을 위한 라우트 ---
@app.route('/submit_au_data_with_image', methods=['POST'])
def submit_au_data_with_image():
    temp_photo_path = None
    teacher_id_from_form = "unspecified_teacher" # 기본값 설정
    photo_storage_url = "upload_failed"
    
    current_au_features = {}
    current_score = 0.0

    try:
        teacher_id_from_form = request.form.get('teacher_id')
        if not teacher_id_from_form:
            print("[app.py] 경고: 폼에서 선생님 ID가 제공되지 않아 'unspecified_teacher'로 처리됩니다.")
            teacher_id_from_form = "unspecified_teacher"

        photo_file = request.files.get('photo')

        if photo_file and photo_file.filename != '':
            unique_filename = f"au_capture_{uuid.uuid4()}{os.path.splitext(photo_file.filename)[1]}"
            temp_upload_dir = os.path.join(app.root_path, UPLOAD_FOLDER)
            temp_photo_path = os.path.join(temp_upload_dir, unique_filename)
            
            os.makedirs(temp_upload_dir, exist_ok=True)
            photo_file.save(temp_photo_path)
            print(f"[app.py] 사진 파일 임시 저장: {temp_photo_path}")

            img = cv2.imread(temp_photo_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(img_rgb) # static_image_mode=True로 초기화된 face_mesh 사용

                if results.multi_face_landmarks:
                    landmarks = np.array([
                        (lm.x * img.shape[1], lm.y * img.shape[0])
                        for lm in results.multi_face_landmarks[0].landmark
                    ], dtype=np.float32)
                    
                    current_au_features = calculate_au(landmarks)
                    
                    if model is not None and feature_cols:
                        user_X = [[current_au_features.get(col, 0.0) for col in feature_cols]]
                        current_score = model.predict(user_X)[0]
                    else:
                        current_score = 0.0
                else:
                    print("[app.py] 이미지에서 얼굴 랜드마크를 찾을 수 없습니다.")
            else:
                print("[app.py] 임시 저장된 이미지를 읽을 수 없습니다.")

            try:
                blob = bucket.blob(f"au_captures/{teacher_id_from_form}/{unique_filename}")
                blob.upload_from_filename(temp_photo_path)
                blob.make_public()
                photo_storage_url = blob.public_url
                print(f"[Firebase Storage] 이미지 업로드 성공: {photo_storage_url}")
            except Exception as e:
                print(f"[Firebase Storage] 이미지 업로드 실패: {e}")
                photo_storage_url = "upload_failed"
        else:
            print("[app.py] 이미지 파일이 전달되지 않았습니다.")
            photo_storage_url = "no_image_provided"

        # 최종 저장할 AU 값은 AU_LANDMARKS에 정의된 것만 (혹은 feature_cols에 있는 모든 것)
        # feature_cols에 있는 모든 AU를 저장하도록 변경하여 일관성 유지
        au_values_for_db = {k: float(f"{current_au_features.get(k, 0.0):.4f}") for k in feature_cols if not k.endswith('_w')}


        overall_score_for_db = float(f"{current_score:.4f}")

        if save_user_au_data_to_firestore(teacher_id_from_form, overall_score_for_db, photo_storage_url, au_values_for_db):
            return jsonify({
                "status": "success",
                "message": f"AU 값 {overall_score_for_db}가 성공적으로 Firebase에 저장되었습니다.",
                "photo_url": photo_storage_url,
                "au_detail": au_values_for_db,
                "overall_score": overall_score_for_db
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

# --- Firebase 저장 함수 (사용자 데이터) ---
def save_user_au_data_to_firestore(teacher_id, overall_score, photo_storage_url, au_detail_values):
    """주어진 AU 값, 선생님 정보, 사진 Storage URL, AU 세부 값을 Firestore 'user_au_results' 컬렉션에 저장합니다."""
    try:
        doc_ref = db.collection('user_au_results').add({
            'teacher_id': teacher_id,
            'overall_score': overall_score,
            'au_detail_values': au_detail_values,
            'photo_url': photo_storage_url,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print(f"[Firebase] 사용자 데이터 저장 성공: {overall_score} (선생님: {teacher_id}, 사진 URL: {photo_storage_url}) 문서 ID: {doc_ref[1].id}")
        return True
    except Exception as e:
        print(f"[Firebase] 사용자 AU 데이터 저장 중 오류 발생: {e}")
        return False

# --- 선생님 데이터 업로드 함수 추가 ---
def upload_teacher_au_data():
    """선생님 이미지 파일에서 AU 값을 추출하여 Firestore 'teacher_au_references' 컬렉션에 저장합니다."""
    teachers = ['emma', 'olivia', 'sophia']
    total_rounds = 10 # 선생님 이미지 라운드 수 (1부터 10까지)

    print("\n--- 선생님 AU 데이터 업로드 시작 ---")
    for teacher_id in teachers:
        for i in range(1, total_rounds + 1):
            image_path = os.path.join(app.static_folder, 'images', 'teachers', teacher_id, f'{teacher_id}{i}.png')
            
            if not os.path.exists(image_path):
                print(f"경고: {image_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue

            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"경고: {image_path} 이미지를 읽을 수 없습니다. 건너뜁니다.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(img_rgb) # static_image_mode=True로 초기화된 face_mesh 사용

                au_features = {}
                score = 0.0

                if results.multi_face_landmarks:
                    landmarks = np.array([
                        (lm.x * img.shape[1], lm.y * img.shape[0])
                        for lm in results.multi_face_landmarks[0].landmark
                    ], dtype=np.float32)
                    
                    au_features = calculate_au(landmarks)
                    
                    if model is not None and feature_cols:
                        teacher_X = [[au_features.get(col, 0.0) for col in feature_cols]]
                        score = model.predict(teacher_X)[0]
                    else:
                        score = 0.0
                else:
                    print(f"경고: {teacher_id}의 {i} 라운드 이미지에서 얼굴 랜드마크를 찾을 수 없습니다.")
                    
                # Firestore에 저장
                # AU_LANDMARKS에 정의된 AU와 그 _w 버전만 저장
                au_values_for_db = {k: float(f"{au_features.get(k, 0.0):.4f}") for k in feature_cols if not k.endswith('_w')}

                doc_ref = db.collection('teacher_au_references').add({
                    'teacher_id': teacher_id,
                    'round_number': i,
                    'overall_score': float(f"{score:.4f}"),
                    'au_detail_values': au_values_for_db,
                    'image_path_static': f'/static/images/teachers/{teacher_id}/{teacher_id}{i}.png',
                    'timestamp': firestore.SERVER_TIMESTAMP # 업로드 시간 기록
                })
                print(f"성공: {teacher_id} 라운드 {i} AU 데이터 저장 완료. 점수: {score:.4f}")

            except Exception as e:
                print(f"오류: {teacher_id} 라운드 {i} AU 데이터 처리 중 오류 발생: {e}")
    print("--- 선생님 AU 데이터 업로드 완료 ---")


# --- 앱 실행 ---
if __name__ == '__main__':
    # 웹캠이 열려있다면 닫기 (서버 재시작 시 필요)
    if global_cap is not None and global_cap.isOpened():
        global_cap.release() # 서버 종료 시 웹캠 자원 명시적 해제
        global_cap = None # None으로 설정하여 다음 시작 시 재초기화되도록 함



    app.run(debug=True, host='0.0.0.0', port=5000)