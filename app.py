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

# atexit 모듈 추가
import atexit 
from urllib.parse import urlparse # URL 파싱을 위해 추가

# Firebase 관련 임포트
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_strong_secret_key_here' # 세션 사용을 위한 secret key 설정 (보안상 강력한 키로 변경 필요)

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
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

stream_face_mesh_instance = None 

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
AU_LANDMARKS = {
    'AU01': [336, 296], 
    'AU02': [334, 298], 
    'AU04': [9, 8],     
    'AU05': [159, 144], 
    'AU06': [205, 206], 
    'AU07': [159, 145], 
    'AU09': [19, 219],  
    'AU10': [11, 10],   
    'AU11': [61, 291],  
    'AU12': [308, 78],  
    'AU14': [41, 13],   
    'AU15': [84, 17],   
    'AU17': [13, 15],   
    'AU20': [32, 262],  
    'AU23': [13, 14],   
    'AU24': [13, 14],   
    'AU25': [13, 14],   
    'AU26': [10, 152],  
    'AU28': [13, 14],   
    'AU43': [145, 374]  
}

def calculate_au(landmarks):
    au_dict = {}
    if not landmarks.any():
        for col in feature_cols: au_dict[col] = 0.0
        return au_dict
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else: au_dict[au] = 0.0
    for au_key in AU_LANDMARKS.keys():
        au_dict[f"{au_key}_w"] = au_dict.get(au_key, 0.0) * 0.8
    final_au_features = {}
    for col in feature_cols: final_au_features[col] = au_dict.get(col, 0.0)
    return final_au_features

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
            static_image_mode=False, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    while global_cap.isOpened():
        success, frame = global_cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- Flask 라우팅 ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/game_mode')
def game_mode(): return render_template('game_mode.html')
@app.route('/rehab_mode')
def rehab_mode(): return render_template('rehab_mode.html')
@app.route('/focus')
def focus(): return render_template('focus.html')
@app.route('/complex')
def complex(): return render_template('complex.html')
@app.route('/complex_fit')
def complex_fit():
    teacher_id = request.args.get('teacher')
    current_set = request.args.get('current_set', type=int, default=1)
    if not teacher_id: return redirect(url_for('complex'))
    return render_template('complex_fit.html', teacher=teacher_id, current_set=current_set) 
@app.route('/next_set_guidance') 
def next_set_guidance():
    teacher_id = request.args.get('teacher_id')
    current_set = request.args.get('current_set', type=int, default=1)
    if not teacher_id: return redirect(url_for('index'))
    return render_template('next_set_guidance.html', teacher_id=teacher_id, next_set_number=current_set)
@app.route('/feedback')
def feedback():
    teacher_id = request.args.get('teacher_id')
    return render_template('feedback.html', teacher_id=teacher_id)
@app.route('/video_feed')
def video_feed():
    @after_this_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 사용자별 임시 데이터 저장을 위한 전역 딕셔너리 (메모리 기반) ---
user_session_data = {} 

# 서버 시작 시 또는 사용자가 특정 세션을 시작할 때 생성되는 고유 ID를 세션에 저장
@app.before_request
def make_session_id():
    if 'user_session_id' not in session:
        session['user_session_id'] = str(uuid.uuid4())
        user_session_data[session['user_session_id']] = [] 

# 서버 종료 시 (Ctrl+C) 실행될 함수
def cleanup_user_data():
    global user_session_data
    print("\n[Cleanup] 서버 종료 감지: 임시 사용자 데이터 삭제를 시작합니다.")
    
    # 각 세션의 데이터를 순회하며 photo_url을 추출하여 Storage에서 삭제
    for session_id, rounds_data in list(user_session_data.items()):
        for round_data in rounds_data:
            photo_url = round_data.get('photo_url')
            if photo_url and photo_url != "no_image_provided" and photo_url != "upload_failed":
                try:
                    # Storage URL에서 파일 경로를 정확하게 추출
                    # bucket.blob()은 'au_captures/teacher_id/filename.jpg'와 같은 버킷 내 경로를 기대
                    
                    # public URL에서 버킷 이름과 파일 경로를 분리하여 추출
                    # 예: https://storage.googleapis.com/YOUR_BUCKET_NAME/path/to/file.jpg
                    # 또는 https://firebasestorage.googleapis.com/v0/b/YOUR_BUCKET_NAME/o/path%2Fto%2Ffile.jpg?...
                    
                    # 가장 간단하고 일반적인 public URL에서 경로 추출 (bucket.name을 사용)
                    if bucket.name in photo_url: # 버킷 이름이 URL에 포함되어 있는지 확인
                        # URL에서 버킷 이름 뒤의 경로만 추출
                        # 예: "smilefit-350ea.firebasestorage.app/au_captures/sophia/..."
                        # 또는 "storage.googleapis.com/smilefit-350ea.appspot.com/au_captures/sophia/..."
                        
                        # Firebase Storage의 일반적인 URL 형식:
                        # https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{path_to_file}?alt=media
                        # 또는 레거시 GCS URL:
                        # https://storage.googleapis.com/{bucket_name}/{path_to_file}
                        
                        # urlparse를 사용하여 path를 추출한 후, 버킷 이름을 제거
                        parsed_url = urlparse(photo_url)
                        full_path_in_url = parsed_url.path.lstrip('/') # /v0/b/bucket_name/o/path/to/file
                        
                        # path_to_file 부분만 추출
                        # 예: 'v0/b/smilefit-350ea.firebasestorage.app/o/au_captures/sophia/...'
                        # 버킷 이름 이후 'o/' 부분을 찾아 그 뒤부터 경로로 사용
                        if '/o/' in full_path_in_url:
                            file_path_in_storage = full_path_in_url.split('/o/', 1)[1]
                        else: # 구글 스토리지 레거시 URL 또는 다른 형식
                            file_path_in_storage = full_path_in_url.replace(f"{bucket.name}/", '', 1)
                        
                        # URL 인코딩된 슬래시 '%2F'를 다시 '/'로 디코딩
                        file_path_in_storage = file_path_in_storage.replace('%2F', '/')

                        blob = bucket.blob(file_path_in_storage)
                        if blob.exists(): 
                            blob.delete()
                            print(f"[Cleanup] Storage 파일 삭제 성공: {file_path_in_storage}")
                        else:
                            print(f"[Cleanup] Storage 파일 없음 (이미 삭제되었거나 경로 오류): {file_path_in_storage}")
                    else:
                        print(f"[Cleanup] Storage 파일 삭제 실패: URL 형식 불일치 (버킷 이름 없음) {photo_url}")

                except Exception as e:
                    print(f"[Cleanup] Storage 파일 삭제 실패 ({file_path_in_storage if 'file_path_in_storage' in locals() else photo_url}): {e}")
    
    user_session_data = {} 
    print("[Cleanup] 모든 임시 사용자 데이터가 메모리에서 삭제되었습니다.")
# 앱 종료 시 cleanup_user_data 함수를 실행하도록 등록
atexit.register(cleanup_user_data)


# --- AU 데이터 제출을 위한 라우트 (Firebase 저장 대신 메모리 임시 저장) ---
@app.route('/submit_au_data_with_image', methods=['POST'])
def submit_au_data_with_image():
    temp_photo_path = None
    teacher_id_from_form = "unspecified_teacher" 
    photo_storage_url = "upload_failed" 
    
    current_au_features = {}
    current_score = 0.0

    try:
        teacher_id_from_form = request.form.get('teacher_id')
        round_number_from_form = request.form.get('round_number', type=int)

        photo_file = request.files.get('photo')

        # Firebase Storage에 이미지 업로드 (데이터는 메모리에만 저장되지만, 이미지는 Storage에 임시로 업로드)
        if photo_file and photo_file.filename != '':
            unique_filename = f"au_capture_temp_{uuid.uuid4()}{os.path.splitext(photo_file.filename)[1]}"
            temp_photo_path = os.path.join(UPLOAD_FOLDER, unique_filename) 
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            photo_file.save(temp_photo_path)
            
            try:
                blob = bucket.blob(f"au_captures/{teacher_id_from_form}/{unique_filename}")
                blob.upload_from_filename(temp_photo_path)
                blob.make_public() # 공개적으로 접근 가능하게 설정
                photo_storage_url = blob.public_url
                print(f"[Firebase Storage] 이미지 업로드 성공: {photo_storage_url}")
            except Exception as e:
                print(f"[Firebase Storage] 이미지 업로드 실패: {e}")
                photo_storage_url = "upload_failed"
        else:
            print("[app.py] 이미지 파일이 전달되지 않았습니다.")
            photo_storage_url = "no_image_provided"

        # MediaPipe 및 XGBoost 로직은 그대로 유지
        img = cv2.imread(temp_photo_path) # 임시 저장된 파일에서 읽어옴
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb) 
            if results.multi_face_landmarks:
                landmarks = np.array([(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in results.multi_face_landmarks[0].landmark], dtype=np.float32)
                current_au_features = calculate_au(landmarks)
                if model is not None and feature_cols:
                    user_X = [[current_au_features.get(col, 0.0) for col in feature_cols]]
                    current_score = model.predict(user_X)[0]
                else: current_score = 0.0
            else: print("[app.py] 이미지에서 얼굴 랜드mark를 찾을 수 없습니다.")
        else: print("[app.py] 임시 저장된 이미지를 읽을 수 없습니다.")

        au_values_for_db = {k: float(f"{current_au_features.get(k, 0.0):.4f}") for k in feature_cols if not k.endswith('_w')}
        overall_score_for_db = float(f"{current_score:.4f}")

        # 사용자 세션 ID를 가져와 데이터 저장
        current_session_id = session.get('user_session_id')
        if not current_session_id:
            print("[submit_au_data_with_image] 경고: 세션 ID 없음, 데이터 저장 불가.")
            return jsonify({"status": "error", "message": "세션 ID가 없어 데이터를 저장할 수 없습니다."}), 500

        # 데이터 저장 로직을 메모리 기반으로 변경
        round_data = {
            'teacher_id': teacher_id_from_form,
            'round_number': round_number_from_form,
            'overall_score': overall_score_for_db,
            'au_detail_values': au_values_for_db,
            'photo_url': photo_storage_url, 
            'timestamp': datetime.now().isoformat() # 서버 시간 기록 (Firebase Timestamp 대신)
        }
        
        if current_session_id not in user_session_data:
             user_session_data[current_session_id] = []
        user_session_data[current_session_id].append(round_data)
        
        print(f"[Memory Storage] 사용자 데이터 저장 성공: 세션 {current_session_id}, 라운드 {round_number_from_form}, 점수 {overall_score_for_db}")

        return jsonify({
            "status": "success",
            "message": "데이터가 메모리에 임시 저장되었습니다.",
            "photo_url": photo_storage_url,
            "au_detail": au_values_for_db,
            "overall_score": overall_score_for_db
        }), 200

    except Exception as e:
        print(f"[app.py] 요청 처리 중 예상치 못한 오류 발생: {e}")
        return jsonify({"status": "error", "message": f"서버 처리 중 오류 발생: {e}"}), 500
    finally:
        if temp_photo_path and os.path.exists(temp_photo_path):
            os.remove(temp_photo_path)
            # print(f"[app.py] 임시 사진 파일 삭제됨: {temp_photo_path}") # 디버그 로그 필요시 활성화

# --- 피드백 페이지에서 임시 데이터를 가져오는 새로운 라우트 추가 ---
@app.route('/get_user_feedback_data')
def get_user_feedback_data():
    current_session_id = session.get('user_session_id')
    if not current_session_id or current_session_id not in user_session_data:
        return jsonify({"user_data": []}), 200 
    
    user_data = user_session_data.get(current_session_id, [])
    user_data.sort(key=lambda x: x.get('round_number', 0))
    
    return jsonify({"user_data": user_data}), 200

# --- 선생님 데이터 업로드 함수 (기존과 동일) ---
def upload_teacher_au_data():
    """선생님 이미지 파일에서 AU 값을 추출하여 Firestore 'teacher_au_references' 컬렉션에 저장합니다."""
    teachers = ['emma', 'olivia', 'sophia']
    total_rounds_per_teacher = 10 

    print("\n--- 선생님 AU 데이터 업로드 시작 ---")
    for teacher_id in teachers:
        for i in range(1, total_rounds_per_teacher + 1):
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
                results = face_mesh.process(img_rgb) 

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
                    else: score = 0.0
                else: print(f"경고: {teacher_id}의 {i} 라운드 이미지에서 얼굴 랜드mark를 찾을 수 없습니다.")
                    
                au_values_for_db = {k: float(f"{au_features.get(k, 0.0):.4f}") for k in feature_cols if not k.endswith('_w')}

                doc_ref = db.collection('teacher_au_references').add({
                    'teacher_id': teacher_id,
                    'round_number': i, 
                    'overall_score': float(f"{score:.4f}"),
                    'au_detail_values': au_values_for_db,
                    'image_path_static': f'/static/images/teachers/{teacher_id}/{teacher_id}{i}.png',
                    'timestamp': firestore.SERVER_TIMESTAMP 
                })
                print(f"성공: {teacher_id} 라운드 {i} AU 데이터 저장 완료. 점수: {score:.4f}")

            except Exception as e:
                print(f"오류: {teacher_id} 라운드 {i} AU 데이터 처리 중 오류 발생: {e}")
    print("--- 선생님 AU 데이터 업로드 완료 ---")


# --- 앱 실행 ---
if __name__ == '__main__':
    if global_cap is not None and global_cap.isOpened():
        global_cap.release() 
        global_cap = None 

    # upload_teacher_au_data() # <-- 실행 후 주석 처리하거나 제거하세요!

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)