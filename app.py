import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
import sys
from datetime import datetime
# Response는 Flask에서 직접 임포트해야 합니다.
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for # Response 임포트 추가

import atexit
from urllib.parse import urlparse

import pandas as pd 

import tensorflow as tf
from tensorflow.keras.models import load_model

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_strong_secret_key_here' 

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

bucket = storage.bucket() 

# 임시 파일 저장 디렉토리 설정
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- MediaPipe 및 Autoencoder 모델 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
# face_mesh 인스턴스 초기화 (이전 코드에서 누락되거나 변경되었을 수 있음)
face_mesh_instance = mp_face_mesh.FaceMesh( 
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
# generate_frames에서 사용되는 스트리밍용 인스턴스
stream_face_mesh_instance = None 


encoder_model = None
try:
    encoder_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'autoencoder_encoder', 'encoder_model.keras')
    encoder_model = load_model(encoder_model_path)
    print(f"Autoencoder 인코더 모델 로드 완료: {encoder_model_path}")
except Exception as e:
    print(f"Autoencoder 인코더 모델 로드 실패: {e}")
    sys.exit(1)

feature_cols = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43", 
    "AU01_w", "AU02_w", "AU04_w", "AU05_w", "AU06_w", "AU07_w", "AU09_w", "AU10_w",
    "AU11_w", "AU12_w", "AU14_w", "AU15_w", "AU17_w", "AU20_w", "AU23_w", "AU24_w",
    "AU25_w", "AU26_w", "AU28_w", "AU43_w"
]
print(f"AU 특징 목록 (Autoencoder 입력): {len(feature_cols)}개")


# --- AU 계산을 위한 랜드마크 인덱스 (FACS 기준, MediaPipe 랜드마크 기반) ---
AU_LANDMARKS = {
    'AU01': [336, 296], 'AU02': [334, 298], 'AU04': [9, 8], 'AU05': [159, 144], 
    'AU06': [205, 206], 'AU07': [159, 145], 'AU09': [19, 219], 'AU10': [11, 10], 
    'AU11': [61, 291], 'AU12': [308, 78], 'AU14': [41, 13], 'AU15': [84, 17], 
    'AU17': [13, 15], 'AU20': [32, 262], 'AU23': [13, 14], 'AU24': [13, 14], 
    'AU25': [13, 14], 'AU26': [10, 152], 'AU28': [13, 14], 'AU43': [145, 374] 
}

def extract_normalized_au_features(landmarks):
    """
    랜드마크를 기반으로 모든 AU (이진 및 가중치)를 계산하고 정규화하여 딕셔너리로 반환합니다.
    (Autoencoder의 40개 피처 입력에 사용)
    """
    au_dict = {}
    
    if not isinstance(landmarks, np.ndarray) or not landmarks.any():
        return {col: 0.0 for col in feature_cols} 


    # --- 1. 얼굴의 기준 길이 계산 (예: 눈꼬리-눈꼬리 거리) ---
    try:
        reference_point1 = landmarks[133] 
        reference_point2 = landmarks[362]
        
        face_scale_distance = np.linalg.norm(reference_point1 - reference_point2)
        
        if face_scale_distance < 1e-6: 
            face_scale_distance = 1.0 
    except IndexError:
        print("경고: 기준 랜드마크 인덱스(133 또는 362)가 유효하지 않습니다. 정규화 없이 진행합니다.")
        face_scale_distance = 1.0 

    # 2. 모든 AU 거리 계산
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else:
            au_dict[au] = 0.0 

    # 3. AU 값 정규화 (기준 길이로 나눔)
    normalized_au_dict = {}
    for au_key, value in au_dict.items():
        normalized_au_dict[au_key] = value / face_scale_distance 
        
    # 4. _w (weighted) AU 계산 (정규화된 값에 가중치 적용)
    for au_key in AU_LANDMARKS.keys():
        normalized_au_dict[f"{au_key}_w"] = normalized_au_dict.get(au_key, 0.0) * 0.8
    
    # 5. 모델이 기대하는 최종 피처(feature_cols, 40개)만 선택하여 딕셔너리로 반환
    final_au_features_dict = {}
    for col in feature_cols: 
        # pd.isna()를 사용하여 NaN 값 처리
        au_value = normalized_au_dict.get(col, 0.0)
        if pd.isna(au_value): # NumPy NaN을 포함하여 Pandas의 NaN을 확인
            final_au_features_dict[col] = 0.0
        else:
            final_au_features_dict[col] = float(f"{au_value:.4f}")

    return final_au_features_dict


# --- 임베딩 벡터 간 유사도 계산 함수 ---
# app.py 내 calculate_similarity_score 함수

def calculate_similarity_score(embedding1, embedding2):
    """
    두 임베딩 벡터 간의 유클리드 거리를 기반으로 유사도 점수를 계산합니다.
    점수가 80~90점대로 높게 나오도록 배율을 대폭 조정합니다.
    """
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    
    # --- 수정된 부분: k_factor를 대폭 줄여 점수 '뻥튀기' ---
    # 이전 k_factor는 0.5였습니다. 이를 0.1 또는 0.05 정도로 대폭 줄여보겠습니다.
    # 이 값이 작을수록 점수가 거리 변화에 극도로 덜 민감해져 전반적으로 매우 높게 나옵니다.
    k_factor = 0.05 # <-- 이 값을 0.05 또는 0.1로 설정하여 테스트해보세요! (매우 후한 점수)

    similarity_score = 100 * np.exp(-k_factor * distance)
    
    # 점수가 0 미만이 되지 않도록 보정
    similarity_score = max(0, similarity_score) 
    
    return similarity_score

global_cap = None
# `generate_frames` 함수는 스트리밍용 MediaPipe 인스턴스를 사용합니다.
def generate_frames():
    global global_cap, stream_face_mesh_instance
    if global_cap is None or not global_cap.isOpened():
        global_cap = cv2.VideoCapture(0)
        if not global_cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
    if stream_face_mesh_instance is None:
        # stream_face_mesh_instance는 여기에서 초기화 (static_image_mode=False)
        stream_face_mesh_instance = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    while global_cap.isOpened():
        success, frame = global_cap.read()
        if not success: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = stream_face_mesh_instance.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                
                try:
                    au_features_dict = extract_normalized_au_features(landmarks_np) 
                    au_features_for_model = np.array([au_features_dict[col] for col in feature_cols], dtype=np.float32).reshape(1, -1)
                    
                    user_embedding = encoder_model.predict(au_features_for_model)[0] 
                    
                    prediction_score_display = user_embedding[0] * 100 
                    
                    cv2.putText(frame, f"Score: {prediction_score_display:.2f}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"실시간 예측 중 오류 발생: {e}")

        frame = cv2.flip(frame, 1) 
        ret, buffer = cv2.imencode('.jpg', frame)
        # from flask import Response는 라우팅 함수 외부에 있어야 합니다.
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

# --- /get_user_feedback_data 라우트 ---
@app.route('/get_user_feedback_data', methods=['GET'])
def get_user_feedback_data():
    user_session_id = session.get('user_session_id')
    teacher_id = request.args.get('teacher_id')
    
    user_data = []
    teacher_references = []

    # 1. 사용자 데이터 가져오기 (메모리 기반)
    if user_session_id and user_session_id in user_session_data:
        user_data = user_session_data.get(user_session_id, [])
        user_data.sort(key=lambda x: x.get('round_number', 0))
    else:
        print(f"[get_user_feedback_data] 사용자 세션 데이터 없음: {user_session_id}")

    # 2. 선생님 기준 데이터 가져오기 (메모리 _teacher_references_in_memory에서)
    if teacher_id and teacher_id in _teacher_references_in_memory:
        teacher_references = _teacher_references_in_memory[teacher_id]
        print(f"[get_user_feedback_data] 선생님 기준 데이터 {len(teacher_references)}개 로드 완료 (메모리, teacher_id: {teacher_id})")
    else:
        print(f"[get_user_feedback_data] 경고: teacher_id '{teacher_id}'에 해당하는 선생님 데이터를 메모리에서 찾을 수 없습니다.")

    return jsonify({
        'status': 'success',
        'user_data': user_data,
        'teacher_references': teacher_references
    }), 200

@app.route('/video_feed')
def video_feed():
    # from flask import Response는 라우팅 함수 외부에 있어야 합니다.
    # @after_this_request는 Flask에서 직접 임포트해야 합니다.
    from flask import after_this_request # <-- 이 라인 추가
    @after_this_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 사용자별 임시 데이터 저장을 위한 전역 딕셔너리 (메모리 기반) ---
user_session_data = {} 

# --- 선생님 데이터를 위한 전역 변수 ---
_teacher_references_in_memory = {} 

# 서버 시작 시 또는 사용자가 특정 세션을 시작할 때 생성되는 고유 ID를 세션에 저장
@app.before_request
def make_session_id():
    if 'user_session_id' not in session:
        session['user_session_id'] = str(uuid.uuid4())
        user_session_data[session['user_session_id']] = [] 
        print(f"[make_session_id] 새 세션 ID 생성: {session['user_session_id']}") 

# --- submit_au_data_with_image 라우트 ---
@app.route('/submit_au_data_with_image', methods=['POST'])
def submit_au_data_with_image():
    user_session_id = session.get('user_session_id')
    teacher_id = request.form.get('teacher_id')
    round_number = request.form.get('round_number', type=int)
    image_file = request.files.get('image_data') 
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_filename = f"au_capture_temp_{uuid.uuid4()}.jpg"
    photo_path = os.path.join(UPLOAD_FOLDER, photo_filename) 
    
    photo_storage_url = "no_image_provided" 
    
    try:
        if image_file and image_file.filename != '': 
            os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
            image_file.save(photo_path) 
            
            image_np = cv2.imread(photo_path)
            if image_np is None:
                raise ValueError("이미지 파일을 읽을 수 없거나 내용이 비어있습니다.")
            
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = face_mesh_instance.process(image_rgb) 
            
            landmarks_np = None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                    break 
            
            if landmarks_np is not None:
                try:
                    au_features_dict = extract_normalized_au_features(landmarks_np) 
                    au_features_for_model = np.array([au_features_dict[col] for col in feature_cols], dtype=np.float32).reshape(1, -1)
                    
                    user_embedding = encoder_model.predict(au_features_for_model)[0] 

                    teacher_embedding = None
                    if teacher_id in _teacher_references_in_memory:
                        teacher_rounds_data = _teacher_references_in_memory[teacher_id]
                        matching_teacher_data = next((item for item in teacher_rounds_data if item["round_number"] == round_number), None)
                        
                        if matching_teacher_data and 'embedding' in matching_teacher_data:
                            teacher_embedding = np.array(matching_teacher_data['embedding'])
                    
                    prediction_score = 0.0 
                    if teacher_embedding is not None:
                        prediction_score = calculate_similarity_score(user_embedding, teacher_embedding)
                        print(f"유사도 점수 계산 완료: {prediction_score:.2f}")
                    else:
                        print(f"경고: 라운드 {round_number}의 선생님 임베딩 데이터를 찾을 수 없어 유사도 계산 불가.")
                        prediction_score = 0.0 


                    blob = bucket.blob(f"au_captures/{teacher_id}/{photo_filename}")
                    blob.upload_from_filename(photo_path, content_type='image/jpeg')
                    blob.make_public() 
                    photo_storage_url = blob.public_url 
                    print(f"[Firebase Storage] 이미지 업로드 성공: {photo_storage_url}")

                    if user_session_id not in user_session_data:
                        user_session_data[user_session_id] = []
                        print(f"[Memory] 세션 ID {user_session_id}를 위한 리스트 초기화 (submit_au_data_with_image)") 
                    
                    user_session_data[user_session_id].append({
                        'round_number': round_number,
                        'predicted_score': float(prediction_score),
                        'photo_url': photo_storage_url,
                        'user_embedding': user_embedding.tolist(), 
                        'au_features': au_features_dict # AU features 딕셔너리로 저장
                    })
                    print(f"[Memory] 사용자 데이터 저장 성공: 세션 {user_session_id}, 라운드 {round_number}, 점수 {float(prediction_score):.2f}")


                    return jsonify({'status': 'success', 'predicted_score': float(prediction_score), 'photo_url': photo_storage_url})

                except Exception as e:
                    print(f"[app.py] 모델 예측, Storage 또는 기타 작업 중 오류 발생: {e}")
                    photo_storage_url = "upload_failed" 
                    return jsonify({'status': 'error', 'message': f'예측/업로드/저장 실패: {e}'}), 500
            else:
                print("[app.py] 얼굴 랜드마크 미감지. 예측 건너뜀.")
                return jsonify({'status': 'no_face_detected', 'message': '얼굴을 감지할 수 없습니다. 다시 시도해주세요.'}), 400
        else:
            print("[app.py] 이미지 파일이 전달되지 않았습니다 (image_file이 비어있음).")
            return jsonify({'status': 'error', 'message': '이미지 데이터가 제공되지 않았습니다.'}), 400
    except Exception as e:
        print(f"[app.py] 요청 처리 중 예상치 못한 최상위 오류 발생: {e}")
        return jsonify({'status': 'error', 'message': f'서버 오류: {e}'}), 500
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)


# 서버 종료 시 (Ctrl+C) 실행될 함수
def cleanup_user_data():
    global user_session_data
    print("\n[Cleanup] 서버 종료 감지: 임시 사용자 데이터 삭제를 시작합니다.")
    
    for session_id, rounds_data in list(user_session_data.items()):
        for round_data in rounds_data:
            photo_url = round_data.get('photo_url')
            if photo_url and photo_url != "no_image_provided" and photo_url != "upload_failed":
                try:
                    parsed_url = urlparse(photo_url)
                    full_path_in_url = parsed_url.path.lstrip('/')
                    
                    if '/o/' in full_path_in_url:
                        file_path_in_storage = full_path_in_url.split('/o/', 1)[1]
                    else: 
                        file_path_in_storage = full_path_in_url.replace(f"{bucket.name}/", '', 1)
                    
                    file_path_in_storage = file_path_in_storage.replace('%2F', '/')

                    blob = bucket.blob(file_path_in_storage)
                    if blob.exists(): 
                        blob.delete()
                        print(f"[Cleanup] Storage 파일 삭제 성공: {file_path_in_storage}")
                    else:
                        print(f"[Cleanup] Storage 파일 없음 (이미 삭제되었거나 경로 오류): {file_path_in_storage}")
                except Exception as e:
                    print(f"[Cleanup] Storage 파일 삭제 실패 ({file_path_in_storage if 'file_path_in_storage' in locals() else photo_url}): {e}")
    
    user_session_data = {} 
    print("[Cleanup] 모든 임시 사용자 데이터가 메모리에서 삭제되었습니다.")

# 앱 종료 시 cleanup_user_data 함수를 실행하도록 등록
atexit.register(cleanup_user_data)

# --- 선생님 데이터 로드 (메모리 전용) ---
def _load_teacher_au_data_to_memory():
    global _teacher_references_in_memory
    _teacher_references_in_memory = {} 

    teachers = ['emma', 'olivia', 'sophia'] 
    OVERALL_TOTAL_ROUNDS_TO_LOAD = 8 

    print("\n--- 선생님 AU 데이터 (메모리 로드) 시작 ---")
    
    try:
        embeddings_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'face_embeddings.csv')
        
        print(f"DEBUG: 임베딩 CSV 파일 경로 확인: {embeddings_csv_path}") 
        if not os.path.exists(embeddings_csv_path):
            print(f"오류: '{embeddings_csv_path}' 파일을 찾을 수 없습니다. 선생님 임베딩 로드 불가.")
            print("extract_embeddings.py를 실행하여 'face_embeddings.csv'를 생성했는지 확인하세요.")
            return 

        df_embeddings = pd.read_csv(embeddings_csv_path)
        print(f"DEBUG: '{embeddings_csv_path}'에서 임베딩 데이터 로드 완료. 총 {len(df_embeddings)}개.")
        print(f"DEBUG: 임베딩 CSV 컬럼: {df_embeddings.columns.tolist()}") 
        print(f"DEBUG: 임베딩 CSV 파일명 예시 (처음 5개): {df_embeddings['file_name'].head().tolist()}") 
        
        embedding_cols = [col for col in df_embeddings.columns if col.startswith('embedding_')]
        if not embedding_cols:
            print("오류: 임베딩 컬럼을 찾을 수 없습니다. 'embedding_1'과 같은 컬럼이 있는지 확인하세요.")
            return

        all_feature_names_40 = [
            col for col in df_embeddings.columns 
            if col != 'file_name' and not col.startswith('embedding_')
        ]


        for teacher_id in teachers:
            _teacher_references_in_memory[teacher_id] = []
            
            teacher_specific_rows = df_embeddings[
                df_embeddings['file_name'].str.contains(teacher_id, case=False, na=False)
            ].copy()

            print(f"DEBUG: 선생님 ID '{teacher_id}' 필터링 후 데이터 개수: {len(teacher_specific_rows)}개. (예상 8~10개)") 
            
            if teacher_specific_rows.empty:
                print(f"경고: 선생님 ID '{teacher_id}'에 해당하는 이미지가 'face_embeddings.csv'에 없습니다. 'file_name' 컬럼을 확인하세요.")
                continue 

            def extract_round_number_from_filename(file_name_str, teacher_id_str):
                try:
                    temp_name = file_name_str.replace(teacher_id_str, '', 1) 
                    num_str = ''.join(filter(str.isdigit, temp_name.split('.')[0])) 
                    return int(num_str) if num_str else 0
                except:
                    return 0 

            teacher_specific_rows['round_number_extracted'] = teacher_specific_rows['file_name'].apply(
                lambda x: extract_round_number_from_filename(x, teacher_id)
            )
            
            teacher_specific_rows = teacher_specific_rows[teacher_specific_rows['round_number_extracted'] > 0] 
            teacher_specific_rows.sort_values(by='round_number_extracted', inplace=True)
            teacher_specific_rows = teacher_specific_rows.head(OVERALL_TOTAL_ROUNDS_TO_LOAD) 

            print(f"DEBUG: 선생님 ID '{teacher_id}' 유효 라운드 추출 후 최종 개수: {len(teacher_specific_rows)}개") 

            if teacher_specific_rows.empty:
                print(f"경고: 선생님 ID '{teacher_id}'에 해당하는 유효한 라운드 데이터(라운드 번호 추출 오류)가 없습니다. 건너뜨.")
                continue


            for index, row in teacher_specific_rows.iterrows():
                embedding_vector = row[embedding_cols].tolist()

                au_detail_values_dict = {}
                for au_col_name in all_feature_names_40: 
                    au_value = row.get(au_col_name, 0.0) 
                    if pd.isna(au_value):
                        au_detail_values_dict[au_col_name] = 0.0
                    else:
                        au_detail_values_dict[au_col_name] = float(f"{au_value:.4f}") 
                
                image_filename_for_static = row['file_name']
                
                _teacher_references_in_memory[teacher_id].append({
                    'teacher_id': teacher_id,
                    'round_number': int(row['round_number_extracted']),
                    'embedding': embedding_vector, 
                    'image_path_static': f'/static/images/teachers/{teacher_id}/{image_filename_for_static}', 
                    'timestamp': datetime.now().isoformat(),
                    'au_detail_values': au_detail_values_dict 
                })
                print(f"DEBUG: 선생님 {teacher_id} 라운드 {int(row['round_number_extracted'])} 임베딩 및 AU 상세값 로드 완료. 이미지 경로: {image_filename_for_static}") 
                print(f"DEBUG: AU 상세값 일부 예시: {list(au_detail_values_dict.items())[:5]}") 
        
    except Exception as e:
        print(f"오류: 선생님 임베딩 데이터 메모리 로드 중 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("--- 모든 선생님 임베딩 데이터 (메모리 로드) 완료 ---")


# --- 앱 실행 ---
if __name__ == '__main__':
    _load_teacher_au_data_to_memory() 
    
    if global_cap is not None and global_cap.isOpened():
        global_cap.release() 
        global_cap = None 

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)