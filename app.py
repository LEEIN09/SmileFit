# app.py (최종 완성본)

# --- 1. 기본 및 Firebase 라이브러리 임포트 ---
from flask import Flask, session, send_from_directory, request, jsonify, render_template
from flask_mail import Mail, Message
import base64
import uuid
import os
from datetime import datetime
import traceback # 오류 추적을 위해 추가
import traceback
import firebase_admin
from firebase_admin import credentials, storage, db

# --- 2. 분석용 라이브러리 임포트 ---
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from keras.models import load_model

from gemini_helper import get_gemini_feedback 

# --- ▼▼▼▼▼ [추가] 임시 데이터 저장을 위한 설정 ▼▼▼▼▼ ---
import atexit
from urllib.parse import urlparse

# 사용자별 임시 데이터를 메모리에 저장할 전역 딕셔너리
user_session_data = {}

# 서버 종료 시 실행될 자동 정리 함수
# app.py

# app.py

def cleanup_user_data():
    global user_session_data
    print("\n[Cleanup] 서버 종료 감지: 임시 사용자 데이터 및 Storage 파일 삭제를 시작합니다.")
    
    bucket = storage.bucket() 
    
    for session_id, rounds_data in list(user_session_data.items()):
        for round_data in rounds_data:
            photo_url = round_data.get('photo_url')
            print(f"  [Cleanup DEBUG] 처리 중인 URL: {photo_url}") 

            # --- ▼▼▼▼▼ [수정된 URL 유효성 체크 조건] ▼▼▼▼▼ ---
            # 'storage.googleapis.com' 또는 'firebasestorage.googleapis.com' 둘 다 허용
            if photo_url and ("storage.googleapis.com" in photo_url or "firebasestorage.googleapis.com" in photo_url): 
            # --- ▲▲▲▲▲ [수정된 URL 유효성 체크 조건] 여기까지 ▲▲▲▲▲ ---
                try:
                    parsed_url = urlparse(photo_url)
                    path_in_url = parsed_url.path 
                    
                    file_path_in_storage = None
                    
                    bucket_name_in_path = '/' + bucket.name
                    
                    if bucket_name_in_path in path_in_url:
                        file_path_in_storage = path_in_url.split(bucket_name_in_path + '/', 1)[1]
                    elif '/o/' in path_in_url: 
                        file_path_in_storage = path_in_url.split('/o/', 1)[1]
                    else:
                        print(f"  [Cleanup DEBUG] Warning: 예상치 못한 URL 형식 - {path_in_url}")
                        file_path_in_storage = None 

                    if file_path_in_storage:
                        file_path_in_storage = file_path_in_storage.replace('%2F', '/')
                        if '?' in file_path_in_storage:
                            file_path_in_storage = file_path_in_storage.split('?', 1)[0]

                    print(f"  [Cleanup DEBUG] 파싱된 Storage 경로: {file_path_in_storage}") 
                    
                    if file_path_in_storage:
                        blob = bucket.blob(file_path_in_storage)
                        print(f"  [Cleanup DEBUG] Blob 존재 여부 확인 중: {blob.name}") 
                        if blob.exists():
                            blob.delete()
                            print(f"  ✅ Storage 파일 삭제 성공: {file_path_in_storage}")
                        else:
                            print(f"  ℹ️ Storage 파일 없음 (이미 삭제되었거나 경로 오류): {file_path_in_storage}")
                    else:
                        print(f"  [Cleanup DEBUG] 파일 경로 파싱 실패 또는 비어있음: {photo_url}")

                except Exception as e:
                    print(f"  ❌ Storage 파일 삭제 실패 ({photo_url}): {e}")
                    traceback.print_exc() 
            else:
                # 이 부분이 이제는 제대로 작동해야 합니다.
                print(f"  [Cleanup DEBUG] 유효한 Firebase Storage URL이 아님 또는 비어있음 (형식 불일치): {photo_url}") 
        
    user_session_data = {}
    print("[Cleanup] 모든 임시 데이터 처리가 완료되었습니다.")

# 앱 종료 시 cleanup_user_data 함수를 실행하도록 등록
atexit.register(cleanup_user_data)
# --- ▲▲▲▲▲ [추가] 여기까지 ▲▲▲▲▲ ---

# --- 3. Flask 앱 및 Firebase 초기화 ---
app = Flask(__name__)
app.secret_key = 'your-very-secret-key-for-session' # <-- 이 라인 추가!

# Firebase 인증키 경로 설정 및 초기화
try:
    # --- ▼▼▼▼▼ [수정된 부분] 절대 경로로 변경 ▼▼▼▼▼ ---
    # 현재 파일(app.py)이 있는 폴더의 절대 경로를 가져옴
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 절대 경로와 파일 이름을 합쳐서 최종 경로를 만듦
    cred_path = os.path.join(base_dir, "serviceAccountKey.json")
    # --- ▲▲▲▲▲ [수정된 부분] 여기까지 ▲▲▲▲▲ ---
    
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'smilefit-350ea.firebasestorage.app',
        'databaseURL': 'https://smilefit-350ea-default-rtdb.firebaseio.com/'
    })
    print("✅ Firebase 앱이 성공적으로 초기화되었습니다.")
except Exception as e:
    print(f"❌ Firebase 초기화 실패: {e}. 'serviceAccountKey.json' 파일이 있는지 확인하세요.")


# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 선생님 데이터 로드 함수 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
def _load_teacher_au_data_to_memory():
    global _teacher_references_in_memory
    _teacher_references_in_memory = {}
    teachers = ['emma', 'olivia', 'sophia']
    print("\n--- 💡 선생님 기준 데이터 메모리 로드 시작 ---")
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'face_embeddings.csv')
        if not os.path.exists(csv_path):
            print(f"❌ 오류: '{csv_path}' 파일을 찾을 수 없습니다.")
            return
        df_embeddings = pd.read_csv(csv_path)
        embedding_cols = [col for col in df_embeddings.columns if col.startswith('embedding_')]
        au_feature_cols = [col for col in df_embeddings.columns if col != 'file_name' and not col.startswith('embedding_')]

        for teacher_id in teachers:
            _teacher_references_in_memory[teacher_id] = []
            teacher_df = df_embeddings[df_embeddings['file_name'].str.contains(teacher_id, na=False)].copy()
            if teacher_df.empty: continue
            
            def extract_round_num(filename):
                try: return int(''.join(filter(str.isdigit, filename.split('.')[0])))
                except: return 0
            
            teacher_df['round_number'] = teacher_df['file_name'].apply(extract_round_num)
            teacher_df = teacher_df[teacher_df['round_number'] > 0].sort_values('round_number')

            for _, row in teacher_df.iterrows():
                au_details = {col: float(row[col]) for col in au_feature_cols}
                _teacher_references_in_memory[teacher_id].append({
                    'teacher_id': teacher_id, 'round_number': int(row['round_number']),
                    'embedding': row[embedding_cols].tolist(), 'au_detail_values': au_details,
                    'image_path_static': f'/static/images/teachers/{teacher_id}/{row["file_name"]}'
                })
            print(f"👍 '{teacher_id}' 선생님 데이터 {len(teacher_df)}개 메모리 로드 완료.")
    except Exception as e:
        print(f"❌ 오류: 선생님 데이터 로드 중 오류 발생: {e}")
    print("--- ✅ 모든 선생님 기준 데이터 메모리 로드 완료 ---\n")


# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ AU 분석 핵심 함수 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
# MediaPipe 및 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
try:
    encoder_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'models', 'autoencoder_encoder', 'encoder_model.keras')
    encoder_model = load_model(encoder_model_path)
    print("✅ Autoencoder 인코더 모델 로드 완료.")
except Exception as e:
    print(f"❌ Autoencoder 인코더 모델 로드 실패: {e}")
    encoder_model = None
feature_cols = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43", "AU01_w", "AU02_w", "AU04_w", "AU05_w", "AU06_w", "AU07_w", "AU09_w", "AU10_w", "AU11_w", "AU12_w", "AU14_w", "AU15_w", "AU17_w", "AU20_w", "AU23_w", "AU24_w", "AU25_w", "AU26_w", "AU28_w", "AU43_w"]
AU_LANDMARKS = {'AU01': [336, 296], 'AU02': [334, 298], 'AU04': [9, 8], 'AU05': [159, 144], 'AU06': [205, 206], 'AU07': [159, 145], 'AU09': [19, 219], 'AU10': [11, 10], 'AU11': [61, 291], 'AU12': [308, 78], 'AU14': [41, 13], 'AU15': [84, 17], 'AU17': [13, 15], 'AU20': [32, 262], 'AU23': [13, 14], 'AU24': [13, 14], 'AU25': [13, 14], 'AU26': [10, 152], 'AU28': [13, 14], 'AU43': [145, 374]}
def extract_normalized_au_features(landmarks):
    au_dict = {}
    if not isinstance(landmarks, np.ndarray) or not landmarks.any(): return {col: 0.0 for col in feature_cols}
    try:
        p_left_eye, p_right_eye = landmarks[133], landmarks[362]
        face_scale_distance = np.linalg.norm(p_left_eye - p_right_eye)
        if face_scale_distance < 1e-6: face_scale_distance = 1.0
    except IndexError: face_scale_distance = 1.0
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1, p2 = landmarks[indices[0]], landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else: au_dict[au] = 0.0
    final_au_features = {}
    for au_key in AU_LANDMARKS.keys():
        normalized_value = au_dict.get(au_key, 0.0) / face_scale_distance
        if pd.isna(normalized_value): normalized_value = 0.0
        final_au_features[au_key] = float(f"{normalized_value:.4f}")
        final_au_features[f"{au_key}_w"] = float(f"{normalized_value * 0.8:.4f}")
    return final_au_features
def calculate_similarity_score(embedding1, embedding2):
    if embedding1 is None or embedding2 is None: return 0.0
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    k_factor = 0.05
    similarity_score = 100 * np.exp(-k_factor * distance)
    return max(0, similarity_score)
def analyze_user_image(image_bytes, teacher_embedding):
    if encoder_model is None: raise Exception("Autoencoder 모델이 로드되지 않았습니다.")
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None: raise ValueError("이미지 데이터를 디코딩할 수 없습니다.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(image_rgb)
    if not results.multi_face_landmarks: return {'status': 'no_face_detected', 'message': '얼굴을 감지할 수 없습니다.'}
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    au_features_dict = extract_normalized_au_features(landmarks_np)
    au_features_for_model = np.array([au_features_dict.get(col, 0.0) for col in feature_cols], dtype=np.float32).reshape(1, -1)
    user_embedding = encoder_model.predict(au_features_for_model)[0]
    score = calculate_similarity_score(user_embedding.tolist(), teacher_embedding)
    return {'status': 'success', 'predicted_score': score, 'user_embedding': user_embedding.tolist(), 'au_features': au_features_dict}


# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ API 엔드포인트 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================

# --- [추가] AU 데이터 제출 API ---
@app.route('/submit_au_data_with_image', methods=['POST'])
def submit_au_data_with_image():
    # 세션 ID를 사용하여 사용자 구분 (app_au.py 방식)
    # Flask의 세션을 사용하려면 secret_key 설정이 필요합니다.
    if 'user_session_id' not in session:
        session['user_session_id'] = str(uuid.uuid4())
    user_session_id = session['user_session_id']

    if 'image_data' not in request.files:
        return jsonify({'message': '이미지 파일이 없습니다.'}), 400

    image_file = request.files['image_data']
    teacher_id = request.form.get('teacher_id')
    round_number = request.form.get('round_number', type=int)

    try:
        teacher_rounds_data = _teacher_references_in_memory.get(teacher_id)
        if not teacher_rounds_data:
            return jsonify({'message': f"'{teacher_id}' 선생님의 데이터를 찾을 수 없습니다."}), 404

        matching_teacher_data = next((item for item in teacher_rounds_data if item["round_number"] == round_number), None)
        if not matching_teacher_data:
            return jsonify({'message': f"라운드 {round_number}의 기준 데이터를 찾을 수 없습니다."}), 404
            
        teacher_embedding = matching_teacher_data.get('embedding')
        
        image_bytes = image_file.read()
        analysis_result = analyze_user_image(image_bytes, teacher_embedding)

        if analysis_result.get('status') != 'success':
            return jsonify(analysis_result), 400

        # Firebase Storage에 이미지 업로드 (이 부분은 유지)
        bucket = storage.bucket()
        filename = f"{user_session_id}_{round_number}_{uuid.uuid4()}.jpg"
        blob = bucket.blob(f"user_images_temp/{teacher_id}/{filename}") # 임시 폴더에 저장
        blob.upload_from_string(image_bytes, content_type='image/jpeg')
        blob.make_public()
        photo_url = blob.public_url

        gemini_feedback = "" # Gemini 피드백은 나중에 생성하므로 여기서는 빈 문자열로 초기

        # --- ▼▼▼▼▼ [수정] Firebase DB 대신 메모리에 저장 ▼▼▼▼▼ ---
        if user_session_id not in user_session_data:
            user_session_data[user_session_id] = []
        
        data_to_save = {
            'teacher_id': teacher_id, 'round_number': round_number,
            'photo_url': photo_url, 'predicted_score': analysis_result['predicted_score'],
            'au_features': analysis_result['au_features'],
            'user_embedding': analysis_result['user_embedding'],
            'gemini_feedback': gemini_feedback
        }
        user_session_data[user_session_id].append(data_to_save)
        print(f"✅ 데이터 임시 저장 성공 (세션 ID: {user_session_id}, 라운드: {round_number})")
        # --- ▲▲▲▲▲ [수정] 여기까지 ▲▲▲▲▲ ---

        return jsonify({
            'message': '데이터 임시 저장 성공',
            'predicted_score': analysis_result['predicted_score'],
            'photo_url': photo_url
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'서버 내부 오류: {e}'}), 500

# --- [추가] 피드백 데이터 조회 API ---

@app.route('/get_user_feedback_data', methods=['GET'])
def get_user_feedback_data():
    user_session_id = session.get('user_session_id')
    teacher_id = request.args.get('teacher_id')

    if not user_session_id:
        return jsonify({'message': '세션 정보가 없습니다. 운동을 다시 시작해주세요.'}), 400

    try:
        # --- ▼▼▼▼▼ [수정] Firebase DB 대신 메모리에서 조회 ▼▼▼▼▼ ---
        user_results = user_session_data.get(user_session_id, [])
        # --- ▲▲▲▲▲ [수정] 여기까지 ▲▲▲▲▲ ---

        # 선생님 기준 데이터는 그대로 메모리에서 조회
        teacher_results = _teacher_references_in_memory.get(teacher_id, [])

        updated_user_results = []
        for user_record in user_results:
            # 피드백이 없거나 (or ""이거나) 오류 메시지일 경우에만 생성 시도
            if 'gemini_feedback' not in user_record or not user_record['gemini_feedback'] or \
               "피드백 생성에 문제가 발생했습니다" in user_record['gemini_feedback'] or \
               "선생님 데이터가 부족하여" in user_record['gemini_feedback']:

                print(f"[Gemini Lazy Load] 라운드 {user_record.get('round_number')}에 대한 Gemini 피드백 생성 시도...")

                # 해당 라운드의 선생님 데이터 찾기
                matching_teacher_data = next((item for item in teacher_results if item["round_number"] == user_record.get('round_number')), None)

                if matching_teacher_data and 'au_detail_values' in matching_teacher_data:
                    try:
                        # get_gemini_feedback 함수 호출 (gemini_helper.py에서 임포트한 함수)
                        generated_feedback = get_gemini_feedback(
                            user_record.get('predicted_score', 0.0),
                            user_record.get('au_features', {}),
                            matching_teacher_data['au_detail_values'],
                            # app.py의 전역 변수 feature_cols를 참조합니다.
                            # feature_cols는 위에 이미 정의되어 있습니다.
                            ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", 
                            "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", 
                            "AU25", "AU26", "AU28", "AU43", "AU01_w", "AU02_w", "AU04_w", "AU05_w", 
                            "AU06_w", "AU07_w", "AU09_w", "AU10_w", "AU11_w", "AU12_w", "AU14_w", 
                            "AU15_w", "AU17_w", "AU20_w", "AU23_w", "AU24_w", "AU25_w", "AU26_w", 
                            "AU28_w", "AU43_w"] 
                            # feature_cols 변수를 직접 넘겨도 되지만, 명시적으로 리스트를 넘겨도 무방합니다.
                        )
                        user_record['gemini_feedback'] = generated_feedback
                        print(f"[Gemini Lazy Load] 라운드 {user_record.get('round_number')} 피드백 생성 완료.")
                    except Exception as e:
                        print(f"[Gemini Lazy Load] 라운드 {user_record.get('round_number')} 피드백 생성 중 오류: {e}")
                        user_record['gemini_feedback'] = "피드백 생성에 문제가 발생했습니다. 다시 시도해 주세요."
                else:
                    print(f"[Gemini Lazy Load] 라운드 {user_record.get('round_number')}의 선생님 AU 데이터 부족으로 피드백 생성 불가.")
                    user_record['gemini_feedback'] = "선생님 데이터가 부족하여 피드백을 생성할 수 없습니다."
            updated_user_results.append(user_record)

        # 메모리의 user_session_data도 업데이트 (다음 요청 시 재사용)
        if user_session_id in user_session_data:
            user_session_data[user_session_id] = updated_user_results
        # --- ▲▲▲▲▲ [추가] 여기까지 ▲▲▲▲▲ ---

        return jsonify({'user_data': updated_user_results, 'teacher_references': teacher_results}) # 업데이트된 데이터 반환
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'데이터 조회 실패: {e}'}), 500

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 기존 라우트 및 메일 기능 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/tower_defense_game.html")
def tower_defense_game():
    return render_template("tower_defense_game.html")

# ... (기타 모든 기존 라우트들: /models, /pages, /scripts 등) ...
@app.route('/models/<path:filename>')
def serve_models(filename): return send_from_directory('static/models', filename)
@app.route('/pages/<path:filename>')
def load_page(filename): return send_from_directory('templates/pages', filename)
@app.route('/scripts/<path:filename>')
def serve_script(filename): return send_from_directory('static/scripts', filename)
@app.route('/static/images/<path:filename>')
def serve_image(filename): return send_from_directory('static/images', filename)
@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename): return send_from_directory('static/sounds', filename)
@app.route('/icons/<path:filename>')
def icons(filename): return send_from_directory('static/icons', filename)
@app.route('/manifest.json')
def manifest(): return send_from_directory('static', 'manifest.json')
@app.route('/serviceWorker.js')
def service_worker(): return send_from_directory('static', 'serviceWorker.js')


# 이메일 기능 유지
app.config.update(
    MAIL_SERVER='smtp.gmail.com', MAIL_PORT=587, MAIL_USE_TLS=True,
    MAIL_USERNAME='your_email@gmail.com', MAIL_PASSWORD='your_app_password',
    MAIL_DEFAULT_SENDER='your_email@gmail.com'
)
mail = Mail(app)

@app.route('/send_email', methods=['POST'])
def send_email():
    # ... (기존 이메일 코드) ...
    data = request.json
    email, images = data['email'], data['images']
    msg = Message('SMILE FIT 결과 이미지', recipients=[email])
    msg.body = "첨부된 사진은 SMILE FIT에서 촬영된 결과입니다."
    for i, img_data_url in enumerate(images):
        header, base64_data = img_data_url.split(',', 1)
        img_bytes = base64.b64decode(base64_data)
        msg.attach(f'image_{i+1}.png', 'image/png', img_bytes)
    try:
        mail.send(msg)
        return jsonify({'message': '✅ 이메일 전송 완료!'})
    except Exception as e:
        print("이메일 전송 오류:", e)
        return jsonify({'message': '❌ 이메일 전송 실패'}), 500


# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 서버 실행 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
if __name__ == "__main__":
    # 서버 시작 전, 선생님 데이터를 메모리에 로드
    _load_teacher_au_data_to_memory()
    
    # Flask 앱 실행
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 