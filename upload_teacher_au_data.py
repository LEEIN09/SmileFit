import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import mediapipe as mp
import numpy as np
import os
import json # feature_cols 로드를 위해 필요
import xgboost as xgb # XGBoost 모델 로드를 위해 추가

# --- Firebase 초기화 ---
# serviceAccountKey.json 파일 경로 (현재 스크립트와 같은 폴더에 있어야 함)
service_account_key_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred)
        print("[Firebase] Firebase 앱이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"[Firebase] Firebase 앱 초기화 실패: {e}")
        print("Firebase 초기화에 실패했습니다. 스크립트를 종료합니다.")
        exit() # 초기화 실패 시 스크립트 종료

db = firestore.client() # Firestore 클라이언트 초기화

# --- MediaPipe 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, # 이미지 처리용이므로 True
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- feature_cols 로드 ---
feature_cols = []
try:
    # feature_cols.json 파일의 인코딩이 UTF-8인지 확인하세요.
    with open('feature_cols.json', 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    print("feature_cols.json 로드 완료.")
except Exception as e:
    print(f"feature_cols.json 로드 실패: {e}")
    print("feature_cols 파일 없이는 AU 계산이 어렵습니다. 스크립트를 종료합니다.")
    exit() # feature_cols 로드 실패 시 스크립트 종료

# --- XGBoost 모델 로드 (모든 선생님 AU 데이터의 overall_score 예측용) ---
overall_score_model = None # 모든 선생님에 대한 점수 계산용 모델
try:
    overall_score_model = xgb.XGBRegressor()
    overall_score_model.load_model('expression_similarity_model.json') # <-- app.py와 동일한 모델 사용
    print("Overall Score 예측 모델 로드 완료.")
except Exception as e:
    print(f"Overall Score 예측 모델 로드 실패: {e}. 선생님 overall_score는 임의 계산됩니다.")

# AU 계산을 위한 랜드마크 인덱스 (FACS 기준, MediaPipe 랜드마크 기반)
AU_LANDMARKS = {
    'AU01': [336, 296], # 이마 안쪽 (inner brow raiser)
    'AU02': [334, 298], # 이마 바깥쪽 (outer brow raiser)
    'AU04': [9, 8],     # 눈썹 내림 (brow furrower)
    'AU06': [205, 206],  # 광대 (cheek raiser)
    'AU12': [308, 78],  # 입꼬리 (lip corner puller)
    'AU25': [13, 14] # 입술 개방 (lips part)
}

# app.py의 calculate_au 함수와 동일 (복사해 옴)
def calculate_au(landmarks, image_shape, feature_cols): # feature_cols 인자 추가
    """MediaPipe 랜드마크 기반 AU 계산 (거리 기반)"""
    au_dict = {}
    
    # landmarks가 비어있으면 0으로 채움
    if not (isinstance(landmarks, np.ndarray) and landmarks.any()):
        if feature_cols: # feature_cols가 존재할 때만 채움
            for col in feature_cols:
                au_dict[col] = 0.0
        return au_dict
    
    # 랜드마크를 픽셀 값으로 변환 (정규화된 값 * 이미지 크기)
    # Python MediaPipe 랜드마크는 0.0 ~ 1.0 정규화된 값입니다.
    pixel_landmarks = np.array([
        (lm.x * image_shape[1], lm.y * image_shape[0])
        for lm in landmarks # MediaPipe Result 객체의 landmarks는 직접 lm.x, lm.y 접근 가능
    ], dtype=np.float32)

    # 얼굴 크기 변화에 둔감하도록, 특정 기준선으로 AU 값을 정규화 (눈 사이 거리)
    # 랜드마크 인덱스가 유효한지 먼저 확인
    if 133 < len(pixel_landmarks) and 362 < len(pixel_landmarks):
        left_eye_inner = pixel_landmarks[133]
        right_eye_inner = pixel_landmarks[362]
        eye_distance = np.linalg.norm(left_eye_inner - right_eye_inner)
        normalization_factor = eye_distance if eye_distance > 0.01 else 1 # 0으로 나누는 것 방지
    else:
        normalization_factor = 1 # 눈 랜드마크 없으면 정규화 안함

    # 기본 AU 계산 (랜드마크 간 유클리드 거리)
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(pixel_landmarks) and indices[1] < len(pixel_landmarks):
            p1 = pixel_landmarks[indices[0]]
            p2 = pixel_landmarks[indices[1]]
            distance = np.linalg.norm(p1 - p2)
            au_dict[au] = distance / normalization_factor # 정규화된 거리
        else:
            au_dict[au] = 0.0
    
    # 가중치 컬럼 생성 (모델이 요구하는 형식에 맞춤)
    # feature_cols에 'AU01_w'와 같은 키가 있다면, 해당 AU 값 * 0.8 등으로 생성
    final_au_features = {} # 먼저 딕셔너리 초기화
    for col in feature_cols:
        if col.endswith('_w'):
            base_au = col.replace('_w', '')
            final_au_features[col] = au_dict.get(base_au, 0.0) * 0.8 # 예시 가중치
        else:
            final_au_features[col] = au_dict.get(col, 0.0) # au_dict에 있는 값 사용

    return final_au_features


# --- 선생님 이미지 경로 설정 ---
TEACHERS_IMAGE_BASE_PATH = 'static/images/teachers' # static/images/teachers 폴더 경로
TEACHERS = ['emma', 'olivia', 'sophia'] # 선생님 이름 목록
TOTAL_ROUNDS = 10 # 각 선생님별 라운드 수

# --- Firestore에 저장할 컬렉션 이름 ---
COLLECTION_NAME = 'teacher_au_data' # 선생님 AU 데이터 저장 컬렉션

# --- 메인 스크립트 실행 부분 ---
if __name__ == "__main__":
    print("선생님 AU 데이터 추출 및 Firestore 업로드를 시작합니다.")
    
    all_data_uploaded_successfully = True
    
    # 이미 Firebase 앱이 초기화되었는지 확인 (위에서 exit() 하지 않았다면)
    if not firebase_admin._apps:
        print("Firebase 앱이 초기화되지 않았습니다. 스크립트를 종료합니다.")
        exit()

    # feature_cols가 없으면 스크립트 종료 (위에서 exit() 하지 않았다면)
    if not feature_cols:
        print("feature_cols이 로드되지 않았습니다. 스크립트를 종료합니다.")
        exit()
        
    for teacher_name in TEACHERS:
        for round_num in range(1, TOTAL_ROUNDS + 1):
            image_filename = f"{teacher_name}{round_num}.png"
            image_path = os.path.join(TEACHERS_IMAGE_BASE_PATH, teacher_name, image_filename)
            
            if not os.path.exists(image_path):
                print(f"경고: 이미지를 찾을 수 없습니다: {image_path}. 이 라운드는 건너뜁니다.")
                all_data_uploaded_successfully = False
                continue
            
            print(f"처리 중: 선생님 '{teacher_name}', 라운드 {round_num} ({image_path})")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"오류: 이미지 로드 실패: {image_path}")
                all_data_uploaded_successfully = False
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(image_rgb)
            
            overall_score = 0.0
            au_detail_values = {}
            
            if results.multi_face_landmarks:
                landmarks_list = results.multi_face_landmarks[0].landmark # MediaPipe 랜드마크 리스트
                
                # AU 계산 (정규화된 랜드마크와 이미지 shape 전달)
                calculated_au_features = calculate_au(landmarks_list, image.shape, feature_cols=feature_cols) # feature_cols 인자 전달
                au_detail_values = {k: float(f"{v:.4f}") for k, v in calculated_au_features.items()}
                
                # overall_score는 XGBoost 모델로 예측 (모델이 있다면)
                if overall_score_model is not None and feature_cols: # <-- overall_score_model 사용
                    user_X = np.array([[calculated_au_features.get(col, 0.0) for col in feature_cols]])
                    overall_score = float(f"{overall_score_model.predict(user_X)[0]:.4f}")
                    print(f"   XGBoost 예측 Score: {overall_score}")
                else:
                    # 모델이 없으면 AU 상세 값 기반으로 임의 계산 (app.py와 유사하게)
                    if 'AU12' in au_detail_values:
                        # AU12 값을 0-100 스케일로 변환 (예시)
                        # 실제 모델이 없는 경우, 이 값은 단순 참고용
                        overall_score = float(f"{(au_detail_values['AU12'] * 50):.4f}") 
                        if overall_score > 100: overall_score = 100
                        if overall_score < 0: overall_score = 0
                    else:
                        overall_score = 0.0
                    print(f"   (모델 없음) 임의 계산 Score: {overall_score}")

            else:
                print(f"얼굴 감지 실패: {image_path}")
                overall_score = 0.0 
                au_detail_values = {col: 0.0 for col in feature_cols} # 모든 AU를 0으로
            
            # Firestore에 데이터 저장
            try:
                doc_ref = db.collection(COLLECTION_NAME).add({
                    'teacher_id': teacher_name,
                    'round_number': round_num,
                    'overall_score': overall_score,
                    'au_detail_values': au_detail_values,
                    'image_path_static': f'/static/images/teachers/{teacher_name}/{image_filename}', # 웹 경로 저장
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                print(f"성공: '{teacher_name}', 라운드 {round_num} 데이터 Firestore 저장 완료. 문서 ID: {doc_ref[1].id}")
            except Exception as e:
                print(f"오류: '{teacher_name}', 라운드 {round_num} Firestore 저장 실패: {e}")
                all_data_uploaded_successfully = False
    
    if all_data_uploaded_successfully:
        print("\n모든 선생님 AU 데이터 추출 및 Firestore 업로드 완료.")
    else:
        print("\n일부 데이터 업로드에 실패했습니다. 로그를 확인해주세요.")