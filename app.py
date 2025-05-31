from flask import Flask, send_from_directory, request, jsonify, render_template
from flask_mail import Mail, Message
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("au_regression_model.h5", compile=False)

@app.route('/predict_aus', methods=['POST'])
def predict_aus():
    data = request.get_json()

    # 입력 데이터: vectors는 (10, 136) 형태의 리스트
    vectors = np.array(data.get("vectors", []))

    if vectors.ndim != 2 or vectors.shape[1] != 136:
        return jsonify({"error": "Invalid input shape"}), 400

    # 예측
    predictions = model.predict(vectors)

    return jsonify({"aus": predictions.tolist()})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory('static/models', filename)

@app.route('/pages/<path:filename>')
def load_page(filename):
    return send_from_directory('templates/pages', filename)

@app.route('/scripts/<path:filename>')
def serve_script(filename):
    return send_from_directory('static/scripts', filename)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)

@app.route('/icons/<path:filename>')
def icons(filename):
    return send_from_directory('static/icons', filename)

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/serviceWorker.js')
def service_worker():
    return send_from_directory('static', 'serviceWorker.js')

# 이메일 기능 유지
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='your_email@gmail.com',
    MAIL_PASSWORD='your_app_password',
    MAIL_DEFAULT_SENDER='your_email@gmail.com'
)

mail = Mail(app)

@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.json
    email = data['email']
    images = data['images']

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

if __name__ == "__main__":
    app.run(debug=True)
