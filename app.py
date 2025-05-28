from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_mail import Mail, Message
import base64
import io

app = Flask(__name__, static_url_path='/models', static_folder='static/models')

@app.route('/icons/<path:filename>')
def icons(filename):
    return send_from_directory('static/icons', filename)

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/serviceWorker.js')
def service_worker():
    return send_from_directory('static', 'serviceWorker.js')

@app.route('/pages/<path:filename>')
def load_page(filename):
    return send_from_directory('templates/pages', filename)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/scripts/<path:filename>')
def serve_script(filename):
    return send_from_directory('static/scripts', filename)

# ✅ 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_follow')
def game_follow():
    return render_template('game_follow.html')  # 표정 따라하기 모드

@app.route('/game_feedback')
def game_feedback():
    return render_template('game_feedback.html')

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
    return render_template('complex_fit.html')

@app.route('/focus_fit')
def focus_fit():
    return render_template('focus_fit.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/game_mode')
def game_mode():
    return render_template('game_mode.html')

@app.route('/game_emotion')
def game_emotion():
    return render_template('game_emotion.html')

@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='your_email@gmail.com',
    MAIL_PASSWORD='your_app_password',  # 앱 비밀번호 사용 (Google 계정 보안 설정 필요)
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

    # 이미지 첨부
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
