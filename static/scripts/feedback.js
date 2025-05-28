export function init() {

  console.log("✅ feedback.js init() 호출됨");

  const imageGrid = document.getElementById('imageGrid');
  const emailInput = document.getElementById('emailInput');
  const sendBtn = document.getElementById('sendBtn');
  const statusDiv = document.getElementById('status');

  // 이미지 표시
  const images = JSON.parse(sessionStorage.getItem('capturedImages')) || [];

  images.forEach((src) => {
    const img = document.createElement('img');
    img.src = src;
    imageGrid.appendChild(img);
  });

  // 이메일 전송
  sendBtn.onclick = () => {
    const email = emailInput.value.trim();
    if (!email) {
      alert("이메일을 입력해 주세요.");
      return;
    }

    fetch('/send_email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email, images: images })
    })
      .then(res => res.json())
      .then(data => {
        statusDiv.innerText = data.message;
      })
      .catch(err => {
        statusDiv.innerText = "전송 실패 😢";
        console.error(err);
      });
  };

  // 홈 버튼 SPA 처리
  const homeBtn = document.querySelector('.home-button');
  if (homeBtn) {
    homeBtn.onclick = (e) => {
      e.preventDefault();
      loadPage('index');
    };
  }

  document.body.classList.add('loaded');
}
