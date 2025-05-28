
let stream = null;  // 캠 스트림 저장용

export function init() {
  console.log("✅ focus_fit.js init() 호출됨");

  const video = document.getElementById('video');
  const canvas = document.getElementById('guide-canvas');
  const ctx = canvas.getContext('2d');
  const referenceImg = document.getElementById('reference-img');
  const roundText = document.getElementById('round');
  const checkMark = document.getElementById('check-mark');
  const submitBtn = document.getElementById('submit-btn');

  const TOTAL_ROUNDS = 10;
  let currentRound = 1;
  let capturedImages = [];

  const selectedExercise = sessionStorage.getItem('selectedExercise');
  if (!selectedExercise) {
    alert("운동 정보가 없습니다. 이전 페이지로 이동합니다.");
    loadPage('focus');
    return;
  }

  // 캠 접근
  navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
    stream = s;
    video.srcObject = stream;
  }).catch(err => {
    console.error("❌ 캠 접근 실패:", err);
    alert("카메라 권한을 허용해주세요.");
  });

  // 가이드 원
  function drawEllipse() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.ellipse(canvas.width / 2, canvas.height / 2, 80, 100, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.4)';
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  setInterval(drawEllipse, 100);

  // 기준 이미지 갱신
  function updateReferenceImage() {
    referenceImg.src = `/static/images/expression/${selectedExercise}/${currentRound}.png`;
    roundText.textContent = currentRound;
  }

  // 사진 제출
  submitBtn.onclick = () => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 160;
    tempCanvas.height = 120;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const dataUrl = tempCanvas.toDataURL('image/png');
    capturedImages.push(dataUrl);

    checkMark.style.display = 'block';
    setTimeout(() => {
      checkMark.style.display = 'none';
    }, 1000);

    currentRound++;
    if (currentRound > TOTAL_ROUNDS) {
      submitBtn.textContent = '운동 완료';
      submitBtn.style.backgroundColor = '#8e24aa';
      submitBtn.style.boxShadow = '0 4px 10px rgba(142, 36, 170, 0.4)';
      submitBtn.style.color = 'white';
      submitBtn.onclick = () => {
        sessionStorage.setItem('capturedImages', JSON.stringify(capturedImages));
        sessionStorage.setItem('selectedExercise', selectedExercise);
        sessionStorage.setItem('mode', 'focus');
        loadPage('feedback');
      };
    } else {
      updateReferenceImage();
    }
  };

  // 로딩 완료
  document.body.classList.add('loaded');
  updateReferenceImage();

  // 홈 버튼 이벤트 (SPA 방식)
  const homeBtn = document.querySelector('.home-button');
  if (homeBtn) {
    homeBtn.onclick = (e) => {
      e.preventDefault();
      loadPage('index');
    };
  }
}

// ✅ 캠 종료 함수 (SPA 이동 시 자동 호출됨)
export function cleanup() {
  const video = document.getElementById('video');
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    console.log("📷 캠 스트림 정리 완료 (focus_fit)");
  }
  stream = null;
}
