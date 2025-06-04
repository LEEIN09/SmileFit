
let stream = null;  // 캠 스트림 전역 변수

export function init() {
  console.log("✅ game_emotion.js init() 호출됨");

  const TOTAL_ROUNDS = 25;
  let currentRound = 0;
  let totalScore = 0;

  const referenceImg = document.getElementById('referenceImg');
  const video = document.getElementById('video');
  const roundDisplay = document.getElementById('roundDisplay');
  const refDisplay = document.getElementById('refEmotionDisplay');
  const userDisplay = document.getElementById('userEmotionDisplay');

  const allIndices = Array.from({ length: 50 }, (_, i) => i + 1);
  const selectedIndices = allIndices.sort(() => Math.random() - 0.5).slice(0, TOTAL_ROUNDS);

  function getDetectorOptions() {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: 224,        // 얼굴을 더 잘 인식하게 해주는 해상도
      scoreThreshold: 0.3    // 너무 낮으면 잡음, 너무 높으면 인식 실패
    });
  }

  function updateUI() {
    const imgNum = selectedIndices[currentRound];
    referenceImg.src = `static/images/e_game/e${imgNum}.png`;
    roundDisplay.innerText = `${currentRound + 1} / ${TOTAL_ROUNDS} 라운드`;
    document.getElementById("score").innerHTML = `이번 점수: -`;
    document.getElementById("totalScore").innerHTML = `총점: <b>${totalScore} / ${TOTAL_ROUNDS * 10}</b>`;
  }

  function drawGuideEllipse() {
    const canvas = document.getElementById("guideCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(0, 255, 0, 0.6)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.ellipse(canvas.width / 2, canvas.height / 2, canvas.width * 0.25, canvas.height * 0.38, 0, 0, 2 * Math.PI);
    ctx.stroke();
  }

  function cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }

  async function startVideo() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      await video.play();
      drawGuideEllipse();
    } catch (err) {
      alert("카메라 접근 실패");
    }
  }

  function waitForImageLoad(imgElement) {
    return new Promise((resolve, reject) => {
      if (imgElement.complete && imgElement.naturalHeight !== 0) resolve();
      imgElement.onload = resolve;
      imgElement.onerror = () => reject(new Error("이미지 로드 실패"));
    });
  }

  async function tryRecognizeReferenceEmotion(maxAttempts = 3) {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        await waitForImageLoad(referenceImg);
        const result = await faceapi
          .detectSingleFace(referenceImg, getDetectorOptions())
          .withFaceExpressions();
        if (result) return result;
      } catch (_) {}
    }
    return null;
  }

  async function tryRecognizeUserEmotion(canvas, maxAttempts = 3) {
    for (let i = 0; i < maxAttempts; i++) {
      const result = await faceapi
        .detectSingleFace(canvas, getDetectorOptions())
        .withFaceExpressions();
      if (result) return result;
    }
    return null;
  }

  document.getElementById("captureBtn").onclick = async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 225;
    canvas.getContext("2d").drawImage(video, 0, 0, 300, 225);

    const userResult = await tryRecognizeUserEmotion(canvas);
    if (!userResult) {
      alert("❌ 사용자 얼굴 인식 실패 - 다시 시도해주세요");
      return;
    }

    const refResult = await tryRecognizeReferenceEmotion();
    if (!refResult) {
      alert("❌ 기준 이미지 인식 실패 - 새로운 이미지로 다시 시도합니다");
      selectedIndices[currentRound] = allIndices.find(i => !selectedIndices.includes(i));
      updateUI();
      return;
    }


    const refVec = Object.values(refResult.expressions);
    const userVec = Object.values(userResult.expressions);
    const sim = cosineSimilarity(refVec, userVec);
    const score = Math.max(3, Math.round(sim * 10));
    totalScore += score;

    const topEmotion = exp => Object.entries(exp).sort((a, b) => b[1] - a[1])[0][0];
    refDisplay.innerHTML = `기준 감정: <b>${topEmotion(refResult.expressions)}</b>`;
    userDisplay.innerHTML = `당신 감정: <b>${topEmotion(userResult.expressions)}</b>`;
    document.getElementById("score").innerHTML = `이번 점수: <b>${score} / 10</b>`;
    document.getElementById("totalScore").innerHTML = `총점: <b>${totalScore} / ${TOTAL_ROUNDS * 10}</b>`;

    if (currentRound < TOTAL_ROUNDS - 1) {
      currentRound++;
      setTimeout(updateUI, 2500);
    } else {
      setTimeout(() => {
        sessionStorage.setItem("totalScore", totalScore);
        window.location.href = "game_feedback.html";
      }, 2500);
    }
  };

  // 초기화 실행
  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models'),
  ]).then(() => {
    startVideo();
    updateUI();
  });
}

export function cleanup() {
  const video = document.getElementById("video");
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    console.log("📷 game_emotion 캠 스트림 종료됨");
  }
  stream = null;
}
