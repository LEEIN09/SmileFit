
let camera;

export function init() {
  console.log("✅ game_follow.js 실행됨");

  const interval = setInterval(() => {
    const imgA = document.getElementById("imageA");
    const imgB = document.getElementById("imageB");
    const video = document.querySelector(".input_video");
    const canvas = document.querySelector(".output_canvas");
    const ctx = canvas.getContext("2d");

    if (imgA && imgB && video && canvas && ctx) {
      clearInterval(interval);
      drawGuideEllipse();
      updateUI();

      // ✅ 캠 재시작
      camera = new Camera(video, {
        onFrame: async () => {
          await faceMesh.send({ image: video });
        },
        width: 300,
        height: 225,
      });
      camera.start();
      console.log("📷 캠 시작됨");
    }
  }, 100);
}

// SPA 전용 로딩 효과
window.addEventListener('DOMContentLoaded', () => {
  document.body.classList.add('loaded');
});

function delayedNav(url) {
  document.body.classList.remove('loaded');
  document.body.classList.add('fade-out');
  setTimeout(() => {
    window.location.href = url;
  }, 400);
}

const TOTAL_ROUNDS = 25;
let currentRound = 0;
let totalScore = 0;
const capturedImages = [];

const video = document.querySelector('.input_video');
const canvas = document.querySelector('.output_canvas');
const ctx = canvas.getContext('2d');
let latestLandmarks = null;
let userA = null;
let userB = null;
let teacherA = null;
let teacherB = null;

const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true });
faceMesh.onResults(results => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  if (results.multiFaceLandmarks.length > 0) {
    latestLandmarks = structuredClone(results.multiFaceLandmarks[0]);
  }
});

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

const weightMap = { eyes: 0.2, mouth: 0.4, eyebrows: 0.2, nose: 0.1, jaw: 0.1 };
const regions = {
  eyes: [33, 133, 362, 263, 159, 386],
  mouth: [78, 308, 13, 14, 17, 87, 317],
  eyebrows: [65, 55, 295, 285],
  nose: [1, 2, 98, 327],
  jaw: [152, 234, 454]
};

function normalizeLandmarks(landmarks) {
  const leftEye = landmarks[33], rightEye = landmarks[263];
  const centerX = (leftEye.x + rightEye.x) / 2;
  const centerY = (leftEye.y + rightEye.y) / 2;
  const scale = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
  return landmarks.map(p => ({
    x: (p.x - centerX) / scale,
    y: (p.y - centerY) / scale,
    z: (p.z || 0) / scale
  }));
}

function computeDelta(a, b) {
  return a.map((p, i) => ({
    x: p.x - b[i].x,
    y: p.y - b[i].y,
    z: p.z - b[i].z
  }));
}

function getWeightedDeltaScore(d1, d2) {
  let sum = 0, totalWeight = 0;
  for (const [region, indices] of Object.entries(regions)) {
    const weight = weightMap[region];
    let regionSum = 0;
    for (const i of indices) {
      const dx = d1[i].x - d2[i].x;
      const dy = d1[i].y - d2[i].y;
      const dz = d1[i].z - d2[i].z;
      regionSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    sum += (regionSum / indices.length) * weight;
    totalWeight += weight;
  }
  return Math.max(1, Math.round(10 - (sum / totalWeight) * 50));
}

const staticFaceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
staticFaceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true });

async function getImageLandmarksAsync(imgElement) {
  return new Promise((resolve) => {
    staticFaceMesh.onResults(res => {
      if (res.multiFaceLandmarks && res.multiFaceLandmarks.length > 0) {
        resolve(res.multiFaceLandmarks[0]);
      } else {
        resolve(null);
      }
    });

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = 300;
    tmpCanvas.height = 225;
    const tmpCtx = tmpCanvas.getContext("2d");

    const runWhenReady = () => {
      tmpCtx.drawImage(imgElement, 0, 0, tmpCanvas.width, tmpCanvas.height);
      staticFaceMesh.send({ image: tmpCanvas });
    };

    if (imgElement.complete) {
      runWhenReady();
    } else {
      imgElement.onload = runWhenReady;
    }
  });
}

async function preloadTeacherLandmarks() {
  try {
    const imgA = document.getElementById("imageA");
    const imgB = document.getElementById("imageB");

    teacherA = await getImageLandmarksAsync(imgA);
    teacherB = await getImageLandmarksAsync(imgB);

    if (!teacherA || !teacherB) {
      throw new Error("기준 이미지 얼굴 인식 실패");
    }

    console.log("✅ 기준 이미지 랜드마크 추출 완료");
    document.getElementById("captureA").disabled = false;

  } catch (err) {
    alert("⚠️ 기준 이미지에서 얼굴 인식 실패: " + err.message);
  }
}

function updateUI() {
  console.log("🚀 updateUI 실행됨");
  const imgA = document.getElementById("imageA");
  const imgB = document.getElementById("imageB");

  if (!imgA || !imgB) {
    console.warn("❌ imageA 또는 imageB가 DOM에 없습니다.");
    return;
  }

  const imgNum1 = currentRound * 2 + 1;
  const imgNum2 = currentRound * 2 + 2;

  imgA.src = `/static/images/f_game/${imgNum1}.png`;
  imgB.src = `/static/images/f_game/${imgNum2}.png`;

  document.getElementById("round").innerText = `${currentRound + 1} / ${TOTAL_ROUNDS} 라운드`;
  // document.getElementById("score").innerText = `이번 점수: -`;
  document.getElementById("captureA").disabled = true;
  document.getElementById("captureB").disabled = true;

  imgB.onload = () => preloadTeacherLandmarks();
}

    document.getElementById("captureA").onclick = async () => {
      if (!latestLandmarks) return alert("❌ 얼굴 인식 실패");
      userA = structuredClone(latestLandmarks);
      capturedImages.push(canvas.toDataURL("image/png"));
      document.getElementById("captureA").disabled = true;
      document.getElementById("captureB").disabled = false;
    };

    document.getElementById("captureB").onclick = async () => {
      if (!latestLandmarks || !userA || !teacherA || !teacherB) return alert("❌ 데이터 부족");
      userB = structuredClone(latestLandmarks);
      capturedImages.push(canvas.toDataURL("image/png"));

      const refDelta = computeDelta(normalizeLandmarks(teacherB), normalizeLandmarks(teacherA));
      const imgDelta = computeDelta(normalizeLandmarks(userB), normalizeLandmarks(userA));
      const score = getWeightedDeltaScore(refDelta, imgDelta);

      totalScore += score;
      document.getElementById("score").innerText = `이번 점수: ${score} / 10`;
      document.getElementById("totalScore").innerText = `총점: ${totalScore} / ${TOTAL_ROUNDS * 10}`;

      if (currentRound < TOTAL_ROUNDS - 1) {
        currentRound++;
        setTimeout(updateUI, 1000);
      } else {
        setTimeout(() => {
          sessionStorage.setItem("capturedImages", JSON.stringify(capturedImages));
          sessionStorage.setItem("totalScore", totalScore);
          window.location.href = "game_feedback.html";
        }, 1000);
      }
    };

export function cleanup() {
  if (camera && typeof camera.stop === 'function') {
    camera.stop();
    camera = null;
    console.log("📷 game_follow 카메라 종료됨");
  }
}
