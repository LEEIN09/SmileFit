let faceMeshLoaded = false;
let currentSessionTotalScore_Follow = 0;
let userA_Landmarks = null;
let userB_Landmarks = null;
let teacherA_Landmarks = null;
let teacherB_Landmarks = null;

const followVideo = document.getElementById("follow-video");
const followImageA = document.getElementById("follow-imageA");
const followImageB = document.getElementById("follow-imageB");
const followCaptureABtn = document.getElementById("follow-captureA_Btn");
const followCaptureBBtn = document.getElementById("follow-captureB_Btn");
const followScoreDisplay = document.getElementById("follow-scoreDisplay");
const followGuideCanvas = document.getElementById("follow-guideCanvas");
const followSessionDisplay = document.getElementById("session-total-score-display-follow");

const baseImagePathFollow = "/static/images/f_game/";
let currentFollowIndex = 0;

async function loadFollowModels() {
  if (faceMeshLoaded) return true;
  const model = new FaceMesh({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
  model.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  await model.initialize();
  faceMeshLoaded = true;
  return model;
}

async function startFollowCamera(faceMesh) {
  const camera = new Camera(followVideo, {
    onFrame: async () => {
      await faceMesh.send({ image: followVideo });
    },
    width: 300,
    height: 225
  });
  await camera.start();
}

function drawFollowGuide(canvas, video) {
  const ctx = canvas.getContext("2d");
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(0, 255, 0, 0.6)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.ellipse(canvas.width / 2, canvas.height / 2, canvas.width * 0.25, canvas.height * 0.38, 0, 0, 2 * Math.PI);
  ctx.stroke();
}

async function loadFollowImages() {
  currentFollowIndex = Math.floor(Math.random() * 25) * 2 + 1;
  followImageA.src = `${baseImagePathFollow}${currentFollowIndex}.png`;
  followImageB.src = `${baseImagePathFollow}${currentFollowIndex + 1}.png`;
  followScoreDisplay.innerText = "이번 점수: -";
}

function normalize(landmarks) {
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const centerX = (leftEye.x + rightEye.x) / 2;
  const centerY = (leftEye.y + rightEye.y) / 2;
  const scale = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) || 1;
  return landmarks.map(p => ({ x: (p.x - centerX) / scale, y: (p.y - centerY) / scale, z: (p.z || 0) / scale }));
}

function computeDelta(landA, landB) {
  return landA.map((p, i) => ({ x: p.x - landB[i].x, y: p.y - landB[i].y, z: p.z - landB[i].z }));
}

function computeScore(delta1, delta2) {
  const error = delta1.reduce((sum, d1, i) => {
    const d2 = delta2[i];
    return sum + Math.sqrt((d1.x - d2.x) ** 2 + (d1.y - d2.y) ** 2 + (d1.z - d2.z) ** 2);
  }, 0);
  const avgError = error / delta1.length;
  return Math.max(1, Math.min(10, Math.round(10 - avgError * 50)));
}

export async function setupFollowMode() {
  const faceMesh = await loadFollowModels();
  faceMesh.onResults(async results => {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      latestLandmarks = structuredClone(results.multiFaceLandmarks[0]);
    }
  });

  await startFollowCamera(faceMesh);
  drawFollowGuide(followGuideCanvas, followVideo);
  await loadFollowImages();

  followCaptureABtn.onclick = () => {
    if (!latestLandmarks) return alert("얼굴 인식 실패");
    userA_Landmarks = structuredClone(latestLandmarks);
    followCaptureABtn.disabled = true;
    followCaptureBBtn.disabled = false;
    followScoreDisplay.innerText = "첫 번째 표정 저장됨";
  };

  followCaptureBBtn.onclick = () => {
    if (!latestLandmarks || !userA_Landmarks) return alert("얼굴 인식 실패 또는 첫 번째 표정 없음");
    userB_Landmarks = structuredClone(latestLandmarks);

    // 임시 기준 이미지 랜드마크 (향후 static 분석으로 교체 가능)
    teacherA_Landmarks = userA_Landmarks;
    teacherB_Landmarks = userB_Landmarks.map(p => ({ x: p.x * 1.02, y: p.y * 1.02, z: p.z }));

    const normTeacherA = normalize(teacherA_Landmarks);
    const normTeacherB = normalize(teacherB_Landmarks);
    const normUserA = normalize(userA_Landmarks);
    const normUserB = normalize(userB_Landmarks);

    const deltaT = computeDelta(normTeacherB, normTeacherA);
    const deltaU = computeDelta(normUserB, normUserA);
    const score = computeScore(deltaT, deltaU);

    followScoreDisplay.innerHTML = `이번 점수: <b>${score} / 10</b>`;
    currentSessionTotalScore_Follow += score;
    followSessionDisplay.innerText = `총 점수: ${currentSessionTotalScore_Follow}`;

    followCaptureABtn.disabled = false;
    followCaptureBBtn.disabled = true;
    loadFollowImages();
  };
}

import { loadUnityGame } from './unity_loader.js';

export async function init() {
  console.log("🧠 표정 따라하기 모드 init()");
  await loadUnityGame();

  const loaded = await loadFollowModels();
  if (!loaded) return;

  const videoReady = await startVideoForFollowMode();
  if (!videoReady) return;

  await setupSingleFollowExercise_Follow();

  const btnA = document.getElementById('follow-captureA_Btn');
  const btnB = document.getElementById('follow-captureB_Btn');
  if (btnA && btnB) {
    btnA.removeEventListener('click', processFollowExpression_CaptureA);
    btnA.addEventListener('click', processFollowExpression_CaptureA);
    btnB.removeEventListener('click', processFollowExpression_CaptureB);
    btnB.addEventListener('click', processFollowExpression_CaptureB);
  }
}

export function closeFacialModal(mode = null, reason = "manual_close") {
  const modal = document.getElementById("facial-recognition-modal");
  if (modal) modal.style.display = "none";

  const score = currentSessionTotalScore ?? 0;

  // ✅ Unity로 점수 또는 중단 메시지 전송
  if (window.unityGameInstance) {
    if (reason.startsWith("session_ended_by_unity")) {
      console.log("✅ Unity로 최종 점수 전송:", score);
      window.unityGameInstance.SendMessage('CostManagerObject', 'ReceiveFacialScore', score);
    } else {
      console.log("🛑 Unity에 운동 중단 알림 전송");
      window.unityGameInstance.SendMessage('CostManagerObject', 'FacialExerciseAborted', 0);
    }
  }

  // ✅ 캠 스트림 정리
  const video = (mode === 'emotion_expression')
    ? document.getElementById("emotion-video")
    : document.getElementById("follow-video");

  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }

  // ✅ 점수 초기화
  currentSessionTotalScore = 0;
}

window.closeFacialModal = closeFacialModal;
window.ShowFacialRecognitionUI_JS = ShowFacialRecognitionUI_JS;