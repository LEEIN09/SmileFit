

// ✅ 필수 인덱스 (68개)
const REQUIRED_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109, 10, 338, 297, 332,
  284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21
].slice(0, 68);

// ✅ 단일 이미지에서 136 벡터 추출
async function extractNormalizedLandmarksFromImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = src;

    img.onload = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });
      faceMesh.setOptions({
        staticImageMode: true,
        refineLandmarks: true,
        maxNumFaces: 1,
        minDetectionConfidence: 0.5,
      });

      faceMesh.onResults((results) => {
        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
          console.warn("😢 얼굴 인식 실패:", src);
          resolve(null);
          return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        const selected = REQUIRED_INDICES.map(i => [landmarks[i].x, landmarks[i].y]);
        const nose = selected[30];
        const centered = selected.map(([x, y]) => [x - nose[0], y - nose[1]]);
        const left = centered[36];
        const right = centered[45];
        const dist = Math.hypot(left[0] - right[0], left[1] - right[1]) || 1e-6;
        const normalized = centered.map(([x, y]) => [x / dist, y / dist]);
        resolve(normalized.flat());
      });

      await faceMesh.send({ image: img });
    };

    img.onerror = (err) => {
      console.error("❌ 이미지 로드 실패:", src);
      reject(err);
    };
  });
}

// ✅ 이미지 리스트 처리 함수
async function processImages(imageList) {
  const results = [];

  for (let i = 0; i < imageList.length; i++) {
    const vector = await extractNormalizedLandmarksFromImage(imageList[i]);
    if (vector) results.push(vector);
    else results.push(Array(136).fill(0));
  }

  return results;
}

// ✅ AU 예측 API 호출
async function predictAUs(vectors) {
  const res = await fetch("/predict_aus", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vectors })
  });

  if (!res.ok) {
    console.error("❌ 예측 API 실패");
    return null;
  }

  const data = await res.json();
  return data.aus;
}


function getPenaltyFromDiff(diff) {
  if (diff >= 1.0) return 0.8;
  if (diff >= 0.7) return 0.6;
  if (diff >= 0.4) return 0.4;
  if (diff >= 0.1) return 0.2;
  // 0.1 미만은 아주 미세한 차이 → 평균 감점 방식 사용
  return diff * 5; // 0.1 미만이면 최대 0.5점 감점 (0.1 * 5 = 0.5)
}

function computeAUScore(userAU, refAU) {
  const diffs = refAU.map((val, i) => Math.abs(val - userAU[i]));
  let totalPenalty = 0;

  for (let diff of diffs) {
    totalPenalty += getPenaltyFromDiff(diff);
  }

  const maxScore = 10;
  const score = Math.max(0, maxScore - totalPenalty);
  return score;
}

// 라운드별 유사도 시각화
function visualizeAUComparison(userAUs, refAUs) {
  const scores = [];

  for (let i = 0; i < 10; i++) {
    const score = computeAUScore(userAUs[i], refAUs[i]);
    scores.push(score);
  }

  const labels = Array.from({ length: 10 }, (_, i) => `Round ${i + 1}`);

  const canvas = document.createElement("canvas");
  canvas.id = "au-chart";
  canvas.style.width = "100px";
  canvas.style.height = "60px";
  document.getElementById("app").appendChild(canvas);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '표정 점수 (10점 만점)',
        data: scores,
        backgroundColor: 'rgba(54, 162, 235, 0.6)'
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 10 }
      }
    }
  });
}

function showLoadingMessage(message) {
  let existing = document.getElementById("loading-text");
  if (!existing) {
    existing = document.createElement("p");
    existing.id = "loading-text";
    existing.style.textAlign = "center";
    existing.style.fontSize = "18px";
    document.getElementById("app").appendChild(existing);
  }
  existing.textContent = message;
}


async function startAnalysis() {
  let loading;
  const teacher = sessionStorage.getItem("selectedTeacher");
  const userBase64List = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");

  if (!teacher || userBase64List.length !== 10) {
    alert("❌ 저장된 이미지 데이터가 부족합니다.");
    return;
  }

  const referenceUrls = Array.from({ length: 10 }, (_, i) =>
    `/static/images/teachers/${teacher}/${teacher}${i + 1}.png`
  );


  // 👁‍🗨 전처리 및 예측
  showLoadingMessage("📊 사용자 이미지 랜드마크 추출 중...");
  const userVectors = await processImages(userBase64List);
  showLoadingMessage("📊 기준 이미지 랜드마크 추출 중...");
  const refVectors = await processImages(referenceUrls);

  showLoadingMessage("📊 사용자 AU 추출 중...");
  const userAUs = await predictAUs(userVectors);
  showLoadingMessage("📊 사용자 AU 추출 중...");
  const refAUs = await predictAUs(refVectors);

  loading = document.getElementById("loading-text");
  if (!userAUs || !refAUs) {
    if (loading) loading.remove();
    alert("❌ AU 예측 실패");
    return;
  }

  if (loading) loading.remove();
  visualizeAUComparison(userAUs, refAUs);
}

async function testSingleImageAU() {
  const teacher = sessionStorage.getItem("selectedTeacher");
  const userBase64List = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");

  const refUrl = `/static/images/teachers/${teacher}/${teacher}1.png`;
  const userImg = userBase64List[0];

  console.log("🧪 기준 이미지:", refUrl);
  console.log("🧪 사용자 이미지 (Base64):", userImg.slice(0, 100) + "...");

  const refVec = await extractNormalizedLandmarksFromImage(refUrl);
  const userVec = await extractNormalizedLandmarksFromImage(userImg);

  const refAU = await predictAUs([refVec]);
  const userAU = await predictAUs([userVec]);

  console.log("🔹 기준 AU 벡터:", refAU[0]);
  console.log("🔸 사용자 AU 벡터:", userAU[0]);

  const diff = refAU[0].map((val, i) => Math.abs(val - userAU[0][i]));
  console.log("⚠️ 차이:", diff);
}


export function init() {
  console.log("✅ complex_feedback.js init 실행됨");

  const btn = document.getElementById("analyze-btn");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "분석 중...";

      await testSingleImageAU();        // 테스트 먼저
      await startAnalysis();            // 분석 이후

      btn.style.display = "none";
    });
  }
}

export function cleanup() {
  const canvas = document.getElementById("au-chart");
  if (canvas) canvas.remove();
  const loadingText = document.getElementById("loading-text");
  if (loadingText) loadingText.remove();
}
