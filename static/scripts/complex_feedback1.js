

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

let faceMeshInstance = null;

async function setupFaceMesh() {
  if (!faceMeshInstance) {
    faceMeshInstance = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });
    faceMeshInstance.setOptions({
      staticImageMode: true,
      refineLandmarks: true,
      maxNumFaces: 1,
      minDetectionConfidence: 0.5,
    });

    // ✅ 결과 핸들러는 따로 필요 없음 (Promise로 처리)
    await faceMeshInstance.initialize(); // 미리 로드
  }
}

// ✅ 단일 이미지에서 136 벡터 추출
async function extractNormalizedLandmarksFromImage(src) {
  await setupFaceMesh();  // 최초 1회만 로드

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

      faceMeshInstance.onResults((results) => {
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

      await faceMeshInstance.send({ image: img });
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
  if (diff >= 1.0) return 2;
  if (diff >= 0.7) return 1.5;
  if (diff >= 0.4) return 1;
  if (diff >= 0.1) return 0.5;
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

  const ctx = document.getElementById("chart").getContext("2d");

  new Chart(ctx, {
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
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 10
        }
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

function logAUComparison(userAUs, refAUs) {
  console.log("📋 AU 벡터 비교 로그:");

  for (let i = 0; i < 10; i++) {
    const round = `ROUND ${i + 1}`;
    const refStr = refAUs[i].map(v => v.toFixed(4)).join(", ");
    const usrStr = userAUs[i].map(v => v.toFixed(4)).join(", ");
    console.log(`\n🔹 ${round}`);
    console.log(` 기준 AU: [${refStr}]`);
    console.log(` 사용자 AU: [${usrStr}]`);
  }
}


function renderImages(referenceUrls, userBase64List) {
  const refContainer = document.getElementById("reference-images");
  const userContainer = document.getElementById("user-images");

  // 초기화
  refContainer.innerHTML = "";
  userContainer.innerHTML = "";

  for (let i = 0; i < 10; i++) {
    // 기준 이미지 (URL)
    const refImg = document.createElement("img");
    refImg.src = referenceUrls[i];
    refImg.alt = `기준 이미지 ${i + 1}`;
    refImg.style.width = "9vw";
    refImg.style.minWidth = "60px";
    refImg.style.aspectRatio = "3 / 4";
    refImg.style.objectFit = "cover";
    refImg.style.borderRadius = "6px";
    refImg.style.border = "2px solid #ccc";
    refImg.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
    refContainer.appendChild(refImg);

    // 사용자 이미지 (base64)
    const userImg = document.createElement("img");
    userImg.src = userBase64List[i];
    userImg.alt = `사용자 이미지 ${i + 1}`;
    userImg.style.width = "9vw";
    userImg.style.minWidth = "60px";
    userImg.style.aspectRatio = "3 / 4";
    userImg.style.objectFit = "cover";
    userImg.style.borderRadius = "6px";
    userImg.style.border = "2px solid #ff4081";
    userImg.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
    userContainer.appendChild(userImg);
  }
}


const AU_MUSCLE_MAP = {
  1: "이마근 (눈썹 올리기)",
  2: "전두근 (눈썹 내측 올리기)",
  4: "추미근 (찡그리기)",
  5: "상안검거근 (눈 크게 뜨기)",
  6: "눈둘레근 (눈웃음)",
  7: "눈 감기근 (눈 감기)",
  9: "비근 (코 찡그리기)",
  10: "상순거근 (윗입술 들기)",
  12: "대관골근 (웃기)",
  14: "협근 (입 모으기)",
  15: "하순거근 (입꼬리 내리기)",
  17: "턱근 (입 다물기)",
  20: "입윤근 (입 내밀기)",
  23: "구륜근 (입 오므리기)",
  24: "구륜근 하부 (뾰로통)",
  25: "하악개근 (입 벌리기)",
  26: "악관절근 (턱 크게 벌리기)",
  27: "턱관절 상하근 (최대 개구)",
  28: "측두근 (어금니 꽉 물기)",
  45: "눈확근 (부드럽게 눈감기)"
};


function renderTopMuscles(userAUs) {
  const muscleList = document.getElementById("top-muscle-list");
  muscleList.innerHTML = "";

  // AU별 평균 구하기
  const auLength = userAUs[0].length;
  const auSums = Array(auLength).fill(0);

  for (const vec of userAUs) {
    vec.forEach((v, i) => auSums[i] += v);
  }

  const auAverages = auSums.map(sum => sum / userAUs.length);

  // AU 번호와 평균값 묶기
  const entries = auAverages.map((avg, i) => ({
    au: i + 1,
    name: AU_MUSCLE_MAP[i + 1] || `AU${i + 1}`,
    value: avg
  }));

  // 평균값 기준 정렬 후 상위 10개 추출
  const top10 = entries
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  // 리스트 렌더링
  top10.forEach((item, idx) => {
    const li = document.createElement("li");
    li.textContent = `${idx + 1}. ${item.name} (${item.value.toFixed(3)})`;
    muscleList.appendChild(li);
  });
}


async function startAnalysis() {
  const btn = document.getElementById("analyze-btn");

  const teacher = sessionStorage.getItem("selectedTeacher");
  const userBase64List = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");

  // 기준 이미지 URL 생성
  const referenceUrls = Array.from({ length: 10 }, (_, i) =>
    `/static/images/teachers/${teacher}/${teacher}${i + 1}.png`
  );

  // ✅ 이미지 표시
  renderImages(referenceUrls, userBase64List);


  try {
    const teacher = sessionStorage.getItem("selectedTeacher");
    const userBase64List = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");

    if (!teacher || userBase64List.length !== 10) {
      alert("❌ 저장된 이미지 데이터가 부족합니다.");
      if (btn) {
        btn.disabled = false;
        btn.textContent = "다시 시도하기";
      }
      return;
    }

    const referenceUrls = Array.from({ length: 10 }, (_, i) =>
      `/static/images/teachers/${teacher}/${teacher}${i + 1}.png`
    );

    // 🔹 전처리 및 예측 과정
    showLoadingMessage("이미지 랜드마크 추출 중...");
    const userVectors = await processImages(userBase64List);
    const refVectors = await processImages(referenceUrls);

    showLoadingMessage("AU 데이터 추출 중...");
    const userAUs = await predictAUs(userVectors);
    const refAUs = await predictAUs(refVectors);

    const loading = document.getElementById("loading-text");
    if (loading) loading.remove();

    if (!userAUs || !refAUs) {
      alert("❌ AU 예측 실패");
      if (btn) {
        btn.disabled = false;
        btn.textContent = "다시 시도하기";
      }
      return;
    }

    // 🔍 AU 비교 로그 출력
    logAUComparison(userAUs, refAUs);

    visualizeAUComparison(userAUs, refAUs);

    renderTopMuscles(userAUs);


  } catch (err) {
    console.error("❌ 예외 발생:", err);
    alert("예측 중 오류가 발생했습니다. 다시 시도해주세요.");
    const loading = document.getElementById("loading-text");
    if (loading) loading.remove();
    if (btn) {
      btn.disabled = false;
      btn.textContent = "다시 시도하기";
    }
  }
}


export function init() {
  console.log("✅ complex_feedback.js init 실행됨");

  const btn = document.getElementById("analyze-btn");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "분석 중...";
      await startAnalysis();     
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
