// complex_feedback.js (SPA 대응 개선 버전) - 기준 이미지와의 일치율 기반 평가
const Chart = window.Chart;
const FaceMesh = window.FaceMesh;

// ✅ 표정별 관련 랜드마크 인덱스 정의
const EXPRESSION_LANDMARKS = {
  "눈_크게_뜨기": [159, 145, 386, 374, 70, 63],
  "눈썹_내리기": [66, 105, 107, 55, 65, 52],
  "찡그리기": [8, 9, 168, 66, 105],
  "눈_꼭_감기": [159, 145, 386, 374, 133, 362],
  "볼_올림": [205, 425, 50, 280, 101, 330],
  "입꼬리_당김": [61, 291, 78, 308, 50, 280],
  "입꼬리_내림": [61, 291, 164, 393],
  "입술_내밀기": [0, 17, 84, 314, 13, 14, 87, 317],
  "입술_옆으로_당김": [61, 291, 78, 308, 14, 87],
  "윗입술_올림": [13, 14, 0, 61, 291],
  "입_벌리기": [13, 14, 87, 317, 152],
  "볼_부풀리기": [205, 425, 50, 280, 101, 330],
  "아랫입술_깨물기": [17, 84, 152, 164, 393]
};

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
    await faceMeshInstance.initialize();
  }
}

async function extractLandmarks(src) {
  await setupFaceMesh();

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
          resolve(null);
          return;
        }

        const landmarks = results.multiFaceLandmarks[0].map(pt => [pt.x, pt.y]);
        resolve(landmarks);
      });

      await faceMeshInstance.send({ image: img });
    };

    img.onerror = (err) => reject(err);
  });
}

async function extractLandmarksFromImages(imageList) {
  const result = [];
  for (let src of imageList) {
    const vector = await extractLandmarks(src);
    result.push(vector || []);
  }
  return result;
}

function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val ** 2, 0));
  const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val ** 2, 0));
  return dot && norm1 && norm2 ? dot / (norm1 * norm2) : 0;
}

function computeDiffVector(base, target, indices) {
  return indices.map(i => {
    const [bx, by] = base[i];
    const [tx, ty] = target[i];
    return Math.sqrt((bx - tx) ** 2 + (by - ty) ** 2);
  });
}

function compareExpressionSimilarity(refLandmarks, userLandmarks) {
  const result = [];
  for (let i = 0; i < refLandmarks.length - 1; i++) {
    const similarities = [];
    for (let expr in EXPRESSION_LANDMARKS) {
      const indices = EXPRESSION_LANDMARKS[expr];
      const refVec = computeDiffVector(refLandmarks[0], refLandmarks[i + 1], indices);
      const userVec = computeDiffVector(userLandmarks[0], userLandmarks[i + 1], indices);
      const sim = cosineSimilarity(refVec, userVec);
      similarities.push(sim);
    }
    const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    result.push(avgSim);
  }
  return result;
}

function renderChart(values) {
  const ctx = document.getElementById("chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: values.map((_, i) => `라운드 ${i + 1}`),
      datasets: [{
        label: "일치율 (%)",
        data: values.map(v => Math.round(v * 100)),
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      }
    }
  });
}

function getFeedback(avgScore) {
  if (avgScore > 0.9) return "완벽에 가까운 표정 일치! 멋져요!";
  if (avgScore > 0.7) return "훌륭한 표정 따라하기! 조금만 더 힘을 줘볼까요?";
  return "표정이 다소 약했어요. 더 힘을 줘서 따라해보세요!";
}

export async function init() {
  document.body.classList.add("loaded");
  // const dateSpan = document.getElementById("report-date");
  // const nameInput = document.getElementById("user-name");

  // const today = new Date().toLocaleDateString("ko-KR");
  // const name = nameInput?.value.trim() || "";
  
  // dateSpan.textContent = name ? `${today} - ${name}` : today;


  document.getElementById("analyze-btn").addEventListener("click", async () => {

  const dateSpan = document.getElementById("report-date");
  const nameInput = document.getElementById("user-name");

  const today = new Date().toLocaleDateString("ko-KR");
  const name = nameInput?.value.trim() || "";
  
  dateSpan.textContent = name ? `${today} - ${name}` : today;

    const teacher = sessionStorage.getItem("selectedTeacher");
    const refImages = [`/static/images/teachers/${teacher}/neutral.png`];
    for (let i = 1; i <= 10; i++) refImages.push(`/static/images/teachers/${teacher}/${teacher}${i}.png`);

    const userNeutral = JSON.parse(sessionStorage.getItem("neutralImage"));
    const userImages = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");
    const userAll = [userNeutral, ...userImages];

    const refLandmarks = await extractLandmarksFromImages(refImages);
    const userLandmarks = await extractLandmarksFromImages(userAll);

    const similarityScores = compareExpressionSimilarity(refLandmarks, userLandmarks);
    const avgScore = similarityScores.reduce((a, b) => a + b, 0) / similarityScores.length;

    renderChart(similarityScores);
    document.getElementById("summary-text").textContent = getFeedback(avgScore);

    const refContainer = document.getElementById("reference-images");
    refImages.slice(1).forEach(src => {
      const img = document.createElement("img");
      img.src = src;
      refContainer.appendChild(img);
    });

    const userContainer = document.getElementById("user-images");
    userImages.forEach(src => {
      const img = document.createElement("img");
      img.src = src;
      userContainer.appendChild(img);
    });

    // 사용자 전체 근육 사용량 계산 (neutral 대비 1~10)
    const topMuscles = Object.keys(EXPRESSION_LANDMARKS).map(expr => {
      const indices = EXPRESSION_LANDMARKS[expr];
      let total = 0;
      for (let i = 1; i < userLandmarks.length; i++) {
        const vec = computeDiffVector(userLandmarks[0], userLandmarks[i], indices);
        total += vec.reduce((a, b) => a + b, 0);
      }
      return { expr, total };
    }).sort((a, b) => b.total - a.total).slice(0, 5);

    const muscleList = document.getElementById("top-muscle-list");
    topMuscles.forEach(m => {
      const li = document.createElement("li");
      li.textContent = `${m.expr} (${m.total.toFixed(2)})`;
      muscleList.appendChild(li);
    });

    // 동작 카운팅 기반 top5
    const expressionCount = {};
    for (let i = 1; i < userLandmarks.length; i++) {
      let bestMatch = null;
      let bestSim = -1;
      for (let expr in EXPRESSION_LANDMARKS) {
        const indices = EXPRESSION_LANDMARKS[expr];
        const vec = computeDiffVector(userLandmarks[0], userLandmarks[i], indices);
        const refVec = computeDiffVector(refLandmarks[0], refLandmarks[i], indices);
        const sim = cosineSimilarity(refVec, vec);
        if (sim > bestSim) {
          bestSim = sim;
          bestMatch = expr;
        }
      }
      if (bestMatch) {
        expressionCount[bestMatch] = (expressionCount[bestMatch] || 0) + 1;
      }
    }
    const sortedExpressions = Object.entries(expressionCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const exprList = document.getElementById("top-expression-list");
    sortedExpressions.forEach(([expr, count]) => {
      const li = document.createElement("li");
      li.textContent = `${expr} (${count}회)`;
      exprList.appendChild(li);
    });

    document.getElementById("report").style.display = "block";
  });
}