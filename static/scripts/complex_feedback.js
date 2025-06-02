// complex_feedback.js (SPA 대응 개선 버전) - 기준 이미지와의 일치율 기반 평가
const Chart = window.Chart;
const FaceMesh = window.FaceMesh;

// ✅ 표정별 관련 랜드마크 인덱스 정의

const MAX_CHANGES = {
  "전두근": 0.136,
  "안륜근": 0.047,
  "추미근": 0.072,
  "콧방울올림근": 0.143,
  "대관골근": 0.048,
  "익돌근": 0.09,
  "구륜근": 0.017,
  "볼근": 0.017,
};

const MUSCLE_TO_ACTION = {
  "전두근": "오른쪽 눈썹 올리기",
  "안륜근": "눈 작게 뜨기",
  "추미근": "미간 조이기",
  "콧방울올림근": "찡그리기",
  "대관골근": "입꼬리 올리기",
  "익돌근": "입 벌리기",
  "구륜근": "입술 오므리기",
  "볼근": "보조개 만들기"
};

// 근육별 제어 기준 정의
const MUSCLE_RULES = {
  "전두근": { points: [334, 386], direction: "increase" },
  "안륜근": { points: [386, 374], direction: "decrease" },
  "추미근": { points: [107, 336], direction: "decrease" },
  "콧방울올림근": { points: [285, 437], direction: "decrease" },
  "대관골근": { points: [291, 446], direction: "decrease" },
  "익돌근": { points: [1, 152], direction: "increase" },
  "구륜근": { points: [61, 291], direction: "decrease" },
  "볼근": { points: [61, 291], direction: "increase", stable: [13, 14] }  // AND 조건
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

    // 거리 계산 유틸 함수
  function computeDist(landmarks, [i1, i2]) {
    const [x1, y1] = landmarks[i1];
    const [x2, y2] = landmarks[i2];
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
  }

  function evaluateRoundScores(refLandmarks, userLandmarks) {
    const roundScores = [];

    for (let i = 1; i < userLandmarks.length; i++) {
      const perMuscleScores = [];

      for (let muscle in MUSCLE_RULES) {
        const rule = MUSCLE_RULES[muscle];
        const maxChange = MAX_CHANGES[muscle] || 1;

        const refDist = computeDist(refLandmarks[0], rule.points);
        const refExprDist = computeDist(refLandmarks[i], rule.points);
        const refDiff = refExprDist - refDist;
        const refRatio = Math.abs(refDiff) / maxChange;

        const userDist = computeDist(userLandmarks[0], rule.points);
        const userExprDist = computeDist(userLandmarks[i], rule.points);
        const userDiff = userExprDist - userDist;
        const userRatio = Math.abs(userDiff) / maxChange;

        // 볼근 예외 처리
        if (muscle === "볼근" && rule.stable) {
          const refStable = Math.abs(computeDist(refLandmarks[i], rule.stable) - computeDist(refLandmarks[0], rule.stable));
          const userStable = Math.abs(computeDist(userLandmarks[i], rule.stable) - computeDist(userLandmarks[0], rule.stable));
          if (refStable > 0.01 || userStable > 0.01) {
            continue; // 중앙 입술 변화 크면 제외
          }
        }

        const score = 1 - Math.abs(refRatio - userRatio);  // 유사할수록 높음
        perMuscleScores.push(score);
      }

      const roundAvg = perMuscleScores.reduce((a, b) => a + b, 0) / perMuscleScores.length;
      roundScores.push(roundAvg);
    }

    return roundScores;
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

function getFeedbackPerMuscle(refLandmarks, userLandmarks) {
  const underused = [];

  for (let muscle in MUSCLE_RULES) {
    const rule = MUSCLE_RULES[muscle];
    const maxChange = MAX_CHANGES[muscle] || 1;

    let refSum = 0;
    let userSum = 0;

    for (let i = 1; i < userLandmarks.length; i++) {
      const refDist = computeDist(refLandmarks[0], rule.points);
      const refExprDist = computeDist(refLandmarks[i], rule.points);
      const refRatio = Math.abs(refExprDist - refDist) / maxChange;

      const userDist = computeDist(userLandmarks[0], rule.points);
      const userExprDist = computeDist(userLandmarks[i], rule.points);
      const userRatio = Math.abs(userExprDist - userDist) / maxChange;

      refSum += refRatio;
      userSum += userRatio;
    }

    const refAvg = refSum / (userLandmarks.length - 1);
    const userAvg = userSum / (userLandmarks.length - 1);

    const ratioDiff = userAvg / (refAvg || 1);

    if (ratioDiff < 0.7) {  // 기준치보다 60% 미만 사용 시 "활용 부족"
      underused.push({
        muscle,
        action: MUSCLE_TO_ACTION[muscle],
        percent: Math.round(ratioDiff * 100)
      });
    }
  }

  if (underused.length === 0) {
    return "모든 사용된 근육들이 잘 활성화 되었어요! 👏";
  }

  let message = "다음 근육이 상대적으로 덜 활성화 되었어요:\n";
  underused.forEach(({ muscle, action, percent }) => {
    message += `- ${action} (${percent}% 활용)\n`;
  });

  message += "이 동작들을 조금 더 집중적으로 연습해볼까요?";
  return message;
}

const pieColors = [
  "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#CCCCCC"
];

export async function init() {
  document.body.classList.add("loaded");
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

    const similarityScores = evaluateRoundScores(refLandmarks, userLandmarks);
    const avgScore = similarityScores.reduce((a, b) => a + b, 0) / similarityScores.length;

    renderChart(similarityScores);
    document.getElementById("summary-text").textContent = getFeedbackPerMuscle(refLandmarks, userLandmarks);

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

    // 근육 사용량 계산
    const topMuscles = Object.entries(MUSCLE_RULES).map(([muscle, rule]) => {
      let usageSum = 0;
      let count = 0;

      for (let i = 1; i < userLandmarks.length; i++) {
        const neutralDist = computeDist(userLandmarks[0], rule.points);
        const exprDist = computeDist(userLandmarks[i], rule.points);
        const diff = exprDist - neutralDist;

        let activated = false;
        if (rule.direction === "increase" && diff > 0) activated = true;
        if (rule.direction === "decrease" && diff < 0) activated = true;

        // 볼근은 보조 조건 추가: 중앙 입술 변화가 거의 없어야 함
        if (muscle === "볼근") {
          const lipChange = computeDist(userLandmarks[0], rule.stable) - computeDist(userLandmarks[i], rule.stable);
          if (Math.abs(lipChange) > 0.01) activated = false;
        }

        // 구륜근은 입 벌리기와의 간섭 방지 조건 추가
        if (muscle === "구륜근") {
          const lipToNose = computeDist(userLandmarks[i], [14, 1]);
          if (lipToNose > MAX_CHANGES["익돌근"] / 5) activated = false;
        }

        if (activated) {
          const ratio = Math.abs(diff) / (MAX_CHANGES[muscle] || 1);
          usageSum += ratio;
          count++;
        }
      }

      const avgUsage = count > 0 ? usageSum / count : 0;
      return { expr: muscle, usage: avgUsage };
    })
    .sort((a, b) => b.usage - a.usage)
    .slice(0, 5);

    // 결과 출력
    const topMusclesFull = Object.entries(MUSCLE_RULES).map(([muscle, rule]) => {
      let usage = 0;
      for (let i = 1; i < userLandmarks.length; i++) {
        const neutralDist = computeDist(userLandmarks[0], rule.points);
        const exprDist = computeDist(userLandmarks[i], rule.points);
        const diff = exprDist - neutralDist;

        let activated = false;
        if (rule.direction === "increase" && diff > 0) activated = true;
        if (rule.direction === "decrease" && diff < 0) activated = true;

        if (muscle === "볼근") {
          const lipChange = computeDist(userLandmarks[0], rule.stable) - computeDist(userLandmarks[i], rule.stable);
          if (Math.abs(lipChange) > 0.01) activated = false;
        }

        if (muscle === "구륜근") {
          const verticalChange = computeDist(userLandmarks[0], [14, 1]) - computeDist(userLandmarks[i], [14, 1]);
          const maxJaw = MAX_CHANGES["익돌근"] || 1;
          if (Math.abs(verticalChange) > maxJaw / 5) activated = false;
        }

        const ratio = Math.abs(diff) / (MAX_CHANGES[muscle] || 1);
        if (activated) usage += ratio;
      }
      return { expr: muscle, usage };
    }).sort((a, b) => b.usage - a.usage);

    // ✅ Pie chart용 데이터 준비
    const totalUsage = topMusclesFull.reduce((sum, m) => sum + m.usage, 0);
    const pieData = [];
    const pieLabels = [];

    topMusclesFull.slice(0, 5).forEach(m => {
      const percent = (m.usage / totalUsage) * 100;
      pieLabels.push(m.expr);
      pieData.push(percent);
    });

    const othersUsage = topMusclesFull.slice(5).reduce((sum, m) => sum + m.usage, 0);
    if (othersUsage > 0) {
      const othersPercent = (othersUsage / totalUsage) * 100;
      pieLabels.push("기타");
      pieData.push(othersPercent);
    }

    document.getElementById("report").style.display = "block";

    // ✅ 원형 차트 렌더링
    const pieCtx = document.getElementById("topMusclePieChart").getContext("2d");
    new Chart(pieCtx, {
      type: "pie",
      data: {
        labels: pieLabels,
        datasets: [{
          data: pieData.map(v => parseFloat(v.toFixed(1))),
          backgroundColor: [
            "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#CCCCCC"
          ]
        }]
      },
      options: {
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.parsed;
                return `${label}: ${value.toFixed(1)}%`;
              }
            }
          }
        }
      }
    });

    const labelContainer = document.getElementById("pie-labels");
    labelContainer.innerHTML = "";

    pieLabels.forEach((label, i) => {
      const percent = pieData[i].toFixed(1);
      const color = pieColors[i];  // 차트 색상 배열

      const li = document.createElement("li");
      li.style.display = "flex";
      li.style.alignItems = "center";
      li.style.marginBottom = "8px";

      li.innerHTML = `
        <span style="display:inline-block; width: 12px; height: 12px; background-color: ${color}; margin-right: 8px; border-radius: 2px;"></span>
        <strong style="margin-right: 6px;">${label}</strong>
        <span style="font-size: 14px; color: #555;">${percent}%</span>
      `;

      document.getElementById("pie-labels").appendChild(li);
    });

    // 동작 카운팅 기반 top5
    const exprList = document.getElementById("top-expression-list");
    topMuscles.forEach((m, index) => {
      const action = MUSCLE_TO_ACTION[m.expr] || m.expr;
      const li = document.createElement("li");
      li.textContent = `${index + 1}. ${action}`;
      li.style.marginBottom = "8px";
      exprList.appendChild(li);
    });

    
  });
}