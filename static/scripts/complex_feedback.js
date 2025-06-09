// complex_feedback.js

// Mediapipe FaceMesh 및 Chart.js 인스턴스 (HTML에서 전역으로 로드된다고 가정)
const Chart = window.Chart;
const FaceMesh = window.FaceMesh;

// 표정별 관련 랜드마크 인덱스 및 근육별 제어 기준 (기존 정의 유지)
const MAX_CHANGES = {
  "전두근": 0.136,
  "안륜근": 0.047,
  "추미근": 0.072,
  "상순비익거근": 0.143,
  "대관골근": 0.048,
  "익돌근": 0.09,
  "상순절치근": 0.017,
  "협근": 0.017,
};

const MUSCLE_TO_ACTION = {
  "전두근": "눈썹 올리기",
  "안륜근": "눈 강하게 감기",
  "추미근": "미간 조이기",
  "상순비익거근": "찡그리기",
  "대관골근": "입꼬리 올리기",
  "익돌근": "입 벌리기",
  "상순절치근": "입술 오므리기",
  "협근": "보조개 만들기"
};

const MUSCLE_RULES = {
  "전두근": { points: [334, 386], direction: "increase" },
  "안륜근": { points: [386, 374], direction: "decrease" },
  "추미근": { points: [107, 336], direction: "decrease" },
  "상순비익거근": { points: [285, 437], direction: "decrease" },
  "대관골근": { points: [291, 446], direction: "decrease" },
  "익돌근": { points: [1, 152], direction: "increase" }, // 코 끝(1)과 턱(152) 사이의 거리로 되돌림
  "상순절치근": { points: [61, 291], direction: "decrease" },
  "협근": { points: [61, 291], direction: "increase", stable: [13, 14] }
};


const pieColors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#CCCCCC", "#FF9F40", "#C9CBCF"];
let faceMeshInstance = null;

// 라운드 수 상수
const NORMAL_MODE_TOTAL_ROUNDS = 10;
const AU_MODE_TOTAL_ROUNDS = 8;

// FaceMesh 설정
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
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log("✅ MediaPipe FaceMesh가 성공적으로 초기화 및 준비되었습니다.");
  }
}

// 이미지에서 랜드마크 추출
async function extractLandmarks(src) {
  await setupFaceMesh();
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = src;

    img.onload = () => {
      setTimeout(async () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        faceMeshInstance.onResults((results) => {
          if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            console.warn(`랜드마크 추출 실패: ${src.substring(0, 50)}...`);
            resolve(null);
            return;
          }
          const landmarks = results.multiFaceLandmarks[0].map(pt => [pt.x, pt.y]);
          resolve(landmarks);
        });

        await faceMeshInstance.send({ image: canvas });
      }, 50);
    };

    img.onerror = (err) => {
      console.error(`이미지 로드 오류: ${src.substring(0, 50)}...`, err);
      reject(err);
    };
  });
}

// 이미지 리스트에서 랜드마크 추출
async function extractLandmarksFromImages(imageList) {
  const result = [];
  for (let src of imageList) {
    try {
      const vector = await extractLandmarks(src);
      result.push(vector);
    } catch (error) {
      console.error(`extractLandmarksFromImages 오류 (${src}):`, error);
      result.push(null);
    }
  }
  return result;
}

// 거리 계산 유틸 함수
function computeDist(landmarks, [i1, i2]) {
  if (!landmarks || !landmarks[i1] || !landmarks[i2]) {
    return 0;
  }
  const [x1, y1] = landmarks[i1];
  const [x2, y2] = landmarks[i2];
  return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
}


// 라운드별 점수 평가 (Normal 모드용)
function evaluateRoundScores(refLandmarks, userLandmarks) {
  const roundScores = [];
  if (!userLandmarks[0] || !refLandmarks[0]) {
    console.error("중립 표정 랜드마크가 없어 점수 평가를 진행할 수 없습니다.");
    const numRounds = Math.min(refLandmarks.length -1, userLandmarks.length - 1);
    return Array(numRounds > 0 ? numRounds : 0).fill(null);
  }

  const numExpressionRounds = Math.min(refLandmarks.length - 1, userLandmarks.length - 1);

  for (let i = 1; i <= numExpressionRounds; i++) {
    if (!refLandmarks[i] || !userLandmarks[i]) {
      roundScores.push(null);
      continue;
    }

    const perMuscleScores = [];
    for (let muscle in MUSCLE_RULES) {
      const rule = MUSCLE_RULES[muscle];
      const maxChange = MAX_CHANGES[muscle] || 1;

      const refDistNeutral = computeDist(refLandmarks[0], rule.points);
      const refDistExpr = computeDist(refLandmarks[i], rule.points);
      const refDiff = refDistExpr - refDistNeutral;

      const userDistNeutral = computeDist(userLandmarks[0], rule.points);
      const userDistExpr = computeDist(userLandmarks[i], rule.points);
      const userDiff = userDistExpr - userDistNeutral;
      
      const refRatio = maxChange !== 0 ? Math.abs(refDiff) / maxChange : 0;
      const userRatio = maxChange !== 0 ? Math.abs(userDiff) / maxChange : 0;
      
      let score = 0;
      if ((rule.direction === "increase" && refDiff > 0 && userDiff > 0) ||
          (rule.direction === "decrease" && refDiff < 0 && userDiff < 0) ||
          (refDiff === 0 && userDiff === 0)) {
            score = 1 - Math.abs(refRatio - userRatio);
      } else if (Math.sign(refDiff) !== Math.sign(userDiff) && refDiff !== 0 && userDiff !== 0) {
            score = 0;
      } else {
            score = Math.max(0, 0.5 - Math.abs(refRatio - userRatio));
      }
      
      if (muscle === "협근" && rule.stable) {
          const refStableDistNeutral = computeDist(refLandmarks[0], rule.stable);
          const refStableDistExpr = computeDist(refLandmarks[i], rule.stable);
          const userStableDistNeutral = computeDist(userLandmarks[0], rule.stable);
          const userStableDistExpr = computeDist(userLandmarks[i], rule.stable);

          const refStableChange = Math.abs(refStableDistExpr - refStableDistNeutral);
          const userStableChange = Math.abs(userStableDistExpr - userStableDistNeutral);

          if (refStableChange > 0.015 || userStableChange > 0.015) {
             continue;
          }
      }
      perMuscleScores.push(Math.max(0, Math.min(1, score)));
    }
    const roundAvg = perMuscleScores.length > 0 ? perMuscleScores.reduce((a, b) => a + b, 0) / perMuscleScores.length : 0;
    roundScores.push(roundAvg);
  }
  return roundScores;
}

// 막대 차트 렌더링
function renderChart(values, totalRoundsForMode) {
  const ctx = document.getElementById("chart").getContext("2d");
  const labels = Array.from({ length: totalRoundsForMode }, (_, i) => `라운드 ${i + 1}`);
  
  const dataValues = values.map(v => (v === null || v === undefined) ? 0 : Math.round(v * 100));

  if (window.myBarChart instanceof Chart) {
    window.myBarChart.destroy();
  }
  window.myBarChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.slice(0, dataValues.length),
      datasets: [{
        label: "일치율 (%)",
        data: dataValues,
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 100 } }
    }
  });
}

// 종합 피드백 생성 (Normal 모드용)
function generateSummaryFeedback(avgScore, usedMusclesCount) {
  const lines = [];
  lines.push("오늘도 얼굴 운동 하느라 수고 많았어요! 😊");

  if (usedMusclesCount >= 5) {
    lines.push("다양한 근육들을 골고루 사용하셨습니다. 👏");
  } else if (usedMusclesCount > 0) {
    lines.push("몇 가지 주요 근육을 잘 사용하셨어요. 다음엔 더 다양하게 도전해봐요! 💪");
  } else {
    lines.push("앗, 근육 사용이 감지되지 않았어요. 다음에는 선생님 표정을 조금 더 적극적으로 따라해볼까요? 😉");
  }

  if (avgScore >= 0.7) {
    lines.push("선생님의 사진을 매우 잘 따라했습니다. 👍");
  } else if (avgScore >= 0.4) {
    lines.push("선생님 표정을 잘 따라하려고 노력하셨네요! 조금만 더 힘내봐요! 😊");
  } else {
     lines.push("괜찮아요! 처음엔 어려울 수 있어요. 꾸준히 연습하면 분명 좋아질 거예요. 😊");
  }
  lines.push("항상 스마일핏과 함께 즐겁고 활기찬 운동 되시길 바랍니다! 🌟");
  return lines.join("<br><br>");
}


// 클라이언트 측 근육 사용량 분석 (Normal 모드용)
function calculateClientSideMuscleAnalysis(userLandmarks, muscleRules, maxChanges, muscleToAction) {
  if (!userLandmarks || !userLandmarks[0]) {
    return { topMusclesFull: [], activatedMusclesList: [] };
  }

  const topMusclesFull = Object.entries(muscleRules).map(([muscle, rule]) => {
    let totalNormalizedDiff = 0;
    let activationCount = 0;

    for (let i = 1; i < userLandmarks.length; i++) {
      if (!userLandmarks[i]) continue;

      const neutralDist = computeDist(userLandmarks[0], rule.points);
      const exprDist = computeDist(userLandmarks[i], rule.points);
      const diff = exprDist - neutralDist;
      
      let activatedInRound = false;
      if (rule.direction === "increase" && diff > 0.005) activatedInRound = true;
      if (rule.direction === "decrease" && diff < -0.005) activatedInRound = true;

      if (muscle === "협근" && rule.stable) {
        const lipChange = Math.abs(computeDist(userLandmarks[0], rule.stable) - computeDist(userLandmarks[i], rule.stable));
        if (lipChange > 0.015) activatedInRound = false;
      }
      if (muscle === "상순절치근") {
          const jawOpeningPoints = MUSCLE_RULES["익돌근"].points;
          const jawNeutral = computeDist(userLandmarks[0], jawOpeningPoints);
          const jawExpr = computeDist(userLandmarks[i], jawOpeningPoints);
          const jawDiff = jawExpr - jawNeutral;
          if (jawDiff > (MAX_CHANGES["익돌근"] || 0.09) * 0.3) {
             activatedInRound = false;
          }
      }

      if (activatedInRound) {
        totalNormalizedDiff += Math.abs(diff) / (maxChanges[muscle] || 1);
        activationCount++;
      }
    }
    const usage = activationCount > 0 ? totalNormalizedDiff / activationCount : 0;
    return { expr: muscle, usage: usage };
  }).sort((a, b) => b.usage - a.usage);


  const activatedMusclesList = [];
  Object.entries(muscleRules).forEach(([muscle, rule]) => {
    let activatedOnce = false;
    for (let i = 1; i < userLandmarks.length; i++) {
      if (!userLandmarks[i] || !userLandmarks[0]) continue;
      const neutralDist = computeDist(userLandmarks[0], rule.points);
      const exprDist = computeDist(userLandmarks[i], rule.points);
      const diff = exprDist - neutralDist;
      let activatedInRound = false;
      if (rule.direction === "increase" && diff > 0.005) activatedInRound = true;
      if (rule.direction === "decrease" && diff < -0.005) activatedInRound = true;

      if (muscle === "협근" && rule.stable) {
        const lipChange = Math.abs(computeDist(userLandmarks[0], rule.stable) - computeDist(userLandmarks[i], rule.stable));
        if (lipChange > 0.015) activatedInRound = false;
      }
      if (muscle === "상순절치근") {
          const jawOpeningPoints = MUSCLE_RULES["익돌근"].points;
          const jawNeutral = computeDist(userLandmarks[0], jawOpeningPoints);
          const jawExpr = computeDist(userLandmarks[i], jawOpeningPoints);
          const jawDiff = jawExpr - jawNeutral;
          if (jawDiff > (MAX_CHANGES["익돌근"] || 0.09) * 0.3) {
             activatedInRound = false;
          }
      }

      if (activatedInRound) {
        activatedOnce = true;
        break;
      }
    }
    if (activatedOnce) {
      activatedMusclesList.push({ muscle: muscle, action: muscleToAction[muscle] || "-" });
    }
  });
  return { topMusclesFull, activatedMusclesList };
}

// 사용된 근육 리스트 DOM 업데이트
function populateActivatedMusclesList(activatedMusclesListData) {
  const exprList = document.getElementById("top-expression-list");
  exprList.innerHTML = "";
  if (!activatedMusclesListData || activatedMusclesListData.length === 0) {
    exprList.innerHTML = "<li>데이터가 없습니다.</li>";
    return;
  }
  activatedMusclesListData.forEach(item => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${item.muscle}</strong> – ${item.action}`;
    li.style.marginBottom = "6px";
    exprList.appendChild(li);
  });
}

// 파이 차트 및 라벨 렌더링
function renderPieChartWithLabels(dataForPie, serverProvidedLabels = null) {
  let pieDataValues = [];
  let pieLabelTexts = [];

  if (serverProvidedLabels) {
    pieDataValues = dataForPie;
    pieLabelTexts = serverProvidedLabels;
  } else {
    const validMuscles = dataForPie.filter(m => m.usage > 0);
    if (validMuscles.length === 0) {
        document.getElementById("pie-chart-container").innerHTML = "<p>분석된 근육 사용량 데이터가 없습니다.</p>";
        return;
    }
    const totalUsage = validMuscles.reduce((sum, m) => sum + m.usage, 0);
    if (totalUsage === 0) {
        document.getElementById("pie-chart-container").innerHTML = "<p>계산된 근육 사용량이 없습니다.</p>";
        return;
    }
    
    const top5Muscles = validMuscles.slice(0, 5);
    top5Muscles.forEach(m => {
      const percent = (m.usage / totalUsage) * 100;
      pieLabelTexts.push(m.expr);
      pieDataValues.push(parseFloat(percent.toFixed(1)));
    });

    const othersUsage = validMuscles.slice(5).reduce((sum, m) => sum + m.usage, 0);
    if (othersUsage > 0) {
      const othersPercent = (othersUsage / totalUsage) * 100;
      pieLabelTexts.push("기타");
      pieDataValues.push(parseFloat(othersPercent.toFixed(1)));
    }
  }

  if (pieDataValues.length === 0) {
    document.getElementById("pie-chart-container").innerHTML = "<p>표시할 근육 사용량 데이터가 없습니다.</p>";
    return;
  }

  const pieCtx = document.getElementById("topMusclePieChart").getContext("2d");
  if (window.myPieChart instanceof Chart) {
    window.myPieChart.destroy();
  }
  window.myPieChart = new Chart(pieCtx, {
    type: "pie",
    data: {
      labels: pieLabelTexts,
      datasets: [{ data: pieDataValues, backgroundColor: pieColors.slice(0, pieDataValues.length) }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
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
  pieLabelTexts.forEach((label, i) => {
    const percent = pieDataValues[i].toFixed(1);
    const color = pieColors[i % pieColors.length];
    const li = document.createElement("li");
    li.style.cssText = "display: flex; align-items: center; margin-bottom: 8px;";
    li.innerHTML = `
      <span style="display:inline-block; width:12px; height:12px; background-color:${color}; margin-right:8px; border-radius:2px;"></span>
      <strong style="margin-right:6px;">${label}</strong>
      <span style="font-size:14px; color:#555;">${percent}%</span>`;
    labelContainer.appendChild(li);
  });
}


// 페이지 초기화 함수
export async function init() {
  document.body.classList.add("loaded");

  document.getElementById("analyze-btn").addEventListener("click", async () => {
    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = true;

    const urlParams = new URLSearchParams(window.location.search);
    const currentMode = urlParams.get('mode') || sessionStorage.getItem('mode');
    const teacher = urlParams.get('teacher_id') || sessionStorage.getItem('selectedTeacher');

    const reportDateSpan = document.getElementById("report-date");
    const nameInput = document.getElementById("user-name");
    const today = new Date().toLocaleDateString("ko-KR", { year: 'numeric', month: 'long', day: 'numeric' });
    const name = nameInput?.value.trim() || "사용자";
    reportDateSpan.textContent = name ? `${today} - ${name}` : today;

    const reportModeNameSpan = document.getElementById("report-mode-name");
    if (reportModeNameSpan) {
      reportModeNameSpan.textContent = (currentMode === 'au') ? 'AU 분석' : '일반운동';
    }
    document.getElementById("summary-text").innerHTML = "분석 결과를 기다리고 있습니다...";


    if (currentMode === 'normal') {
      analyzeBtn.textContent = "분석 중... (Normal)";
      try {
        const userNeutralSrc = sessionStorage.getItem("neutralImage");
        const userImageSrcs = JSON.parse(sessionStorage.getItem("capturedImages") || "[]");
        
        if (!userNeutralSrc || userImageSrcs.length === 0) {
          throw new Error("일반 모드: 사용자 이미지를 세션 스토리지에서 찾을 수 없습니다. 운동을 다시 진행해주세요.");
        }
        if (userImageSrcs.length !== NORMAL_MODE_TOTAL_ROUNDS) {
             console.warn(`일반 모드: 예상된 라운드 수(${NORMAL_MODE_TOTAL_ROUNDS})와 실제 캡처된 이미지 수(${userImageSrcs.length})가 다릅니다.`);
        }
        const userAllSrcs = [userNeutralSrc, ...userImageSrcs];
        
        const refImagePaths = [`/static/images/teachers/${teacher}/neutral.png`];
        for (let i = 1; i <= NORMAL_MODE_TOTAL_ROUNDS; i++) {
          refImagePaths.push(`/static/images/teachers/${teacher}/${teacher}${i}.png`);
        }

        const refContainer = document.getElementById("reference-images");
        refContainer.innerHTML = '';
        refImagePaths.slice(1).forEach(src => { const img = document.createElement("img"); img.src = src; refContainer.appendChild(img); });
        document.getElementById("reference-images-title").textContent = `기준 사진 (${NORMAL_MODE_TOTAL_ROUNDS}장)`;

        const userContainer = document.getElementById("user-images");
        userContainer.innerHTML = '';
        userImageSrcs.forEach(src => { const img = document.createElement("img"); img.src = src; userContainer.appendChild(img); });
        document.getElementById("user-images-title").textContent = `사용자 사진 (${userImageSrcs.length}장)`;
        document.getElementById("user-images-section").style.display = "block";
        document.getElementById("similarity-chart-title").textContent = "라운드별 유사도";


        const refLandmarks = await extractLandmarksFromImages(refImagePaths);
        const userLandmarks = await extractLandmarksFromImages(userAllSrcs);

        if (!refLandmarks[0] || !userLandmarks[0]) {
          throw new Error("중립 표정 이미지의 얼굴을 인식할 수 없어 분석을 진행할 수 없습니다. 사진을 확인해주세요.");
        }
        const missingRefRounds = refLandmarks.slice(1).map((lm, idx) => lm ? -1 : idx+1).filter(idx => idx !== -1);
        const missingUserRounds = userLandmarks.slice(1).map((lm, idx) => lm ? -1 : idx+1).filter(idx => idx !== -1);
        if (missingRefRounds.length > 0) console.warn(`기준 사진 중 라운드 ${missingRefRounds.join(', ')}의 얼굴 인식이 불안정합니다.`);
        if (missingUserRounds.length > 0) console.warn(`사용자 사진 중 라운드 ${missingUserRounds.join(', ')}의 얼굴 인식이 불안정합니다.`);


        const similarityScores = evaluateRoundScores(refLandmarks, userLandmarks);
        renderChart(similarityScores, NORMAL_MODE_TOTAL_ROUNDS);
        document.getElementById("similarity-chart-container").style.display = "block";


        const { topMusclesFull, activatedMusclesList } = calculateClientSideMuscleAnalysis(userLandmarks, MUSCLE_RULES, MAX_CHANGES, MUSCLE_TO_ACTION);
        renderPieChartWithLabels(topMusclesFull);
        populateActivatedMusclesList(activatedMusclesList);
        document.getElementById("pie-chart-container").style.display = "block";


        const validScores = similarityScores.filter(s => s !== null);
        const avgScore = validScores.length > 0 ? validScores.reduce((a, b) => a + b, 0) / validScores.length : 0;
        const summaryHTML = generateSummaryFeedback(avgScore, topMusclesFull.filter(m => m.usage > 0).length);
        document.getElementById("summary-text").innerHTML = summaryHTML;

        document.getElementById("report").style.display = "block";
        analyzeBtn.textContent = "분석 완료 ✅ (Normal)";

      } catch (err) {
        console.error("Normal 모드 피드백 오류:", err);
        analyzeBtn.textContent = "분석 실패 ❌ (Normal)";
        document.getElementById("summary-text").innerHTML = `<strong style="color:red;">오류 (일반 모드):</strong> ${err.message}. <br>문제가 지속되면 관리자에게 문의하세요.`;
        document.getElementById("report").style.display = "block";
      }

    // ▼▼▼▼▼ [수정된 부분] AU 분석 모드 로직 ▼▼▼▼▼
    } else if (currentMode === 'au') {
      analyzeBtn.textContent = "AU 결과 로딩 중...";
      try {
        // 1. 불필요한 UI 요소 숨기기 (일반 모드용 UI)
        document.getElementById("pie-chart-container").style.display = "none";
        document.getElementById("top-expression-list").parentElement.style.display = "none"; // 부모인 feedback-box를 숨김
        document.getElementById("similarity-chart-title").textContent = "AU 라운드별 점수";

        // 2. 올바른 Endpoint로 서버에 데이터 요청
        const response = await fetch(`/get_user_feedback_data?teacher_id=${teacher}`);
        if (!response.ok) {
          const errorResult = await response.json().catch(() => ({ message: "서버 응답을 확인할 수 없습니다."}));
          throw new Error(`AU 데이터 가져오기 실패 (상태: ${response.status}): ${errorResult.message || '알 수 없는 서버 오류입니다.'}`);
        }
        
        // 3. 서버에서 받은 데이터 파싱
        const feedbackData = await response.json();
        const userResults = feedbackData.user_data.sort((a, b) => a.round_number - b.round_number);

        if (!userResults || userResults.length === 0) {
            throw new Error("서버에서 받아온 운동 기록이 없습니다.");
        }

        // 4. 기준 사진 표시
        const refImagePaths_AU = [];
        for (let i = 1; i <= AU_MODE_TOTAL_ROUNDS; i++) {
          refImagePaths_AU.push(`/static/images/teachers/${teacher}/${teacher}${i}.png`);
        }
        const refContainer = document.getElementById("reference-images");
        refContainer.innerHTML = '';
        refImagePaths_AU.forEach(src => { const img = document.createElement("img"); img.src = src; refContainer.appendChild(img);});
        document.getElementById("reference-images-title").textContent = `기준 사진 (${AU_MODE_TOTAL_ROUNDS}장)`;

        // 5. 사용자 사진 표시
        const userContainer = document.getElementById("user-images");
        userContainer.innerHTML = '';
        const userImageUrls = userResults.map(data => data.photo_url).filter(url => url && url !== 'no_image_provided');
        userImageUrls.forEach(src => { const img = document.createElement("img"); img.src = src; userContainer.appendChild(img); });
        document.getElementById("user-images-title").textContent = `사용자 촬영 사진 (${userImageUrls.length}장)`;
        document.getElementById("user-images-section").style.display = "block";

        // 6. 라운드별 점수 차트 렌더링 (점수 범위를 0-1로 변환)
        const roundScores = userResults.map(d => d.predicted_score / 100); 
        renderChart(roundScores, AU_MODE_TOTAL_ROUNDS);
        document.getElementById("similarity-chart-container").style.display = "block";

        // 7. 종합 피드백 및 점수 요약 생성
        const totalScore = userResults.reduce((sum, data) => sum + data.predicted_score, 0);
        const avgScore = totalScore / userResults.length;
        let overallComment = '';
        if (avgScore >= 80) {
            overallComment = '매우 훌륭합니다! 선생님의 표정을 정확하게 따라 하고 있습니다. 🤩';
        } else if (avgScore >= 60) {
            overallComment = '좋아요! 전반적으로 잘하고 있지만, 일부 동작에서 조금 더 연습하면 완벽해질 거예요. 😊';
        } else {
            overallComment = '수고하셨습니다! 꾸준한 연습을 통해 정확도를 높여봐요. 화이팅! 💪';
        }
        document.getElementById("summary-text").innerHTML = `<strong>평균 점수: ${avgScore.toFixed(1)}점</strong><br><br>${overallComment}`;

        // 8. 라운드별 상세 카드 동적 생성
        const detailsContainer = document.getElementById('au-round-details-container');
        detailsContainer.innerHTML = ''; // 컨테이너 초기화

        const getScoreClassName = (score) => {
            if (score >= 80) return 'score-good';
            if (score >= 60) return 'score-ok';
            return 'score-bad';
        };

        userResults.forEach(userData => {
            const card = document.createElement('div');
            card.className = 'round-card';
            const geminiFeedback = userData.gemini_feedback || "AI 코칭 피드백이 없습니다.";
            const userAuDetails = Object.entries(userData.au_features || {}).map(([key, val]) => `<p>${key}: ${val.toFixed(2)}</p>`).join('');

            card.innerHTML = `
                <h3>ROUND ${userData.round_number}</h3>
                <p>유사도 점수: <strong class="${getScoreClassName(userData.predicted_score)}">${userData.predicted_score.toFixed(2)}</strong></p>
                <img src="${userData.photo_url}" alt="라운드 ${userData.round_number} 사용자 사진">
                <div class="gemini-feedback-box">
                    <p><strong>🤖 Gemini AI 코칭:</strong></p>
                    <p>${geminiFeedback}</p>
                </div>
                <button class="toggle-au-details-btn">AU 상세 값 보기</button>
                <div class="au-details-collapsible">
                    <h4>나의 AU 값:</h4>
                    ${userAuDetails || '<p>데이터 없음</p>'}
                </div>
            `;
            detailsContainer.appendChild(card);

            // 토글 버튼 이벤트 리스너 추가
            const toggleBtn = card.querySelector('.toggle-au-details-btn');
            const collapsible = card.querySelector('.au-details-collapsible');
            toggleBtn.addEventListener('click', () => {
                collapsible.classList.toggle('expanded');
                if (collapsible.classList.contains('expanded')) {
                    collapsible.style.height = collapsible.scrollHeight + 'px';
                    toggleBtn.textContent = 'AU 상세 값 숨기기';
                } else {
                    collapsible.style.height = '0';
                    toggleBtn.textContent = 'AU 상세 값 보기';
                }
            });
        });
        
        document.getElementById('au-round-details-section').style.display = 'block';
        document.getElementById("report").style.display = "block";
        analyzeBtn.textContent = "AU 분석 완료 ✅";

      } catch (err) {
        console.error("AU 모드 피드백 오류:", err);
        analyzeBtn.textContent = "AU 분석 실패 ❌";
        document.getElementById("summary-text").innerHTML = `<strong style="color:red;">오류 (AU 모드):</strong> ${err.message}. <br>문제가 지속되면 관리자에게 문의하세요.`;
        document.getElementById("report").style.display = "block";
      }
    // ▲▲▲▲▲ [수정된 부분] 여기까지 ▲▲▲▲▲
    } else {
      analyzeBtn.textContent = "모드 인식 불가";
      document.getElementById("summary-text").innerHTML = "<strong style='color:red;'>오류:</strong> 알 수 없는 운동 모드입니다. 페이지를 새로고침하거나 운동 선택 페이지로 돌아가 다시 시도해주세요.";
      document.getElementById("report").style.display = "block";
    }
    analyzeBtn.disabled = false;
  });
}