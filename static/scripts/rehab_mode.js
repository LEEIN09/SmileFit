export function init() {
  console.log("✅ rehab_mode.js init() 호출됨");

  // 페이지가 로드되었을 때 로직을 여기서 정의
  document.body.classList.add('loaded');

  // 버튼에 클릭 이벤트 연결
  const focusBtn = document.getElementById("focusBtn");
  const complexBtn = document.getElementById("complexBtn");

  if (focusBtn) {
    focusBtn.onclick = () => {
      document.body.classList.remove('loaded');
      document.body.classList.add('fade-out');
      setTimeout(() => loadPage('focus'), 400);
    };
  }

  if (complexBtn) {
    complexBtn.onclick = () => {
      document.body.classList.remove('loaded');
      document.body.classList.add('fade-out');
      setTimeout(() => loadPage('complex'), 400);
    };
  }
}
