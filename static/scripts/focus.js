export function init() {
  console.log("✅ focus.js init() 호출됨");

  // 카드 클릭 → sessionStorage 저장 + 이동
  const cards = document.querySelectorAll('.exercise-card');
  cards.forEach(card => {
    card.onclick = () => {
      const exercise = card.dataset.exercise;
      if (!exercise) return;

      sessionStorage.setItem('selectedExercise', exercise);
      document.body.classList.remove('loaded');
      document.body.classList.add('fade-out');
      setTimeout(() => {
        loadPage('focus_fit');
      }, 400);
    };
  });


  document.body.classList.add('loaded');
}
