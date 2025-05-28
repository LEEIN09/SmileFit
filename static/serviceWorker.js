const CACHE_NAME = "smilefit-cache-v1";

// 여기에 캐시할 파일 경로를 정확히 써주세요 (존재하는 파일만!)
const urlsToCache = [
  "/",                              // 홈 (Flask의 index.html 템플릿 렌더링)
  "/manifest.json",                 // Flask 라우트로 서빙
  "/serviceWorker.js",             // Flask 라우트로 서빙
  "/scripts/index.js",             // static 안에 있어야 하며 /scripts 경로로 서빙되어야 함
  "/icons/icon-192.png",           // Flask 라우트로 /icons 경로 등록해야 함
  "/icons/icon-512.png"
];

// 설치 단계: 캐시 등록
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(urlsToCache);
    }).catch((err) => {
      console.error("❌ 캐시 등록 실패:", err);
    })
  );
  console.log("✅ Service Worker 설치 완료");
});

// 활성화 단계: 이전 캐시 삭제 가능
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  console.log("🔄 Service Worker 활성화 및 이전 캐시 정리");
});

// 요청 처리: 캐시 우선, 없으면 네트워크
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
