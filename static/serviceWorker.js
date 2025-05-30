const CACHE_NAME = "smilefit-cache-v1";

// 여기에 캐시할 파일 경로를 정확히 써주세요
const urlsToCache = [
  "/",
  "/manifest.json",
  "/serviceWorker.js",
  "/scripts/index.js",
  "/icons/icon-192.png",
  "/icons/icon-512.png"
];

// ✅ 설치 단계
self.addEventListener("install", (event) => {
  self.skipWaiting();  // ✅ 새 SW 즉시 활성화

  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(urlsToCache);
    }).catch((err) => {
      console.error("❌ 캐시 등록 실패:", err);
    })
  );
  console.log("✅ Service Worker 설치 완료");
});

// ✅ 활성화 단계
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    ).then(() => {
      return self.clients.claim(); // ✅ 모든 클라이언트에 즉시 적용
    })
  );
  console.log("🔄 Service Worker 활성화 및 이전 캐시 정리");
});

// 요청 처리
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
