// Bead Counter service worker — offline app shell + cached OpenCV runtime.
// Bump CACHE when you change any shell file so clients pull the new version.
const CACHE = 'beadcounter-v5';

const SHELL = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './cv-worker.js',
  './manifest.webmanifest',
  './icon.svg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Cache-first for everything we can (shell + the big opencv.js from CDN).
// opencv.js comes back as an opaque cross-origin response — still cacheable,
// which is what lets the whole app work offline after the first load.
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  e.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => hit);
    })
  );
});
