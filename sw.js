// Network-first for everything on our origin, falling back to cache so the app opens offline.
const V = "sc-v2";
self.addEventListener("install", e => { e.waitUntil(caches.open(V).then(c => c.addAll(["./", "./index.html", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png"]))); self.skipWaiting(); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  const u = new URL(e.request.url);
  if (u.origin !== location.origin && !u.hostname.endsWith("gstatic.com") && !u.hostname.endsWith("googleapis.com")) return;
  e.respondWith(fetch(e.request, { cache: "no-store" }).then(r => { const c = r.clone(); caches.open(V).then(x => x.put(e.request, c)); return r; }).catch(() => caches.match(e.request)));
});
