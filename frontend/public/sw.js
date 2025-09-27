// Simple service worker to prevent 404 errors
// This is a minimal implementation to avoid console errors

self.addEventListener('install', function(event) {
  console.log('Service Worker installing');
  self.skipWaiting();
});

self.addEventListener('activate', function(event) {
  console.log('Service Worker activating');
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', function(event) {
  // Let the browser handle all fetch requests normally
  return;
});