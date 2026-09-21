// Bead Counter — CV worker. Runs OpenCV (WASM) off the main thread.
//
// Message in:  { type:'count', imageData:ImageData, diameter:Number }
//   imageData  : the photo at processing resolution (RGBA)
//   diameter   : one bead's diameter in pixels, in that SAME image space
// Message out: { type:'result', markers:[{x,y}], peakCount, areaCount,
//                foregroundArea, singleBeadArea }
//
// Two independent estimators, no training data required (beads are uniform size):
//   • peakCount  — distance-transform local maxima → also gives marker positions
//                  the user can tap-correct. This is the headline count.
//   • areaCount  — foreground area ÷ single-bead area. Independent cross-check;
//                  a big gap between the two signals low confidence.

const OPENCV_URL = 'https://docs.opencv.org/4.9.0/opencv.js';

let cvReady = false;
function announceReady() {
  if (cvReady) return;
  cvReady = true;
  postMessage({ type: 'ready' });
}

// OpenCV's WASM build initialises in one of two ways depending on version:
// an Emscripten Module hook, or by exporting `cv` as a Promise. Handle both.
self.Module = { onRuntimeInitialized: announceReady };
try {
  importScripts(OPENCV_URL);
  if (typeof cv !== 'undefined' && cv && typeof cv.then === 'function') {
    cv.then((mod) => { cv = mod; announceReady(); });
  }
} catch (err) {
  postMessage({ type: 'error', message: 'Failed to load OpenCV: ' + err.message });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'count') return;
  if (!cvReady) { postMessage({ type: 'error', message: 'OpenCV not ready yet.' }); return; }
  try {
    postMessage(count(msg.imageData, msg.diameter));
  } catch (err) {
    postMessage({ type: 'error', message: 'Counting failed: ' + (err && err.message || err) });
  }
};

function count(imageData, diameter) {
  const d = Math.max(4, diameter);
  const radius = d / 2;
  const singleBeadArea = Math.PI * radius * radius;

  const src = cv.matFromImageData(imageData);   // RGBA
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, gray, new cv.Size(3, 3), 0);

  // --- 1. Otsu threshold, then orient so beads = 255 --------------------
  const bin = new cv.Mat();
  cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  if (borderMean(bin) > 127) cv.bitwise_not(bin, bin); // border is background

  // --- 2. Clean up: open (despeckle) then close (seal gaps) -------------
  const kOpen = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  const kClose = cv.getStructuringElement(cv.MORPH_ELLIPSE, sizeOdd(d * 0.25));
  cv.morphologyEx(bin, bin, cv.MORPH_OPEN, kOpen);
  cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kClose);

  // --- 3. Fill holes (specular highlights) + drop specks via contours ---
  const mask = cv.Mat.zeros(bin.rows, bin.cols, cv.CV_8UC1);
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bin, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const minArea = 0.15 * singleBeadArea;
  for (let i = 0; i < contours.size(); i++) {
    if (cv.contourArea(contours.get(i)) >= minArea) {
      cv.drawContours(mask, contours, i, new cv.Scalar(255), -1);
    }
  }

  const foregroundArea = cv.countNonZero(mask);
  const areaCount = Math.round(foregroundArea / singleBeadArea);

  // --- 4. Distance transform → local maxima = bead centres -------------
  const dist = new cv.Mat();
  cv.distanceTransform(mask, dist, cv.DIST_L2, 3);

  const dilated = new cv.Mat();
  const kPeak = cv.getStructuringElement(cv.MORPH_ELLIPSE, sizeOdd(d * 0.7));
  cv.dilate(dist, dilated, kPeak);

  // local max (incl. plateaus): dist >= dilated
  const isMax = new cv.Mat();
  cv.compare(dist, dilated, isMax, cv.CMP_GE); // 8U, 255 at maxima

  // ...and tall enough to be a real bead centre, not a thin neck
  const minPeak = Math.max(1, 0.35 * radius);
  const distThresh = new cv.Mat();
  cv.threshold(dist, distThresh, minPeak, 255, cv.THRESH_BINARY); // 32F
  distThresh.convertTo(distThresh, cv.CV_8UC1);

  const peaks = new cv.Mat();
  cv.bitwise_and(isMax, distThresh, peaks);

  // Collapse plateau clusters to one marker each (centroids).
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(peaks, labels, stats, centroids, 8);
  const markers = [];
  for (let i = 1; i < n; i++) { // 0 = background
    markers.push({ x: centroids.doublePtr(i, 0)[0], y: centroids.doublePtr(i, 1)[0] });
  }

  const out = {
    type: 'result',
    markers,
    peakCount: markers.length,
    areaCount,
    foregroundArea,
    singleBeadArea,
  };

  [src, gray, bin, kOpen, kClose, mask, contours, hierarchy, dist, dilated,
   isMax, distThresh, peaks, labels, stats, centroids, kPeak].forEach((m) => {
    try { m.delete(); } catch (_) {}
  });
  return out;
}

// Mean intensity of a thin frame around the image border.
function borderMean(m) {
  const w = m.cols, h = m.rows;
  const b = Math.max(2, Math.round(Math.min(w, h) * 0.03));
  const strips = [
    new cv.Rect(0, 0, w, b),
    new cv.Rect(0, h - b, w, b),
    new cv.Rect(0, 0, b, h),
    new cv.Rect(w - b, 0, b, h),
  ];
  let sum = 0;
  for (const r of strips) {
    const roi = m.roi(r);
    sum += cv.mean(roi)[0];
    roi.delete();
  }
  return sum / strips.length;
}

// Nearest odd Size >= 3 from a scalar length.
function sizeOdd(v) {
  let k = Math.max(3, Math.round(v));
  if (k % 2 === 0) k += 1;
  return new cv.Size(k, k);
}
