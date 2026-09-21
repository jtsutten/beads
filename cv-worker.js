// Bead Counter — CV worker. Runs OpenCV (WASM) off the main thread.
//
// Message in:  { type:'count', imageData:ImageData, line:{x1,y1,x2,y2} }
//   imageData : the photo at processing resolution (RGBA)
//   line      : the calibration stroke the user drew across ONE bead, in that
//               same image space. It gives us two things:
//                 • bead size  = the stroke length (diameter in px)
//                 • bead color = sampled along the stroke (Lab a/b)
//                 • which strand to count = the blob the stroke sits on
//
// Approach (no training data; beads are uniform size on one strand):
//   1. Learn the bead color from the stroke and segment by Lab-chroma distance
//      — this ignores a differently-colored background (e.g. leather) far more
//      robustly than brightness thresholding did.
//   2. Keep ONLY the connected blob the stroke lies on = the strand. Cables,
//      table specks, a plastic bag, etc. are separate blobs and get dropped.
//   3. Count beads as distance-transform peaks within that strand.
// Message out: { type:'result', markers:[{x,y}], peakCount, areaCount }

const OPENCV_URL = 'https://docs.opencv.org/4.9.0/opencv.js';

let cvReady = false;
function announceReady() {
  if (cvReady) return;
  cvReady = true;
  postMessage({ type: 'ready' });
}

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
    postMessage(count(msg.imageData, msg.line));
  } catch (err) {
    postMessage({ type: 'error', message: 'Counting failed: ' + (err && err.message || err) });
  }
};

const CHROMA_T = 20;   // Lab a/b distance from the sampled bead color (bigger = more permissive)

function count(imageData, line) {
  const trash = [];
  const keep = (m) => (trash.push(m), m);

  const d = Math.max(4, Math.hypot(line.x2 - line.x1, line.y2 - line.y1)); // bead diameter (px)
  const radius = d / 2;
  const singleBeadArea = Math.PI * radius * radius;

  const src = keep(cv.matFromImageData(imageData));      // RGBA
  const rgb = keep(new cv.Mat());
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const lab = keep(new cv.Mat());
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);               // 8U: L, a, b (a/b centered ~128)

  const rows = lab.rows, cols = lab.cols;

  // --- sample points along the calibration stroke ----------------------
  // Sample the INNER portion of the stroke (0.2..0.8): endpoints often overshoot
  // the bead onto the background, and those would poison the color sample.
  const pts = [];
  const N = 9;
  for (let i = 0; i <= N; i++) {
    const t = 0.2 + 0.6 * (i / N);
    const x = Math.round(line.x1 + (line.x2 - line.x1) * t);
    const y = Math.round(line.y1 + (line.y2 - line.y1) * t);
    if (x >= 0 && y >= 0 && x < cols && y < rows) pts.push([x, y]);
  }
  if (!pts.length) pts.push([Math.round((line.x1 + line.x2) / 2), Math.round((line.y1 + line.y2) / 2)]);

  // median bead color (a,b) sampled from small patches along the stroke —
  // patches (not single pixels) keep the sample robust to a slightly-off line.
  const as = [], bs = [];
  for (const [x, y] of pts) {
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < cols && yy < rows) {
        const p = lab.ucharPtr(yy, xx); as.push(p[1]); bs.push(p[2]);
      }
    }
  }
  const a0 = median(as), b0 = median(bs);

  // --- segment by Lab-chroma distance to the bead color ----------------
  const chans = new cv.MatVector();
  cv.split(lab, chans);
  const aF = keep(new cv.Mat()), bF = keep(new cv.Mat());
  chans.get(1).convertTo(aF, cv.CV_32F);
  chans.get(2).convertTo(bF, cv.CV_32F);
  chans.delete();
  const a0m = keep(matScalar(rows, cols, a0));
  const b0m = keep(matScalar(rows, cols, b0));
  cv.subtract(aF, a0m, aF);
  cv.subtract(bF, b0m, bF);
  cv.multiply(aF, aF, aF);
  cv.multiply(bF, bF, bF);
  const sum = keep(new cv.Mat());
  cv.add(aF, bF, sum);
  cv.sqrt(sum, sum);                                     // = chroma distance
  const mask = keep(new cv.Mat());
  cv.threshold(sum, mask, CHROMA_T, 255, cv.THRESH_BINARY_INV); // near bead color -> 255
  mask.convertTo(mask, cv.CV_8U);

  // clean: despeckle, then bridge the thin necks between touching beads
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN,
    keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))));
  cv.morphologyEx(mask, mask, cv.MORPH_CLOSE,
    keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, sizeOdd(d * 0.35))));

  // --- keep only the blob the stroke sits on = the strand --------------
  const labels = keep(new cv.Mat());
  const stats = keep(new cv.Mat());
  const cents = keep(new cv.Mat());
  const nLabels = cv.connectedComponentsWithStats(mask, labels, stats, cents, 8);

  // pick the label most of the stroke's sample points fall on
  const votes = {};
  for (const [x, y] of pts) {
    const lb = labels.intAt(y, x);
    if (lb > 0) votes[lb] = (votes[lb] || 0) + 1;
  }
  let strandLabel = 0, bestVotes = -1;
  for (const k in votes) if (votes[k] > bestVotes) { bestVotes = votes[k]; strandLabel = +k; }
  // fallback: if the stroke landed on holes, take the largest blob
  if (!strandLabel) {
    let bestArea = -1;
    for (let i = 1; i < nLabels; i++) {
      const area = stats.intAt(i, cv.CC_STAT_AREA);
      if (area > bestArea) { bestArea = area; strandLabel = i; }
    }
  }

  const strand = keep(new cv.Mat());
  const labelMat = keep(matScalar(rows, cols, strandLabel, cv.CV_32S));
  cv.compare(labels, labelMat, strand, cv.CMP_EQ);       // 8U, 255 on the strand

  const foregroundArea = cv.countNonZero(strand);
  const areaCount = Math.round(foregroundArea / singleBeadArea);

  // --- count beads = distance-transform peaks within the strand --------
  const dist = keep(new cv.Mat());
  cv.distanceTransform(strand, dist, cv.DIST_L2, 3);
  const dilated = keep(new cv.Mat());
  cv.dilate(dist, dilated, keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, sizeOdd(d * 0.6))));
  const isMax = keep(new cv.Mat());
  cv.compare(dist, dilated, isMax, cv.CMP_GE);
  const distThresh = keep(new cv.Mat());
  cv.threshold(dist, distThresh, Math.max(1, 0.3 * radius), 255, cv.THRESH_BINARY);
  distThresh.convertTo(distThresh, cv.CV_8UC1);
  const peaks = keep(new cv.Mat());
  cv.bitwise_and(isMax, distThresh, peaks);

  const pLabels = keep(new cv.Mat());
  const pStats = keep(new cv.Mat());
  const pCents = keep(new cv.Mat());
  const nPeaks = cv.connectedComponentsWithStats(peaks, pLabels, pStats, pCents, 8);
  const markers = [];
  for (let i = 1; i < nPeaks; i++) {
    markers.push({ x: pCents.doublePtr(i, 0)[0], y: pCents.doublePtr(i, 1)[0] });
  }

  trash.forEach((m) => { try { m.delete(); } catch (_) {} });
  return { type: 'result', markers, peakCount: markers.length, areaCount };
}

function matScalar(rows, cols, val, type) {
  const m = new cv.Mat(rows, cols, type || cv.CV_32FC1);
  m.setTo(new cv.Scalar(val));
  return m;
}
function median(arr) {
  const s = arr.slice().sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function sizeOdd(v) {
  let k = Math.max(3, Math.round(v));
  if (k % 2 === 0) k += 1;
  return new cv.Size(k, k);
}
