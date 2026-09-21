// Bead Counter — CV worker. Runs OpenCV (WASM) off the main thread.
//
// Message in:  { type:'count', imageData:ImageData, line:{x1,y1,x2,y2} }
//   imageData : the photo at processing resolution (RGBA), from app.js.
//   line      : the calibration stroke drawn across ONE bead, same image space.
//               Gives bead width d (stroke length), a point on the strand (midpoint), and
//               the strand orientation (strand axis ≈ perpendicular to the stroke).
//
// Approach — trace + boundary count (mirrors test/verify_pipeline.py). Tuned for tiny beads:
//   1. TRACE the strand centerline out from the calibration stroke. At each step we look at a
//      short perpendicular cross-section, estimate the LOCAL fabric colour from its outer ends
//      (immune to global shading / folds), and take the run of pixels near the centre that are
//      far from that local fabric = the strand here. Coasts through low-contrast (e.g. black)
//      beads and thread gaps via direction momentum.
//   2. COUNT bead boundaries: sample colour along the centerline, take the along-strand colour
//      gradient (peaks at each bead-to-bead boundary), and count prominent peaks. This handles
//      random multi-colour beads (keys on boundaries, not colour) and naturally skips smooth
//      thread gaps. Pitch (autocorrelation) sets the peak spacing. Markers land on the peaks.
// Message out: { type:'result', markers:[{x,y}], peakCount, areaCount, debug }
//   areaCount = independent cross-check (span between first/last boundary ÷ pitch).

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

function count(imageData, line) {
  const trash = [];
  const keep = (m) => (trash.push(m), m);
  const done = (r) => { trash.forEach((m) => { try { m.delete(); } catch (_) {} }); return r; };

  const d = Math.max(4, Math.hypot(line.x2 - line.x1, line.y2 - line.y1)); // bead width (px)
  const Ln = Math.hypot(line.x2 - line.x1, line.y2 - line.y1) || 1;
  const ux = (line.x2 - line.x1) / Ln, uy = (line.y2 - line.y1) / Ln;      // across-strand unit
  const ax = -uy, ay = ux;                                                 // along-strand unit
  const midx = (line.x1 + line.x2) / 2, midy = (line.y1 + line.y2) / 2;

  const src = keep(cv.matFromImageData(imageData));
  const rgb = keep(new cv.Mat());
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const lab = keep(new cv.Mat());
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);          // 8U: L, a, b
  const rows = lab.rows, cols = lab.cols;
  const labAt = (x, y) => {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= cols || yi >= rows) return null;
    const p = lab.ucharPtr(yi, xi);
    return [p[0], p[1], p[2]];
  };

  // --- trace the strand centerline -------------------------------------
  const R = Math.max(14, Math.round(1.8 * d));      // cross-section half-width
  const step = Math.max(2, 0.4 * d);
  const NO = 2 * R + 1;
  const coastMax = Math.round(2.5 * d / step);

  const trace = (sign) => {
    const nodes = [];
    let px = midx, py = midy, dx = ax * sign, dy = ay * sign, coast = 0;
    for (let s = 0; s < 5000; s++) {
      const nx = px + dx * step, ny = py + dy * step;
      const perpx = -dy, perpy = dx;
      const V = new Array(NO), ok = new Array(NO);
      const oL = [], oA = [], oB = [];
      for (let i = 0; i < NO; i++) {
        const o = i - R;
        const p = labAt(nx + o * perpx, ny + o * perpy);
        if (p) { V[i] = p; ok[i] = true; if (Math.abs(o) > 0.65 * R) { oL.push(p[0]); oA.push(p[1]); oB.push(p[2]); } }
        else { V[i] = [0, 0, 0]; ok[i] = false; }
      }
      if (oL.length < 6) break;
      const fL = median(oL), fA = median(oA), fB = median(oB);
      const dist = new Array(NO); const oD = [];
      for (let i = 0; i < NO; i++) {
        if (!ok[i]) { dist[i] = 0; continue; }
        const dd = Math.hypot(V[i][0] - fL, V[i][1] - fA, V[i][2] - fB);
        dist[i] = dd;
        if (Math.abs(i - R) > 0.65 * R) oD.push(dd);
      }
      const T = Math.max(12, median(oD) + 8);
      const bead = dist.map((v) => v > T);
      let ci = R;
      if (!bead[ci]) {
        let found = -1;
        for (let off = 1; off <= 4; off++) {
          if (ci + off < NO && bead[ci + off]) { found = ci + off; break; }
          if (ci - off >= 0 && bead[ci - off]) { found = ci - off; break; }
        }
        if (found < 0) { coast++; px = nx; py = ny; if (coast > coastMax) break; continue; }
        ci = found;
      }
      coast = 0;
      let lo = ci, hi = ci;
      while (lo - 1 >= 0 && bead[lo - 1]) lo--;
      while (hi + 1 < NO && bead[hi + 1]) hi++;
      const w = hi - lo + 1;
      if (w > 1.6 * R) break;                         // ran into a fabric flood
      let coff = 0; for (let i = lo; i <= hi; i++) coff += (i - R); coff /= w;
      const npx = nx + coff * perpx, npy = ny + coff * perpy;
      if (npx < 0 || npy < 0 || npx >= cols || npy >= rows) break;
      nodes.push([npx, npy, w]);
      let ndx = npx - px, ndy = npy - py; const nl = Math.hypot(ndx, ndy);
      if (nl > 1e-6) { dx = 0.6 * dx + 0.4 * (ndx / nl); dy = 0.6 * dy + 0.4 * (ndy / nl); const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl; }
      px = npx; py = npy;
    }
    return nodes;
  };

  const back = trace(-1), fwd = trace(1);
  const nodes = back.reverse().concat([[midx, midy, d]], fwd);
  const dbg = (ns, extra) => Object.assign({
    dPx: Math.round(d), nodes: ns.length,
    centerline: sampleLine(ns, 200),
  }, extra || {});
  if (nodes.length < 8) return done({ type: 'result', markers: [], peakCount: 0, areaCount: 0, debug: dbg(nodes) });

  // --- arc length + resample -------------------------------------------
  const arc = [0];
  for (let i = 1; i < nodes.length; i++) arc.push(arc[i - 1] + Math.hypot(nodes[i][0] - nodes[i - 1][0], nodes[i][1] - nodes[i - 1][1]));
  const Ltot = arc[arc.length - 1];
  if (Ltot < 2 * d) return done({ type: 'result', markers: [], peakCount: 0, areaCount: 0, debug: dbg(nodes) });
  const ds = 1;
  const nS = Math.floor(Ltot / ds);
  const px = new Float64Array(nS), py = new Float64Array(nS);
  const interpAt = (s, comp) => {
    let i = 1; while (i < arc.length && arc[i] < s) i++;
    if (i >= arc.length) i = arc.length - 1;
    const t = (arc[i] - arc[i - 1]) > 1e-9 ? (s - arc[i - 1]) / (arc[i] - arc[i - 1]) : 0;
    return nodes[i - 1][comp] + t * (nodes[i][comp] - nodes[i - 1][comp]);
  };
  const Lp = new Float64Array(nS), ap = new Float64Array(nS), bp = new Float64Array(nS);
  const wsig = new Float64Array(nS);
  for (let j = 0; j < nS; j++) {
    px[j] = interpAt(j * ds, 0); py[j] = interpAt(j * ds, 1);
    wsig[j] = interpAt(j * ds, 2);
    const p = labAt(px[j], py[j]) || [0, 0, 0];
    Lp[j] = p[0]; ap[j] = p[1]; bp[j] = p[2];
  }
  const medw = median(Array.from(wsig));

  // --- boundary gradient + smoothing -----------------------------------
  const g = new Float64Array(nS);
  for (let j = 0; j < nS; j++) {
    const jm = Math.max(0, j - 1), jp = Math.min(nS - 1, j + 1);
    const dL = (Lp[jp] - Lp[jm]) / 2, da = (ap[jp] - ap[jm]) / 2, db = (bp[jp] - bp[jm]) / 2;
    g[j] = Math.hypot(dL, da, db);
  }
  const gs = new Float64Array(nS);
  for (let j = 0; j < nS; j++) {
    const jm = Math.max(0, j - 1), jp = Math.min(nS - 1, j + 1);
    gs[j] = (g[jm] + g[j] + g[jp]) / 3;
  }

  // --- pitch (autocorrelation) sets the min peak spacing ---------------
  const [period] = autocPeriod(gs, 0.6 * d / ds, 1.8 * d / ds);
  const pitch = period ? period * ds : d;
  const minDist = Math.max(3, Math.round(0.7 * pitch / ds));

  // --- count = prominent boundary peaks --------------------------------
  let mean = 0; for (let j = 0; j < nS; j++) mean += gs[j]; mean /= nS;
  let vs = 0; for (let j = 0; j < nS; j++) vs += (gs[j] - mean) * (gs[j] - mean);
  const std = Math.sqrt(vs / nS);
  let peaks = portablePeaks(gs, minDist, 0.35 * std);
  peaks = trimTailPeaks(peaks, pitch);              // drop isolated thread-tail peaks
  // PRIMARY = detected boundaries (markers on real beads; skips smooth thread). CROSS-CHECK =
  // same with missed boundaries filled (higher on smooth strands) -> the "double-check" hint.
  const markers = peaks.map((j) => ({ x: px[j], y: py[j] }));
  const area = fillMarkers(peaks, pitch, ds, nS, wsig, medw).length;

  return done({
    type: 'result', markers, peakCount: markers.length, areaCount: area,
    debug: dbg(nodes, { pitchPx: +pitch.toFixed(1), lenPx: Math.round(Ltot), rawPeaks: markers.length }),
  });
}

// --- plain-JS helpers (mirror verify_pipeline.py) --------------------------
function trimTailPeaks(peaks, pitch) {
  peaks = peaks.slice();
  while (peaks.length > 3 && (peaks[1] - peaks[0]) > 2.5 * pitch) peaks.shift();
  while (peaks.length > 3 && (peaks[peaks.length - 1] - peaks[peaks.length - 2]) > 2.5 * pitch) peaks.pop();
  return peaks;
}
function fillMarkers(peaks, pitch, ds, nS, wsig, medw) {
  if (!peaks.length) return [];
  const out = [peaks[0]];
  for (let k = 1; k < peaks.length; k++) {
    const a = peaks[k - 1], b = peaks[k], g = (b - a) * ds;
    let mw = wsig[a];
    if (b > a) { let s = 0; for (let i = a; i <= b; i++) s += wsig[i]; mw = s / (b - a + 1); }
    if (g <= 2.5 * pitch && mw >= 0.6 * medw) {       // bead-width interval: fill missed boundaries
      const n = Math.max(1, Math.round(g / pitch));
      for (let m = 1; m < n; m++) out.push(Math.min(nS - 1, Math.max(0, Math.round(a + (b - a) * m / n))));
    }
    out.push(b);                                     // thin/large gap (thread) => no fill
  }
  return out;
}
function portablePeaks(y, minDist, minProm) {
  const n = y.length, maxima = [];
  for (let i = 1; i < n - 1; i++) if (y[i] >= y[i - 1] && y[i] > y[i + 1]) maxima.push(i);
  const prom = new Map();
  for (const i of maxima) {
    let j = i - 1, lmin = y[i];
    while (j >= 0 && y[j] <= y[i]) { if (y[j] < lmin) lmin = y[j]; j--; }
    j = i + 1; let rmin = y[i];
    while (j < n && y[j] <= y[i]) { if (y[j] < rmin) rmin = y[j]; j++; }
    prom.set(i, y[i] - Math.max(lmin, rmin));
  }
  const cand = maxima.filter((i) => prom.get(i) >= minProm).sort((a, b) => prom.get(b) - prom.get(a));
  const acc = [];
  for (const i of cand) { if (acc.every((k) => Math.abs(i - k) >= minDist)) acc.push(i); }
  return acc.sort((a, b) => a - b);
}

function autocPeriod(sig, pmin, pmax) {
  const n = sig.length;
  if (n < 2 * pmax) return [null, 0];
  let mean = 0; for (let i = 0; i < n; i++) mean += sig[i]; mean /= n;
  const s = new Float64Array(n); let varsum = 0;
  for (let i = 0; i < n; i++) { s[i] = sig[i] - mean; varsum += s[i] * s[i]; }
  if (varsum < 1e-6) return [null, 0];
  const lo = Math.max(1, Math.floor(pmin)), hi = Math.min(n - 2, Math.floor(pmax));
  if (hi <= lo) return [null, 0];
  const ac = new Float64Array(hi + 2);
  for (let lag = lo - 1; lag <= hi + 1; lag++) {
    let sum = 0; for (let i = 0; i + lag < n; i++) sum += s[i] * s[i + lag];
    ac[lag] = sum / varsum;
  }
  let gmax = 0; for (let k = lo; k <= hi; k++) if (ac[k] > gmax) gmax = ac[k];
  if (gmax <= 0) return [null, 0];
  const thresh = 0.5 * gmax;
  let k = -1;
  for (let j = lo; j <= hi; j++) if (ac[j] >= thresh && ac[j] >= ac[j - 1] && ac[j] >= ac[j + 1]) { k = j; break; }
  if (k < 0) { k = lo; for (let j = lo; j <= hi; j++) if (ac[j] > ac[k]) k = j; }
  while ((k >> 1) >= lo && ac[k >> 1] >= 0.55 * ac[k]) k = k >> 1;   // octave guard
  const y0 = ac[k - 1], y1 = ac[k], y2 = ac[k + 1];
  const den = y0 - 2 * y1 + y2;
  const delta = Math.abs(den) > 1e-9 ? 0.5 * (y0 - y2) / den : 0;
  return [k + delta, ac[k]];
}

function sampleLine(nodes, maxN) {
  const out = [], stepN = Math.max(1, Math.floor(nodes.length / maxN));
  for (let i = 0; i < nodes.length; i += stepN) out.push({ x: Math.round(nodes[i][0]), y: Math.round(nodes[i][1]) });
  return out;
}
function median(arr) {
  if (!arr.length) return 0;
  const s = arr.slice().sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
