// Bead Counter — CV worker. Runs OpenCV (WASM) off the main thread.
//
// Message in:  { type:'count', imageData:ImageData, line:{x1,y1,x2,y2} }
//   imageData : the photo at PROCESSING resolution (RGBA) — app.js already downscaled it
//               to PROC_SIDE, so this worker does NOT resize.
//   line      : the calibration stroke drawn across ONE bead, in that same image space.
//               Gives bead width d (stroke length), a point on the strand (midpoint), and
//               the strand orientation (strand axis ≈ perpendicular to the stroke).
//
// Approach — corridor + centerline + pitch (mirrors test/verify_pipeline.py):
//   1. Background colour model from patches beside the strand + image corners; a pixel is
//      "strand" when it is far from EVERY background sample. Works for multi-colour strands
//      (beads have no single colour, but the table does) and rejects busy backgrounds.
//   2. Trace the strand CENTERLINE outward from the stroke midpoint, staying in a corridor
//      around the strand — the busy background is never inspected.
//   3. Count by PITCH: build a 1-D signal along the centerline (cross-strand width + mean L),
//      find the dominant period by autocorrelation, count = arc_length / period. Robust to
//      tiny beads (aggregates the whole strand) and to colour changes (keys on spacing).
// Message out: { type:'result', markers:[{x,y}], peakCount, areaCount }
//   areaCount here = the cross-check count from the 2nd-strongest signal (UI confidence hint).

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

const BG_T = 22;    // Lab distance: a pixel farther than this from every background sample = strand
const DS = 0.5;     // centerline resample spacing (px)

function count(imageData, line) {
  const trash = [];
  const keep = (m) => (trash.push(m), m);
  const done = (result) => { trash.forEach((m) => { try { m.delete(); } catch (_) {} }); return result; };

  const d = Math.max(4, Math.hypot(line.x2 - line.x1, line.y2 - line.y1)); // bead width (px)
  const L = Math.hypot(line.x2 - line.x1, line.y2 - line.y1) || 1;
  const ux = (line.x2 - line.x1) / L, uy = (line.y2 - line.y1) / L;        // across-strand unit
  const ax = -uy, ay = ux;                                                 // along-strand unit
  const midx = (line.x1 + line.x2) / 2, midy = (line.y1 + line.y2) / 2;

  const src = keep(cv.matFromImageData(imageData));
  const rgb = keep(new cv.Mat());
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const lab = keep(new cv.Mat());
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);          // 8U: L, a, b
  const rows = lab.rows, cols = lab.cols;

  const labAt = (x, y) => {                          // median-ish Lab at rounded (x,y)
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= cols || yi >= rows) return null;
    const p = lab.ucharPtr(yi, xi);
    return [p[0], p[1], p[2]];
  };
  const patch = (x, y, rad) => {                     // median Lab over a small patch
    const Ls = [], As = [], Bs = [];
    for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
      const s = labAt(x + dx, y + dy);
      if (s) { Ls.push(s[0]); As.push(s[1]); Bs.push(s[2]); }
    }
    return Ls.length ? [median(Ls), median(As), median(Bs)] : null;
  };

  // --- background colour references -------------------------------------
  const bgRefs = [];
  for (const k of [1.3, 1.7, 2.2]) for (const sgn of [1, -1]) {
    const s = patch(midx + sgn * k * d * ux, midy + sgn * k * d * uy, 3);
    if (s) bgRefs.push(s);
  }
  for (const [cx, cy] of [[6, 6], [cols - 7, 6], [6, rows - 7], [cols - 7, rows - 7],
                          [(cols / 2) | 0, 6], [(cols / 2) | 0, rows - 7]]) {
    const s = patch(cx, cy, 4);
    if (s) bgRefs.push(s);
  }
  if (!bgRefs.length) bgRefs.push([240, 128, 128]);
  const refs = dedupRefs(bgRefs, 6);                 // drop near-identical bg samples (fewer passes)

  // --- strand mask: min SQUARED Lab distance to any background ref > BG_T² --
  const chans = new cv.MatVector();
  cv.split(lab, chans);
  const L32 = keep(new cv.Mat()), a32 = keep(new cv.Mat()), b32 = keep(new cv.Mat());
  chans.get(0).convertTo(L32, cv.CV_32F);
  chans.get(1).convertTo(a32, cv.CV_32F);
  chans.get(2).convertTo(b32, cv.CV_32F);
  chans.delete();

  const minDist = keep(matScalar(rows, cols, 1e18)); // squared distance (no per-ref sqrt)
  for (const [Lr, ar, br] of refs) {
    const dl = new cv.Mat(), da = new cv.Mat(), db = new cv.Mat(), dsum = new cv.Mat();
    const sL = matScalar(rows, cols, Lr), sa = matScalar(rows, cols, ar), sb = matScalar(rows, cols, br);
    cv.subtract(L32, sL, dl); cv.multiply(dl, dl, dl);
    cv.subtract(a32, sa, da); cv.multiply(da, da, da);
    cv.subtract(b32, sb, db); cv.multiply(db, db, db);
    cv.add(dl, da, dsum); cv.add(dsum, db, dsum);     // squared distance
    cv.min(minDist, dsum, minDist);
    dl.delete(); da.delete(); db.delete(); dsum.delete(); sL.delete(); sa.delete(); sb.delete();
  }
  const mask = keep(new cv.Mat());
  const mtmp = new cv.Mat();
  cv.threshold(minDist, mtmp, BG_T * BG_T, 255, cv.THRESH_BINARY);   // strand = 255
  mtmp.convertTo(mask, cv.CV_8U);                    // fresh 8U mat (avoid in-place type change)
  mtmp.delete();
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN,
    keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))));
  cv.morphologyEx(mask, mask, cv.MORPH_CLOSE,
    keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, sizeOdd(d * 0.3))));

  const mb = mask.data;                              // Uint8Array, row-major
  const Ldata = new Uint8Array(rows * cols);         // L channel for the brightness signal
  { const lc = new cv.Mat(); L32.convertTo(lc, cv.CV_8U); Ldata.set(lc.data); lc.delete(); }
  const fg = (x, y) => {
    const xi = Math.round(x), yi = Math.round(y);
    return xi >= 0 && yi >= 0 && xi < cols && yi < rows && mb[yi * cols + xi] > 0;
  };

  // --- centerline tracing ----------------------------------------------
  const step = Math.max(1, d * 0.4);
  const half = Math.max(2, Math.round(d * 1.2));
  const offs = [];
  for (let o = -half; o <= half; o++) offs.push(o);
  const ci = offs.indexOf(0) < 0 ? (offs.length >> 1) : offs.indexOf(0);

  const trace = (sign) => {
    const nodes = [];
    let px = midx, py = midy, dx = ax * sign, dy = ay * sign;
    for (let s = 0; s < 6000; s++) {
      const nx = px + dx * step, ny = py + dy * step;
      const perpx = -dy, perpy = dx;
      const vals = offs.map((o) => fg(nx + o * perpx, ny + o * perpy));
      if (vals.reduce((a, v) => a + (v ? 1 : 0), 0) < 2) break;
      const run = contiguous(vals, ci);
      if (!run) break;
      const [lo, hi] = run;
      let coff = 0; for (let i = lo; i <= hi; i++) coff += offs[i]; coff /= (hi - lo + 1);
      const width = hi - lo + 1;
      const npx = nx + coff * perpx, npy = ny + coff * perpy;
      if (npx < 0 || npy < 0 || npx >= cols || npy >= rows) break;
      nodes.push([npx, npy, width]);
      let ndx = npx - px, ndy = npy - py;
      const nl = Math.hypot(ndx, ndy);
      if (nl > 1e-6) {
        dx = 0.5 * dx + 0.5 * (ndx / nl); dy = 0.5 * dy + 0.5 * (ndy / nl);
        const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl;
      }
      px = npx; py = npy;
    }
    return nodes;
  };

  const back = trace(-1), fwd = trace(1);
  const nodes = back.reverse().concat([[midx, midy, d]], fwd);
  if (nodes.length < 3) return done({ type: 'result', markers: [], peakCount: 0, areaCount: 0 });

  // --- resample to uniform arc length ----------------------------------
  const arc = [0];
  for (let i = 1; i < nodes.length; i++) {
    arc.push(arc[i - 1] + Math.hypot(nodes[i][0] - nodes[i - 1][0], nodes[i][1] - nodes[i - 1][1]));
  }
  const Ltot = arc[arc.length - 1];
  if (Ltot < d) return done({ type: 'result', markers: [], peakCount: 0, areaCount: 0 });

  const nS = Math.floor(Ltot / DS);
  const wsig = new Float64Array(nS), lsig = new Float64Array(nS);
  const interpXY = (s) => {                          // point at arc length s
    let i = 1; while (i < arc.length && arc[i] < s) i++;
    if (i >= arc.length) i = arc.length - 1;
    const t = (arc[i] - arc[i - 1]) > 1e-9 ? (s - arc[i - 1]) / (arc[i] - arc[i - 1]) : 0;
    return [nodes[i - 1][0] + t * (nodes[i][0] - nodes[i - 1][0]),
            nodes[i - 1][1] + t * (nodes[i][1] - nodes[i - 1][1]),
            nodes[i - 1][2] + t * (nodes[i][2] - nodes[i - 1][2])];
  };
  for (let j = 0; j < nS; j++) {
    const [x, y, w] = interpXY(j * DS);
    wsig[j] = w;
    const xi = Math.min(cols - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(rows - 1, Math.max(0, Math.round(y)));
    lsig[j] = Ldata[yi * cols + xi];
  }

  // --- pitch by autocorrelation ----------------------------------------
  // signals that oscillate once per bead: L and (negated) width. |dL/ds| is NOT used — it
  // has two edges per bead, so its fundamental is pitch/2 and would double the count.
  const pmin = 0.55 * d / DS, pmax = 2.4 * d / DS;
  const negW = wsig.map((v) => -v);
  const cands = [];
  for (const [name, sig] of [['width', negW], ['L', lsig]]) {
    const [p, strength] = autocPeriod(sig, pmin, pmax);
    if (p) cands.push([strength, p, name]);
  }
  if (!cands.length) return done({ type: 'result', markers: [], peakCount: 0, areaCount: 0 });
  cands.sort((A, B) => B[0] - A[0]);
  const pitch = cands[0][1] * DS;
  const cnt = Math.max(1, Math.round(Ltot / pitch));
  const alt = cands.length > 1 ? Math.round(Ltot / (cands[1][1] * DS)) : cnt;

  // --- markers: evenly spaced along the centerline ---------------------
  const markers = [];
  for (let j = 0; j < cnt; j++) {
    const [x, y] = interpXY((j + 0.5) * Ltot / cnt);
    markers.push({ x, y });
  }
  return done({ type: 'result', markers, peakCount: markers.length, areaCount: alt });
}

// --- plain-JS helpers (mirror verify_pipeline.py) --------------------------
function autocPeriod(sig, pmin, pmax) {
  const n = sig.length;
  if (n < 2 * pmax) return [null, 0];
  let mean = 0; for (let i = 0; i < n; i++) mean += sig[i]; mean /= n;
  const s = new Float64Array(n);
  let varsum = 0;
  for (let i = 0; i < n; i++) { s[i] = sig[i] - mean; varsum += s[i] * s[i]; }
  if (varsum < 1e-6) return [null, 0];
  const lo = Math.max(1, Math.floor(pmin)), hi = Math.min(n - 2, Math.floor(pmax));
  if (hi <= lo) return [null, 0];
  const ac = new Float64Array(hi + 2);
  const ac0 = varsum || 1e-9;
  for (let lag = lo - 1; lag <= hi + 1; lag++) {
    let sum = 0; for (let i = 0; i + lag < n; i++) sum += s[i] * s[i + lag];
    ac[lag] = sum / ac0;
  }
  let gmax = 0; for (let k = lo; k <= hi; k++) if (ac[k] > gmax) gmax = ac[k];
  if (gmax <= 0) return [null, 0];
  const thresh = 0.5 * gmax;
  let k = -1;
  for (let j = lo; j <= hi; j++) {
    if (ac[j] >= thresh && ac[j] >= ac[j - 1] && ac[j] >= ac[j + 1]) { k = j; break; }
  }
  if (k < 0) { k = lo; for (let j = lo; j <= hi; j++) if (ac[j] > ac[k]) k = j; }
  while ((k >> 1) >= lo && ac[k >> 1] >= 0.55 * ac[k]) k = k >> 1;   // octave guard
  const y0 = ac[k - 1], y1 = ac[k], y2 = ac[k + 1];
  const denom = y0 - 2 * y1 + y2;
  const delta = Math.abs(denom) > 1e-9 ? 0.5 * (y0 - y2) / denom : 0;
  return [k + delta, ac[k]];
}

function contiguous(vals, ci) {
  const n = vals.length;
  if (!vals[ci]) {
    let found = -1;
    for (const off of [1, -1, 2, -2]) if (ci + off >= 0 && ci + off < n && vals[ci + off]) { found = ci + off; break; }
    if (found < 0) return null;
    ci = found;
  }
  let lo = ci, hi = ci;
  while (lo - 1 >= 0 && vals[lo - 1]) lo--;
  while (hi + 1 < n && vals[hi + 1]) hi++;
  return [lo, hi];
}

function matScalar(rows, cols, val) {
  const m = new cv.Mat(rows, cols, cv.CV_32FC1);
  m.setTo(new cv.Scalar(val));
  return m;
}
function dedupRefs(refs, tol) {
  const out = [];
  for (const r of refs) {
    if (!out.some((o) => Math.hypot(o[0] - r[0], o[1] - r[1], o[2] - r[2]) < tol)) out.push(r);
  }
  return out;
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
