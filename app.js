// Bead Counter — UI controller.
// Flow: capture photo → zoom/tap to calibrate one bead → CV worker detects beads →
// review & tap-to-correct. Everything runs on-device; no network after load.

const PROC_SIDE = 2400; // resolution handed to the CV worker (speed vs. accuracy)
const FULL_SIDE = 4096; // resolution retained purely so the calibrate loupe stays crisp

const el = (id) => document.getElementById(id);
const steps = {
  capture: el('step-capture'),
  calibrate: el('step-calibrate'),
  result: el('step-result'),
};
function showStep(name) {
  Object.values(steps).forEach((s) => s.classList.remove('is-active'));
  steps[name].classList.add('is-active');
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// --- state -----------------------------------------------------------------
const srcCanvas = document.createElement('canvas'); // proc-resolution source of truth
const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
const fullCanvas = document.createElement('canvas'); // hi-res source, only for the loupe
const fullCtx = fullCanvas.getContext('2d', { willReadFrequently: true });
let diameter = 0;           // bead diameter in proc pixels
let markers = [];           // [{x,y}] in proc pixels
let lastAreaCount = null;

// --- CV worker -------------------------------------------------------------
let worker;
let workerReady = false;
let pendingCount = false;

function initWorker() {
  worker = new Worker('cv-worker.js');
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'ready') {
      workerReady = true;
      setStatus('ready', 'CV ready');
    } else if (m.type === 'result') {
      pendingCount = false;
      el('spinner').hidden = true;
      markers = m.markers.map((p) => ({ x: p.x, y: p.y }));
      lastAreaCount = m.areaCount;
      renderResult();
      showStep('result');
    } else if (m.type === 'error') {
      pendingCount = false;
      el('spinner').hidden = true;
      setStatus('error', 'CV error');
      alert(m.message || 'Something went wrong in the vision code.');
    }
  };
  worker.onerror = (err) => {
    setStatus('error', 'CV failed');
    console.error(err);
  };
}
function setStatus(kind, text) {
  const s = el('cvStatus');
  s.className = 'status status--' + kind;
  s.textContent = text;
}

// --- coordinate helper: pointer event → canvas pixel coords ----------------
// (used by the result step, whose canvas fills its display box 1:1)
function toCanvasXY(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (canvas.width / r.width);
  const y = (ev.clientY - r.top) * (canvas.height / r.height);
  return { x, y };
}

// ===========================================================================
// STEP 1 — capture
// ===========================================================================
el('fileInput').addEventListener('change', (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const longest = Math.max(img.width, img.height);
    // hi-res copy retained so the calibrate magnifier can show real detail
    const fScale = Math.min(1, FULL_SIDE / longest);
    fullCanvas.width = Math.round(img.width * fScale);
    fullCanvas.height = Math.round(img.height * fScale);
    fullCtx.drawImage(img, 0, 0, fullCanvas.width, fullCanvas.height);
    // processing copy handed to the worker; everything (line, markers) is in this space
    const pScale = Math.min(1, PROC_SIDE / longest);
    srcCanvas.width = Math.round(img.width * pScale);
    srcCanvas.height = Math.round(img.height * pScale);
    srcCtx.drawImage(img, 0, 0, srcCanvas.width, srcCanvas.height);
    URL.revokeObjectURL(img.src);
    startCalibration();
  };
  img.src = URL.createObjectURL(file);
  ev.target.value = ''; // allow re-picking the same file
});

// ===========================================================================
// STEP 2 — calibrate (zoom + tap two edges of one bead)
// ===========================================================================
const calCanvas = el('calCanvas');
const calCtx = calCanvas.getContext('2d');
const loupe = el('loupe');
const loupeCtx = loupe.getContext('2d');

let points = [];            // 0..2 endpoints in proc-image space [{x,y}]
let activePt = -1;          // index of the endpoint currently being placed/dragged
const pointers = new Map(); // pointerId → {x,y} in canvas pixels (for pinch)
let pinch = null;           // { dist, imgMid:{x,y} } captured at pinch start
let view = { scale: 1, cx: 0, cy: 0 }; // cx,cy = image point shown at canvas center

const LOUPE_WIN = 30;       // proc pixels shown across the loupe (smaller = more zoom)
const HIT_R = 26;           // endpoint grab radius, in canvas pixels

function startCalibration() {
  calCanvas.width = srcCanvas.width;
  calCanvas.height = srcCanvas.height;
  points = [];
  activePt = -1;
  pointers.clear();
  pinch = null;
  view = { scale: 1, cx: srcCanvas.width / 2, cy: srcCanvas.height / 2 };
  diameter = 0;
  el('calCount').disabled = true;
  el('calReadout').textContent = '';
  loupe.hidden = true;
  drawCal();
  showStep('calibrate');
}

// --- view transform (image proc-space ⇄ canvas pixels) ---------------------
function visibleRect() {
  const CW = calCanvas.width, CH = calCanvas.height;
  const sw = CW / view.scale, sh = CH / view.scale;
  const sx = clamp(view.cx - sw / 2, 0, CW - sw);
  const sy = clamp(view.cy - sh / 2, 0, CH - sh);
  return { sx, sy, sw, sh };
}
function imgToCanvas(p) {
  const { sx, sy } = visibleRect();
  return { x: (p.x - sx) * view.scale, y: (p.y - sy) * view.scale };
}
function canvasToImg(c) {
  const { sx, sy } = visibleRect();
  return { x: sx + c.x / view.scale, y: sy + c.y / view.scale };
}
function eventToCanvas(ev) {
  const r = calCanvas.getBoundingClientRect();
  return {
    x: (ev.clientX - r.left) * (calCanvas.width / r.width),
    y: (ev.clientY - r.top) * (calCanvas.height / r.height),
  };
}

function drawCal() {
  const CW = calCanvas.width, CH = calCanvas.height;
  const { sx, sy, sw, sh } = visibleRect();
  calCtx.imageSmoothingEnabled = false;
  calCtx.clearRect(0, 0, CW, CH);
  calCtx.drawImage(srcCanvas, sx, sy, sw, sh, 0, 0, CW, CH);

  if (points.length) {
    const cpts = points.map(imgToCanvas);
    const lw = Math.max(2, CW / 300);
    if (cpts.length === 2) {
      calCtx.lineWidth = lw;
      calCtx.strokeStyle = '#ff5c8a';
      calCtx.beginPath();
      calCtx.moveTo(cpts[0].x, cpts[0].y);
      calCtx.lineTo(cpts[1].x, cpts[1].y);
      calCtx.stroke();
    }
    for (const c of cpts) {
      calCtx.fillStyle = '#ffffff';
      calCtx.strokeStyle = '#ff5c8a';
      calCtx.lineWidth = lw;
      calCtx.beginPath();
      calCtx.arc(c.x, c.y, lw * 2.2, 0, Math.PI * 2);
      calCtx.fill();
      calCtx.stroke();
    }
  }
}

// --- magnifier loupe -------------------------------------------------------
function drawLoupe(pImg, canvasX) {
  const ratio = fullCanvas.width / srcCanvas.width; // proc → full scale
  const L = loupe.width;
  const fwin = LOUPE_WIN * ratio;
  const fx = pImg.x * ratio, fy = pImg.y * ratio;
  loupeCtx.imageSmoothingEnabled = false;
  loupeCtx.clearRect(0, 0, L, L);
  loupeCtx.drawImage(fullCanvas, fx - fwin / 2, fy - fwin / 2, fwin, fwin, 0, 0, L, L);
  // crosshair at the exact endpoint
  loupeCtx.strokeStyle = '#ff5c8a';
  loupeCtx.lineWidth = 1.5;
  loupeCtx.beginPath();
  loupeCtx.moveTo(L / 2, L / 2 - 10); loupeCtx.lineTo(L / 2, L / 2 + 10);
  loupeCtx.moveTo(L / 2 - 10, L / 2); loupeCtx.lineTo(L / 2 + 10, L / 2);
  loupeCtx.stroke();
  loupeCtx.beginPath();
  loupeCtx.arc(L / 2, L / 2, 5, 0, Math.PI * 2);
  loupeCtx.stroke();
  // keep the loupe on the opposite side from the finger
  loupe.classList.toggle('loupe--right', canvasX < calCanvas.width / 2);
  loupe.hidden = false;
}

// --- endpoint helpers ------------------------------------------------------
function hitEndpoint(c) {
  let hit = -1, best = HIT_R;
  points.forEach((p, i) => {
    const cc = imgToCanvas(p);
    const dd = Math.hypot(cc.x - c.x, cc.y - c.y);
    if (dd < best) { best = dd; hit = i; }
  });
  return hit;
}
function updateReadout() {
  if (points.length === 2) {
    diameter = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
    el('calCount').disabled = diameter < 4;
    el('calReadout').textContent = `bead ≈ ${Math.round(diameter)} px`;
  } else {
    diameter = 0;
    el('calCount').disabled = true;
    el('calReadout').textContent = points.length === 1
      ? 'now tap the other edge' : 'tap one edge of a bead';
  }
}

// --- pointer handling (1 finger = place/drag, 2 fingers = pinch/pan) --------
calCanvas.addEventListener('pointerdown', (ev) => {
  ev.preventDefault();
  const c = eventToCanvas(ev);
  pointers.set(ev.pointerId, c);
  calCanvas.setPointerCapture(ev.pointerId);

  if (pointers.size >= 2) { beginPinch(); loupe.hidden = true; activePt = -1; return; }

  activePt = hitEndpoint(c);
  if (activePt < 0) {
    const img = canvasToImg(c);
    if (points.length < 2) { points.push(img); activePt = points.length - 1; }
    else { activePt = nearerEndpoint(c); points[activePt] = img; } // retarget the nearer end
  }
  drawLoupe(points[activePt], c.x);
  drawCal();
  updateReadout();
});

calCanvas.addEventListener('pointermove', (ev) => {
  if (!pointers.has(ev.pointerId)) return;
  const c = eventToCanvas(ev);
  pointers.set(ev.pointerId, c);

  if (pointers.size >= 2) { updatePinch(); drawCal(); return; }
  if (activePt < 0) return;
  points[activePt] = canvasToImg(c);
  drawLoupe(points[activePt], c.x);
  drawCal();
  updateReadout();
});

function endPointer(ev) {
  pointers.delete(ev.pointerId);
  if (pointers.size < 2) pinch = null;
  if (pointers.size === 0) { activePt = -1; loupe.hidden = true; drawCal(); }
}
calCanvas.addEventListener('pointerup', endPointer);
calCanvas.addEventListener('pointercancel', endPointer);

function nearerEndpoint(c) {
  const d0 = Math.hypot(imgToCanvas(points[0]).x - c.x, imgToCanvas(points[0]).y - c.y);
  const d1 = Math.hypot(imgToCanvas(points[1]).x - c.x, imgToCanvas(points[1]).y - c.y);
  return d0 <= d1 ? 0 : 1;
}

// pinch: anchor the image point under the two-finger midpoint while scaling
function twoPointers() { return Array.from(pointers.values()); }
function beginPinch() {
  const [a, b] = twoPointers();
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  pinch = {
    dist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
    scale: view.scale,
    imgMid: canvasToImg(mid),
  };
}
function updatePinch() {
  if (!pinch) { beginPinch(); return; }
  const [a, b] = twoPointers();
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const dist = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
  const CW = calCanvas.width, CH = calCanvas.height;
  view.scale = clamp(pinch.scale * (dist / pinch.dist), 1, 14);
  // choose cx,cy so canvasToImg(mid) == pinch.imgMid (keeps the pinch point put + pans)
  view.cx = pinch.imgMid.x + CW / (2 * view.scale) - mid.x / view.scale;
  view.cy = pinch.imgMid.y + CH / (2 * view.scale) - mid.y / view.scale;
  const sw = CW / view.scale, sh = CH / view.scale;
  view.cx = clamp(view.cx, sw / 2, CW - sw / 2);
  view.cy = clamp(view.cy, sh / 2, CH - sh / 2);
}

el('calRetake').addEventListener('click', () => { showStep('capture'); el('fileInput').click(); });
el('calCount').addEventListener('click', runCount);

// ===========================================================================
// STEP 3 — count + review + correct
// ===========================================================================
function runCount() {
  if (!workerReady) { alert('The vision engine is still loading — try again in a moment.'); return; }
  if (pendingCount) return;
  if (points.length !== 2) { alert('Mark across one bead first — tap each edge.'); return; }
  pendingCount = true;
  el('spinner').hidden = false;
  const imageData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const line = { x1: points[0].x, y1: points[0].y, x2: points[1].x, y2: points[1].y };
  worker.postMessage({ type: 'count', imageData, line }, [imageData.data.buffer]);
}

const resCanvas = el('resCanvas');
const resCtx = resCanvas.getContext('2d');

function renderResult() {
  resCanvas.width = srcCanvas.width;
  resCanvas.height = srcCanvas.height;
  resCtx.drawImage(srcCanvas, 0, 0);
  const r = Math.max(4, diameter / 2);
  for (const m of markers) {
    resCtx.lineWidth = Math.max(2, diameter / 12);
    resCtx.strokeStyle = '#ff5c8a';
    resCtx.beginPath();
    resCtx.arc(m.x, m.y, r * 0.72, 0, Math.PI * 2);
    resCtx.stroke();
    resCtx.fillStyle = '#ffffffcc';
    resCtx.beginPath();
    resCtx.arc(m.x, m.y, resCtx.lineWidth * 0.9, 0, Math.PI * 2);
    resCtx.fill();
  }
  el('countBig').textContent = markers.length;
  el('countLabel').textContent = markers.length === 1 ? 'bead' : 'beads';
  const note = (lastAreaCount != null && Math.abs(lastAreaCount - markers.length) >= 3)
    ? `cross-check: ~${lastAreaCount} — double-check the markers`
    : (lastAreaCount != null ? `cross-check: ~${lastAreaCount}` : '');
  el('estNote').textContent = note;
}

// tap: hit a marker → remove; empty space → add
resCanvas.addEventListener('pointerup', (ev) => {
  const p = toCanvasXY(resCanvas, ev);
  const hitR = Math.max(6, diameter * 0.5);
  let hit = -1, best = Infinity;
  markers.forEach((m, i) => {
    const dd = Math.hypot(m.x - p.x, m.y - p.y);
    if (dd < hitR && dd < best) { best = dd; hit = i; }
  });
  if (hit >= 0) markers.splice(hit, 1);
  else markers.push({ x: p.x, y: p.y });
  renderResult();
});

el('resRecount').addEventListener('click', runCount);
el('resRestart').addEventListener('click', () => { showStep('capture'); el('fileInput').click(); });

// --- boot ------------------------------------------------------------------
initWorker();
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW failed', e));
}
