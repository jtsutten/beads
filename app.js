// Bead Counter — UI controller.
// Flow: capture photo → drag to calibrate one bead → CV worker detects beads →
// review & tap-to-correct. Everything runs on-device; no network after load.

const MAX_SIDE = 1200; // cap processing resolution (speed vs. accuracy)

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

// --- state -----------------------------------------------------------------
const srcCanvas = document.createElement('canvas'); // proc-resolution source of truth
const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
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
    const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    srcCanvas.width = w;
    srcCanvas.height = h;
    srcCtx.drawImage(img, 0, 0, w, h);
    URL.revokeObjectURL(img.src);
    startCalibration();
  };
  img.src = URL.createObjectURL(file);
  ev.target.value = ''; // allow re-picking the same file
});

// ===========================================================================
// STEP 2 — calibrate (drag a line across one bead)
// ===========================================================================
const calCanvas = el('calCanvas');
const calCtx = calCanvas.getContext('2d');
let calLine = null;   // {x1,y1,x2,y2}
let dragging = false;

function startCalibration() {
  calCanvas.width = srcCanvas.width;
  calCanvas.height = srcCanvas.height;
  calLine = null;
  diameter = 0;
  el('calCount').disabled = true;
  el('calReadout').textContent = '';
  drawCal();
  showStep('calibrate');
}

function drawCal() {
  calCtx.drawImage(srcCanvas, 0, 0);
  if (calLine) {
    const { x1, y1, x2, y2 } = calLine;
    calCtx.lineWidth = Math.max(2, srcCanvas.width / 300);
    calCtx.strokeStyle = '#ff5c8a';
    calCtx.beginPath();
    calCtx.moveTo(x1, y1);
    calCtx.lineTo(x2, y2);
    calCtx.stroke();
    for (const [px, py] of [[x1, y1], [x2, y2]]) {
      calCtx.fillStyle = '#ffffff';
      calCtx.beginPath();
      calCtx.arc(px, py, calCtx.lineWidth * 1.6, 0, Math.PI * 2);
      calCtx.fill();
    }
  }
}

calCanvas.addEventListener('pointerdown', (ev) => {
  ev.preventDefault();
  const p = toCanvasXY(calCanvas, ev);
  calLine = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
  dragging = true;
  calCanvas.setPointerCapture(ev.pointerId);
});
calCanvas.addEventListener('pointermove', (ev) => {
  if (!dragging) return;
  const p = toCanvasXY(calCanvas, ev);
  calLine.x2 = p.x; calLine.y2 = p.y;
  drawCal();
});
calCanvas.addEventListener('pointerup', () => {
  dragging = false;
  if (!calLine) return;
  const len = Math.hypot(calLine.x2 - calLine.x1, calLine.y2 - calLine.y1);
  if (len < 4) { calLine = null; drawCal(); return; } // treat as stray tap
  diameter = len;
  el('calCount').disabled = false;
  el('calReadout').textContent = `bead ≈ ${Math.round(diameter)} px`;
  drawCal();
});

el('calRetake').addEventListener('click', () => { showStep('capture'); el('fileInput').click(); });
el('calCount').addEventListener('click', runCount);

// ===========================================================================
// STEP 3 — count + review + correct
// ===========================================================================
function runCount() {
  if (!workerReady) { alert('The vision engine is still loading — try again in a moment.'); return; }
  if (pendingCount) return;
  pendingCount = true;
  el('spinner').hidden = false;
  const imageData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  worker.postMessage({ type: 'count', imageData, diameter }, [imageData.data.buffer]);
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
    ? `area estimate: ~${lastAreaCount} — double-check the markers`
    : (lastAreaCount != null ? `area estimate: ~${lastAreaCount}` : '');
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
