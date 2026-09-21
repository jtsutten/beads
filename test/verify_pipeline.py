"""Mirror of cv-worker.js count() in Python/OpenCV, for validating the detector.

Keep this in sync with cv-worker.js if the algorithm changes.

Approach (corridor + centerline + pitch — background-agnostic 1-D counting):
  1. Learn a BACKGROUND colour model (patches beside the strand + image corners) and a
     bead colour from the calibration stroke. A pixel is "strand" when it is far from
     every background sample -> works for multi-colour strands, where the beads have no
     single colour but the table does.
  2. Trace the strand CENTERLINE outward from the stroke midpoint, following the strand
     ribbon and staying in a corridor around it -> the busy background is never inspected.
  3. Count by PITCH: build a 1-D signal along the centerline (cross-strand width, mean L,
     and boundary gradient), find the dominant period by autocorrelation, and take
     count = arc_length / period. Robust to tiny beads (aggregates the whole strand) and
     to colour changes (keys on spacing, not appearance).

Usage:
    python3 test/verify_pipeline.py IMAGE x1 y1 x2 y2 [--annot out.png] [--proc 2400]
    python3 test/verify_pipeline.py --batch DIR [--proc 2400]
        batch: reads DIR/*.{jpg,jpeg,png}; true count parsed from a *_count-NN.* or
        strand-NN filename; calibration lines read from DIR/lines.json
        ({"<filename>": [x1,y1,x2,y2]} in each image's ORIGINAL pixel space).
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

PROC_SIDE = 2400  # must match app.js


# --------------------------------------------------------------------------- helpers
def _resize_to_proc(img, line, proc_side):
    h, w = img.shape[:2]
    scale = min(1.0, proc_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, [c * scale for c in line], scale


def _patch_lab(lab, x, y, rad=2):
    h, w = lab.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return None
    p = lab[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].reshape(-1, 3)
    return np.median(p, axis=0)


def _dedup_refs(refs, tol):
    out = []
    for r in refs:
        r = np.asarray(r, float)
        if not any(np.linalg.norm(o - r) < tol for o in out):
            out.append(r)
    return out


def _contiguous(vals, ci):
    """Span of the True run in bool array `vals` that contains index ci (or a neighbour)."""
    n = len(vals)
    if not vals[ci]:
        for off in (1, -1, 2, -2):
            if 0 <= ci + off < n and vals[ci + off]:
                ci = ci + off
                break
        else:
            return None
    lo = hi = ci
    while lo - 1 >= 0 and vals[lo - 1]:
        lo -= 1
    while hi + 1 < n and vals[hi + 1]:
        hi += 1
    return lo, hi


def _autoc_period(sig, pmin, pmax):
    """Fundamental period of `sig` in [pmin,pmax] via autocorrelation + parabolic refine.

    Picks the FIRST prominent autocorrelation peak (the fundamental), not the global max
    in range -- the global max often lands on 2x the pitch (a harmonic), which halves the
    count.
    """
    s = np.asarray(sig, float)
    if len(s) < 2 * pmax or s.std() < 1e-6:
        return None, 0.0
    s = s - s.mean()
    n = len(s)
    ac = np.correlate(s, s, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = max(1, int(pmin)), min(n - 2, int(pmax))
    if hi <= lo:
        return None, 0.0
    gmax = float(ac[lo:hi + 1].max())
    if gmax <= 0:
        return None, 0.0
    thresh = 0.5 * gmax
    k = None
    for j in range(lo, hi + 1):                          # first local max above threshold
        if ac[j] >= thresh and ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1]:
            k = j
            break
    if k is None:
        k = int(np.argmax(ac[lo:hi + 1])) + lo
    # octave guard: if HALF the chosen lag is also a strong peak (and still a plausible
    # pitch >= pmin), the chosen lag was a 2x harmonic -> peel down to the fundamental.
    while k // 2 >= lo and ac[k // 2] >= 0.55 * ac[k]:
        k = k // 2
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    return k + delta, float(ac[k])


# --------------------------------------------------------------------------- detector
def count_beads(img_bgr, line, annot_path=None, proc_side=PROC_SIDE, debug=False):
    img, line, _ = _resize_to_proc(img_bgr, line, proc_side)
    x1, y1, x2, y2 = line
    rows, cols = img.shape[:2]

    d = max(4.0, float(np.hypot(x2 - x1, y2 - y1)))       # bead width across the strand
    L = np.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / L, (y2 - y1) / L                 # across-strand unit
    ax, ay = -uy, ux                                      # along-strand unit
    mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- colour models -----------------------------------------------------
    bead_samples = []
    for i in range(10):
        t = 0.2 + 0.6 * (i / 9)
        s = _patch_lab(lab, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        if s is not None:
            bead_samples.append(s)
    bead_mean = np.median(bead_samples, axis=0) if bead_samples else np.array([128, 128, 128.])

    bg_refs = []
    for k in (1.6, 2.2, 3.0):                             # patches beside the strand (far enough
        for sgn in (1, -1):                              # to clear tiny strands)
            s = _patch_lab(lab, mid[0] + sgn * k * d * ux, mid[1] + sgn * k * d * uy, rad=3)
            if s is not None:
                bg_refs.append(s)
    for (cx, cy) in ((6, 6), (cols - 7, 6), (6, rows - 7), (cols - 7, rows - 7),
                     (cols // 2, 6), (cols // 2, rows - 7)):  # corners + top/bottom mid
        s = _patch_lab(lab, cx, cy, rad=4)
        if s is not None:
            bg_refs.append(s)
    bg_refs = _dedup_refs(bg_refs, 6.0) if bg_refs else [np.array([240, 128, 128.])]
    # drop any "background" sample that is actually bead-coloured (patch landed on the strand),
    # else the beads get labelled background and the mask comes back empty
    kept = [r for r in bg_refs if np.linalg.norm(r - bead_mean) > 15]
    if kept:
        bg_refs = kept

    # strand = far from EVERY background ref  OR  close to the bead colour (rescue)
    flat = lab.reshape(-1, 3)
    dist2 = np.full(flat.shape[0], 1e18, np.float32)
    for ref in bg_refs:
        dist2 = np.minimum(dist2, ((flat - ref) ** 2).sum(axis=1))
    dist2 = dist2.reshape(rows, cols)
    dist2_bead = ((lab - bead_mean) ** 2).sum(axis=2)
    BG_T, BEAD_T = 22.0, 20.0
    mask = ((dist2 > BG_T * BG_T) | (dist2_bead < BEAD_T * BEAD_T)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    kc = max(3, int(round(d * 0.3)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    mb = mask > 0

    # --- centerline tracing -----------------------------------------------
    step = max(1.0, d * 0.4)
    half = max(2, int(round(d * 1.2)))
    offs = np.arange(-half, half + 1, 1.0)

    def cross_section(pos, direction):
        perp = np.array([-direction[1], direction[0]])
        xs = pos[0] + offs * perp[0]
        ys = pos[1] + offs * perp[1]
        xi = np.round(xs).astype(int)
        yi = np.round(ys).astype(int)
        ok = (xi >= 0) & (xi < cols) & (yi >= 0) & (yi < rows)
        vals = np.zeros(len(offs), bool)
        vals[ok] = mb[yi[ok], xi[ok]]
        return perp, vals

    def trace(sign):
        nodes = []
        pos = mid.astype(float).copy()
        direction = np.array([ax * sign, ay * sign], float)
        for _ in range(6000):
            nxt = pos + direction * step
            perp, vals = cross_section(nxt, direction)
            if vals.sum() < 2:
                break
            ci = int(np.argmin(np.abs(offs)))
            run = _contiguous(vals, ci)
            if run is None:
                break
            lo, hi = run
            c_off = offs[lo:hi + 1].mean()
            width = (hi - lo + 1) * 1.0
            newpos = nxt + perp * c_off
            if not (0 <= newpos[0] < cols and 0 <= newpos[1] < rows):
                break
            nodes.append((newpos.copy(), width))
            nd = newpos - pos
            nl = np.hypot(*nd)
            if nl > 1e-6:
                direction = 0.5 * direction + 0.5 * (nd / nl)
                direction /= (np.hypot(*direction) or 1)
            pos = newpos
        return nodes

    back = trace(-1)
    fwd = trace(1)
    nodes = list(reversed(back)) + [(mid.astype(float), d)] + fwd
    if len(nodes) < 3:
        return _finish([], 0, img, d, annot_path)

    pts = np.array([n[0] for n in nodes])
    widths = np.array([n[1] for n in nodes])

    # --- resample to uniform arc length -----------------------------------
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    arc = np.concatenate([[0], np.cumsum(seg)])
    Ltot = arc[-1]
    if Ltot < d:
        return _finish([], 0, img, d, annot_path)
    ds = 0.5
    s_samples = np.arange(0, Ltot, ds)
    px = np.interp(s_samples, arc, pts[:, 0])
    py = np.interp(s_samples, arc, pts[:, 1])
    wsig = np.interp(s_samples, arc, widths)

    # brightness (L) along the centerline; boundary gradient = |dL/ds|
    Lc = lab[:, :, 0]
    xi = np.clip(np.round(px).astype(int), 0, cols - 1)
    yi = np.clip(np.round(py).astype(int), 0, rows - 1)
    lsig = Lc[yi, xi].astype(float)

    # --- pitch by autocorrelation -----------------------------------------
    # signals that oscillate ONCE per bead: L (bright centre / dark neck) and cross-strand
    # width (bicone necks pinch). The |dL/ds| gradient is deliberately NOT used -- it has two
    # edges per bead, so its fundamental is pitch/2 and it would double the count.
    pmin, pmax = 0.55 * d / ds, 2.4 * d / ds
    cands = []
    for name, sig in (("width", -wsig), ("L", lsig)):
        p, strength = _autoc_period(sig, pmin, pmax)
        if p:
            cands.append((strength, p, name))
    if not cands:
        return _finish([], 0, img, d, annot_path)
    cands.sort(reverse=True)
    pitch_px = cands[0][1] * ds                            # back to image px
    count = max(1, int(round(Ltot / pitch_px)))
    # cross-check count from the 2nd-strongest signal
    alt = int(round(Ltot / (cands[1][1] * ds))) if len(cands) > 1 else count

    # --- markers: evenly spaced along the centerline -----------------------
    markers = []
    for j in range(count):
        s = (j + 0.5) * Ltot / count
        markers.append((float(np.interp(s, arc, pts[:, 0])),
                        float(np.interp(s, arc, pts[:, 1]))))

    if debug:
        print(f"    Ltot={Ltot:.0f}px pitch={pitch_px:.1f}px signal={cands[0][2]} "
              f"strength={cands[0][0]:.2f} count={count} alt={alt} nodes={len(nodes)}")
    return _finish(markers, alt, img, d, annot_path, pts)


def _finish(markers, alt, img, d, annot_path, centerline=None):
    if annot_path is not None:
        out = img.copy()
        if centerline is not None and len(centerline) > 1:
            for i in range(len(centerline) - 1):
                cv2.line(out, tuple(centerline[i].astype(int)),
                         tuple(centerline[i + 1].astype(int)), (255, 200, 0), 1)
        for (x, y) in markers:
            cv2.circle(out, (int(x), int(y)), int(max(6, d * 0.35)), (0, 255, 0), 2)
        cv2.imwrite(annot_path, out)
    return len(markers), alt, 0


# --------------------------------------------------------------------------- eval harness
def _true_count(fn):
    for pat in (r"_count-(\d+)", r"count-(\d+)", r"strand-(\d+)"):
        m = re.search(pat, os.path.basename(fn))
        if m:
            return int(m.group(1))
    return None


def _batch(dirpath, proc_side, annot=True):
    lines = {}
    lj = os.path.join(dirpath, "lines.json")
    if os.path.exists(lj):
        with open(lj) as f:
            lines = json.load(f)
    files = sorted(sum([glob.glob(os.path.join(dirpath, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")], []))
    files = [f for f in files if not os.path.basename(f).startswith(("annot", "prev", "dbg", "mask"))]
    errs = []
    print(f"{'image':40s} {'px/bead':>8s} {'true':>5s} {'got':>5s} {'err':>5s}  alt")
    for f in files:
        base = os.path.basename(f)
        true = _true_count(f)
        line = lines.get(base)
        if line is None:
            print(f"{base:40s} {'—':>8s}  (no calibration line in lines.json)")
            continue
        img = cv2.imread(f)
        _, sl, scale = _resize_to_proc(img, line, proc_side)
        pxbead = np.hypot(sl[2] - sl[0], sl[3] - sl[1])
        annot_path = os.path.join(dirpath, "annot-" + os.path.splitext(base)[0] + ".png") if annot else None
        got, alt, _ = count_beads(img, line, annot_path=annot_path, proc_side=proc_side)
        err = (got - true) if true is not None else None
        if err is not None:
            errs.append(abs(err))
        es = f"{err:+d}" if err is not None else "  ?"
        ts = str(true) if true is not None else "?"
        print(f"{base:40s} {pxbead:8.1f} {ts:>5s} {got:>5d} {es:>5s}  {alt}")
    if errs:
        print(f"\n  MAE = {np.mean(errs):.2f} over {len(errs)} images "
              f"(within-5: {sum(e<=5 for e in errs)}/{len(errs)})")


if __name__ == "__main__":
    args = sys.argv[1:]
    proc = PROC_SIDE
    if "--proc" in args:
        i = args.index("--proc")
        proc = int(args[i + 1])
        del args[i:i + 2]
    if "--batch" in args:
        i = args.index("--batch")
        _batch(args[i + 1], proc)
    elif len(args) >= 5:
        path = args[0]
        line = tuple(float(v) for v in args[1:5])
        annot = args[args.index("--annot") + 1] if "--annot" in args else None
        got, alt, _ = count_beads(cv2.imread(path), line, annot_path=annot, proc_side=proc, debug=True)
        print(f"{path}: count={got}  cross-check={alt}")
    else:
        print(__doc__)
