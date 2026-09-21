"""Mirror of cv-worker.js count() in Python/OpenCV, for validating the detector.

Keep this in sync with cv-worker.js if the algorithm changes.

Approach — trace + boundary count (tuned for tiny multi-colour beads):
  1. TRACE the strand centerline out from the calibration stroke. Each step looks at a short
     perpendicular cross-section, estimates the LOCAL fabric colour from its outer ends (immune
     to global shading/folds), and takes the run of pixels near the centre that are far from
     that local fabric = the strand here. Coasts through low-contrast (black) beads / thread
     gaps via direction momentum.
  2. COUNT bead boundaries: sample colour along the centerline, take the along-strand colour
     gradient (peaks at each bead boundary), count prominent peaks. Handles random multi-colour
     beads (keys on boundaries, not colour) and skips smooth thread gaps. Pitch (autocorr) sets
     the peak spacing; markers land on the peaks.

Usage:
    python3 test/verify_pipeline.py IMAGE x1 y1 x2 y2 [--annot out.png] [--proc 3000]
    python3 test/verify_pipeline.py --batch DIR [--proc 3000]
        batch: DIR/*.{jpg,jpeg,png}; true count from *_count-NN.* / strand-NN filename;
        calibration lines from DIR/lines.json ({"<filename>": [x1,y1,x2,y2]} in ORIGINAL px).
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

PROC_SIDE = 3000  # must match app.js


def _resize_to_proc(img, line, proc_side):
    h, w = img.shape[:2]
    scale = min(1.0, proc_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, [c * scale for c in line], scale


def _median(a):
    return float(np.median(a)) if len(a) else 0.0


def autoc_period(sig, pmin, pmax):
    s = np.asarray(sig, float)
    n = len(s)
    if n < 2 * pmax:
        return None
    s = s - s.mean()
    varsum = float((s * s).sum())
    if varsum < 1e-6:
        return None
    lo, hi = max(1, int(pmin)), min(n - 2, int(pmax))
    if hi <= lo:
        return None
    ac = np.zeros(hi + 2)
    for lag in range(lo - 1, hi + 2):
        ac[lag] = float((s[:n - lag] * s[lag:]).sum()) / varsum
    gmax = ac[lo:hi + 1].max()
    if gmax <= 0:
        return None
    thresh = 0.5 * gmax
    k = -1
    for j in range(lo, hi + 1):
        if ac[j] >= thresh and ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1]:
            k = j
            break
    if k < 0:
        k = lo + int(np.argmax(ac[lo:hi + 1]))
    while (k >> 1) >= lo and ac[k >> 1] >= 0.55 * ac[k]:
        k = k >> 1
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    den = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    return k + delta


def _trim_tail_peaks(peaks, pitch):
    """Drop leading/trailing peaks separated from the bead cluster by a big gap (thread tails)."""
    peaks = list(peaks)
    while len(peaks) > 3 and (peaks[1] - peaks[0]) > 2.5 * pitch:
        peaks.pop(0)
    while len(peaks) > 3 and (peaks[-1] - peaks[-2]) > 2.5 * pitch:
        peaks.pop()
    return peaks


def _fill_markers(peaks, pitch, ds, nS, wsig, medw):
    """Keep detected boundary peaks; between two peaks whose interval is bead-WIDTH (not thin
    thread) fill any missed boundaries; skip filling across thread gaps/tails (thin, or big).
    Returns marker sample indices, so the count matches real beads on strong- and weak-boundary
    strands alike without counting thread as beads."""
    if not peaks:
        return []
    out = [peaks[0]]
    for k in range(1, len(peaks)):
        a, b = peaks[k - 1], peaks[k]
        g = (b - a) * ds
        mid_w = float(np.mean(wsig[a:b + 1])) if b > a else wsig[a]
        if g <= 2.5 * pitch and mid_w >= 0.6 * medw:  # bead-width interval: fill missed boundaries
            n = max(1, round(g / pitch))
            for m in range(1, n):
                out.append(int(min(nS - 1, max(0, round(a + (b - a) * m / n)))))
        out.append(b)                                 # thin/large gap (thread) => no fill
    return out


def portable_peaks(y, min_dist, min_prom):
    n = len(y)
    maxima = [i for i in range(1, n - 1) if y[i] >= y[i - 1] and y[i] > y[i + 1]]
    prom = {}
    for i in maxima:
        j = i - 1; lmin = y[i]
        while j >= 0 and y[j] <= y[i]:
            lmin = min(lmin, y[j]); j -= 1
        j = i + 1; rmin = y[i]
        while j < n and y[j] <= y[i]:
            rmin = min(rmin, y[j]); j += 1
        prom[i] = y[i] - max(lmin, rmin)
    cand = sorted((i for i in maxima if prom[i] >= min_prom), key=lambda i: -prom[i])
    acc = []
    for i in cand:
        if all(abs(i - k) >= min_dist for k in acc):
            acc.append(i)
    return sorted(acc)


def count_beads(img_bgr, line, annot_path=None, proc_side=PROC_SIDE, debug=False):
    img, line, _ = _resize_to_proc(img_bgr, line, proc_side)
    x1, y1, x2, y2 = line
    rows, cols = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    d = max(4.0, float(np.hypot(x2 - x1, y2 - y1)))
    Ln = np.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / Ln, (y2 - y1) / Ln
    ax, ay = -uy, ux
    midx, midy = (x1 + x2) / 2, (y1 + y2) / 2

    def lab_at(x, y):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < cols and 0 <= yi < rows:
            return lab[yi, xi]
        return None

    R = int(max(14, round(1.8 * d)))
    step = max(2.0, 0.4 * d)
    offs = list(range(-R, R + 1))
    coast_max = round(2.5 * d / step)

    def trace(sign):
        nodes = []
        px, py = midx, midy
        dx, dy = ax * sign, ay * sign
        coast = 0
        for _ in range(5000):
            nx, ny = px + dx * step, py + dy * step
            perpx, perpy = -dy, dx
            V = []; ok = []; oL = []; oA = []; oB = []
            for o in offs:
                p = lab_at(nx + o * perpx, ny + o * perpy)
                if p is not None:
                    V.append(p); ok.append(True)
                    if abs(o) > 0.65 * R:
                        oL.append(p[0]); oA.append(p[1]); oB.append(p[2])
                else:
                    V.append((0, 0, 0)); ok.append(False)
            if len(oL) < 6:
                break
            fL, fA, fB = _median(oL), _median(oA), _median(oB)
            dist = [0.0] * len(offs); oD = []
            for i, o in enumerate(offs):
                if not ok[i]:
                    continue
                dd = float(np.sqrt((V[i][0] - fL) ** 2 + (V[i][1] - fA) ** 2 + (V[i][2] - fB) ** 2))
                dist[i] = dd
                if abs(o) > 0.65 * R:
                    oD.append(dd)
            T = max(12.0, _median(oD) + 8)
            bead = [v > T for v in dist]
            ci = R
            if not bead[ci]:
                found = -1
                for off in range(1, 5):
                    if ci + off < len(offs) and bead[ci + off]:
                        found = ci + off; break
                    if ci - off >= 0 and bead[ci - off]:
                        found = ci - off; break
                if found < 0:
                    coast += 1; px, py = nx, ny
                    if coast > coast_max:
                        break
                    continue
                ci = found
            coast = 0
            lo = hi = ci
            while lo - 1 >= 0 and bead[lo - 1]:
                lo -= 1
            while hi + 1 < len(offs) and bead[hi + 1]:
                hi += 1
            w = hi - lo + 1
            if w > 1.6 * R:
                break
            coff = sum(offs[lo:hi + 1]) / w
            npx, npy = nx + coff * perpx, ny + coff * perpy
            if not (0 <= npx < cols and 0 <= npy < rows):
                break
            nodes.append((npx, npy, w))
            ndx, ndy = npx - px, npy - py
            nl = np.hypot(ndx, ndy)
            if nl > 1e-6:
                dx = 0.6 * dx + 0.4 * (ndx / nl); dy = 0.6 * dy + 0.4 * (ndy / nl)
                dl = np.hypot(dx, dy) or 1; dx /= dl; dy /= dl
            px, py = npx, npy
        return nodes

    nodes = list(reversed(trace(-1))) + [(midx, midy, d)] + trace(1)
    if len(nodes) < 8:
        return _finish([], 0, img, d, annot_path, nodes)

    pts = np.array([(n[0], n[1]) for n in nodes])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    arc = np.concatenate([[0], np.cumsum(seg)])
    Ltot = arc[-1]
    if Ltot < 2 * d:
        return _finish([], 0, img, d, annot_path, nodes)

    ds = 1.0
    ss = np.arange(0, Ltot, ds)
    px = np.interp(ss, arc, pts[:, 0]); py = np.interp(ss, arc, pts[:, 1])
    wsig = np.interp(ss, arc, np.array([n[2] for n in nodes]))
    medw = float(np.median(wsig))
    Lp = np.array([lab[min(rows - 1, max(0, int(round(y)))), min(cols - 1, max(0, int(round(x))))][0] for x, y in zip(px, py)])
    ap = np.array([lab[min(rows - 1, max(0, int(round(y)))), min(cols - 1, max(0, int(round(x))))][1] for x, y in zip(px, py)])
    bp = np.array([lab[min(rows - 1, max(0, int(round(y)))), min(cols - 1, max(0, int(round(x))))][2] for x, y in zip(px, py)])
    g = np.sqrt(np.gradient(Lp) ** 2 + np.gradient(ap) ** 2 + np.gradient(bp) ** 2)
    gs = np.convolve(g, np.ones(3) / 3, "same")

    period = autoc_period(gs, 0.6 * d / ds, 1.8 * d / ds)
    pitch = period * ds if period else d
    min_dist = max(3, round(0.7 * pitch / ds))
    peaks = portable_peaks(gs, min_dist, 0.35 * float(np.std(gs)))
    peaks = _trim_tail_peaks(peaks, pitch)          # drop isolated thread-tail peaks
    # PRIMARY count = detected bead boundaries (markers land on real beads; naturally skips
    # smooth thread tails/gaps). CROSS-CHECK = the same with missed boundaries filled in
    # (higher on smooth strands); a big gap between the two is the UI's "double-check" hint.
    markers = [(px[j], py[j]) for j in peaks]
    area = len(_fill_markers(peaks, pitch, ds, len(px), wsig, medw))
    if debug:
        print(f"    nodes={len(nodes)} Ltot={Ltot:.0f} pitch={pitch:.1f} count={len(markers)} cross-check(filled)={area}")
    return _finish(markers, area, img, d, annot_path, nodes)


def _finish(markers, area, img, d, annot_path, nodes):
    if annot_path is not None:
        out = img.copy()
        for i in range(len(nodes) - 1):
            cv2.line(out, (int(nodes[i][0]), int(nodes[i][1])), (int(nodes[i + 1][0]), int(nodes[i + 1][1])), (255, 200, 0), 1)
        for (x, y) in markers:
            cv2.circle(out, (int(x), int(y)), int(max(3, d * 0.4)), (0, 0, 255), -1)
        cv2.imwrite(annot_path, out)
    return len(markers), area, 0


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
    files = sorted(sum([glob.glob(os.path.join(dirpath, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
    files = [f for f in files if not os.path.basename(f).startswith(("annot", "prev", "dbg", "mask", "crop", "markers", "end-", "annot-view"))]
    errs = []
    print(f"{'image':34s} {'px/bead':>8s} {'true':>5s} {'got':>5s} {'err':>5s}  x-check")
    for f in files:
        base = os.path.basename(f)
        line = lines.get(base)
        if line is None:
            print(f"{base:34s}   (no calibration line in lines.json)")
            continue
        true = _true_count(f)
        img = cv2.imread(f)
        _, sl, _ = _resize_to_proc(img, line, proc_side)
        pxbead = float(np.hypot(sl[2] - sl[0], sl[3] - sl[1]))
        ap = os.path.join(dirpath, "annot-" + os.path.splitext(base)[0] + ".png") if annot else None
        got, area, _ = count_beads(img, line, annot_path=ap, proc_side=proc_side)
        err = (got - true) if true is not None else None
        if err is not None:
            errs.append(abs(err))
        print(f"{base:34s} {pxbead:8.1f} {str(true) if true is not None else '?':>5s} {got:>5d} "
              f"{(f'{err:+d}' if err is not None else '  ?'):>5s}  {area}")
    if errs:
        print(f"\n  MAE = {np.mean(errs):.2f} over {len(errs)} images (within-5: {sum(e<=5 for e in errs)}/{len(errs)})")


if __name__ == "__main__":
    args = sys.argv[1:]
    proc = PROC_SIDE
    if "--proc" in args:
        i = args.index("--proc"); proc = int(args[i + 1]); del args[i:i + 2]
    if "--batch" in args:
        i = args.index("--batch"); _batch(args[i + 1], proc)
    elif len(args) >= 5:
        path = args[0]
        line = tuple(float(v) for v in args[1:5])
        annot = args[args.index("--annot") + 1] if "--annot" in args else None
        got, area, _ = count_beads(cv2.imread(path), line, annot_path=annot, proc_side=proc, debug=True)
        print(f"{path}: count={got}  cross-check={area}")
    else:
        print(__doc__)
