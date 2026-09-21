"""Mirror of cv-worker.js count() in Python/OpenCV, for validating the detector.

Keep this in sync with cv-worker.js if the algorithm changes.

Usage:
    python3 test/verify_pipeline.py IMAGE x1 y1 x2 y2 [--annot out.png]
where (x1,y1)-(x2,y2) is the calibration stroke across one bead.
"""
import sys

import cv2
import numpy as np

CHROMA_T = 20


def size_odd(v):
    k = max(3, round(v))
    return k + 1 if k % 2 == 0 else k


def count_beads(img_bgr, line, annot_path=None):
    x1, y1, x2, y2 = line
    d = max(4.0, float(np.hypot(x2 - x1, y2 - y1)))
    radius = d / 2
    single = np.pi * radius * radius

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    rows, cols = lab.shape[:2]

    # sample the INNER portion of the stroke (0.2..0.8) so endpoints that
    # overshoot the bead onto the background don't poison the color sample
    pts = []
    for i in range(10):
        t = 0.2 + 0.6 * (i / 9)
        x = round(x1 + (x2 - x1) * t); y = round(y1 + (y2 - y1) * t)
        if 0 <= x < cols and 0 <= y < rows:
            pts.append((x, y))
    aa, bb = [], []
    for x, y in pts:
        patch = lab[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        aa.extend(patch[:, :, 1].ravel()); bb.extend(patch[:, :, 2].ravel())
    a0 = np.median(aa); b0 = np.median(bb)

    A = lab[:, :, 1].astype(np.float32); B = lab[:, :, 2].astype(np.float32)
    chroma = np.sqrt((A - a0) ** 2 + (B - b0) ** 2)
    mask = (chroma < CHROMA_T).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_odd(d * 0.35),) * 2))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    votes = {}
    for x, y in pts:
        lb = int(labels[y, x])
        if lb > 0:
            votes[lb] = votes.get(lb, 0) + 1
    if votes:
        strand_label = max(votes, key=votes.get)
    else:
        strand_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    strand = (labels == strand_label).astype(np.uint8) * 255

    fg = int(cv2.countNonZero(strand))
    area_count = round(fg / single)

    dist = cv2.distanceTransform(strand, cv2.DIST_L2, 3)
    k = size_odd(d * 0.6)
    dil = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ismax = (dist >= dil).astype(np.uint8) * 255
    peaks = cv2.bitwise_and(ismax, (dist > max(1, 0.3 * radius)).astype(np.uint8) * 255)
    pn, _, _, pc = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    markers = [(pc[i, 0], pc[i, 1]) for i in range(1, pn)]

    if annot_path:
        out = img_bgr.copy()
        for (x, y) in markers:
            cv2.circle(out, (int(x), int(y)), int(max(6, radius * 0.6)), (0, 255, 0), 3)
        cv2.imwrite(annot_path, out)

    return len(markers), area_count, fg


if __name__ == "__main__":
    if len(sys.argv) >= 6:
        path = sys.argv[1]
        line = tuple(float(v) for v in sys.argv[2:6])
        annot = None
        if "--annot" in sys.argv:
            annot = sys.argv[sys.argv.index("--annot") + 1]
        peak, area, fg = count_beads(cv2.imread(path), line, annot)
        print(f"{path}: peak={peak}  area={area}  fg_px={fg}")
    else:
        print(__doc__)
