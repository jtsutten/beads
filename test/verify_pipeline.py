"""Mirror of cv-worker.js count() in Python/OpenCV, to validate accuracy
against known-count fixtures before trusting the JS version.

Keep this in sync with cv-worker.js if the algorithm changes.
"""
import math
import sys

import cv2
import numpy as np


def size_odd(v):
    k = max(3, round(v))
    if k % 2 == 0:
        k += 1
    return (k, k)


def border_mean(m):
    h, w = m.shape
    b = max(2, round(min(w, h) * 0.03))
    strips = [m[0:b, :], m[h - b:h, :], m[:, 0:b], m[:, w - b:w]]
    return sum(float(s.mean()) for s in strips) / len(strips)


def count_beads(img_bgr, diameter):
    d = max(4, diameter)
    radius = d / 2
    single = math.pi * radius * radius

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if border_mean(binimg) > 127:
        binimg = cv2.bitwise_not(binimg)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size_odd(d * 0.25))
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, k_open)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, k_close)

    mask = np.zeros_like(binimg)
    contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.15 * single
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, -1)

    fg_area = int(cv2.countNonZero(mask))
    area_count = round(fg_area / single)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    k_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size_odd(d * 0.7))
    dilated = cv2.dilate(dist, k_peak)
    is_max = (dist >= dilated).astype(np.uint8) * 255
    min_peak = max(1.0, 0.35 * radius)
    dist_thresh = (dist > min_peak).astype(np.uint8) * 255
    peaks = cv2.bitwise_and(is_max, dist_thresh)

    n, _, _, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    peak_count = n - 1
    return peak_count, area_count, fg_area, single


if __name__ == "__main__":
    cases = [("test/strand-42.png", 42, 44), ("test/strand-24.png", 24, 44)]
    if len(sys.argv) > 1:
        cases = [(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))]
    for path, truth, dia in cases:
        img = cv2.imread(path)
        peak, area, fg, single = count_beads(img, dia)
        err = abs(peak - truth)
        flag = "OK" if err <= 5 else "OFF"
        print(f"{path}: true={truth}  peak={peak} (err {err}, {flag})  "
              f"area={area}  fg_px={fg}  bead_px≈{single:.0f}")
