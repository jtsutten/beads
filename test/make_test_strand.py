"""Generate a synthetic bead-strand image for testing the counter.

Uniform-size, touching beads on a plain background, with varied colors and
specular highlights — mimicking the real use case. Prints the true count.

Usage: python3 test/make_test_strand.py [count] [out.png]
"""
import math
import random
import sys

from PIL import Image, ImageDraw

count = int(sys.argv[1]) if len(sys.argv) > 1 else 42
out = sys.argv[2] if len(sys.argv) > 2 else "test/strand-%d.png" % count

random.seed(7)
D = 44                      # bead diameter (px)
R = D / 2
pad = 80
amp = 70                    # curve amplitude
W = int(pad * 2 + (count - 1) * D + D)
H = int(pad * 2 + amp * 2 + D)

img = Image.new("RGB", (W, H), (244, 244, 240))   # plain off-white background
dr = ImageDraw.Draw(img)

cy0 = H / 2
pts = []
for i in range(count):
    cx = pad + R + i * D                           # centers exactly D apart => touching
    cy = cy0 + amp * math.sin(i / count * math.pi * 2)
    pts.append((cx, cy))

# faint string
dr.line(pts, fill=(150, 150, 150), width=3)

palette = [
    (198, 96, 120), (120, 140, 205), (110, 175, 130), (210, 175, 90),
    (150, 110, 190), (90, 170, 180), (200, 120, 80),
]
for (cx, cy) in pts:
    base = random.choice(palette)
    # some beads bi-color (like tourmaline): split fill
    if random.random() < 0.35:
        c2 = random.choice(palette)
        dr.pieslice([cx - R, cy - R, cx + R, cy + R], 90, 270, fill=base)
        dr.pieslice([cx - R, cy - R, cx + R, cy + R], -90, 90, fill=c2)
    else:
        dr.ellipse([cx - R, cy - R, cx + R, cy + R], fill=base)
    # specular highlight (varied shine)
    if random.random() < 0.8:
        hr = R * random.uniform(0.15, 0.3)
        hx, hy = cx - R * 0.35, cy - R * 0.35
        dr.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255))

img.save(out)
print("wrote %s  (%dx%d)  true count = %d  diameter ~ %d px" % (out, W, H, count, D))
