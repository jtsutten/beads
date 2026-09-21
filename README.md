# Bead Counter

Take a phone photo of a strand of beads, get the count. Runs **entirely on your
phone** (OpenCV compiled to WebAssembly, in the browser) — no backend, no accounts,
no cost. It's a static Progressive Web App you install by opening a URL.

Built for a specific, well-scoped case: beads on one strand that are **the same
size** (color and shine may vary), touching each other, up to ~70 per strand.
See `../.claude/plans/wild-purring-lark.md` for the full spec.

## How it works

1. **Take / choose a photo** of the strand.
2. **Calibrate** — drag a line across *one* bead so the app knows the bead size.
3. It **detects** the beads (distance-transform peaks) and shows a marker on each,
   plus an independent area-based cross-check.
4. **Tap to correct** — tap empty space to add a bead, tap a marker to remove one.
   The number updates live, so you can always land on the exact count.

Because the beads are uniform in size, the count comes from geometry (bead centers
and total area), not from AI guessing — that's why it's accurate and free.

### Capture tips (this is what keeps it accurate)
- Lay the strand on a **plain, contrasting** background (white paper, solid cloth).
- **Even lighting**, minimal harsh glare.
- Shoot from **directly above**, strand roughly straight, single layer (not piled).

## Run it

### Try locally on a computer (quickest sanity check)
```bash
cd beads
python3 -m http.server 8000
# open http://localhost:8000  (localhost is a "secure context", so the
# service worker + file picker work; pick one of the test/strand-*.png images)
```

### Use it on your phone — deploy free to GitHub Pages
A phone needs **HTTPS** for install + camera, which Pages gives you for free.
```bash
cd beads
git init && git add . && git commit -m "Bead Counter"
# create an empty GitHub repo, then:
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
In the repo: **Settings → Pages → Build from branch → `main` / root**. After a
minute it's live at `https://<you>.github.io/<repo>/`.

- Send that URL to your friend.
- On Android Chrome: open it → menu **⋮ → Add to Home screen** → it installs like
  an app (works offline after the first load).

Cloudflare Pages / Netlify work the same way if you prefer (drag-and-drop the
folder). All free.

## Testing / accuracy

The counting algorithm in `cv-worker.js` is mirrored in `test/verify_pipeline.py`
(same steps, Python OpenCV) and checked against known-count fixtures:

```bash
python3 test/make_test_strand.py 42 test/strand-42.png   # regenerate a fixture
python3 test/verify_pipeline.py                            # run the checks
```
On the synthetic fixtures (uniform touching beads, varied colors, highlights) the
detector matches the true count exactly for 8–70 beads. Real photos are harder —
that's what the capture tips and tap-to-correct step are for. **Keep the Python
mirror in sync with `cv-worker.js` if you change the algorithm.**

## Files
| File | Role |
|------|------|
| `index.html` / `styles.css` | UI shell (3 steps: capture → calibrate → result) |
| `app.js` | UI flow, camera input, calibration drag, marker overlay, tap-to-correct |
| `cv-worker.js` | OpenCV pipeline (segment → distance-transform peaks + area), off the main thread |
| `manifest.webmanifest` / `sw.js` / `icon.svg` | PWA install + offline caching |
| `test/` | Fixture generator + Python mirror of the pipeline |

## Known limits (by design, for the POC)
- One strand, single layer, uniform bead size. Piles and mixed bead types are future work.
- Needs one calibration drag per photo (auto-sizing is a planned follow-up).
- Transparent/glass beads and busy backgrounds degrade auto-detection — lean on tap-to-correct.
