# Bead Counter — Handoff

**For:** the next session (a higher-capability model) picking up this project.
**Status:** working PWA end-to-end. Detector was reworked in session 3 (2026-09-21) and now
works on the real tourmaline strand. **§3–§5 below describe the OLD (superseded) detector;
read the update box first.**

> ### ⏩ UPDATE 2026-09-21 (session 3) — detector rewritten, works on real photos
> The old chroma-threshold/single-component/peak detector (§3) is **gone**. New detector
> (`cv-worker.js`, mirrored in `test/verify_pipeline.py`), validated on the user's real
> tourmaline photo (`work/tourmaline-orig.png`, ~10px beads) → **165** (hand count ~155–160),
> markers ~one-per-bead. Was returning **0** before.
>
> **How it works (no global mask — light, scales to full res):**
> 1. **Trace** the strand centerline out from the calibration stroke. Each step looks at a short
>    perpendicular cross-section, estimates the **local fabric** colour from its outer ends
>    (immune to global shading/folds — the thing that made global thresholds flood), and takes
>    the near-centre run far from local fabric = strand here. Coasts through black beads / thread
>    gaps via direction momentum.
> 2. **Count = prominent peaks of the along-centerline colour GRADIENT** (= bead boundaries).
>    Random multi-colour beads have *no periodic colour signal*, but the boundary gradient *is*
>    periodic (pitch ~11px). Autocorrelation of the gradient gives pitch → sets peak min-spacing.
>    Tail-trim drops isolated thread-tail peaks. `areaCount` = same with missed boundaries filled
>    in (width-gated) = the UI "double-check" cross-check. Markers land on real boundaries.
>
> **Also new:** calibration is pinch-zoom + pan + **magnifier loupe + two-tap endpoints** (draw
> across a tiny bead accurately); `PROC_SIDE=3000`; **`#debug`** URL overlays the centerline and
> shows d/pitch/nodes. SW cache at **v5**.
>
> **Key lessons:** (a) chroma (Lab a,b) is shading-robust and separates colourful beads from
> neutral fabric; L-distance floods on folds. (b) **RESOLUTION is the binding constraint** —
> ~10px beads → exact (±5) counting is not achievable by any method; ~±10–15 is the ceiling.
> Push the user to capture higher-res / in segments. (c) Do **not** pivot to ML: no training
> data, SAM doesn't count touching beads, cloud APIs violate $0/on-device. EdgeSAM still
> reserved only if a clean-background classical path proves insufficient.
>
> **Open:** waiting on the user's tap-correct feedback (systematically over/under?) to decide
> whether to enable boundary-fill by default. Eval: `python3 test/verify_pipeline.py --batch DIR`
> (photos `*_count-NN.jpg` + `DIR/lines.json`).

---

## 1. What this is (goal + hard constraints)

A personal tool for **two people** (the user + a friend): take a phone photo of a **strand of
beads** and get the **count**. Full spec: `.claude/plans/wild-purring-lark.md` (read it).

**Hard constraints — do not violate without asking the user:**
- **Android** phones (both users). No iOS needed.
- **$0 recurring cost. No backend server. No paid API calls.** The user was explicit: *"I
  cannot pay anything for this."* → everything runs **on-device** (in the phone browser) and
  is served from **free static hosting**.
- **Just a count** (no bead identification). Colors may vary within a strand; **bead size is
  uniform** per strand. Beads **touch** on the strand. Up to ~70 beads.
- **Accuracy target:** within ~5 of true count on a ~70-bead strand (~93%). A **manual
  tap-to-correct** step exists to close the last few, but auto-detection must get close.
- **Must work on BUSY / TEXTURED backgrounds** — a wooden table with grain, a cluttered
  workspace, etc. The user explicitly does **not** want to have to stage a clean, flat,
  consistent background: *"i want this to work on a wooden table, or something with grain... the
  workspace may be busy and it's hard to keep it perfectly flat background."* This is a
  first-class requirement, not a nice-to-have. It **rules out** relying on a plain-background
  capture protocol and is the strongest argument for object/segmentation-based detection (§5C)
  over any global color/brightness thresholding: wood grain is warm-and-textured and will
  defeat chroma thresholds, and clutter produces exactly the table false positives seen now.

The user is a backend software engineer, **new to mobile dev**. That's why we chose a static
PWA (web tech they know) over native — the "app" part is deliberately trivial; the hard part
is the computer vision.

---

## 2. Where everything lives

- **Repo:** https://github.com/jtsutten/beads (public; user is `jtsutten`).
- **Live app:** https://jtsutten.github.io/beads/ (GitHub Pages, free HTTPS).
- **Deploy = `git push origin main`.** Pages rebuilds in ~1 min. **Gotcha:** the service
  worker caches the shell, so after changing any shell file you **must bump `CACHE` in
  `sw.js`** (currently `beadcounter-v2`) or returning phones keep the old code. Confirm a
  deploy is live by fetching the file with a cache-buster (see how the git history / prior
  session polled `...github.io/beads/cv-worker.js?cb=<ts>`).
- `gh` CLI is authed as `jtsutten`. **Creating public repos is blocked by a safety
  classifier** for the assistant — the user ran the "make public + enable Pages" commands
  themselves via the `!` prefix. Keep that pattern for any public-surface action.

### File map
| File | Role |
|------|------|
| `index.html`, `styles.css` | UI shell: 3 steps — capture → calibrate → result. |
| `app.js` | UI flow. File-input camera, **drag-to-calibrate** (draws a line across one bead), spawns the worker, renders markers on a canvas, **tap-to-add / tap-to-remove** correction. Sends `{imageData, line}` to the worker. Downscales photo to `MAX_SIDE=1200` before processing (markers come back in that same coordinate space). |
| `cv-worker.js` | **The algorithm.** OpenCV (WASM) loaded from `docs.opencv.org/4.9.0/opencv.js` in a Web Worker. This is what to fix. |
| `sw.js`, `manifest.webmanifest`, `icon.svg` | PWA install + offline caching. |
| `test/make_test_strand.py` | Generates synthetic strands (round beads, plain bg) — **too easy**, doesn't reflect reality. |
| `test/verify_pipeline.py` | **Python/OpenCV mirror of `cv-worker.js`.** Keep it in sync; it's how you iterate fast without a phone. Takes an image + calibration line. |
| `work/` | **gitignored.** Scratch: the user's real screenshot, inpainted versions, annotated outputs. Not committed (privacy). |

---

## 3. Current algorithm (in `cv-worker.js`, mirrored in `test/verify_pipeline.py`)

Input: the photo (RGBA, ≤1200px) + the calibration `line` {x1,y1,x2,y2} the user drew across
one bead. The line gives **size** (its length = bead diameter `d`), **color** (sampled along
it), and **which strand** (the blob it sits on).

1. Convert to **Lab**. Sample bead color `(a0,b0)` = median Lab a/b over **5×5 patches along
   the inner 60% of the stroke** (inner portion avoids endpoints that overshoot onto
   background).
2. **Segment by chroma distance:** `mask = sqrt((a-a0)² + (b-b0)²) < CHROMA_T` (`CHROMA_T=20`).
   Idea: beads are near-neutral in Lab, leather is warm → separable.
3. Morphology: open 3×3 (despeckle), close `~0.35·d` (bridge the thin necks between bicones).
4. `connectedComponentsWithStats`; **select the component the stroke sits on** (majority vote
   of the sample points' labels; fallback = largest component).
5. Count = **distance-transform peaks within that one component** (local maxima via dilation,
   gated by `dist > 0.3·radius`, separated by a `~0.6·d` kernel). Marker = each peak centroid.
6. Also returns a loose **area cross-check** (`foregroundArea / (π·(d/2)²)`), shown in the UI
   as a confidence hint.

---

## 4. What works vs. what fails

**Works:** PWA install/offline, camera, calibration drag, tap-to-correct, worker/WASM load,
deploy pipeline. On a **clean, inpainted** version of the user's real photo the mirror
isolates the strand and marks one-per-bead down most of it (~22 markers) with nothing on the
table. Synthetic round-bead fixtures count exactly (but they're unrealistically easy).

**Fails on the real phone photo (user's latest report):**
- **Misses ~half the strand:** gets a clean run of ~10–13 beads, then stops. → Strongly
  implies the **strand mask fragments into multiple connected components** (color drifts along
  the strand due to translucency/shadow/lighting, or necks aren't bridged), and step 4 keeps
  only the fragment under the stroke. True count is roughly double what it reports.
- **Too many table false positives:** markers on table specks. → The single global
  `CHROMA_T` threshold is **simultaneously too loose** (matches near-neutral table specks /
  translucent beads showing table color) **and too tight** (drops the shadowed far half of the
  strand). If the selected component leaks into table regions of similar color, peaks land
  there too.

**Root cause (the real lesson):** a **one-color, fixed-threshold, single-component** classical
segmentation is too brittle for uncontrolled phone photos. This is exactly the failure mode
the original research flagged. Squeezing thresholds won't fix it robustly.

**Test-data caveat:** we never got a *raw* photo file — only a **screenshot** (with the app's
pink markers baked into the pixels, which poisoned color sampling until inpainted). The
inpainted `work/real-clean.png` is the best proxy we have. **First order of business: get the
user to commit 2–3 raw photos + hand counts into `work/` so iteration is measurable.**

---

## 5. What to do next (ranked)

**A. Build a real eval first (do this before tuning anything).**
Ask the user for 2–3 **raw** photos (not screenshots) of different strands on their actual
table, each with a **true hand count**. Drop them in `work/` (gitignored). Extend
`test/verify_pipeline.py` into a batch eval that prints error per image. Without this you're
flying blind — the synthetic fixtures lie.

**B. Cheap, high-value fixes within the current approach (try these first — but note the
busy-background requirement in §1 limits how far color-only methods can go; wood grain is
warm and textured and a two-class color model will still leak. Collinearity gating (B2) is the
background-agnostic win; the durable fix is likely §5C):**
1. **Two-class color model (bead vs. background), not a one-sided threshold.** Sample the
   **background** color too — from image corners, and/or add a "tap the table" gesture — then
   classify each pixel to the *nearer* of {bead, background(s)} in Lab. This adapts to the
   actual table color and rejects it by construction, instead of hoping an absolute threshold
   lands right. Likely the single biggest win.
2. **Collinearity / path gating.** The beads lie along one smooth curve. Fit the strand path
   (PCA line or polyline through the selected component / stroke), then **reject any peak not
   near that path.** Kills scattered table false positives regardless of segmentation.
3. **Stop losing half the strand:** after component selection, **merge nearby components that
   lie along the strand's axis** (or grow the mask along the strand direction), so a color
   drift in the middle doesn't truncate the count. An elongated/directional closing kernel
   oriented along the stroke-perpendicular (strand runs perpendicular to the calibration line)
   helps bridge gaps.
4. **GrabCut** (available in opencv.js): seed foreground = the stroke, background = image
   border, run GrabCut to segment the strand object. Handles gradual color drift and rejects
   background better than a global chroma threshold.

**C. The robust path (recommend to the user):** an **on-device promptable segmentation
model** — e.g. **MobileSAM / EdgeSAM exported to ONNX**, run via **ONNX Runtime Web
(WASM/WebGPU)** in the browser. Prompt it with the calibration tap/point → it segments the
strand object precisely, independent of color drift and table clutter. Then count beads within
that mask (peaks / pitch from calibration). This is **still free and on-device** (fits the
constraints), just a bigger lift (bundle a model, WebGPU on Android Chrome, ~tens of MB
cached). This is the most likely route to reliably hitting the ≤5/70 target on messy photos.
Confirm with the user before committing to the added complexity/model size.

**D. UX safety nets (exactness matters, auto is hard):**
- Make manual correction faster (e.g., drag along the strand and count line crossings).
- Guided capture (on-screen frame; "lay strand straight, plain background").
- These are legitimate given it's a 2-person tool and tapping to fix a few is acceptable — but
  don't let them paper over a detector that misses half the strand.

---

## 6. Gotchas / notes
- **opencv.js quirks:** no `ximgproc` (so no `thinning`/skeleton). Scalar ops need a filled
  `Mat` (see `matScalar` helper). `connectedComponentsWithStats` centroids via
  `cents.doublePtr(i,0/1)`; labels via `labels.intAt(y,x)`. Delete every `Mat` (the code
  collects them in a `trash` array).
- **Coordinate space:** everything (line, markers) is in the downscaled ≤1200px image space;
  `app.js` maps pointer events to canvas pixels via `toCanvasXY`. Keep that invariant.
- **Bicone beads:** the user's beads are faceted bicones (diamond profile), touching tip-to-tip
  with thin necks — not spheres. Peak-in-strand counting is fine *if* the strand is one clean
  mask; the neck constrictions actually aid separation.
- **Colors vary within a strand** (user said so — e.g. bi-color tourmaline). Any single-color
  seed is fragile; the two-class model and/or ML segmentation address this.
- **SW cache bump** on every shell change (§2). Users must fully relaunch to pick up updates.
- Keep `test/verify_pipeline.py` in lockstep with `cv-worker.js` — it's the only fast
  iteration loop (no phone required).

---

## 7. Quick start for the next session
```bash
cd ~/workspace/beads
# 1) reproduce current behavior on the proxy image:
python3 test/verify_pipeline.py work/real-clean.png $(cat work/line2.txt) --annot work/annot.png
# 2) get raw photos + true counts from the user into work/  (build the eval — §5A)
# 3) implement §5B(1) two-class color model, measure, iterate
# 4) local UI test:  python3 -m http.server 8000  → http://localhost:8000
# 5) deploy: edit code, bump CACHE in sw.js, git push origin main, wait ~1 min
```
The user is about to test again and can provide raw photos + counts — **ask for them.**
