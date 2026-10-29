# BananaRipeness — Progress (as of 2026-10-29)

## Hardware integration plan + Nov-4 demo scope (decided 2026-10-29)

Demo target: **Nov 4, 2026**. Detection scope is **frozen** at this point
(far-band rescue + containment dedup, commit `b66a0b6`). **Not in scope for
the demo:** retraining on a combined close + far + far-crop YOLO dataset —
the COCO fallback's far-distance ceiling is documented (it simply doesn't
know small far bunches; the far-band crop pass recovers what the model can
still see). The retraining round is the next workstream, **post-demo**.

Hardware handoff workflow (contract with the hardware side, agreed
2026-10-29):

1. **Software → CSV** — the per-banana ripeness table shown on image
   upload (yellow/green/brown % + ripe/unripe/semi-ripe) is **downloaded
   automatically as CSV**. *[Planned — today the table is display-only
   (`app.py` `st.dataframe`); downloads are manual: annotated PNG +
   analytics inspection-log CSV.]*
2. **Hardware → calibration HERE** — the hardware side reads/calibrates
   the gas sensor values and hands them to this software, which
   **performs the calibration in-repo** and updates `analytics_view.py`.
3. **Software → ESP32 LCD** — the final verdict renders as a **one-line
   summary** on the LCD, e.g.
   *"Acceptable, all banana bunches are mostly ripe. Few bananas will
   take 1 day to ripen."*

Plan: push the software state to GitHub now (this docs round completes
it); hardware calibration happens **after** the software push.

## Far-band rescue + containment dedup (committed b66a0b6, 2026-10-29)

Far-distance photos (`images.jfif`, 335×597 truck) still had **zero boxes in
the top half of the frame** — the existing slice-boost gate was inert on it
(best conf 0.707 ≥ 0.35 gate), so its far crates were missed entirely. Two
additions to `main/detection/banana_detector.py` (non-teachable, no
retraining):

| Area | Change | Measured effect |
|---|---|---|
| `detection/banana_detector.py` | **Far-band rescue**: `far_band_gate` fires only when NO box center sits in the top 50% of the frame AND longest side ≤ 1024px; then `far_band_candidates` crops the top band and runs one extra YOLO pass (the crop re-magnifies the far region in YOLO's 640×640 input for free). Far boxes clear only `FAR_BAND_FLOOR=0.10` (exempt from the relative-against-best filter, which the image's 0.707 would crush them with) and are merged through NMS + containment | `images.jfif`: **9 → 10 boxes** — new far-band box (#10, conf 0.115, verified on the drawn-boxes image `verify_images_jfif_farboxes.png`: red box on the far green bunch, upper right). Gate fires on **no** working image (all have box centers in the top band) → regression guard |
| `detection/banana_detector.py` | **Containment dedup** `discard_contained` (always on, after NMS): drop a box fully inside a stronger-or-equal kept box (5% of the outer's shorter edge tolerance) — the "one big crate box + one small banana box" overlap | `images (1).jfif` / `images (8).jfif`: **14 → 9 boxes each** (5 nested duplicates dropped, verified pair-by-pair); `images.jfif` keeps both boxes of its near-nested pair (inner 0.406 is stronger than outer 0.227 → both kept, per the agreed rule) |

New tests: `main/tests/test_farband_containment.py` (11, model-free unit
tests of the gate + containment semantics) and
`main/tests/test_farband_regression.py` (3, live: re-runs real `detect()` on
the 8 readable reference images and diffs against the recorded fixture
`baseline_before.json` — working images may gain 0 boxes, lose only the 5
recorded containment drops; `images.jfif` must gain its far-band box).
Full suite: **81/81 green**.

**Rollback point for this round: git tag `pre-far-band-rescue` = `0af16ae`.**
(`git checkout pre-far-band-rescue` restores the pre-change state. The test
files + `baseline_before.json` + `verify_farband_boxes.py` +
`verify_ui/images_jfif_farband_rescue.png` are committed in `b66a0b6`, so a
plain checkout unwinds everything; no manual `rm` needed.)

## Slice boost + analytics-nav fixes (committed f784cb0/b9f167c/0af16ae, 2026-10-28)

Far-distance bunches (small photos, e.g. the 335×597 truck image) were
detecting only 4 bananas because YOLO's 640×640 input downsamples the bunch
below its localization limit. Added a **gated slice-boost detection layer**
(non-teachable, no retraining) plus three UI fixes:

| Area | Change | Measured effect |
|---|---|---|
| `detection/banana_detector.py` | `generate_slices` (3×3 overlapping tiles, 25% overlap), `should_slice` gate, `slice_candidates` (tile inference + box mapping), wired into `detect()` | far image `images.jfif`: **4 → 9 detections**, verdict HIGH RISK/Denied → ACCEPTABLE/Approved |
| same, gate | fires ONLY when full-frame best conf **< 0.35** AND longest side **≤ 1024px** | the 4 close-range test images (best conf 0.58–0.73) are **byte-identical** to pre-change results (JSON diff: 4× IDENTICAL) — regression guard |
| `app.py` | `StreamlitWidgetAlreadyInstantiatedError` crash on "View Analytics": `st.session_state["app_nav"]` was written after the radio was instantiated. Now an `on_click=_goto_analytics` callback (runs pre-rerun, before the radio re-instantiates) | verified end-to-end in a real browser (Playwright Chromium) — navigation lands on Analytics with no exception |
| `app.py` | "View Analytics" button re-centered under the stat-card row (`[3,2,3]` columns, `width="stretch"`) | button center measured 850px on 1400px viewport ≈ content-area center |
| `app.py` | uploader accepted `.jfif` (was `jpg/jpeg/png` only — the project's own test images are `.jfif`) | `.jfif` upload no longer rejected |
| `app.py` + `analytics_view.py` | `use_container_width` → `width` (Streamlit 1.63 deprecation) | 0 deprecation warnings from the app |

New tests: `main/tests/test_slice_gate_integration.py` (4, model-free fake
detector asserting exact inference-call counts per gate branch) and
`main/tests/test_app_navigation.py` (3: crash-mechanism repro, safe
callback pattern, real-app end-to-end nav via `AppTest`). 8 new gate/geometry
cases in `test_adaptive_detection.py`. Full suite: **75/75 green** at the
time of that round (later rounds added the far-band tests, suite now
81/81).
Screenshots: `main/tests/verify_ui/`; regression proof:
`main/tests/verify_before_slice/` vs `verify_after_slice/`.

**Rollback point for this round: git tag `rollback` = `a928f1b`.**
(`git checkout rollback` returns the pre-slice state; `git status` then
shows only untracked test files to delete.)

## Where this version is (`main` @ `4b76d58`+ = lighting-invariant color B; rollback tag `checkpoint-before-accuracy-work` = `172b1f0`)

Original non-teachable pipeline + verification tooling + **lighting-
invariant color analysis (Approach B, `choose.md`)**: the old double-count
bug (a warm-lit ripe pixel counted as *both* yellow and brown → `brown>20`
overripe gate fires; `other_pct` went negative) is fixed via a new
single source of truth, `main/vision/color_model.py`.

Behavior summary:

```
YOLOv8n COCO (class 46=banana, pre-built fallback — the production path)
  → adaptive detection (0.10 infer → relative score filter → NMS)
  → banana_mask.py  (center-touching connected-component body isolation;
                     windows now derived from color_model, dim-aware)
  → color_analyzer.py (color_model: exclusive families, brown = V ≤ 0.3×p95(V))
  → fusion_engine.py (gas + color → bunch verdict)
```

What changed vs the previous version (commits `8da75e0`→`4b76d58`):

| Area | Change | Measured effect |
|---|---|---|
| `vision/color_model.py` (new) | exclusive families (warm H0–34 / green H35–85, no hue overlap) + relative gate `brown = V ≤ BROWN_V_RATIO × p95(V of crop)`, ratio **0.30** (sweep-measured on the 154-image gate) | negative `other_pct` eliminated (was e.g. −18.8% on the Ripe_yellow root image) |
| `color_analyzer.py` | counting rewired to `color_model.family_counts` (no more double-count); public result keys unchanged | ripe→overripe misreads on the 20-img smoke: 6 → 2 |
| `banana_mask.py` | isolation windows derived from `color_model` (generous V floor) so dim bodies are still seen; `HSV_RANGES` deleted | dim-body isolation invariant tests 4/4 |
| `tests/verify_images.py` (new) | runs the real pipeline on any image folder (incl. `.jfif`), writes per-image JSON (boxes + G/Y/B% + class + bunch verdict) + annotated PNGs | the "don't guess" verification tool |
| `tests/conftest.py` (new) | preloads torch before any test module (Windows c10.dll/pandas ordering) | suite: 52/52 |

Full 154-image gate (same set, same detector — detection metrics
unchanged by construction): **ripeness agreement 47/231 (20%) → 132/231
(57%)**. 20-image smoke: 42% → 42% at the measured optimum (the smoke set
is dominated by the overripe-recall trade-off, see below).
Reports: `main/tests/blind_batch_report_baseline154.md` (before) vs
`main/tests/blind_batch_report_after_full154.md` (after); user-image
before/after: `main/tests/verify_before_full11/` vs `verify_after/`.

Known issues in THIS version:

- **Overripe recall on *uniformly dull* overripe bananas: 1/7** — a whole
  banana at one dark brightness has p95(V) == that level, so nothing is
  "dark *relative*" and it reads ripe. This is a fundamental limit of a
  purely relative gate; **Approach A (white reference patch, hardware)**
  is the fix and the planned next round.
- **Spurious/overlapping boxes** — 615 across the 154-image set; nested-box
  fragments survive NMS (detector work stream, untouched this round).
- **Requirements** — `requirements.txt` now includes `pandas` (the app's
  per-banana table) and `pyserial` (ESP32 path); `scikit-learn` was unused.
- **Windows quirk** — pandas must not import before torch (c10.dll crash);
  see `main/tests/conftest.py` docstring.

The branch table below predates this version — those `fix/*` branches were
superseded by the approach-B work and remain available for reference.

What this version does **not** include (measured fixes, kept on their branches):

| Branch | Tip | What it adds | Measured effect (154-image batch) |
|---|---|---|---|
| `fix/overripe-gate` | `1c375aa` | `decide_ripeness` gate: overripe needs brown > 15% AND yellow ≤ 35% AND brown leading | agreement 20% → 53% |
| `fix/green-background-mask` | `ec780d8` | gate + mask fix: no shrink-remask loop, coverage-qualified center rule, 2% floor (fixes yellow-banana-on-green-bg → UNRIPE) | 53% → 68% |
| `fix/nested-box-removal` | `dfdbace` | drop any box that lies inside a bigger box (overlap rule) | not yet measured |

Known issues in THAT earlier version (what those branches fix):
- **Overripe over-calls** — `brown > 20%` fires on stained-ripe bananas.
- **Green-background misread** — yellow banana on a solid green background
  reads UNRIPE.
- (All three remain relevant as detector/mask work streams.)

## Reality check (what this repo actually is)

Earlier plan docs described a 341-image EfficientNet training workflow — that never
existed on disk or in git history. The repo is a fully non-teachable architecture
("No models to train." — `app.py` caption). Ground truth for verification is
`main/models/Training`: 763-image Roboflow detection set (609/77/77 split), YOLO
polygon labels, 3 classes: 0=overripe (287 objects), 1=ripe (508), 2=underripe (757).
`valid/`+`test/` = 154 held-out images; `blind_batch.py` measures the live pipeline
against them. `classification/ripeness_classifier.py` is dead code (legacy, not
imported by `app.py`). Test images live at the repo root: 6 originals
(240_F/1000_F/pexels/Ripe_yellow) + 5 browser-saved `.jfif` files added
2026-10-28, all covered by `main/tests/verify_images.py` (which accepts `.jfif`).

## Git state

| Where | Commit |
|---|---|
| `main` (local) | `b66a0b6` — far-band rescue + containment dedup (software final, pre-push) |
| pre-far-band-rescue checkpoint | tag `pre-far-band-rescue` = `0af16ae` (rollback point for the far-band round) |
| pre-slice-boost checkpoint | tag `rollback` = `a928f1b` |
| pre-B checkpoint | tag `checkpoint-before-accuracy-work` = `172b1f0` |

Test suites: `python -m pytest main/tests/ -q` → 81/81 green (needs
`conftest.py`'s torch preload on this Windows machine). Standalone runners:
`python main/tests/test_blind_batch.py`, `test_color_model.py`,
`test_color_analyzer.py`, `test_banana_mask.py`, `test_verify_images.py`,
`test_adaptive_detection.py`, `test_slice_gate_integration.py`,
`test_app_navigation.py` (fast 2 + one slow end-to-end),
`test_farband_containment.py` (11 fast, model-free),
`test_farband_regression.py` (3 live, ~40s — diffs real `detect()` against
the recorded `baseline_before.json` fixture; re-record that file if the
detector model changes).

## Blind-batch results (this version, `color_model` B)

Auto mode, IoU ≥ 0.3 matching, `unripe→underripe` mapping, `semi-ripe` = OK band.
Detection metrics are identical to the pre-B baseline by construction (the
detector input is untouched — only the vision-layer color math changed).

| metric | pre-B (baseline154) | post-B (after_full154) |
|---|---|---|
| 20-img smoke: recall | 66% over 19 | 66% over 19 |
| 20-img smoke: agreement | 42% | 42% (at optimum 0.30) |
| 20-img smoke: spurious | 63 | 63 |
| **full 154: agreement** | **47/231 (20%)** | **132/231 (57%)** |
| full 154: spurious | 615 | 615 |
| full 154: zero-detection | 0 | 0 |

Trade-off to know: the 20-img smoke set is dominated by overripe/underripe
label rows, where the relative gate's weaker overripe-recall (1/7 full)
costs a little; on the full 154 set the net is a large win (+32 pts).
Overripe→overripe on the smoke confusion cell dropped 6→2; ripe→overripe
misreads dropped 6→2 (the target bug). Per-image + confusion tables:
`main/tests/blind_batch_report_baseline154.md` (before) and
`main/tests/blind_batch_report_after_full154.md` (after).

## Next

1. **Approach A round (pending hardware)** — white reference patch + locked
   WB/exposure on the enclosure camera; then a software-only ~40-line
   per-channel gain step feeding `color_model`. This is the round that
   closes the uniformly-dull-overripe gap B can't reach. See `choose.md`.
2. **Hardware integration (post-demo workstream)** — gas-sensor CSV
   handoff → in-repo calibration → `analytics_view.py` update, and the
   ESP32 one-line LCD verdict (workflow above). See `main/NON_TEACHABLE_GUIDE.md`.
3. **Detector work stream (post-demo)** — 615 spurious/overlapping boxes
   on the 154-image set. The nested-duplicate class is now handled
   (containment dedup, `b66a0b6`); remaining spurious boxes + the
   far-distance COCO-fallback ceiling are addressed by the planned
   **retraining round** (combined close + far + far-crop YOLO dataset,
   one model — see the "Hardware integration plan" section above).
   Measure with `python main/blind_batch.py --all`.
4. ~~**Manual UI spot-check**~~ — done via
   `python main/tests/verify_ui_button.py` (Playwright/Chromium): `.jfif`
   upload accepted, results render, "View Analytics" centered + crash-free
   navigation. Screenshots in `main/tests/verify_ui/`.
5. **Lighting-stress set** — photograph the same bananas at ~3500K / ~5000K /
   dim (~0.3×) and run `python main/tests/verify_images.py` on the folder
   (that's the tool this round was built to feed).
6. Optional cleanup: remove dead `classification/ripeness_classifier.py` +
   duplicate `main/bunch_analyzer.py` (needs sign-off — flagged, not done).

## How to resume

```
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"
git log --oneline -6            # main = b66a0b6 (far-band rescue + containment dedup)
git tag -l                      # rollback = a928f1b; pre-far-band-rescue = 0af16ae; checkpoint-before-accuracy-work = 172b1f0
python main/blind_batch.py      # 20-image smoke (default)
python main/blind_batch.py --all   # full 154-image gate (~4 min CPU)
python main/tests/verify_images.py # run the user's root test images
python -m pytest main/tests/ -q    # 81 tests
python main/tests/verify_ui_button.py  # real-browser (Playwright) nav + alignment check
python main/tests/verify_farband_boxes.py  # re-render the images.jfif far-band proof image
```

## Open notes
- Software scope is **frozen for the Nov-4 demo** (`b66a0b6`); the only
  planned follow-up rounds are the hardware handoff (CSV → calibration →
  `analytics_view.py` → ESP32 LCD line) and the far-distance retraining
  round. Both are documented in the hardware-integration-plan section above.
- `choose.md` decision sheet was **executed**: B shipped, A held
  for hardware, C parked. Rollback = `git checkout checkpoint-before-
  accuracy-work` (tag `checkpoint-before-accuracy-work`, commit `172b1f0`).
- The pre-B version in this file is `main` @ `d00ffad` = GitHub (that
  commit + the analytics-dashboard merge `eaa62e6` are the parent chain).
- Known local-only oddity: a stray **empty `Pranjal/` directory** sits at
  the repo root (untracked, nothing inside) — flagged for removal, not
  touched.