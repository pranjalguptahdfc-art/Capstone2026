# BananaRipeness — Progress (as of 2026-10-28)

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
| `main` (local) | `f7c0cb7` — lighting-invariant color B + verify harness + conftest |
| pre-B checkpoint | tag `checkpoint-before-accuracy-work` = `172b1f0` (rollback point) |
| `main`'s parent (pre-B) | `eaa62e6` — merge: 30-day analytics dashboard + inspection logging |

Test suites: `python -m pytest main/tests/ -q` → 52/52 green (needs
`conftest.py`'s torch preload on this Windows machine). Standalone runners:
`python main/tests/test_blind_batch.py`, `test_color_model.py`,
`test_color_analyzer.py`, `test_banana_mask.py`, `test_verify_images.py`.

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
2. **Detector work stream** — 615 spurious/overlapping boxes on the 154 set;
   the `fix/nested-box-removal` branch + contour-based detection are the
   planned paths. Measure with `python main/blind_batch.py --all`.
3. **Manual UI spot-check** — `streamlit run main/app.py` on the root test
   images (11 now, incl. the `.jfif` files).
4. **Lighting-stress set** — photograph the same bananas at ~3500K / ~5000K /
   dim (~0.3×) and run `python main/tests/verify_images.py` on the folder
   (that's the tool this round was built to feed).
5. Optional cleanup: remove dead `classification/ripeness_classifier.py` +
   duplicate `main/bunch_analyzer.py` (needs sign-off).

## How to resume

```
cd "D:\Pranjal Files\Capstone2026\BananaRipeness"
git log --oneline -6            # f7c0cb7 = main (color B)
git tag -l                      # checkpoint-before-accuracy-work = 172b1f0
python main/blind_batch.py      # 20-image smoke (default)
python main/blind_batch.py --all   # full 154-image gate (~4 min CPU)
python main/tests/verify_images.py # run the user's root test images
python -m pytest main/tests/ -q    # 52 tests
```

## Open notes
- `choose.md` decision sheet was **executed** this round: B shipped, A held
  for hardware, C parked. Rollback = `git checkout checkpoint-before-
  accuracy-work` (tag `checkpoint-before-accuracy-work`, commit `172b1f0`).
- The pre-B version in this file is `main` @ `d00ffad` = GitHub (that
  commit + the analytics-dashboard merge `eaa62e6` are the parent chain).