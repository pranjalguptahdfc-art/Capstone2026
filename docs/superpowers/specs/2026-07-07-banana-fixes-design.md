# Banana Bunch Fixes — Design Spec (2026-07-07)

## Context

The repo evolved past the original guide: it is now YOLOv8 (ultralytics) +
Streamlit + a gas-sensor fusion pipeline. The active app path
(`main/app.py`) is still **non-teachable**: no training at runtime.
Ripeness = HSV color analysis of each detection box
(`main/vision/color_analyzer.py`); detection = `BananaDetector.detect()`
(`main/detection/banana_detector.py`) with a fixed confidence threshold.

Two confirmed bugs:

1. **Background bleeds into ripeness.**
   `analyze_region()` computes green/yellow/brown percentages over **all**
   pixels in the bounding box. A green stand, yellow shelf, or brown crate
   inside the box changes the result (proven in user test: yellow banana,
   green background → "UNRIPE").
2. **One confidence threshold can't fit all images.**
   Fixed conf 0.20 misses occluded/back bananas; dropping to 0.10 catches
   them but introduces duplicate overlapping boxes on the same banana.

## Hard constraint

**Non-teachable stays non-teachable.** No model retraining, no new
dependencies — pure OpenCV post-processing only.

## Design decisions

### Fix 1 — banana mask before color analysis (new `main/vision/banana_mask.py`)

- Inside each box crop: blur → HSV → OR of the three banana-color windows
  (the same green/yellow/brown HSV ranges the analyzer already uses) →
  morphology close/open → keep the **connected component that touches the
  crop center** (a banana is the solid blob in the middle of its own box;
  background lives on the edges) → fill interior holes.
- `analyze_region()` computes percentages **only over mask pixels** when the
  mask covers ≥ 10% of the crop; otherwise it falls back to today's
  whole-crop behavior (legacy safety net). Background pixels simply don't
  count toward any banana color, so they land in `other_pct`.
- Last-resort fallback: if no color window matches (e.g. white banana),
  use a low-saturation/high-value window.
- Known limitation (accepted, documented): a *same-color* green background
  cannot be separated by color. Center-touching component still excludes
  most of it because it touches the crop edges, not the center.

### Fix 2 — adaptive detection (in `BananaDetector`)

A single pipeline that works at both ends of the image-quality spectrum:

1. **Infer permissively** — always run the first YOLO pass at `0.10`
   (regardless of the requested threshold) so occluded bananas appear.
2. **Relative score filter** — keep boxes with
   `score ≥ max(best_score × 0.30, 0.08)`. The image's own best detection
   sets the quality bar, so junk is dropped on cluttered images while a
   genuinely weak-but-real small bunch (best ≈ 0.10) survives on easy
   images.
3. **Score-ordered NMS** (IoU ≥ 0.50) — kills duplicate boxes on the same
   banana; distinct adjacent bananas have IoU < 0.5 so a tight bunch
   keeps one box per banana.
4. The three-pass fallback chain (contrast-enhanced → all-classes) keeps
   running only when the pipeline finds nothing, and gets the same
   relative-filter + NMS treatment.
5. **UI**: the confidence slider becomes a *manual mode* (a floor applied
   after the pipeline). Default is **Auto** mode — no floor, the adaptive
   pipeline decides. New `banana_id`s are numbered by descending
   confidence so labels stay stable.

Constants live on `BananaDetector` (`adaptive_conf=0.10`,
`relative_keep_ratio=0.30`, `absolute_floor=0.08`, `nms_iou=0.50`) so
calibration is a one-line change, no retraining.

## Files touched

| File | Change |
|---|---|
| `main/vision/banana_mask.py` | **NEW** — mask + banana_crop + validity helpers |
| `main/vision/color_analyzer.py` | mask-aware `analyze_region`, new `background_suppressed` key |
| `main/detection/banana_detector.py` | relative filter + score NMS + adaptive pass-1 conf in `detect()` |
| `main/app.py` | Auto/Manual detection mode in sidebar; updated "how it works" captions |
| `main/tests/test_background_resilience.py` | **NEW** — mask + analyzer tests |
| `main/tests/test_adaptive_detection.py` | **NEW** — filter + NMS tests |
| `main/NON_TEACHABLE_GUIDE.md` | "How it works" + design-decision rows updated |
| `learn/project-notes.md` | new dated entry recording the fixes + constants |

Tests follow the existing convention: plain standalone Python scripts with
asserts (no pytest), run as `python main/tests/<name>.py`.

## Rollback point

Tag `baseline-pre-fixes` + branch `checkpoint/pre-fixes` at the pre-fix
commit. Not pushed (no remote for this repo state).
