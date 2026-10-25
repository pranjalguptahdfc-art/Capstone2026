# Banana Bunch Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix background-induced ripeness errors and make detection work across image quality without per-image manual tuning — pure OpenCV post-processing, no retraining.

**Architecture:** (1) A banana-mask module isolates the banana body inside each detection box (center-touching connected component of banana-colored pixels) and the color analyzer computes percentages over that mask only. (2) The detector infers at a low confidence, then applies a relative score filter + score-ordered NMS so occluded bananas are kept while duplicates are removed; the UI gets an "auto" mode.

**Tech Stack:** Python, OpenCV, ultralytics YOLOv8, Streamlit, NumPy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-07-banana-fixes-design.md`

## Global Constraints

- NON-TEACHABLE: no model retraining, no new pip dependencies, no pretrained-model swap.
- Detection boxes are always `[x1, y1, x2, y2]` ints in the original image's pixel space.
- Detection dicts keep exactly these keys: `banana_id` (int), `bbox` (list of 4 ints), `confidence` (float rounded to 4). Nothing else may be added to them.
- Analyzer result dicts keep all existing keys (`green_pct`, `yellow_pct`, `brown_pct`, `other_pct`, `dominant_color`, `ripeness_class`, `pixel_counts`); new fields are additive only.
- Tests are plain standalone Python scripts (asserts, exit non-zero on failure) following the `main/tests/` convention; run as `python main/tests/<file>.py`.
- Every task ends with a git commit using the repo's attribution rules.
- Rollback point exists: tag `baseline-pre-fixes`, branch `checkpoint/pre-fixes`.

---

### Task 1: Banana mask module (`main/vision/banana_mask.py`)

**Files:**
- Create: `main/vision/banana_mask.py`
- Test: `main/tests/test_background_resilience.py`

**Interfaces:**
- Consumes: `vision.color_analyzer.HSV_RANGES` (the existing green/yellow/brown HSV dicts).
- Produces:
  - `banana_mask(crop_bgr: np.ndarray) -> np.ndarray` — single-channel uint8 mask, banana=255, same size as crop.
  - `banana_crop(crop_bgr: np.ndarray) -> np.ndarray` — crop shrunken to the mask's bounding rect (≥ 95% of crop if mask not solid; the whole crop otherwise).
  - `is_solid_mask(mask: np.ndarray) -> bool` — True when mask covers ≥ 95% of the crop.
  - `is_valid_mask(mask: np.ndarray, min_ratio: float = 0.10) -> bool` — True when the largest component touches the crop center and covers ≥ min_ratio of the crop.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_background_resilience.py`:

```python
"""
test_background_resilience.py
-----------------------------
Unit tests for background-resilient ripeness analysis.

Run:
    python main/tests/test_background_resilience.py
"""

import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/ (matches test_pipeline.py)
sys.path.insert(0, str(PROJECT_ROOT))

from vision import banana_mask
from vision.color_analyzer import analyze_region

BAYER   = (46, 204, 255)   # BGR ripe yellow banana skin  (H~22 -> yellow window)
BGREEN  = (40, 160, 40)    # BGR green (unripe) banana    (H~60 -> green window)
BG_BLUE = (200, 140, 40)   # BGR neutral "background" fill (H~105, NOT a banana color)

def make_banana_on_background(bg_bgr=BG_BLUE, size=160):
    """Yellow banana block on a solid neutral (non-banana-color) background.

    Drawn as a filled rectangle so its bounding box is fully banana -> after
    banana_crop shrinks to the mask bbox it is deterministically "solid".
    """
    crop = np.full((size, size, 3), bg_bgr, dtype=np.uint8)
    cv2.rectangle(crop, (30, 50), (130, 110), BAYER, -1)
    return crop

def test_mask_is_center_touching_blob():
    crop = make_banana_on_background()
    mask = banana_mask.banana_mask(crop)
    assert mask.shape == crop.shape[:2]
    assert banana_mask.is_valid_mask(mask)

def test_background_suppressed():
    crop = make_banana_on_background()
    x1, y1 = 0, 0
    x2, y2 = crop.shape[1], crop.shape[0]
    result = analyze_region(crop, [x1, y1, x2, y2])
    assert result["background_suppressed"] is True
    assert result["other_pct"] > 60.0, f"background not suppressed: other={result['other_pct']}%"

def test_no_mask_falls_back():
    crop = np.full((160, 160, 3), BAYER, dtype=np.uint8)  # banana-colored all over
    result = analyze_region(crop, [0, 0, 160, 160])
    assert result["background_suppressed"] is False
    assert result["yellow_pct"] > 80.0

def test_banana_crop_shrinks_to_mask():
    crop = make_banana_on_background()
    shrunken = banana_mask.banana_crop(crop)
    assert shrunken.size < crop.size
    assert banana_mask.is_solid_mask(banana_mask.banana_mask(shrunken))

def test_green_banana_still_unripe():
    # A green banana on a green background must still classify unripe
    crop = np.full((160, 160, 3), BGREEN, dtype=np.uint8)
    result = analyze_region(crop, [0, 0, 160, 160])
    assert result["ripeness_class"] in ("unripe", "semi-ripe")

if __name__ == "__main__":
    failed = 0
    for fn in [
        test_mask_is_center_touching_blob,
        test_background_suppressed,
        test_no_mask_falls_back,
        test_banana_crop_shrinks_to_mask,
        test_green_banana_still_unripe,
    ]:
        try:
            fn()
            print(f"  [PASS]  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL]  {fn.__name__}: {exc}")
    print(f"\n{5 - failed}/5 tests passed.")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python main/tests/test_background_resilience.py`
Expected: `[FAIL] test_mask_is_center_touching_blob` — "No module named 'vision.banana_mask'" style ImportError.

- [ ] **Step 3: Write minimal implementation**

Create `main/vision/banana_mask.py`:

```python
"""
banana_mask.py
--------------
Background-resilient banana extraction for the ripeness pipeline.

The detector gives a bounding box; the box also contains background
(the market stand, crate, other bananas). This module isolates the
banana BODY — the solid blob that touches the box center — so color
analysis can ignore the background.

Pure OpenCV post-processing. No model, no training.
"""

import cv2
import numpy as np

from .color_analyzer import HSV_RANGES

# Saturation window covering banana skin greens/yellows/browns.
_BANANA_S = np.array([0, 25, 0], dtype=np.uint8)

# Last-resort window for banana colors the three HSV ranges miss
# (very pale / white / over-bleached bananas).
_LAST_RESORT_LOWER = np.array([0, 0, 180], dtype=np.uint8)
_LAST_RESORT_UPPER = np.array([179, 20, 255], dtype=np.uint8)


def banana_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (banana=255) for the banana body inside a crop.

    The banana is the solid blob that touches the crop center; background
    typically touches the crop edges. If no banana-colored pixels exist
    (all-white banana, extreme overexposure) fall back to a
    low-saturation/high-value window.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for _name, rng in HSV_RANGES.items():
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, rng["lower"], rng["upper"]))
    if np.count_nonzero(mask) == 0:
        mask = cv2.inRange(hsv, _LAST_RESORT_LOWER, _LAST_RESORT_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep the largest connected component only if it touches the crop center.
    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        return mask
    h, w = labels.shape
    cy, cx = h // 2, w // 2
    cy0 = labels[0, 0]
    best = _best_component(labels, cy, cx, cy0)
    if best is None:
        return mask
    return (labels == best).astype(np.uint8) * 255


def _best_component(labels, cy, cx, cy0):
    """Label of the largest component touching the crop center, else None."""
    center_cands = {labels[cy, cx]}
    if 0 in center_cands:
        center_cands.remove(0)
    if not center_cands:
        # Center pixel is background — fall back to the largest non-background
        # component as a whole (no center constraint).
        sizes = np.bincount(labels.flatten())
        best = int(np.argmax(sizes[1:]) + 1) if sizes.size > 1 else None
        return best if (best != cy0 and sizes[best] > 0) else None
    sizes = np.bincount(labels.flatten())
    return max(center_cands, key=lambda c: sizes[c])


def is_valid_mask(mask: np.ndarray, min_ratio: float = 0.10) -> bool:
    """True when a component touches the crop center and covers >= min_ratio."""
    if mask.size == 0:
        return False
    labels, n = cv2.connectedComponents(mask)
    if n <= 1:
        return bool(np.count_nonzero(mask) / mask.size >= min_ratio)
    h, w = labels.shape
    center = labels[h // 2, w // 2]
    if center == 0:
        return False
    return np.count_nonzero(labels == center) / mask.size >= min_ratio


def is_solid_mask(mask: np.ndarray) -> bool:
    """True when the mask covers most of the crop (no background to remove)."""
    return mask.size > 0 and np.count_nonzero(mask) / mask.size >= 0.95


def banana_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Shrink a crop to its mask bounding rect when the mask is valid.

    Falls back to the original crop when the mask is solid or invalid,
    so the analyzer never receives an empty region.
    """
    mask = banana_mask(crop_bgr)
    if not is_valid_mask(mask) or is_solid_mask(mask):
        return crop_bgr
    coords = np.where(mask > 0)
    y1, y2 = coords[0].min(), coords[0].max() + 1
    x1, x2 = coords[1].min(), coords[1].max() + 1
    out = crop_bgr[y1:y2, x1:x2]
    if out.size == 0:
        return crop_bgr
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python main/tests/test_background_resilience.py`
Expected: `[PASS]` for all 5 tests, exit 0.

- [ ] **Step 5: Commit**

```
git add main/vision/banana_mask.py main/tests/test_background_resilience.py
git commit -m "feat: add background-resilient banana mask for ripeness analysis"
```

---

### Task 2: Mask-aware color analysis in `color_analyzer.py`

**Files:**
- Modify: `main/vision/color_analyzer.py` (`analyze_region` + `_empty_result`)
- Test: `main/tests/test_background_resilience.py` (tests from Task 1 become green)

**Interfaces:**
- Consumes: `vision.banana_mask` (Task 1).
- Produces: `analyze_region(image_bgr, bbox)` gains one key — `background_suppressed: bool`. All other keys unchanged; `pixel_counts` gains one entry `banana_pixels`. `analyze_batch` propagates the new key automatically (it copies the whole dict).

- [ ] **Step 1: Confirm Task 1 tests are failing on the analyzer (red)**

Run: `python main/tests/test_background_resilience.py`
Expected: `test_background_suppressed` and `test_banana_crop_shrinks_to_mask` (and any other analyzer-dependent test) FAIL because `analyze_region` does not know about masks.

- [ ] **Step 2: Modify `analyze_region` to mask**

In `main/vision/color_analyzer.py`, replace the body of `analyze_region` from `x1, y1, x2, y2 = bbox` down to (but not including) the final `return {` with:

```python
    x1, y1, x2, y2 = bbox
    region = image_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return _empty_result()

    from .banana_mask import banana_crop  # local import keeps module import order stable
    analyzed = banana_crop(region)
    background_suppressed = analyzed.size < region.size and analyzed.size > 0

    hsv = cv2.cvtColor(analyzed, cv2.COLOR_BGR2HSV)
    if background_suppressed:
        mask = banana_mask(analyzed)  # type: ignore[assignment]
        total = int(np.count_nonzero(mask))
    else:
        mask = None
        total = analyzed.shape[0] * analyzed.shape[1]

    counts = {}
    for color_name, hsv_dict in HSV_RANGES.items():
        in_range = cv2.inRange(hsv, hsv_dict["lower"], hsv_dict["upper"])
        if mask is not None:
            in_range = cv2.bitwise_and(in_range, mask)
        counts[color_name] = int(np.count_nonzero(in_range))
```

Update the final `return {` dict to add the two new fields:

```python
    return {
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "other_pct": other_pct,
        "dominant_color": dominant_color,
        "ripeness_class": ripeness_class,
        "pixel_counts": {
            "green": counts["green"],
            "yellow": counts["yellow"],
            "brown": counts["brown"],
            "banana_pixels": total,
            "total_analyzed": total,
        },
        "background_suppressed": background_suppressed,
    }
```

And in `_empty_result()`, add `"background_suppressed": False,` to the returned dict.

Also update the module docstring to note that analysis masks to the banana body when a valid mask is found.

- [ ] **Step 3: Run test to verify it passes**

Run: `python main/tests/test_background_resilience.py`
Expected: all 5 tests PASS.

Also run the existing pipeline smoke test to confirm no regression:
Run: `python main/tests/test_pipeline.py`
Expected: no new failures vs. baseline (it already runs against the current pipeline).

- [ ] **Step 4: Commit**

```
git add main/vision/color_analyzer.py
git commit -m "feat: analyze banana pixels only in ripeness color analysis"
```

---

### Task 3: Adaptive detection (relative filter + score NMS) in `banana_detector.py`

**Files:**
- Modify: `main/detection/banana_detector.py` (new module-level constants + helper methods + `detect()` rewritten)
- Test: `main/tests/test_adaptive_detection.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (public, testable without the model):
  - `relative_score_filter(detections, relative_keep_ratio=None, absolute_floor=None) -> list[dict]` (module-level function)
  - `suppress_duplicates(detections, iou_threshold=None) -> list[dict]` (module-level function)
  - `detect(image, confidence_threshold=None, floor: bool = True)` — signature change is backward-compatible; `floor=True` means the passed threshold acts as a floor, not a hard cutoff.

Detection dicts keep exactly the keys: `banana_id`, `bbox`, `confidence`.

- [ ] **Step 1: Write the failing test**

Create `main/tests/test_adaptive_detection.py`:

```python
"""
test_adaptive_detection.py
-------------------------
Unit tests for adaptive detection: relative score filter + NMS dedup.

Run:
    python main/tests/test_adaptive_detection.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/ (matches test_pipeline.py)
sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import relative_score_filter, suppress_duplicates

def det(box, conf):
    return {"banana_id": 0, "bbox": box, "confidence": conf}

def test_relative_filter_keeps_real_drops_junk():
    # Best score 0.45 -> threshold = max(0.45*0.30, 0.08) = 0.135
    dets = [det([0, 0, 100, 100], 0.45), det([200, 200, 300, 300], 0.20), det([500, 500, 550, 550], 0.05)]
    kept = relative_score_filter(dets)
    assert len(kept) == 2, "junk box (0.05) is below threshold 0.135; 0.45 and 0.20 survive"

def test_weak_bunch_survives():
    # Best score 0.10 → threshold = max(0.03, 0.08) = 0.08 → 0.09 box survives
    dets = [det([0, 0, 100, 100], 0.10), det([200, 200, 250, 250], 0.09)]
    kept = relative_score_filter(dets)
    assert len(kept) == 2, "genuinely weak small bunch must survive"

def test_nms_suppresses_duplicate():
    dets = [det([0, 0, 100, 100], 0.40), det([5, 5, 105, 105], 0.35)]
    kept = suppress_duplicates(dets)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.40, "keep the higher-confidence box"

def test_nms_keeps_adjacent_bananas():
    # Side-by-side boxes with IoU < 0.5 must both survive
    dets = [det([0, 0, 100, 100], 0.40), det([101, 0, 201, 100], 0.35)]
    kept = suppress_duplicates(dets)
    assert len(kept) == 2

def test_nms_numbers_by_confidence():
    dets = [det([0, 0, 100, 100], 0.40), det([300, 300, 400, 400], 0.20)]
    kept = suppress_duplicates(dets)
    ids = [d["banana_id"] for d in kept]
    assert ids == [1, 2], "banana_id must be 1..N in confidence order"

if __name__ == "__main__":
    tests = [
        test_relative_filter_keeps_real_drops_junk,
        test_weak_bunch_survives,
        test_nms_suppresses_duplicate,
        test_nms_keeps_adjacent_bananas,
        test_nms_numbers_by_confidence,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS]  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL]  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python main/tests/test_adaptive_detection.py`
Expected: `ImportError: cannot import name 'relative_score_filter'`.

- [ ] **Step 3: Implement filter + NMS + rewrite `detect()`**

In `main/detection/banana_detector.py`:

(a) Add module-level constants after `SINGLE_MODEL_PATH`:

```python
# Adaptive detection pipeline (non-teachable post-processing, no retraining)
ADAPTIVE_INFERENCE_CONF = 0.10   # pass-1 inference always runs this low to catch occluded bananas
RELATIVE_KEEP_RATIO = 0.30      # keep boxes at >= best_score * this ratio
ABSOLUTE_FLOOR = 0.08          # ... but never below this
NMS_IOU_THRESHOLD = 0.50       # score-ordered NMS suppression threshold
```

(b) Add two module-level functions (pure — no model needed, unit-testable):

```python
def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def relative_score_filter(
    detections: list,
    relative_keep_ratio: float | None = None,
    absolute_floor: float | None = None,
) -> list:
    """Keep boxes whose score clears max(best * ratio, floor).

    The image's own best detection sets the quality bar, so junk boxes
    on cluttered images are dropped while a weak-but-real bunch on a
    clean image survives.
    """
    if not detections:
        return []
    ratio = RELATIVE_KEEP_RATIO if relative_keep_ratio is None else relative_keep_ratio
    floor = ABSOLUTE_FLOOR if absolute_floor is None else absolute_floor
    best = max(d["confidence"] for d in detections)
    threshold = max(best * ratio, floor)
    return [d for d in detections if d["confidence"] >= threshold]


def suppress_duplicates(detections: list, iou_threshold: float | None = None) -> list:
    """Score-ordered NMS: suppress boxes overlapping an already-kept box.

    Higher-confidence boxes win. Distinct adjacent bananas (IoU <
    threshold) are not suppressed. Re-numbers banana_id in confidence order.
    """
    iou_t = NMS_IOU_THRESHOLD if iou_threshold is None else iou_threshold
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list = []
    for d in ordered:
        if any(_iou(d["bbox"], k["bbox"]) >= iou_t for k in kept):
            continue
        kept.append(d)
    for i, d in enumerate(kept, start=1):
        d["banana_id"] = i
    return kept
```

(c) Rewrite `detect()`'s pass 1 to infer permissively, then filter; and apply the pipeline to pass 2/3 results too. Replace the body of `detect()` after the `if self._model is None:` guard with:

```python
        # Adaptive: infer at a low conf so occluded bananas appear,
        # then filter by the image's own best score + NMS.
        # The requested threshold acts as a floor, not a hard cutoff.
        adaptive_conf = ADAPTIVE_INFERENCE_CONF
        detections = self._run_inference(image, adaptive_conf)
        if detections:
            detections = relative_score_filter(detections, absolute_floor=conf)
            detections = suppress_duplicates(detections)

        # Pass 2: contrast-enhanced image at lower confidence
        if not detections:
            logger.info("Pass 1 found nothing. Trying contrast-enhanced image...")
            enhanced = self._preprocess(image)
            detections = self._run_inference(enhanced, min(adaptive_conf, 0.15))
            if detections:
                detections = relative_score_filter(detections, absolute_floor=conf)
                detections = suppress_duplicates(detections)

        # Pass 3: last resort — try ALL COCO classes on the original image
        if not detections:
            logger.info("Pass 2 found nothing. Trying all-class fallback...")
            detections = self._run_inference_all_classes(image, 0.05)
            if detections:
                detections = relative_score_filter(detections, absolute_floor=conf)
                detections = suppress_duplicates(detections)

        logger.debug(f"Detected {len(detections)} banana(s).")
        return detections
```

Note: `conf` variable from the original code stays — it is the requested threshold, now used as the relative filter's floor.

- [ ] **Step 4: Run test to verify it passes**

Run: `python main/tests/test_adaptive_detection.py`
Expected: all 5 tests PASS.

Run: `python main/tests/test_pipeline.py`
Expected: no new failures (the pipeline test exercises `BananaDetector` interface — confirm the docstring example still holds; update `detect()`'s docstring if it describes the old single-pass behavior).

- [ ] **Step 5: Commit**

```
git add main/detection/banana_detector.py main/tests/test_adaptive_detection.py
git commit -m "feat: adaptive detection with relative score filter and NMS dedup"
```

---

### Task 4: UI — Auto vs Manual detection mode (`main/app.py`)

**Files:**
- Modify: `main/app.py` (sidebar settings block + detector call + "how it works" captions)

**Interfaces:**
- Consumes: `BananaDetector.detect(image, confidence_threshold, floor: bool = True)` from Task 3.
- Produces: sidebar now has "Detection mode" (Auto / Manual) replacing the plain confidence slider. Auto mode = `confidence_threshold=0.10, floor=False`. Manual mode = user slider value as the floor.

- [ ] **Step 1: Modify the sidebar settings block**

In `main/app.py`, replace the `conf_threshold` slider block with:

```python
    with st.sidebar:
        st.header("Settings")
        det_mode = st.radio(
            "Detection mode",
            ["Auto", "Manual"],
            index=0,
            help="Auto adapts to each image (recommended). Manual uses your threshold as a floor.",
        )
        if det_mode == "Auto":
            conf_threshold, _floor = 0.10, False
            st.caption("Adaptive: detects all bananas, then removes low-confidence duplicates.")
        else:
            conf_threshold = st.slider(
                "Confidence floor", min_value=0.05, max_value=0.90,
                value=0.20, step=0.05,
            )
            _floor = True
```

- [ ] **Step 2: Update the detector call and captions**

Replace `detector.detect(bgr_image, confidence_threshold=conf_threshold)` with:

```python
        detections = detector.detect(
            bgr_image,
            confidence_threshold=conf_threshold,
            floor=_floor,
        )
```

Update the "how it works" sidebar captions to reflect the pipeline:

```python
        st.caption("1. YOLOv8 detects the bunch (adaptive: finds all, removes duplicates)")
        st.caption("2. Banana-mask isolates the fruit, ignores the background")
        st.caption("3. HSV color classifies each banana")
        st.caption("4. Fusion engine computes bunch verdict")
```

- [ ] **Step 3: Verify**

Run: `python main/tests/test_pipeline.py` (smoke test still passes)
Then start the app and confirm the sidebar renders:
Run: `streamlit run main/app.py` (from `D:\Pranjal Files\Capstone2026\BananaRipeness`, in the venv)
Expected: app starts, sidebar shows "Detection mode" with Auto/Manual radio, Auto selected by default.

- [ ] **Step 4: Commit**

```
git add main/app.py
git commit -m "feat: auto detection mode in UI (adaptive pipeline), manual as floor"
```

---

### Task 5: Documentation + notes

**Files:**
- Modify: `main/NON_TEACHABLE_GUIDE.md` ("How it works" + "Design Decisions & Trade-offs" sections)
- Modify: `learn/project-notes.md` (new dated entry)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update NON_TEACHABLE_GUIDE.md "How it works"**

In `main/NON_TEACHABLE_GUIDE.md`, replace the "### How it works" numbered list with an updated 5-step flow that includes the banana-mask step and the adaptive-detection step, and add one line under "Non-Teachable" in the summary: "Ripeness masks to the banana body so background color is ignored. Detection adapts per-image (finds all, then removes duplicates) instead of one fixed threshold."

- [ ] **Step 2: Add two rows to the Design Decisions table**

```
| Adaptive detection | Fixed confidence threshold | One threshold can't fit all images (occcluded bananas vs. duplicates). We infer low, then filter by the image's own best score + NMS. |
| Banana-mask color analysis | Whole-box average | Background (green stand, yellow shelf) used to skew ripeness. Now we mask to the banana body (center-touching component) before HSV. |
```

- [ ] **Step 3: Add a new dated entry to `learn/project-notes.md`**

```
## 2026-07-07 — Background & Adaptive-Detection Fixes

### Background fix
- New module main/vision/banana_mask.py — banana_mask(), banana_crop(), is_valid_mask(), is_solid_mask()
- color_analyzer.analyze_region masks to the banana body before HSV. Adds 'background_suppressed' key.
- Fallback: mask solid OR invalid → whole-crop analysis (legacy path)
- Fallback: last-resort low-sat / high-value HSV window for all-white bananas

### Adaptive detection
- banana_detector: ADAPTIVE_INFERENCE_CONF=0.10, RELATIVE_KEEP_RATIO=0.30, ABSOLUTE_FLOOR=0.08, NMS_IOU_THRESHOLD=0.50
- relative_score_filter(dets, ratio, floor) + suppress_duplicates(dets, iou) — module-level, unit-testable
- detect(): pass-1 always infers at 0.10, then filters by max(best*0.3, 0.08) + score NMS (IoU≥0.5). Passes 2/3 also filtered.
- banana_id re-numbered by confidence after NMS.

### UI
- main/app.py: Detection mode = Auto (default) / Manual. Auto = conf 0.10, no floor. Manual = slider as floor.

### Tests
- main/tests/test_background_resilience.py (5 tests, all pass)
- main/tests/test_adaptive_detection.py (5 tests, all pass)

### Rollback
- Tag: baseline-pre-fixes  Branch: checkpoint/pre-fixes
```

- [ ] **Step 4: Commit**

```
git add main/NON_TEACHABLE_GUIDE.md learn/project-notes.md
git commit -m "docs: record background-mask and adaptive-detection fixes"
```

---

### Task 6: Full pipeline smoke test + manual end-to-end check

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Run all tests**

Run:
```
python main/tests/test_background_resilience.py
python main/tests/test_adaptive_detection.py
python main/tests/test_bunch_logic.py
python main/tests/test_pipeline.py
```
Expected: all pass, no new failures.

- [ ] **Step 2: Manual end-to-end check**

Start the app, upload the 4 user-provided test images (or any equivalent images from the dataset):

| Image | Expected before fix | Expected after fix |
|---|---|---|
| Market photo, green background, ripe banana | UNRIPE (bleed) | RIPE / SEMI-RIPE |
| Market photo, occluded bunch, conf 0.20 | back bananas missed | all bananas detected, no duplicate boxes |
| Small bunch, conf 0.10 | overlapping boxes | one box per banana |
| Clean photo, 4-6 bunches | OK | still OK (no regression) |

Record results in `learn/project-notes.md` under the 2026-07-07 entry as "Manual E2E results:" with a one-line-per-image verdict.

- [ ] **Step 3: Commit**

```
git add learn/project-notes.md
git commit -m "test: record manual end-to-end results for both fixes"
```

---

## Self-Review Checklist (run before execution)

- [ ] Task 1 test file exercises `banana_mask` directly AND through `analyze_region`
- [ ] Task 2 keeps all existing `analyze_region` keys; only adds `background_suppressed` + `pixel_counts.banana_pixels`
- [ ] Task 3 functions are module-level (importable without loading the model)
- [ ] Task 3 renumbers `banana_id` by confidence (labels stay stable across images)
- [ ] Task 4 Auto mode = `confidence_threshold=0.10, floor=False`; Manual = slider as floor
- [ ] Task 5 docs + notes updated; no stale "single-pass" or "fixed threshold" claims
- [ ] Task 6 full test suite green; manual E2E recorded
