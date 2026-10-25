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

BAYER = (46, 204, 255)    # BGR ripe yellow banana skin  (HSV ~ H22 S209 V255)
BGREEN = (40, 160, 40)    # BGR green (unripe) banana    (HSV ~ H60 S191 V160)
BGBLUE = (200, 140, 40)   # BGR neutral background fill  (HSV ~ H101, no banana window)


def make_banana_on_background(bg_bgr=BGBLUE, size=160):
    """Yellow banana block on a solid neutral (non-banana-color) background."""
    crop = np.full((size, size, 3), bg_bgr, dtype=np.uint8)
    cv2.rectangle(crop, (30, 50), (130, 110), BAYER, -1)
    return crop


def test_mask_is_center_touching_blob():
    crop = make_banana_on_background()
    mask = banana_mask.banana_mask(crop)
    assert mask.shape == crop.shape[:2]
    assert banana_mask.is_valid_mask(mask)


def test_background_suppressed():
    # Yellow banana on a neutral background must read RIPE, not unknown
    crop = make_banana_on_background()
    result = analyze_region(crop, [0, 0, 160, 160])
    assert result["background_suppressed"] is True
    assert result["ripeness_class"] == "ripe", result
    assert result["yellow_pct"] > 95.0


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
    crop = np.full((160, 160, 3), BGREEN, dtype=np.uint8)
    result = analyze_region(crop, [0, 0, 160, 160])
    assert result["ripeness_class"] in ("unripe", "semi-ripe")


if __name__ == "__main__":
    tests = [
        test_mask_is_center_touching_blob,
        test_background_suppressed,
        test_no_mask_falls_back,
        test_banana_crop_shrinks_to_mask,
        test_green_banana_still_unripe,
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
