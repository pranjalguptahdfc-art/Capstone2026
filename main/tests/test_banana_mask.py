"""
test_banana_mask.py
-------------------
Behavior tests for vision/banana_mask.py against the color_model-derived
windows: the body must be isolated for a dim banana (old yellow window's
V≥80 floor could not see it — the brown window only caught it "accidentally"),
for a bright banana, and via the pale last-resort for whitened overripe
skin.

Pure OpenCV (no model, no I/O): instant.

Run:
    python main/tests/test_banana_mask.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from vision.banana_mask import banana_mask, is_valid_mask  # noqa: E402

# 20×20 blob on a 40×40 crop = 25% of the pixels (> is_valid_mask's 10%
# min_ratio, and the blob touches the crop center)
BLOB = (10, 10, 30, 30)


def _crop_with_body(body_bgr, bg_bgr, blob=BLOB, size=40):
    c = np.zeros((size, size, 3), dtype=np.uint8)
    c[:] = bg_bgr
    c[blob[1]:blob[3], blob[0]:blob[2]] = body_bgr
    return c


def _blob_pixels(mask, blob=BLOB):
    return mask[blob[1]:blob[3], blob[0]:blob[2]]


def _bg_leak(mask, blob=BLOB):
    """Share of OUTSIDE-blob pixels the mask wrongly claims."""
    m = np.ones(mask.shape, bool)
    m[blob[1]:blob[3], blob[0]:blob[2]] = False
    outside = int(m.sum())
    return int(np.count_nonzero(mask & m)) / outside if outside else 0.0


DIM_YELLOW_BGR = (18, 60, 76)     # ripe yellow at ~0.3× exposure (HSV ~22,194,76)
BRIGHT_YELLOW_BGR = (60, 200, 255)  # saturated ripe yellow (HSV ~21,195,255)
PALE_SKIN_BGR = (240, 240, 240)   # whitened overripe skin (S≈0, V=240)
GREY_BG_BGR = (120, 120, 120)     # neutral background (S=0, not in any window)
DARK_BG_BGR = (40, 40, 40)


def test_dim_yellow_body_isolated():
    body = _crop_with_body(DIM_YELLOW_BGR, GREY_BG_BGR)
    m = banana_mask(body)
    assert is_valid_mask(m)
    assert np.count_nonzero(_blob_pixels(m)) / _blob_pixels(m).size > 0.9
    assert _bg_leak(m) < 0.05          # background not claimed


def test_bright_body_still_isolated():
    body = _crop_with_body(BRIGHT_YELLOW_BGR, GREY_BG_BGR)
    m = banana_mask(body)
    assert is_valid_mask(m)
    assert np.count_nonzero(_blob_pixels(m)) / _blob_pixels(m).size > 0.9
    assert _bg_leak(m) < 0.05


def test_green_body_isolated():
    body = _crop_with_body((40, 180, 40), GREY_BG_BGR)   # green banana BGR
    m = banana_mask(body)
    assert is_valid_mask(m)
    assert np.count_nonzero(_blob_pixels(m)) / _blob_pixels(m).size > 0.9


def test_pale_overripe_skin_falls_back():
    # Whitened skin is not in the green/warm windows (S<30); the pale
    # last-resort window must isolate it instead.
    body = _crop_with_body(PALE_SKIN_BGR, DARK_BG_BGR)
    m = banana_mask(body)
    assert is_valid_mask(m)
    assert np.count_nonzero(_blob_pixels(m)) / _blob_pixels(m).size > 0.9


def _main():
    tests = {n: f for n, f in globals().items() if n.startswith("test_") and callable(f)}
    failed = []
    for name, fn in sorted(tests.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}")
    if failed:
        print(f"\n{len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f"\nAll {len(tests)} banana_mask tests passed.")


if __name__ == "__main__":
    _main()
