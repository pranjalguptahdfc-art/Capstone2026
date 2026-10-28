"""
test_color_model.py
-------------------
Behavior tests for vision/color_model.py — the exclusive, lighting-invariant
color families. Pure synthetic arrays, no model, no I/O: instant.

Run:
    python main/tests/test_color_model.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from vision.color_model import compute_v_ref, family_counts  # noqa: E402


def _hsv(h, s, v, shape=(40, 40)):
    arr = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    arr[..., 0] = h
    arr[..., 1] = s
    arr[..., 2] = v
    return arr


def test_ripe_yellow_not_doublecounted_as_brown():
    # The exact regression case: warm-lit ripe yellow, H=22 (inside the old
    # brown/yellow overlap zone H15-25), bright V=180. The old model counted
    # this in BOTH yellow and brown.
    hsv = _hsv(22, 150, 180)
    v_ref = compute_v_ref(hsv, None)
    c = family_counts(hsv, None, v_ref)
    assert c["yellow"] == 1600      # every pixel is yellow
    assert c["brown"] == 0          # ...and no pixel is also brown
    assert c["total"] == 1600


def test_dim_yellow_still_yellow():
    # Dim exposure: same banana at 0.3× brightness. The old yellow floor
    # (V≥80) dropped it entirely → misread. Must stay yellow.
    hsv = _hsv(22, 150, 54)        # 180 × 0.3
    v_ref = compute_v_ref(hsv, None)
    c = family_counts(hsv, None, v_ref)
    assert c["yellow"] == 1600
    assert c["brown"] == 0


def test_brown_spot_is_dark_relative_to_crop():
    # Ripe crop with a dark 10% spot: 90% bright yellow (V=180), 10% dark
    # brown (V=60, H=20). The spot must land in brown, not vanish.
    hsv = _hsv(22, 150, 180)
    hsv[:8] = np.array([20, 90, 60], dtype=np.uint8)   # top rows = spot
    v_ref = compute_v_ref(hsv, None)
    c = family_counts(hsv, None, v_ref)
    assert c["brown"] > 0
    assert c["yellow"] < 1600
    assert c["brown"] + c["yellow"] + c["green"] <= c["total"]  # exclusive


def test_uniform_ripe_crop_has_low_brown():
    # No spots: uniform ripe crop with natural V variation. The brown share
    # must stay well under the 20% overripe gate — the gate's correctness
    # depends on this.
    rng = np.random.default_rng(42)
    hsv = np.zeros((60, 60, 3), dtype=np.uint8)
    hsv[..., 0] = 24 + rng.integers(-2, 3, size=(60, 60))
    hsv[..., 1] = 160
    hsv[..., 2] = np.clip(190 + rng.integers(-40, 40, size=(60, 60)), 0, 255)
    v_ref = compute_v_ref(hsv, None)
    c = family_counts(hsv, None, v_ref)
    assert c["brown"] / c["total"] < 0.20


def test_green_stays_green():
    hsv = _hsv(60, 140, 120)
    v_ref = compute_v_ref(hsv, None)
    c = family_counts(hsv, None, v_ref)
    assert c["green"] == 1600
    assert c["yellow"] == 0 and c["brown"] == 0


def test_mask_restriction_counts_only_masked_pixels():
    # When a pixel mask is given, families only count inside it and total
    # equals the mask area.
    hsv = _hsv(22, 150, 180)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, :20] = 255                      # left half only
    v_ref = compute_v_ref(hsv, mask)
    c = family_counts(hsv, mask, v_ref)
    assert c["total"] == 800
    assert c["yellow"] == 800


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
    print(f"\nAll {len(tests)} color_model tests passed.")


if __name__ == "__main__":
    _main()
