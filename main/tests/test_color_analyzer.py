"""
test_color_analyzer.py
---------------------
Behavior tests for vision/color_analyzer.analyze_region against the
lighting-invariant color model: a warm-lit ripe banana must read as ripe
(not overripe via the old brown/yellow double-count), a dimmed crop must
not degrade to "unknown", and a genuinely-spotted crop must still read
overripe.

Pure OpenCV (no model, no I/O): instant.

Run:
    python main/tests/test_color_analyzer.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from vision.color_analyzer import analyze_region, analyze_batch  # noqa: E402

BRIGHT_YELLOW_BGR = (60, 200, 255)   # saturated ripe yellow
DIM_YELLOW_BGR = (18, 60, 76)        # ripe yellow at ~0.3× exposure (HSV ~22,194,76)
BROWN_SPOT_BGR = (12, 38, 60)        # true brown spot on a BRIGHT body (H~30, S~204, V=60)
NEAR_BLACK_BGR = (5, 15, 15)         # very dark overripe skin / spot
GREY_BG_BGR = (120, 120, 120)        # neutral non-warm background (excluded by mask)


def _scene(body_bgr, bg_bgr=GREY_BG_BGR, body_box=(8, 8, 32, 32),
           spot_box=None, spot_bgr=NEAR_BLACK_BGR):
    """40×40 scene: neutral background, a `body_bgr` blob at body_box, and an
    optional `spot_bgr` region at spot_box (both BGR). bbox passed to
    analyze_region is the whole 40×40 frame so the mask isolates the blob."""
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:] = bg_bgr
    frame[body_box[1]:body_box[3], body_box[0]:body_box[2]] = body_bgr
    if spot_box:
        frame[spot_box[1]:spot_box[3], spot_box[0]:spot_box[2]] = spot_bgr
    return frame


FULL = [0, 0, 40, 40]


BBOX = [5, 5, 35, 35]


def test_ripe_crop_is_ripe_not_overripe():
    r = analyze_region(_scene(BRIGHT_YELLOW_BGR), FULL)
    assert r["ripeness_class"] == "ripe"
    assert r["brown_pct"] < 20          # old double-count tripped this gate
    assert r["other_pct"] >= 0          # exclusive families: never negative


def test_dim_crop_stays_ripe():
    r = analyze_region(_scene(DIM_YELLOW_BGR), FULL)
    assert r["ripeness_class"] in ("ripe", "semi-ripe")
    assert r["ripeness_class"] != "overripe"


def test_spotted_crop_is_overripe():
    # Bright ripe body + a dark spot in the top third → brown crosses the gate.
    r = analyze_region(
        _scene(BRIGHT_YELLOW_BGR, spot_box=(8, 8, 32, 16),
               spot_bgr=BROWN_SPOT_BGR), FULL)
    assert r["brown_pct"] >= 20
    assert r["ripeness_class"] == "overripe"


def test_green_crop_is_unripe():
    green_bgr = (40, 180, 40)          # green banana: G high, R/B low
    r = analyze_region(_scene(green_bgr), FULL)
    assert r["green_pct"] > 50
    assert r["ripeness_class"] == "unripe"


def test_analyze_batch_assigns_ids_and_keeps_alignment():
    frame = _scene(BRIGHT_YELLOW_BGR)
    detections = [
        {"bbox": [8, 8, 32, 32], "banana_id": 1},
        {"bbox": [8, 8, 20, 20], "banana_id": 2},
    ]
    results = analyze_batch(frame, detections)
    assert len(results) == 2
    assert results[0]["banana_id"] == 1 and results[1]["banana_id"] == 2
    for res in results:
        assert {"green_pct", "yellow_pct", "brown_pct", "other_pct",
                "dominant_color", "ripeness_class"} <= set(res)


def test_empty_bbox_gives_unknown():
    r = analyze_region(np.zeros((10, 10, 3), dtype=np.uint8), [0, 0, 0, 0])
    assert r["ripeness_class"] == "unknown"

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
    print(f"\nAll {len(tests)} color_analyzer tests passed.")


if __name__ == "__main__":
    _main()
