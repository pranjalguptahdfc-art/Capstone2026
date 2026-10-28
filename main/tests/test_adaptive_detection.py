"""
test_adaptive_detection.py
-------------------------
Unit tests for adaptive detection: relative score filter + NMS dedup.

Run:
    python main/tests/test_adaptive_detection.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/ (matches test_pipeline.py)
sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import (
    relative_score_filter, suppress_duplicates,
    generate_slices, should_slice,
    SLICE_GATE_BEST_CONF, SLICE_MAX_LONG_SIDE,
    BananaDetector,
)


def det(box, conf):
    return {"banana_id": 0, "bbox": box, "confidence": conf}


def test_relative_filter_keeps_real_drops_junk():
    # Best score 0.45 -> threshold = max(0.45*0.30, 0.08) = 0.135
    dets = [det([0, 0, 100, 100], 0.45), det([200, 200, 300, 300], 0.20),
            det([500, 500, 550, 550], 0.05)]
    kept = relative_score_filter(dets)
    assert len(kept) == 2, "junk box (0.05) is below threshold 0.135; 0.45 and 0.20 survive"


def test_weak_bunch_survives():
    # Best score 0.10 -> threshold = max(0.03, 0.08) = 0.08 -> 0.09 box survives
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


# ── Gated slice boost (far-distance / small bunches) ────────────────────────
# see memory: rollback-pre-slice-detection. Slicing must fire ONLY when the
# full-frame pass is weak AND the image is small enough that its bananas get
# downsampled below what YOLO can localize. Working close-range images
# (best conf ~0.58+) must NOT slice → bit-identical results.

def test_should_slice_fires_on_weak_small_image():
    # the far-bunch case: weak full-frame conf, small image → slice
    assert should_slice(0.18, 335, 597) is True


def test_should_slice_no_on_strong_image():
    # working image: strong conf → do NOT slice (regression guard)
    assert should_slice(0.61, 480, 640) is False


def test_should_slice_no_when_image_large():
    # weak conf but a large image → no slice (full-frame already resolves it,
    # and we don't want 9x inference cost on big photos)
    assert should_slice(0.10, 4000, 3000) is False


def test_should_slice_gate_is_strict():
    # exactly at the conf gate → no slice; just below → slice
    assert should_slice(SLICE_GATE_BEST_CONF, 100, 100) is False
    assert should_slice(SLICE_GATE_BEST_CONF - 0.01, 100, 100) is True


def test_generate_slices_default_grid_is_3x3():
    tiles = generate_slices(600, 600)
    assert len(tiles) == 9


def test_generate_slices_1x1_single_full_tile():
    tiles = generate_slices(100, 80, n_tiles=1)
    assert tiles == [(0, 0, 100, 80)]


def test_generate_slices_tiling_covers_full_extent():
    tiles = generate_slices(300, 300, n_tiles=4, overlap=0.2)
    assert len(tiles) == 16
    x0_min = min(t[0] for t in tiles)
    x1_max = max(t[2] for t in tiles)
    y0_min = min(t[1] for t in tiles)
    y1_max = max(t[3] for t in tiles)
    assert (x0_min, y0_min) == (0, 0), "tiling must start at the origin"
    assert (x1_max, y1_max) == (300, 300), "tiling must reach the far edge"


def test_generate_slices_rectangular_image():
    tiles = generate_slices(400, 200, n_tiles=3)
    assert len(tiles) == 9
    # every tile's top-left origin is inside the image, every box is valid
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= 400
        assert 0 <= y0 < y1 <= 200


if __name__ == "__main__":
    tests = [
        test_relative_filter_keeps_real_drops_junk,
        test_weak_bunch_survives,
        test_nms_suppresses_duplicate,
        test_nms_keeps_adjacent_bananas,
        test_nms_numbers_by_confidence,
        test_should_slice_fires_on_weak_small_image,
        test_should_slice_no_on_strong_image,
        test_should_slice_no_when_image_large,
        test_should_slice_gate_is_strict,
        test_generate_slices_default_grid_is_3x3,
        test_generate_slices_1x1_single_full_tile,
        test_generate_slices_tiling_covers_full_extent,
        test_generate_slices_rectangular_image,
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
