"""
test_farband_containment.py
--------------------------
Unit tests for the far-band rescue pass and containment dedup.

Run:
    python main/tests/test_farband_containment.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import (
    discard_contained,
    far_band_gate,
    FAR_BAND_FRACTION,
    FAR_BAND_FLOOR,
    FAR_BAND_MAX_LONG_SIDE,
)


def det(box, conf):
    return {"banana_id": 0, "bbox": box, "confidence": conf}


# ── far_band_gate ─────────────────────────────────────────────────────────────

def test_far_band_gate_fires_when_top_band_empty():
    """images.jfif: 9 boxes, all in bottom 55%, top 45% has zero centers → gate True."""
    H = 597
    dets = [det([84, 507, 181, 596], 0.707),
            det([183, 425, 326, 548], 0.588)]
    # center of first: (507+596)/2 = 551.5; 551.5/597 = 0.924 → NOT in top 45%
    # center of second: (425+548)/2 = 486.5; 486.5/597 = 0.815 → NOT in top 45%
    assert far_band_gate(dets, H, W=335) is True


def test_far_band_gate_does_not_fire_when_top_band_has_boxes():
    """images (1): 2 boxes have centers in top 40% → gate False."""
    H = 640
    dets = [det([8, 273, 82, 382], 0.341),   # center=327.5, 327.5/640=0.512 → top band? 0.512>0.55? No.
            det([304, 85, 354, 132], 0.233)]  # center=108.5, 108.5/640=0.170 → IN top band
    assert far_band_gate(dets, H=640, W=480) is False


def test_far_band_gate_no_fire_on_strong_image():
    """Working close-range image: 14 boxes, several in top band → gate False."""
    H = 640
    dets = [det([0, 0, 100, 100], 0.614) for _ in range(3)]  # all in top-left
    assert far_band_gate(dets, H=640, W=480) is False


def test_far_band_gate_no_fire_on_large_image():
    """Top band empty but image is large (long side > 1024) → gate False."""
    H = 4000
    dets = [det([0, 3900, 100, 3999], 0.7)]  # box at very bottom only
    assert far_band_gate(dets, H=4000, W=3000) is False


def test_far_band_gate_empty_detections_fires():
    """No boxes at all: far band trivially empty → gate True."""
    assert far_band_gate([], H=500, W=300) is True


# ── discard_contained ─────────────────────────────────────────────────────────

def test_containment_drops_inner_when_outer_stronger():
    """Outer box [0,0,200,200] conf=0.4 contains inner [50,50,150,150] conf=0.3 → drop inner."""
    dets = [det([0, 0, 200, 200], 0.40), det([50, 50, 150, 150], 0.30)]
    result = discard_contained(dets)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.40


def test_containment_keeps_inner_when_inner_stronger():
    """Outer [0,0,200,200] conf=0.227 contains inner [50,50,150,150] conf=0.406 → keep both
    (outer not strong enough to justify discarding the stronger inner box)."""
    dets = [det([0, 0, 200, 200], 0.227), det([50, 50, 150, 150], 0.406)]
    result = discard_contained(dets)
    assert len(result) == 2, "inner box is stronger — both must be kept"


def test_containment_no_false_positive_on_adjacent_boxes():
    """Side-by-side boxes are not containment — both survive."""
    dets = [det([0, 0, 100, 100], 0.40), det([101, 0, 201, 100], 0.35)]
    result = discard_contained(dets)
    assert len(result) == 2


def test_containment_cascade_stops_after_first_discard():
    """Outer A contains B; B contains C. A conf=0.5, B conf=0.4, C conf=0.3.
    Walked strongest-first: A kept; B is contained in A → discarded; C is also
    contained in A (containment is transitive here) → discarded. Result: only A."""
    dets = [
        det([0, 0, 400, 400], 0.50),    # A outer
        det([100, 100, 300, 300], 0.40),  # B middle
        det([150, 150, 250, 250], 0.30),  # C inner
    ]
    result = discard_contained(dets)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.50, "outer A (strongest) must survive"
    assert result[0]["banana_id"] == 1


def test_containment_keeps_box_inside_weaker_kept_box():
    """Two overlapping-but-not-contained boxes stay; a third box contained only
    in the weaker second box is dropped when the weaker box is kept first.
    A [0,0,100,100] conf=0.6 (kept). B [90,0,190,100] conf=0.4 (kept, overlaps A).
    C [120,30,160,70] conf=0.3 is inside B → dropped (B kept first, C contained in B)."""
    dets = [
        det([0, 0, 100, 100], 0.60),
        det([90, 0, 190, 100], 0.40),
        det([120, 30, 160, 70], 0.30),
    ]
    result = discard_contained(dets)
    confs = [d["confidence"] for d in result]
    assert 0.60 in confs and 0.40 in confs, "A and B overlap (IoU<containment) → both kept"
    assert 0.30 not in confs, "C is contained in kept B → discarded"


def test_containment_renumbers_banana_id():
    dets = [det([0, 0, 200, 200], 0.40), det([50, 50, 150, 150], 0.30)]
    result = discard_contained(dets)
    assert [d["banana_id"] for d in result] == [1], "banana_id renumbered 1..N after dedup"


if __name__ == "__main__":
    tests = [
        test_far_band_gate_fires_when_top_band_empty,
        test_far_band_gate_does_not_fire_when_top_band_has_boxes,
        test_far_band_gate_no_fire_on_strong_image,
        test_far_band_gate_no_fire_on_large_image,
        test_far_band_gate_empty_detections_fires,
        test_containment_drops_inner_when_outer_stronger,
        test_containment_keeps_inner_when_inner_stronger,
        test_containment_no_false_positive_on_adjacent_boxes,
        test_containment_cascade_stops_after_first_discard,
        test_containment_keeps_box_inside_weaker_kept_box,
        test_containment_renumbers_banana_id,
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
