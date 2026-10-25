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
