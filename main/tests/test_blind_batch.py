"""
test_blind_batch.py
-------------------
Unit tests for blind_batch.py's pure functions: label parsing,
polygon->bbox conversion, IoU matching, and class mapping.

Run:
    python main/tests/test_blind_batch.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from blind_batch import (
    parse_label_file,
    polygon_bbox,
    iou,
    match_detections,
    map_pred_class,
    ripeness_is_match,
    LABEL_CLASSES,
)


def test_label_classes():
    # data.yaml: names: ['overripe', 'ripe', 'underripe']
    assert LABEL_CLASSES == {0: "overripe", 1: "ripe", 2: "underripe"}


def test_parse_label_file_two_polygons():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n")   # 4-point polygon, class 1
        f.write("2 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n")   # full-frame, class 2
        name = f.name
    try:
        objs = parse_label_file(name)
    finally:
        Path(name).unlink()
    assert len(objs) == 2
    assert objs[0]["cls"] == 1
    assert len(objs[0]["points"]) == 4
    assert objs[1]["points"][0] == (0.0, 0.0)


def test_parse_label_file_real():
    # A real label from the held-out split must parse: every object has
    # an even coordinate count and a class in 0..2.
    label = sorted(
        p for p in (PROJECT_ROOT / "models" / "Training" / "valid" / "labels").iterdir()
        if p.suffix == ".txt"
    )[0]
    for obj in parse_label_file(label):
        assert obj["cls"] in (0, 1, 2)
        assert len(obj["points"]) >= 3  # a polygon needs >= 3 points


def test_polygon_bbox_normalized_to_pixels():
    # Square polygon at normalized (0.2,0.2)-(0.4,0.5) on a 1000x500 image
    pts = [(0.2, 0.2), (0.4, 0.2), (0.4, 0.5), (0.2, 0.5)]
    assert polygon_bbox(pts, 1000, 500) == [200, 100, 400, 250]


def test_polygon_bbox_clips_to_image():
    pts = [(0.0, 0.0), (1.05, 0.0), (1.05, 1.05), (0.0, 1.05)]  # slightly out of frame
    assert polygon_bbox(pts, 100, 100) == [0, 0, 100, 100]


def test_iou_identical_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_quarter_overlap():
    # 50% x-overlap, 50% y-overlap -> IoU = 0.25 / 1.75
    assert abs(iou([0, 0, 10, 10], [5, 5, 15, 15]) - 0.25 / 1.75) < 1e-9


def test_match_detections_greedy_one_to_one():
    # Two truths; preds overlap T0 twice (near-dup) and T1 once.
    truths = [[0, 0, 100, 100], [300, 0, 400, 100]]
    preds = [[2, 2, 102, 102], [5, 5, 105, 105], [302, 2, 402, 102]]
    m = match_detections(preds, truths, iou_threshold=0.3)
    assert sorted(t for t, _ in m) == [0, 1], "both truths matched"
    matched_preds = [p for _, p in m]
    assert len(matched_preds) == 2 and len(set(matched_preds)) == 2, "one pred per truth"


def test_match_detections_empty_when_below_threshold():
    truths = [[0, 0, 100, 100]]
    preds = [[500, 500, 600, 600]]
    assert match_detections(preds, truths, iou_threshold=0.3) == []


def test_match_detections_extra_preds_are_ignored():
    truths = [[0, 0, 100, 100]]
    preds = [[0, 0, 100, 100], [200, 200, 300, 300]]
    assert len(match_detections(preds, truths, iou_threshold=0.3)) == 1


def test_map_pred_class():
    # Live pipeline vocab -> dataset vocab
    assert map_pred_class("unripe") == "underripe"
    assert map_pred_class("underripe") == "underripe"
    assert map_pred_class("ripe") == "ripe"
    assert map_pred_class("overripe") == "overripe"
    assert map_pred_class("semi-ripe") == "semi-ripe"  # its own band
    assert map_pred_class("unknown") == "unknown"


def test_ripeness_is_match():
    # Exact matches
    for c in ("underripe", "ripe", "overripe"):
        assert ripeness_is_match(c, c)
    # semi-ripe is the user's 50/50 band between ripe and underripe
    assert ripeness_is_match("semi-ripe", "underripe")
    assert ripeness_is_match("semi-ripe", "ripe")
    # Mismatches
    assert not ripeness_is_match("semi-ripe", "overripe")
    assert not ripeness_is_match("underripe", "ripe")
    assert not ripeness_is_match("unripe", "overripe")
    assert not ripeness_is_match("unknown", "ripe")
