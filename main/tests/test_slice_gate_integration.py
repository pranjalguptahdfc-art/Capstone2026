"""
test_slice_gate_integration.py
-----------------------------
Fast, model-free integration tests for the gated slice boost in
`BananaDetector.detect()`.

The real YOLO model is replaced with a fake (see _FakeModel below) so we can
assert exactly how many inference calls happen:
  - strong full-frame read  -> 1 call, no slicing (regression guard)
  - weak full-frame read on a small image -> 1 + 9 tile calls
  - weak read on a large image -> 1 call (size gate)

Run:
    python main/tests/test_slice_gate_integration.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import BananaDetector, SLICE_TILES


class _Arr:
    """Mimics torch tensor access: .item() and .tolist()"""

    def __init__(self, value):
        self._v = value

    def item(self):
        return self._v

    def tolist(self):
        return list(self._v)


class _FakeBox:
    def __init__(self, x1, y1, x2, y2, conf, cls_idx):
        self.cls = [_Arr(cls_idx)]
        self.conf = [_Arr(conf)]
        self.xyxy = [_Arr([x1, y1, x2, y2])]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Stand-in for the ultralytics model: `model(img)` -> [Result].

    `make_boxes(w, h)` returns a list of (x1, y1, x2, y2, conf, cls_idx) in
    the coords of whichever image (full frame or tile) was passed in.
    """

    def __init__(self, make_boxes):
        self.calls = 0
        self._make = make_boxes

    def __call__(self, image, **kwargs):
        self.calls += 1
        h, w = image.shape[:2]
        boxes = self._make(w, h)
        if boxes:
            return [_FakeResult([_FakeBox(*b) for b in boxes])]
        return [_FakeResult(None)]


def _make_detector(make_boxes):
    """BananaDetector with a fake model — no YOLO/torch inference."""
    det = BananaDetector.__new__(BananaDetector)
    det._model = _FakeModel(make_boxes)
    det._is_fallback = False
    det._model_label = "FAKE (test)"
    det.confidence_threshold = 0.20
    det.device = "cpu"
    return det


def _img(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_gate_off_strong_small_image_no_slicing():
    """Strong full-frame read (conf 0.60) on a small image: exactly one
    inference call — the slice path must NOT run (regression guard)."""
    det = _make_detector(lambda w, h: [(10, 10, 80, 60, 0.60, 46)])
    r = det.detect(_img(480, 640), confidence_threshold=0.10, floor=False)
    assert det._model.calls == 1, f"expected 1 call, got {det._model.calls}"
    assert len(r) == 1


def test_gate_on_weak_small_image_slices():
    """Weak full-frame read (conf 0.10) on a 335x597 image: 1 full-frame
    call + one per tile, and tile boxes are mapped back into frame."""
    det = _make_detector(lambda w, h: [(0, 0, 60, 40, 0.10, 46)])
    r = det.detect(_img(335, 597), confidence_threshold=0.10, floor=False)
    from detection.banana_detector import generate_slices
    n_tiles = len(generate_slices(335, 597))
    assert det._model.calls == 1 + n_tiles, (
        f"expected 1 + {n_tiles} tile calls, got {det._model.calls}")
    assert len(r) > 1, "sliced candidates from different tiles should survive"
    # every merged box must be within the original frame
    for d in r:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= 334 and 0 <= y1 < y2 <= 596


def test_gate_off_large_image_no_slicing():
    """Weak conf but large image (4000x3000 > 1024): the size gate stops
    us from paying for 9 inference passes on a big photo."""
    det = _make_detector(lambda w, h: [(10, 10, 80, 60, 0.10, 46)])
    det.detect(_img(4000, 3000), confidence_threshold=0.10, floor=False)
    assert det._model.calls == 1, f"expected 1 call, got {det._model.calls}"


def test_gate_on_weak_conf_threshold_is_strict():
    """Conf exactly at the gate (0.35) does NOT slice; just below does."""
    det = _make_detector(lambda w, h: [(10, 10, 80, 60, 0.35, 46)])
    det.detect(_img(400, 400), confidence_threshold=0.10, floor=False)
    assert det._model.calls == 1

    det2 = _make_detector(lambda w, h: [(10, 10, 80, 60, 0.34, 46)])
    det2.detect(_img(400, 400), confidence_threshold=0.10, floor=False)
    assert det2._model.calls == 1 + SLICE_TILES * SLICE_TILES


if __name__ == "__main__":
    tests = [
        test_gate_off_strong_small_image_no_slicing,
        test_gate_on_weak_small_image_slices,
        test_gate_off_large_image_no_slicing,
        test_gate_on_weak_conf_threshold_is_strict,
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
