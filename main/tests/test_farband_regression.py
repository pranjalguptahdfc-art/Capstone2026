"""
test_farband_regression.py
-------------------------
Live regression guard for the far-band rescue + containment dedup.

Re-runs the real `BananaDetector.detect()` on the repo's reference images and
compares the live box set against the recorded PRE-CHANGE fixture
(`baseline_before.json`, captured at tag `pre-far-band-rescue` / commit
0af16ae). This answers the question "did the change break a working image?"
by checking it against the recorded baseline, not by eyeballing it.

Expected deltas vs the baseline (model: YOLOv8n COCO fallback — re-record
`baseline_before.json` if the detector model changes):

  * images.jfif      : +1 far-band box, 0 lost       (far crates now detected)
  * images (1).jfif  : -5 contained duplicates, 0 gained
  * images (8).jfif  : -5 contained duplicates, 0 gained   (same frame as (1))
  * every other readable image : byte-identical (0 gained, 0 lost)

Slow: runs real YOLO inference on CPU (~40s for the readable set). The
model-free unit/integration tests in test_farband_containment.py and
test_slice_gate_integration.py cover the logic fast; this one proves the
end-to-end behavior on the actual reference photos.

Run:
    python main/tests/test_farband_regression.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))
ROOT = PROJECT_ROOT.parent  # repo root, where the fixture + images live

import cv2

try:
    import torch  # noqa: F401  (conftest preloads in pytest; guard for standalone)
except Exception:
    pass

from detection.banana_detector import BananaDetector, FAR_BAND_FRACTION

BASELINE = json.loads((ROOT / "baseline_before.json").read_text())

# Reference images: (filename, expected gained bboxes, expected lost bboxes).
# A bbox here is (x1, y1, x2, y2) in original-image pixel coords, matched by
# box only (confidence drift is not the failure mode we guard here — the
# structural "did a working image gain/lose a box" is what matters).
EXPECTED = {
    # far-distance miss -> the rescue pass must ADD one far-band box and remove nothing
    "images.jfif": {
        "gained": {(260, 234, 283, 253)},
        "lost": set(),
    },
    # working image -> containment dedup removes 5 nested duplicates, adds nothing
    "images (1).jfif": {
        "gained": set(),
        "lost": {
            (88, 336, 220, 427),   # contained in a stronger outer box
            (87, 367, 226, 498),
            (380, 481, 479, 592),
            (0, 271, 22, 379),
            (239, 385, 341, 534),
        },
    },
    "images (8).jfif": {  # byte-identical frame to (1); same 5 drops
        "gained": set(),
        "lost": {
            (88, 336, 220, 427),
            (87, 367, 226, 498),
            (380, 481, 479, 592),
            (0, 271, 22, 379),
            (239, 385, 341, 534),
        },
    },
    # working images that must stay UNCHANGED
    "images (2).jfif": {"gained": set(), "lost": set()},
    "images (3).jfif": {"gained": set(), "lost": set()},
    "240_F_2190749496_4ixxK8FK8592ixkdFzldzJ43Y1YG3byB.jpg": {"gained": set(), "lost": set()},
    "240_F_2196992637_4QbFXSZMvg81TSqcqUHEFcJbaIQ4wSOk.jpg": {"gained": set(), "lost": set()},
    "240_F_2198946922_Sq6idWOIpzrYRgeJvRuBZx44p6acQfjh.jpg": {"gained": set(), "lost": set()},
}


_LIVE_CACHE = None


def _run_live():
    """Run the real detector on the reference set, once per process.

    YOLO inference is slow (~40s for the set), so results are cached at
    module level; the three tests all share one run.
    """
    global _LIVE_CACHE
    if _LIVE_CACHE is not None:
        return _LIVE_CACHE
    det = BananaDetector(device="cpu")
    results = {}
    for name, spec in EXPECTED.items():
        img = cv2.imread(str(ROOT / name))
        if img is None:
            results[name] = {"error": "unreadable"}
            continue
        live = det.detect(img)
        live_keys = {tuple(d["bbox"]) for d in live}
        base_keys = {tuple(b[0]) for b in BASELINE[name]["boxes"]}
        H = img.shape[0]
        gained = live_keys - base_keys
        lost = base_keys - live_keys
        # the gained far box(es) must actually sit in the far band
        gained_in_far = all(
            (b[1] + b[3]) / 2 < H * FAR_BAND_FRACTION for b in gained
        ) if gained else True
        results[name] = {
            "live": len(live),
            "base": len(base_keys),
            "gained": gained,
            "lost": lost,
            "gained_in_far": gained_in_far,
        }
    _LIVE_CACHE = results
    return results


def test_working_images_gain_no_boxes():
    """No working image may GAIN a box (the 'loop hole' guard)."""
    res = _run_live()
    for name, r in res.items():
        if "error" in r:
            continue
        exp = EXPECTED[name]["gained"]
        assert r["gained"] == exp, (
            f"{name}: gained {sorted(r['gained'])} but expected {sorted(exp)}"
        )


def test_no_working_image_loses_unexpected_boxes():
    """Only the recorded containment drops may be lost; nothing else."""
    res = _run_live()
    for name, r in res.items():
        if "error" in r:
            continue
        exp = EXPECTED[name]["lost"]
        assert r["lost"] == exp, (
            f"{name}: lost {sorted(r['lost'])} but expected {sorted(exp)}"
        )


def test_far_image_gained_a_far_band_box():
    """images.jfif must have gained its far-band box, and it must sit up top."""
    res = _run_live()
    r = res["images.jfif"]
    assert "error" not in r, "images.jfif must be readable for this guard"
    assert len(EXPECTED["images.jfif"]["gained"]) == 1, "precondition: one expected far box"
    assert r["gained"] == EXPECTED["images.jfif"]["gained"], f"gained {sorted(r['gained'])}"
    assert r["gained_in_far"], "the gained box must be in the far (top) band"


if __name__ == "__main__":
    tests = [
        test_working_images_gain_no_boxes,
        test_no_working_image_loses_unexpected_boxes,
        test_far_image_gained_a_far_band_box,
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
