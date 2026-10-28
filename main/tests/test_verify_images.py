"""
test_verify_images.py
---------------------
Tests for the verify_images.py harness: per-image JSON + annotated PNG
output, in-frame boxes, empty-folder handling.

Run:
    python main/tests/test_verify_images.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from verify_images import verify_folder          # noqa: E402


def _one_valid_dataset_image() -> Path:
    """A labelled valid/ image — known ground truth, and the folder is
    capped with limit=1 so the test only runs one detection pass."""
    return sorted(
        p for p in (PROJECT_ROOT / "models" / "Training" / "valid" / "images").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[0]


def test_verify_folder_writes_json_and_annotated_png():
    src = _one_valid_dataset_image()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        results = verify_folder(src.parent, out, limit=1)
        assert len(results) == 1
        assert (out / f"{src.stem}.json").exists()
        assert (out / f"{src.stem}_annotated.png").exists()
        r = results[0]
        assert r["image"] == src.name
        assert isinstance(r["n_detections"], int)
        for d in r["detections"]:
            assert {"id", "bbox", "conf", "green_pct", "yellow_pct",
                    "brown_pct", "ripeness_class"} <= set(d)
            x1, y1, x2, y2 = d["bbox"]
            assert 0 <= x1 < x2 and 0 <= y1 < y2   # boxes are in-frame
        assert "bunch" in r and "decision" in r["bunch"]


def test_verify_folder_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        assert verify_folder(Path(td), Path(td), limit=1) == []


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
    print(f"\nAll {len(tests)} verify_images tests passed.")


if __name__ == "__main__":
    _main()
