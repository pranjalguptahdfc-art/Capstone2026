"""
verify_images.py
----------------
Run the real, non-teachable pipeline on an arbitrary image folder and
write out exactly where each detection box landed and what ripeness class
each banana was assigned — so accuracy claims are verified, not guessed.

Same detection + color settings as main/blind_batch.py (adaptive auto mode:
conf 0.10, floor off), plus the app's color-only bunch verdict, so the
numbers line up with both the regression gate and the UI.

Usage:
    python main/tests/verify_images.py [FOLDER] [--out OUTDIR] [--limit N]

FOLDER defaults to the repo root (the user's test images live there, outside
main/). For each image it writes:
    OUTDIR/<stem>.json          per-banana detections + bunch verdict
    OUTDIR/<stem>_annotated.png boxes labelled "#id CLASS" (via annotate_colors)

Per-image JSON:
    {"image": name, "n_detections": n,
     "detections": [
        {"id","bbox","conf","green_pct","yellow_pct","brown_pct",
         "other_pct","dominant_color","ripeness_class"}],
     "bunch": {"bunch_status","color_signal","decision","ripeness_stage",
               "avg_ripeness_pct","avg_green","avg_yellow","avg_brown"}}

Run the module's tests with:  python main/tests/test_verify_images.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

_MAIN = Path(__file__).resolve().parent.parent          # main/
_REPO_ROOT = _MAIN.parent                                # BananaRipeness/
sys.path.insert(0, str(_MAIN))

from detection.banana_detector import BananaDetector          # noqa: E402
from vision.color_analyzer import analyze_batch, annotate_colors  # noqa: E402

# NOTE: utils.analytics_store (pandas) must NOT be imported at module top.
# On Windows its OpenBLAS OpenMP runtime clashes with torch's c10.dll IF
# pandas loads before torch. It is imported lazily inside _color_only_bunch,
# which only runs after the detector (and therefore torch) is loaded.

# Same live-pipeline detection settings as the blind batch + app default.
AUTO_CONF = 0.10
AUTO_FLOOR = False
DEFAULT_OUT = _MAIN / "tests" / "verify"
# .jfif = JPEG files saved by Windows browsers ("images (1).jfif").
IMAGE_EXTS = (".jpg", ".jpeg", ".jfif", ".png")


# ──────────────────────────────────────────────────────────────────────────────
# Bunch verdict (color-only), mirroring app.py::_color_only_result
# ──────────────────────────────────────────────────────────────────────────────

def _bunch_color_signal(color_analyses: List[Dict]) -> str:
    """Per-banana classification counts → a single color signal (app.py:412)."""
    n = len(color_analyses)
    if n == 0:
        return "unknown"
    overripe = sum(1 for a in color_analyses if a["ripeness_class"] == "overripe")
    semi_ripe = sum(1 for a in color_analyses if a["ripeness_class"] == "semi-ripe")
    if overripe > 0:
        return "overripe"
    if semi_ripe / n >= 0.10:
        return "semi-ripe"
    avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
    return "ripe" if avg_yellow > 50 else "unripe"


def _color_only_bunch(color_analyses: List[Dict]) -> Dict:
    """Reproduce the app's no-gas verdict (app.py:357) + the ripeness metrics
    the UI shows (app.py:203). Empty input → the app's UNKNOWN result."""
    n = len(color_analyses)
    if n == 0:
        return {"bunch_status": "UNKNOWN", "color_signal": "unknown",
                "decision": "Denied", "ripeness_stage": "Unknown",
                "avg_ripeness_pct": 0.0, "avg_green": 0.0,
                "avg_yellow": 0.0, "avg_brown": 0.0}

    avg_green = sum(a["green_pct"] for a in color_analyses) / n
    avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
    avg_brown = sum(a["brown_pct"] for a in color_analyses) / n
    overripe_count = sum(1 for a in color_analyses if a["ripeness_class"] == "overripe")
    semi_ripe_count = sum(1 for a in color_analyses if a["ripeness_class"] == "semi-ripe")

    if overripe_count > 0:
        status = "HIGH RISK"
    elif semi_ripe_count / n >= 0.20:
        status = "SEMI-RIPE"
    elif avg_yellow > 50:
        status = "ACCEPTABLE"
    elif avg_green > 50:
        status = "ACCEPTABLE"
    else:
        status = "ACCEPTABLE"

    color_signal = _bunch_color_signal(color_analyses)
    # Deferred: keep pandas (via analytics_store) out of the top-of-module
    # import so it never loads before torch. See the note above the imports.
    from utils.analytics_store import compute_ripeness_metrics
    stage, ripeness_pct, decision = compute_ripeness_metrics(
        avg_green=avg_green, avg_yellow=avg_yellow, avg_brown=avg_brown,
        bunch_status=status, color_signal=color_signal,
    )
    return {
        "bunch_status": status,
        "color_signal": color_signal,
        "decision": decision,
        "ripeness_stage": stage,
        "avg_ripeness_pct": ripeness_pct,
        "avg_green": round(avg_green, 1),
        "avg_yellow": round(avg_yellow, 1),
        "avg_brown": round(avg_brown, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_one(detector: BananaDetector, image_path: Path) -> Dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read {image_path.name}")

    detections = detector.detect(image, confidence_threshold=AUTO_CONF, floor=AUTO_FLOOR)
    analyses = analyze_batch(image, detections)

    det_rows = []
    for det, analysis in zip(detections, analyses):
        det_rows.append({
            "id": det.get("banana_id", 0),
            "bbox": det["bbox"],
            "conf": round(det["confidence"], 4),
            "green_pct": analysis["green_pct"],
            "yellow_pct": analysis["yellow_pct"],
            "brown_pct": analysis["brown_pct"],
            "other_pct": analysis["other_pct"],
            "dominant_color": analysis["dominant_color"],
            "ripeness_class": analysis["ripeness_class"],
        })

    return {
        "image": image_path.name,
        "n_detections": len(detections),
        "detections": det_rows,
        "bunch": _color_only_bunch(analyses),
    }


def verify_folder(
    folder: Path,
    out_dir: Path,
    limit: Optional[int] = None,
    detector: Optional[BananaDetector] = None,
) -> List[Dict]:
    """Run the pipeline on every image in `folder`, write JSON + annotated PNG
    per image into `out_dir`, and return one result dict per image.

    `limit` caps how many images are processed (used by the tests so they only
    run a single detection pass). One shared detector is reused across images.
    """
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(folder.iterdir())
              if p.suffix.lower() in IMAGE_EXTS]
    if limit is not None:
        images = images[:limit]
    if not images:
        return []

    detector = detector if detector is not None else BananaDetector(
        confidence_threshold=AUTO_CONF, device="cpu")
    print(f"Verifying {len(images)} image(s) with {detector.model_label}")

    results: List[Dict] = []
    for i, img in enumerate(images, start=1):
        t0 = time.time()
        try:
            res = _evaluate_one(detector, img)
            res["seconds"] = round(time.time() - t0, 2)
        except Exception as exc:   # keep going — the report should say why
            res = {"image": img.name, "error": str(exc),
                   "n_detections": 0, "detections": [], "bunch": _color_only_bunch([])}
            res["seconds"] = round(time.time() - t0, 2)

        with open(out_dir / f"{img.stem}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        if "error" not in res:
            # Regenerate the annotated PNG only when we have a valid image read
            image = cv2.imread(str(img))
            dets = [{"bbox": d["bbox"], "banana_id": d["id"]} for d in res["detections"]]
            analyses = [{"ripeness_class": d["ripeness_class"]} for d in res["detections"]]
            annotated = annotate_colors(image, dets, analyses)
            cv2.imwrite(str(out_dir / f"{img.stem}_annotated.png"), annotated)

        print(f"[{i}/{len(images)}] {img.name[:48]:48} "
              f"dets={res['n_detections']} "
              f"bunch={res['bunch']['bunch_status']}/{res['bunch']['decision']} "
              f"({res['seconds']}s)" + (f"  ERR: {res['error']}" if "error" in res else ""))
        results.append(res)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=str(_REPO_ROOT),
                    help="folder of images (default: repo root, the user's test images)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output dir for JSON + annotated PNGs (default main/tests/verify)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N images (default: all)")
    args = ap.parse_args()

    results = verify_folder(Path(args.folder), Path(args.out), limit=args.limit)
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\n{n_ok}/{len(results)} image(s) verified. Outputs in {args.out}")


if __name__ == "__main__":
    main()
