"""
blind_batch.py
--------------
Held-out batch runner that measures the LIVE non-teachable pipeline
(YOLOv8n COCO fallback -> adaptive detection -> banana-mask -> HSV
color analysis) against the ground-truth ripeness labels in
`main/models/Training` (Roboflow polygon labels: 0=overripe,
1=ripe, 2=underripe).

This is verification only — no model training, per
NON_TEACHABLE_GUIDE.md. It answers two questions per image:

  1. Detection recall: how many labeled bananas did the pipeline find?
  2. Ripeness agreement: for each found banana, does the HSV class
     match the human label? (semi-ripe is the acceptable 50/50 band
     between ripe and underripe — a mismatch is only when a matched
     banana's mapped class is off.)

Run:
    python main/blind_batch.py            # 20-image smoke batch
    python main/blind_batch.py --all      # every valid/ + test/ image
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent  # main/
sys.path.insert(0, str(_PROJECT_ROOT))

from detection.banana_detector import BananaDetector          # noqa: E402
from vision.color_analyzer import analyze_batch                 # noqa: E402

DATASET = _PROJECT_ROOT / "models" / "Training"
REPORT_PATH = _PROJECT_ROOT / "tests" / "blind_batch_report.md"

# data.yaml: names: ['overripe', 'ripe', 'underripe']
LABEL_CLASSES: Dict[int, str] = {0: "overripe", 1: "ripe", 2: "underripe"}

# The live pipeline's default is auto (adaptive) mode — mirrors the app
# sidebar default: infer permissively, let the relative filter do the work.
AUTO_CONF = 0.10
AUTO_FLOOR = False

MATCH_IOU_THRESHOLD = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Label parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_label_file(path) -> List[Dict]:
    """Parse a YOLO polygon-label file into per-object dicts.

    Each line: `<cls> x1 y1 x2 y2 ...` in normalized coords — every object
    is a polygon (class + an even number of coordinates), not a box.
    """
    objects: List[Dict] = []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 4 or (len(tokens) - 1) % 2 != 0:
            continue
        cls = int(tokens[0])
        coords = [float(t) for t in tokens[1:]]
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        objects.append({"cls": cls, "points": points})
    return objects


def polygon_bbox(points: List, img_w: int, img_h: int) -> List[int]:
    """Tight pixel bounding box of a normalized polygon, clipped to frame."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(int(min(xs) * img_w), img_w))
    y1 = max(0, min(int(min(ys) * img_h), img_h))
    x2 = max(0, min(int(max(xs) * img_w), img_w))
    y2 = max(0, min(int(max(ys) * img_h), img_h))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


# ──────────────────────────────────────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────────────────────────────────────

def iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    preds: List[List[int]],
    truths: List[List[int]],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> List:
    """Greedy one-to-one IoU matching: pairs above the threshold, best first.

    Returns [(truth_idx, pred_idx), ...].  Extra predictions (no truth
    nearby) and extra truths (unmatched) are simply left out.
    """
    scored = [
        (iou(p, t), ti, pi)
        for ti, t in enumerate(truths)
        for pi, p in enumerate(preds)
        if iou(p, t) >= iou_threshold
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    used_pred, used_truth = set(), set()
    pairs = []
    for score, ti, pi in scored:
        if ti in used_truth or pi in used_pred:
            continue
        used_truth.add(ti)
        used_pred.add(pi)
        pairs.append((ti, pi))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Class mapping  (live pipeline vocab -> dataset vocab)
# ──────────────────────────────────────────────────────────────────────────────

def map_pred_class(pred: str) -> str:
    """Map a live-pipeline ripeness class to the dataset vocabulary.

    unripe -> underripe;  semi-ripe keeps its own band.
    """
    return {"unripe": "underripe"}.get(pred, pred)


def ripeness_is_match(pred: str, label: str) -> bool:
    """True when the predicted class agrees with the human label.

    semi-ripe is the user-defined 50/50 band between ripe and underripe,
    so it counts as a match against either; against overripe it does not.
    unknown never matches.
    """
    if pred == "unknown":
        return False
    mapped = map_pred_class(pred)
    if mapped == label:
        return True
    return mapped == "semi-ripe" and label in ("underripe", "ripe")


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_image(detector: BananaDetector, image_path: Path, label_path: Path) -> Dict:
    """Run the live pipeline on one image and score it against its labels."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read {image_path.name}")
    img_h, img_w = image.shape[:2]

    detections = detector.detect(image, confidence_threshold=AUTO_CONF, floor=AUTO_FLOOR)
    truths = [
        {"cls": obj["cls"], "bbox": polygon_bbox(obj["points"], img_w, img_h)}
        for obj in parse_label_file(label_path)
    ]
    truths = [t for t in truths if t["bbox"][2] > t["bbox"][0]]  # drop degenerate
    truth_boxes = [t["bbox"] for t in truths]
    pred_boxes = [d["bbox"] for d in detections]

    pairs = match_detections(pred_boxes, truth_boxes)
    analyses = analyze_batch(image, detections)

    banana_rows = []
    matched_pi = {pi for _, pi in pairs}
    for ti, pi in pairs:
        label = LABEL_CLASSES[truths[ti]["cls"]]
        pred = analyses[pi]["ripeness_class"]
        banana_rows.append({
            "truth_idx": ti + 1,
            "label": label,
            "pred": pred,
            "pred_mapped": map_pred_class(pred),
            "ok": ripeness_is_match(pred, label),
            "green": analyses[pi]["green_pct"],
            "yellow": analyses[pi]["yellow_pct"],
            "brown": analyses[pi]["brown_pct"],
            "conf": detections[pi]["confidence"],
        })

    return {
        "image": image_path.name,
        "n_labelled": len(truths),
        "n_detected": len(detections),
        "n_matched": len(pairs),
        "detection_recall": (len(pairs) / len(truths)) if truths else None,
        "spurious": len(detections) - len(matched_pi),
        "bananas": banana_rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch selection + report
# ──────────────────────────────────────────────────────────────────────────────

def _split_images(split: str) -> Dict[str, List[Path]]:
    """Group a split's (image, label) pairs by the image's dominant label class.

    Pairing is by filename stem; images with no (or empty) label file fall
    under "unlabelled".
    """
    imgs = DATASET / split / "images"
    labs = DATASET / split / "labels"
    by_class: Dict[str, List[Path]] = {}
    for p in sorted(imgs.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lp = labs / (p.stem + ".txt")
        objects = parse_label_file(lp) if lp.exists() else []
        if not objects:
            by_class.setdefault("unlabelled", []).append(p)
            continue
        counts: Dict[str, int] = {}
        for o in objects:
            name = LABEL_CLASSES.get(o["cls"], "unlabelled")
            counts[name] = counts.get(name, 0) + 1
        dom = max(counts, key=lambda c: (counts[c], c))
        by_class.setdefault(dom, []).append(p)
    return by_class


def pick_batch(limit: int = 20) -> List[tuple]:
    """Deterministic smoke batch: cover every overripe image (scarcest,
    7 total), then fill with ripe/underripe/unlabelled. Returns
    [(split, image_path, label_path), ...].
    """
    picked: List[tuple] = []
    for split in ("valid", "test"):
        for p in _split_images(split).get("overripe", []):
            picked.append((split, p, DATASET / split / "labels" / (p.stem + ".txt")))
    if len(picked) < limit:
        pool = []
        for split in ("valid", "test"):
            for cls in ("ripe", "underripe", "unlabelled"):
                for p in _split_images(split).get(cls, []):
                    pool.append((split, p, DATASET / split / "labels" / (p.stem + ".txt")))
        random.Random(42).shuffle(pool)
        picked += pool[: limit - len(picked)]
    return picked[:limit]


def pick_all() -> List[tuple]:
    out = []
    for split in ("valid", "test"):
        by_class = _split_images(split)
        for cls in ("overripe", "ripe", "underripe", "unlabelled"):
            for p in by_class.get(cls, []):
                out.append((split, p, DATASET / split / "labels" / (p.stem + ".txt")))
    return out


def run_batch(items: List[tuple], detector: BananaDetector) -> List[Dict]:
    results = []
    for i, (split, img, lab) in enumerate(items, start=1):
        t0 = time.time()
        try:
            res = evaluate_image(detector, img, lab)
        except Exception as exc:  # keep going — the report should say why
            res = {"image": img.name, "split": split, "error": str(exc)}
        res["split"] = split
        res["seconds"] = round(time.time() - t0, 2)
        results.append(res)
        print(f"[{i}/{len(items)}] {img.stem[:40]} ... {res.get('error') or 'ok'} "
              f"({res['seconds']}s)")
    return results


def write_report(results: List[Dict], path: Path, limit: int) -> None:
    lines = [
        "# Blind-batch report — live pipeline vs. Training labels",
        "",
        f"Batch size: {len(results)} (limit {limit})  ·  splits: valid + test  ·  "
        f"detection: adaptive auto mode (conf {AUTO_CONF}, floor off)  ·  "
        f"match IoU ≥ {MATCH_IOU_THRESHOLD}  ·  generated {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if all("error" in r for r in results):
        lines.append("_All images failed — see errors below._\n")
        for r in results:
            lines.append(f"- {r.get('image')}: {r['error']}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    ok = [r for r in results if "error" not in r]

    # ── Aggregates ──
    with_recall = [r for r in ok if r["n_labelled"] > 0]
    mean_recall = (
        sum(r["detection_recall"] for r in with_recall) / len(with_recall)
        if with_recall else 0.0
    )
    zero_det = [r for r in ok if r["n_detected"] == 0]
    all_bananas = [b for r in ok for b in r["bananas"]]
    matches = [b for b in all_bananas if b["ok"]]
    mismatched = [b for b in all_bananas if not b["ok"]]
    spurious = sum(r["spurious"] for r in ok)

    agreement = (len(matches) / len(all_bananas)) if all_bananas else 0.0

    lines += [
        "## Aggregates",
        "",
        "| metric | value |",
        "|---|---|",
        f"| images | {len(ok)} ok / {len(results) - len(ok)} errored |",
        f"| mean detection recall (images with labels) | {mean_recall:.0%} over {len(with_recall)} |",
        f"| images with zero detections | {len(zero_det)} |",
        f"| matched bananas | {len(all_bananas)} |",
        f"| ripeness agreement | {len(matches)} / {len(all_bananas)} ({agreement:.0%}) |",
        f"| spurious detections (no label overlap) | {spurious} |",
        "",
    ]

    # Confusion: label (human) -> pred (mapped live class)
    lines += ["## Confusion (label row × predicted col)", ""]
    if all_bananas:
        labels_seen = sorted({b["label"] for b in all_bananas} | {"unlabelled"})
        preds_seen = sorted({b["pred_mapped"] for b in all_bananas} | {"unlabelled"})
        header = "| label \\ pred | " + " | ".join(preds_seen) + " |"
        lines += [header, "|" + "---|" * (len(preds_seen) + 1)]
        for lab in labels_seen:
            row = [str(sum(1 for b in all_bananas if b["label"] == lab and b["pred_mapped"] == p))
                   for p in preds_seen]
            lines.append(f"| {lab} | " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "## Mismatches (matched banana whose class disagrees)",
        "",
    ]
    if mismatched:
        lines += [
            "| image | # | label | pred | G% | Y% | B% | det conf |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for b in mismatched:
            # find the image this row came from
            img_name = next(r["image"] for r in ok if b in r["bananas"])
            lines.append(
                f"| {img_name[:30]} | {b['truth_idx']} | {b['label']} | {b['pred']} "
                f"| {b['green']} | {b['yellow']} | {b['brown']} | {b['conf']:.2f} |"
            )
    else:
        lines.append("_None — every matched banana agreed._")
    lines.append("")

    if zero_det:
        lines += [
            "## Zero-detection images",
            "",
        ]
        for r in zero_det:
            n = f"{r['n_labelled']} labelled" if r["n_labelled"] else "unlabelled"
            lines.append(f"- `{r['image']}` ({n})")
        lines.append("")

    lines += [
        "## Per-image detail",
        "",
        "| image | labels | dets | matched | recall | spurious | sec |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        rec = f"{r['detection_recall']:.0%}" if r["detection_recall"] is not None else "—"
        lines.append(
            f"| {r['image'][:40]} | {r['n_labelled']} | {r['n_detected']} "
            f"| {r['n_matched']} | {rec} | {r['spurious']} | {r['seconds']} |"
        )
    for r in results:
        if "error" in r:
            lines.append(f"| {r.get('image', '?')[:40]} | — | — | — | — | — | ERROR: {r['error'][:40]} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="run every valid/+test/ image instead of the 20-image batch")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    detector = BananaDetector(confidence_threshold=AUTO_CONF, device=args.device)
    items = pick_all() if args.all else pick_batch(limit=args.limit)
    print(f"Running {len(items)} image(s) with {detector.model_label}")
    results = run_batch(items, detector)
    write_report(results, REPORT_PATH, args.limit if not args.all else len(items))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
