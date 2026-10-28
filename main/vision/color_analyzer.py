"""
color_analyzer.py
-----------------
HSV color analysis for banana ripeness detection.

No ML model — just OpenCV color thresholding.
Analyzes each detected banana region and classifies it as green (unripe),
yellow (ripe), or brown (overripe) based on pixel percentages.

Color families are exclusive and use a lighting-invariant, relative
brightness gate (see vision/color_model.py): a pixel is never counted in
more than one family, and the brown/yellow split is relative to the crop's
own brightness (p95 of V), so classification stays stable under dim or
warm lighting.
"""

import cv2
import numpy as np
from typing import Optional

from .color_model import compute_v_ref, family_counts


# ── HSV thresholds ──────────────────────────────────────────────────────────

HSV_RANGES = {
    "green": {
        "lower": np.array([35, 30, 30], dtype=np.uint8),
        "upper": np.array([85, 255, 255], dtype=np.uint8),
    },
    "yellow": {
        "lower": np.array([15, 30, 80], dtype=np.uint8),
        "upper": np.array([35, 255, 255], dtype=np.uint8),
    },
    "brown": {
        "lower": np.array([0, 10, 20], dtype=np.uint8),
        "upper": np.array([25, 200, 200], dtype=np.uint8),
    },
}

COLOR_TO_RIPENESS = {
    "green": "unripe",
    "yellow": "ripe",
    "brown": "overripe",
    "semi-ripe": "semi-ripe",
}

# When the banana mask can't isolate the body (the crop is solid banana-colored
# or the mask is invalid), analysis falls back to the central core of the box.
# A banana occupies the center of its own bounding box; a same-coloured
# background (market stand, crate) typically frames the edges. Restricting to
# this core keeps those edge pixels out of the ripeness math.
CENTER_FRAC = 0.6


def analyze_region(
    image_bgr: np.ndarray,
    bbox: list[int],
) -> dict:
    """
    Analyze a single banana region for color distribution.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full BGR image.
    bbox : list[int]
        [x1, y1, x2, y2] bounding box.

    Returns
    -------
    dict with keys:
        green_pct, yellow_pct, brown_pct, other_pct,
        dominant_color, ripeness_class, pixel_counts
    """
    x1, y1, x2, y2 = bbox
    region = image_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return _empty_result()

    from .banana_mask import banana_crop, banana_mask, center_crop

    # Isolate the banana body so background colors (stand, crate, other
    # bananas) do not bleed into the HSV analysis.
    analyzed = banana_crop(region)
    background_suppressed = bool(analyzed.size > 0 and analyzed.size < region.size)

    if background_suppressed:
        # Banana-colored mask cleanly isolated the body — use it.
        hsv = cv2.cvtColor(analyzed, cv2.COLOR_BGR2HSV)
        pixel_mask = banana_mask(analyzed)
        total = int(np.count_nonzero(pixel_mask))
    else:
        # Mask is solid or invalid (e.g. banana and background share a color,
        # or the box is mostly banana). Fall back to the central core so
        # same-colored background framing the edges is excluded.
        analyzed = center_crop(region, CENTER_FRAC)
        hsv = cv2.cvtColor(analyzed, cv2.COLOR_BGR2HSV)
        pixel_mask = None
        total = analyzed.shape[0] * analyzed.shape[1]
    if total == 0:
        empty = _empty_result()
        empty["background_suppressed"] = background_suppressed
        return empty

    counts = family_counts(hsv, pixel_mask, compute_v_ref(hsv, pixel_mask))

    green_pct = round(counts["green"] / total * 100, 1)
    yellow_pct = round(counts["yellow"] / total * 100, 1)
    brown_pct = round(counts["brown"] / total * 100, 1)
    other_pct = round(100 - green_pct - yellow_pct - brown_pct, 1)

    # Brown spots dominate: if >20%, it's overripe regardless of green/yellow
    if brown_pct > 20:
        dominant_color = "brown"
        ripeness_class = "overripe"
    # Both green and yellow are significant → mid-ripening (semi-ripe)
    elif green_pct > 20 and yellow_pct > 35:
        dominant_color = "semi-ripe"
        ripeness_class = "semi-ripe"
    else:
        # Exclusive families: pick the most-present COLOR family only
        # (counts also carries "total", which must not be considered).
        dominant_color = max(
            ("green", "yellow", "brown"), key=lambda c: counts[c]
        )
        ripeness_class = COLOR_TO_RIPENESS.get(dominant_color, "unknown")

    return {
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "other_pct": other_pct,
        "dominant_color": dominant_color,
        "ripeness_class": ripeness_class,
        "pixel_counts": {
            "green": counts["green"],
            "yellow": counts["yellow"],
            "brown": counts["brown"],
            "banana_pixels": total,
            "total_analyzed": total,
        },
        "background_suppressed": background_suppressed,
    }


def analyze_batch(
    image_bgr: np.ndarray,
    detections: list[dict],
) -> list[dict]:
    """
    Analyze color for each detected banana region.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full BGR image.
    detections : list[dict]
        Each dict has "bbox": [x1, y1, x2, y2].

    Returns
    -------
    list of dict — one analysis result per detection.
    """
    results = []
    for det in detections:
        analysis = analyze_region(image_bgr, det["bbox"])
        analysis["banana_id"] = det.get("banana_id", len(results) + 1)
        results.append(analysis)
    return results


def annotate_colors(
    image_bgr: np.ndarray,
    detections: list[dict],
    analyses: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes with color-based labels on the image.

    Box colors: green=unripe, yellow=ripe, red=overripe.
    """
    annotated = image_bgr.copy()
    BGR_COLORS = {
        "unripe":     (0, 200, 0),
        "ripe":       (0, 200, 255),
        "semi-ripe":  (255, 165, 0),
        "overripe":   (0, 60, 220),
        "unknown":    (200, 200, 200),
    }

    for i, (det, analysis) in enumerate(zip(detections, analyses)):
        x1, y1, x2, y2 = det["bbox"]
        cls = analysis.get("ripeness_class", "unknown")
        color = BGR_COLORS.get(cls, BGR_COLORS["unknown"])

        thick = max(2, int(min(image_bgr.shape[:2]) * 0.003))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        label = f"#{det.get('banana_id', i+1)} {cls.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(image_bgr.shape[:2]) * 0.001)
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)

        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                    font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def _empty_result() -> dict:
    return {
        "green_pct": 0.0,
        "yellow_pct": 0.0,
        "brown_pct": 0.0,
        "other_pct": 100.0,
        "dominant_color": "other",
        "ripeness_class": "unknown",
        "pixel_counts": {"green": 0, "yellow": 0, "brown": 0,
                         "banana_pixels": 0, "total_analyzed": 0},
        "background_suppressed": False,
    }
