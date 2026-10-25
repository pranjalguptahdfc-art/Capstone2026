"""
banana_mask.py
--------------
Background-resilient banana extraction for the ripeness pipeline.

The detector gives a bounding box that also contains background (the market
stand, crate, other bananas). This module isolates the banana BODY — the solid
blob that touches the box center — so color analysis can ignore the background.

Pure OpenCV post-processing. No model, no training.
"""

import cv2
import numpy as np

from .color_analyzer import HSV_RANGES

# Last-resort window for banana colors the three HSV ranges miss
# (very pale / white / over-bleached bananas).
_LAST_RESORT_LOWER = np.array([0, 0, 180], dtype=np.uint8)
_LAST_RESORT_UPPER = np.array([179, 20, 255], dtype=np.uint8)


def banana_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (banana=255) for the banana body inside a crop.

    The banana is the solid blob that touches the crop center; background
    typically touches the crop edges. If no banana-colored pixels exist,
    fall back to a low-saturation/high-value window.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for _name, rng in HSV_RANGES.items():
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, rng["lower"], rng["upper"]))
    if np.count_nonzero(mask) == 0:
        mask = cv2.inRange(hsv, _LAST_RESORT_LOWER, _LAST_RESORT_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels = cv2.connectedComponents(mask)
    if n_labels <= 1:
        return mask
    h, w = labels.shape
    cy, cx = h // 2, w // 2
    best = _center_component(labels, cy, cx)
    if best is None:
        return mask
    return (labels == best).astype(np.uint8) * 255


def _center_component(labels, cy, cx):
    """Label of the largest component touching the crop center, else None."""
    cands = {int(labels[cy, cx])}
    cands.discard(0)
    if not cands:
        return None
    sizes = np.bincount(labels.flatten())
    return max(cands, key=lambda c: sizes[c])


def is_valid_mask(mask: np.ndarray, min_ratio: float = 0.10) -> bool:
    """True when a component touches the crop center and covers >= min_ratio."""
    if mask.size == 0:
        return False
    n, labels = cv2.connectedComponents(mask)
    if n <= 1:
        return bool(np.count_nonzero(mask) / mask.size >= min_ratio)
    h, w = labels.shape
    center = labels[h // 2, w // 2]
    if center == 0:
        return False
    return np.count_nonzero(labels == center) / mask.size >= min_ratio


def is_solid_mask(mask: np.ndarray) -> bool:
    """True when the mask covers most of the crop (no background to remove)."""
    return mask.size > 0 and np.count_nonzero(mask) / mask.size >= 0.95


def banana_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Shrink a crop to its mask bounding rect when the mask is valid.

    Falls back to the original crop when the mask is solid or invalid, so the
    analyzer never receives an empty region.
    """
    mask = banana_mask(crop_bgr)
    if not is_valid_mask(mask) or is_solid_mask(mask):
        return crop_bgr
    coords = np.where(mask > 0)
    y1, y2 = coords[0].min(), coords[0].max() + 1
    x1, x2 = coords[1].min(), coords[1].max() + 1
    out = crop_bgr[y1:y2, x1:x2]
    if out.size == 0:
        return crop_bgr
    return out
