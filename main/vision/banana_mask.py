"""
banana_mask.py
--------------
Background-resilient banana extraction for the ripeness pipeline.

The detector gives a bounding box that also contains background (the market
stand, crate, other bananas). This module isolates the banana BODY — the solid
blob that touches the box center — so color analysis can ignore the background.

Pure OpenCV post-processing. No model, no training.

The isolation windows are derived from the single color_model source of
truth (green + warm families, V floors dropped to MIN_V so dim bananas
are still seen). This keeps the mask and the ripeness analysis in the
same color space, so one banana can never be "masked as body" and then
"counted as background" by a different threshold set.
"""

import cv2
import numpy as np

from .color_model import GREEN_H, WARM_H, MIN_S, MIN_V, PALE_LOWER, PALE_UPPER

# Per-family windows used only for BODY ISOLATION (not counting):
# each window is generous on V (dim to bright), so lighting does not
# decide whether the body is seen.
_WINDOWS = [
    # green family
    (np.array([GREEN_H[0], MIN_S, MIN_V], dtype=np.uint8),
     np.array([GREEN_H[1], 255, 255], dtype=np.uint8)),
    # warm family (yellow + brown)
    (np.array([WARM_H[0], MIN_S, MIN_V], dtype=np.uint8),
     np.array([WARM_H[1], 255, 255], dtype=np.uint8)),
]


def banana_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask (banana=255) for the banana body inside a crop.

    Per color window (green / warm), keep only the connected component
    that touches the crop CENTER. The banana body sits in the middle of
    its own box; background of the same color family (a green stand, a
    yellow shelf) forms a ring reaching the edges, not the center, so
    it is dropped. The kept components are unioned.

    Windows are derived from color_model (green + warm families) with
    generous V floors so dim or bright bodies are both seen. If no
    window touches the center (overexposed / whitened / all-pale body),
    fall back to the pale last-resort window.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cy, cx = h // 2, w // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    union = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in _WINDOWS:
        m = cv2.inRange(hsv, lower, upper)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        n, labels = cv2.connectedComponents(m)
        if n <= 1:
            continue  # no pixels of this color family at all
        c = int(labels[cy, cx])
        if c > 0:  # this family touches the center -> part of the banana
            union = cv2.bitwise_or(union, (labels == c).astype(np.uint8) * 255)

    if int(np.count_nonzero(union)) == 0:
        union = cv2.inRange(hsv, PALE_LOWER, PALE_UPPER)
    return union


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


def center_crop(crop_bgr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Return the central `frac` fraction of a crop in each axis.

    Used as the last-resort fallback when the banana mask is solid/invalid:
    the banana body dominates the center of its own box while same-colored
    background (stands, crates) frames the edges.
    """
    h, w = crop_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    rh = max(1, int(h * frac) // 2)
    rw = max(1, int(w * frac) // 2)
    out = crop_bgr[cy - rh:cy + rh + 1, cx - rw:cx + rw + 1]
    return out if out.size else crop_bgr


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
