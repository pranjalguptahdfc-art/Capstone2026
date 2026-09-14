"""
image_utils.py
--------------
Image utility functions for the banana ripeness pipeline.

Functions
---------
crop_banana        — Crop a detected banana from the full image with padding.
resize_for_classifier — Resize a crop to the classifier's input dimensions.
annotate_image     — Draw bounding boxes and ripeness labels on the image.
load_image_cv2     — Load an image from a file path as a BGR numpy array.
cv2_to_pil         — Convert a BGR numpy array to a PIL RGB Image.
pil_to_cv2         — Convert a PIL RGB Image to a BGR numpy array.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette for ripeness classes (BGR for OpenCV)
# ──────────────────────────────────────────────────────────────────────────────
RIPENESS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "unripe":   (0,   200,   0),    # green
    "ripe":     (0,   200, 255),    # yellow (BGR)
    "overripe": (0,    60, 220),    # red
}
_DEFAULT_COLOR_BGR = (200, 200, 200)  # grey for unknown


def _get_color(canonical: Optional[str]) -> Tuple[int, int, int]:
    if canonical is None:
        return _DEFAULT_COLOR_BGR
    return RIPENESS_COLORS_BGR.get(canonical.lower().replace(" ", ""), _DEFAULT_COLOR_BGR)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image_cv2(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    np.ndarray
        BGR image.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If OpenCV could not decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def cv2_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB Image."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image to a BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Cropping
# ──────────────────────────────────────────────────────────────────────────────

def crop_banana(
    image: np.ndarray,
    bbox: List[int],
    padding: float = 0.05,
) -> np.ndarray:
    """
    Crop a detected banana from the full image, with optional padding.

    Parameters
    ----------
    image : np.ndarray
        Full BGR image.
    bbox : list of int
        [x1, y1, x2, y2] bounding box in pixel coordinates.
    padding : float
        Fractional padding relative to the bounding-box size.
        0.05 = 5%.  Default: 0.05.

    Returns
    -------
    np.ndarray
        Cropped BGR banana image.

    Raises
    ------
    ValueError
        If the bounding box is invalid or the resulting crop is empty.
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox}")

    # Apply padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    # Clamp to image size
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)

    crop = image[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        raise ValueError(
            f"Resulting crop is empty after applying padding. "
            f"bbox={bbox}, image size={img_w}×{img_h}"
        )

    return crop


def resize_for_classifier(
    crop: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize a banana crop to the classifier's expected input dimensions.

    Parameters
    ----------
    crop : np.ndarray
        BGR banana crop of any size.
    size : tuple of int
        (width, height) target size.  Default: (224, 224).

    Returns
    -------
    np.ndarray
        Resized BGR image of shape (height, width, 3).
    """
    w, h = size
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation
# ──────────────────────────────────────────────────────────────────────────────

def annotate_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    predictions: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and ripeness labels on the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (will be copied — the original is not modified).
    detections : list of dict
        Each dict: {"banana_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}
    predictions : list of dict or None
        Each dict: {"ripeness_class": str, "canonical": str, "confidence": float, ...}
        Must be in the same order as detections.
        If None, only bounding boxes are drawn (no ripeness labels).

    Returns
    -------
    np.ndarray
        Annotated BGR image (a new array — the input is not modified).
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    for i, det in enumerate(detections):
        banana_id = det["banana_id"]
        x1, y1, x2, y2 = det["bbox"]
        det_conf = det["confidence"]

        # Determine colour from prediction (if available)
        canonical = None
        ripe_label = ""
        ripe_conf = 0.0
        if predictions is not None and i < len(predictions):
            pred = predictions[i]
            canonical = pred.get("canonical")
            ripeness_class = pred.get("ripeness_class", "?")
            ripe_conf = pred.get("confidence", 0.0)
            ripe_label = f"{ripeness_class} {ripe_conf * 100:.0f}%"

        color = _get_color(canonical)

        # Draw bounding box
        thickness = max(2, int(min(img_w, img_h) * 0.003))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label
        label_top = f"#{banana_id} det:{det_conf * 100:.0f}%"
        label_bot = ripe_label if ripe_label else "classifying..."

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(img_w, img_h) * 0.001)
        text_thickness = max(1, thickness - 1)

        # Background for readability
        for j, label in enumerate([label_top, label_bot]):
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            ty = y1 - (j + 1) * (th + 4) if y1 - (j + 1) * (th + 6) >= 0 else y2 + (j + 1) * (th + 6)
            cv2.rectangle(
                annotated,
                (x1, ty - th - 2),
                (x1 + tw + 2, ty + baseline),
                color,
                -1,
            )
            # Black text on coloured background
            cv2.putText(
                annotated,
                label,
                (x1 + 1, ty),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA,
            )

    return annotated
