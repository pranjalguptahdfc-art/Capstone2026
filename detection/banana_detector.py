"""
banana_detector.py
------------------
YOLOv8 banana detection wrapper.

Model loading priority:
  1. models/banana_detector.pt  (banana-specific YOLOv8 — final model)
  2. yolov8n.pt                 (COCO generic fallback — DEVELOPMENT ONLY)

When using the COCO fallback, only detections with class index 46 (banana)
are returned.  The fallback is clearly labelled in logs and the Streamlit UI.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# COCO class index for "banana"
COCO_BANANA_CLASS = 46

# Default path relative to this file's project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "banana_detector.pt"
FALLBACK_MODEL = "yolov8n.pt"


class BananaDetector:
    """
    Wraps a YOLOv8 model for banana detection.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to `banana_detector.pt`.  If the file does not exist the generic
        YOLOv8n COCO model is used as a development fallback.
    confidence_threshold : float
        Minimum detection confidence (0–1).  Default: 0.40.
    device : str
        Inference device: 'cpu', '0', '1', etc.  Default: 'cpu'.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.40,
        device: str = "cpu",
    ):
        # Resolve model path
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        model_path = Path(model_path)

        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._is_fallback = False
        self._model_label = ""

        self._load_model(model_path)

    # ──────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────
    def _load_model(self, model_path: Path) -> None:
        """Load either the banana-specific model or the COCO fallback."""
        from ultralytics import YOLO  # lazy import — ultralytics is heavy

        if model_path.exists():
            logger.info(f"Loading banana-specific detector: {model_path}")
            self._model = YOLO(str(model_path), task="detect")
            self._is_fallback = False
            self._model_label = f"Banana-specific YOLOv8 ({model_path.name})"
        else:
            warning_msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  WARNING: Using generic YOLOv8n COCO model.             ║\n"
                "║  This is a DEVELOPMENT FALLBACK.                        ║\n"
                "║  Performance is NOT equivalent to banana_detector.pt.   ║\n"
                "║  Place banana_detector.pt in models/ to use the         ║\n"
                "║  banana-specific detector.                               ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
            logger.warning(warning_msg)
            self._model = YOLO(FALLBACK_MODEL, task="detect")
            self._is_fallback = True
            self._model_label = "YOLOv8n COCO FALLBACK (class 46 = banana)"

        logger.info(f"Detector ready: {self._model_label}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    @property
    def is_fallback(self) -> bool:
        """True when the generic COCO model is being used."""
        return self._is_fallback

    @property
    def model_label(self) -> str:
        """Human-readable label describing which detector is loaded."""
        return self._model_label

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run banana detection on an image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (as returned by cv2.imread).
        confidence_threshold : float, optional
            Override the instance-level threshold for this call.

        Returns
        -------
        list of dict
            Each dict:
            {
                "banana_id"  : int,
                "bbox"       : [x1, y1, x2, y2],   # pixel coords, clipped
                "confidence" : float,                # 0–1
            }
        """
        if self._model is None:
            raise RuntimeError("Detector model is not loaded.")

        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        img_h, img_w = image.shape[:2]

        results = self._model(
            image,
            conf=conf,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        banana_id = 1

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_idx = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())

                # When using the COCO fallback, only keep banana detections
                if self._is_fallback and cls_idx != COCO_BANANA_CLASS:
                    continue

                # Raw bounding box (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Clip to image boundaries
                x1 = max(0, min(int(x1), img_w - 1))
                y1 = max(0, min(int(y1), img_h - 1))
                x2 = max(0, min(int(x2), img_w - 1))
                y2 = max(0, min(int(y2), img_h - 1))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "banana_id": banana_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(conf_score, 4),
                    }
                )
                banana_id += 1

        logger.debug(f"Detected {len(detections)} banana(s).")
        return detections

    def __repr__(self) -> str:
        return (
            f"BananaDetector("
            f"model={self._model_label!r}, "
            f"conf={self.confidence_threshold}, "
            f"device={self.device!r})"
        )
