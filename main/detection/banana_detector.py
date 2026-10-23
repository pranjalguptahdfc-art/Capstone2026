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

# COCO class index for "banana"
COCO_BANANA_CLASS = 46

# CLAHE settings to enhance dark/overripe banana visibility
_CLAHE_CLIP = 2.0
_CLAHE_TILE = (8, 8)

# Default path relative to this file's project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "banana_detector_single.pt"
FALLBACK_MODEL = _PROJECT_ROOT / "models" / "yolov8n.pt"
SINGLE_MODEL_PATH = DEFAULT_MODEL_PATH  # alias for test compatibility


class BananaDetector:
    """
    Wraps a YOLOv8 model for banana detection.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to `banana_detector.pt`.  If the file does not exist the generic
        YOLOv8n COCO model is used as a development fallback.
    confidence_threshold : float
        Minimum detection confidence (0–1).  Default: 0.20.
    device : str
        Inference device: 'cpu', '0', '1', etc.  Default: 'cpu'.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.20,
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
    # Image pre-processing
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Enhance contrast so dark/overripe bananas are more visible to YOLO."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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

        Uses two passes: (1) original image at the requested confidence,
        then (2) a contrast-enhanced version at lower confidence if the
        first pass finds nothing — this helps detect dark/overripe bananas.

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

        # Pass 1: original image at requested confidence (banana class only)
        detections = self._run_inference(image, conf)

        # Pass 2: contrast-enhanced image at lower confidence
        if not detections:
            logger.info("Pass 1 found nothing. Trying contrast-enhanced image...")
            enhanced = self._preprocess(image)
            detections = self._run_inference(enhanced, min(conf, 0.15))

        # Pass 3: last resort — try ALL COCO classes on the original image
        # A very dark/rotten single banana might not match the banana class,
        # but could still be detected as a generic object
        if not detections:
            logger.info("Pass 2 found nothing. Trying all-class fallback...")
            detections = self._run_inference_all_classes(image, 0.05)

        logger.debug(f"Detected {len(detections)} banana(s).")
        return detections

    def _run_inference(
        self,
        image: np.ndarray,
        conf: float,
    ) -> List[Dict[str, Any]]:
        """Run a single YOLO inference pass and return filtered detections."""
        img_h, img_w = image.shape[:2]

        results = self._model(
            image,
            conf=conf,
            iou=0.4,
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

                if self._is_fallback and cls_idx != COCO_BANANA_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, min(int(x1), img_w - 1))
                y1 = max(0, min(int(y1), img_h - 1))
                x2 = max(0, min(int(x2), img_w - 1))
                y2 = max(0, min(int(y2), img_h - 1))

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

        return detections

    def _run_inference_all_classes(
        self,
        image: np.ndarray,
        conf: float,
    ) -> List[Dict[str, Any]]:
        """Last-resort: detect any object, filter by shape/aspect ratio of a banana."""
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        results = self._model(
            image,
            conf=conf,
            iou=0.3,
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
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                area = w * h

                # Banana-shaped objects: aspect ratio 0.3–3.0, 2–40% of image area
                aspect = w / h if h > 0 else 0
                if not (0.3 <= aspect <= 3.0 and 0.02 <= area / img_area <= 0.4):
                    continue

                x1 = max(0, min(int(x1), img_w - 1))
                y1 = max(0, min(int(y1), img_h - 1))
                x2 = max(0, min(int(x2), img_w - 1))
                y2 = max(0, min(int(y2), img_h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "banana_id": banana_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box.conf[0].item()), 4),
                    }
                )
                banana_id += 1

        return detections

    def __repr__(self) -> str:
        return (
            f"BananaDetector("
            f"model={self._model_label!r}, "
            f"conf={self.confidence_threshold}, "
            f"device={self.device!r})"
        )
