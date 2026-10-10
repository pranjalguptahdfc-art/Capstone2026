"""
ripeness_classifier.py
----------------------
EfficientNetB0-based banana ripeness classifier.

PROPERTIES VERIFIED PROGRAMMATICALLY FROM models/EfficientnetBo.h5:
===================================================================
1. Input Shape:          (None, 224, 224, 3)
2. Output Shape:         (None, 3)
3. Output Activation:    Linear (Logits — softmax applied post-inference)
4. Internal Rescaling:   NONE
5. External Preprocess:  RGB, 224×224, rescale img / 255.0
6. Class Mapping:
       Index 0 → Over Ripe  (canonical: overripe)
       Index 1 → Ripe       (canonical: ripe)
       Index 2 → Unripe     (canonical: unripe)
7. Custom Objects Req:   YES — DepthwiseConv2D legacy 'groups=1' compatibility fix for Keras 3.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants — VERIFIED FROM PROGRAMMATIC INSPECTION & REPOSITORY CODE
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: Dict[int, str] = {
    0: "Over Ripe",
    1: "Ripe",
    2: "Unripe",
}

_CANONICAL_MAP: Dict[str, str] = {
    "Over Ripe": "overripe",
    "Ripe":      "ripe",
    "Unripe":    "unripe",
}

INPUT_SIZE = (224, 224)   # (width, height)
NUM_CLASSES = 3

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "EfficientnetBo.h5"
FALLBACK_MODEL_PATH = _PROJECT_ROOT / "models" / "banana_ripeness.h5"


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from raw logit vector."""
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


class RipenessClassifier:
    """
    EfficientNetB0-based banana ripeness classifier.

    Handles loading Keras .h5 models trained in Keras 2 on Keras 3 runtimes
    by providing a custom DepthwiseConv2D layer that strips legacy parameters.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        input_size: tuple = INPUT_SIZE,
    ):
        if model_path is None:
            if DEFAULT_MODEL_PATH.exists():
                self.model_path = DEFAULT_MODEL_PATH
            else:
                self.model_path = FALLBACK_MODEL_PATH
        else:
            self.model_path = Path(model_path)

        self.input_size = input_size
        self._model = None
        self._loaded = False

    def is_available(self) -> bool:
        """Return True if the model file exists on disk."""
        return self.model_path.exists()

    def load(self) -> "RipenessClassifier":
        """
        Explicitly load the Keras model from disk using custom objects for Keras 3 compatibility.
        """
        if self._loaded:
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Ripeness model not found at: {self.model_path}\n"
                "Please place EfficientnetBo.h5 at models/EfficientnetBo.h5."
            )

        import tensorflow as tf
        from keras.layers import DepthwiseConv2D

        class FixedDepthwiseConv2D(DepthwiseConv2D):
            """Legacy Keras 2 -> Keras 3 compatibility fix for DepthwiseConv2D groups parameter."""
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                config.pop("groups", None)
                return super().from_config(config)

        custom_objects = {"DepthwiseConv2D": FixedDepthwiseConv2D}

        try:
            logger.info(f"Loading ripeness classifier from {self.model_path} …")
            self._model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects=custom_objects,
                compile=False,
            )
            self._loaded = True
            logger.info(f"Classifier loaded successfully from {self.model_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ripeness model from {self.model_path}: {exc}"
            ) from exc

        return self

    @staticmethod
    def preprocess(image_array: np.ndarray, input_size: tuple = INPUT_SIZE) -> np.ndarray:
        """
        Preprocess a crop for model inference:
        1. Resize to input_size (224x224)
        2. Convert color to RGB
        3. Scale pixel values by / 255.0
        4. Add batch dimension (1, 224, 224, 3)
        """
        import cv2

        w, h = input_size
        resized = cv2.resize(image_array, (w, h))

        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        arr = resized.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classify the ripeness of a single banana crop.
        Converts model output logits → probabilities via softmax.
        """
        if not self._loaded:
            self.load()

        processed = self.preprocess(image_crop, self.input_size)
        raw_output = self._model.predict(processed, verbose=0)[0]  # shape (3,)

        # Model outputs raw logits -> convert to probabilities via softmax
        probs = _softmax(raw_output)

        class_index = int(np.argmax(probs))
        confidence = float(probs[class_index])
        ripeness_class = CLASS_NAMES.get(class_index, f"class_{class_index}")
        canonical = _CANONICAL_MAP.get(ripeness_class, ripeness_class.lower())

        return {
            "class_index": class_index,
            "ripeness_class": ripeness_class,
            "canonical": canonical,
            "confidence": round(confidence, 4),
            "all_scores": [round(float(p), 4) for p in probs],
            "raw_logits": [round(float(l), 4) for l in raw_output],
        }

    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Classify a list of banana crops."""
        return [self.predict(crop) for crop in crops]

    def inspect_model(self) -> Dict[str, Any]:
        """Return diagnostic properties of the loaded model."""
        if not self._loaded:
            self.load()

        return {
            "model_path": str(self.model_path),
            "input_shape": tuple(self._model.input_shape) if self._model else None,
            "output_shape": tuple(self._model.output_shape) if self._model else None,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "input_size": self.input_size,
            "output_type": "Logits -> Softmax probabilities",
            "preprocessing": "RGB, 224x224, pixel / 255.0",
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"RipenessClassifier(model={self.model_path.name!r}, status={status!r})"
