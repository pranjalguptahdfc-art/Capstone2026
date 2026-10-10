"""
bunch_analyzer.py
-----------------
Non-teachable bunch-level ripeness analyzer.

No ML model. Uses the FusionEngine to combine:
  - Gas sensor reading (optional)
  - Color analysis from vision/color_analyzer.py

This replaces the old BunchClassifier (removed).
"""

from typing import Optional

from vision.color_analyzer import analyze_batch, annotate_colors
from fusion.fusion_engine import FusionEngine
from sensor.gas_reader import GasReader


class BunchAnalyzer:
    """Non-teachable bunch analyzer using color + gas fusion."""

    def __init__(self, use_gas: bool = False):
        self.use_gas = use_gas
        self._engine = FusionEngine()
        self._gas_reader = GasReader(demo=True) if use_gas else None

    def analyze_image(self, image_bgr, detections: list) -> dict:
        """
        Full analysis pipeline for one image.

        Parameters
        ----------
        image_bgr : np.ndarray
            BGR image.
        detections : list[dict]
            Banana detections from banana_detector.py.

        Returns
        -------
        dict with detections, color_analyses, annotated_image, bunch_verdict,
             gas_reading (optional).
        """
        # Color analysis
        color_analyses = analyze_batch(image_bgr, detections)

        # Gas reading
        gas_reading = None
        if self.use_gas and self._gas_reader:
            gas_reading = self._gas_reader.read()

        # Fusion
        if gas_reading:
            bunch = self._engine.evaluate(gas_reading, color_analyses)
        else:
            # Build a fake gas reading for color-only mode
            fake_gas = {"ethylene_ppm": 0.0, "raw_adc": 0, "status": "N/A", "timestamp": ""}
            bunch = self._engine._no_data_result(fake_gas)
            bunch = self._color_only_fallback(color_analyses)

        # Annotated image
        annotated = annotate_colors(image_bgr, detections, color_analyses)

        return {
            "detections": detections,
            "color_analyses": color_analyses,
            "annotated_image": annotated,
            "bunch_verdict": bunch,
            "gas_reading": gas_reading,
        }

    def _color_only_fallback(self, color_analyses):
        n = len(color_analyses)
        if n == 0:
            return {"bunch_status": "UNKNOWN", "confidence": "LOW"}

        avg_green = sum(a["green_pct"] for a in color_analyses) / n
        avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
        avg_brown = sum(a["brown_pct"] for a in color_analyses) / n

        if avg_brown > 20:
            return {"bunch_status": "HIGH RISK", "confidence": "MEDIUM",
                    "reasoning": f"High brown ({avg_brown:.1f}%) - overripe."}
        elif avg_yellow > 50:
            return {"bunch_status": "ACCEPTABLE", "confidence": "MEDIUM",
                    "reasoning": f"Mostly yellow ({avg_yellow:.1f}%) - ripe."}
        elif avg_green > 50:
            return {"bunch_status": "ACCEPTABLE", "confidence": "MEDIUM",
                    "reasoning": f"Mostly green ({avg_green:.1f}%) - unripe."}
        return {"bunch_status": "ACCEPTABLE", "confidence": "LOW", "reasoning": "Mixed levels."}
