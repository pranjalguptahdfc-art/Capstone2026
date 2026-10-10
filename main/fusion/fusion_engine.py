from collections import Counter
from typing import Any


class FusionEngine:
    """Combines gas + color signals into bunch-level ripeness verdict."""

    def __init__(self, brown_risk_threshold=20.0, yellow_ok_threshold=50.0):
        self.brown_risk_threshold = brown_risk_threshold
        self.yellow_ok_threshold = yellow_ok_threshold

    def evaluate(self, gas_reading, color_analyses):
        gas_status = gas_reading["status"]
        gas_ppm = gas_reading["ethylene_ppm"]
        n = len(color_analyses)
        if n == 0:
            return self._no_data_result(gas_reading)

        avg_green = sum(a["green_pct"] for a in color_analyses) / n
        avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
        avg_brown = sum(a["brown_pct"] for a in color_analyses) / n
        color_signal = self._color_signal(avg_green, avg_yellow, avg_brown)

        if gas_status == "LOW":
            verdict, confidence, reasoning = self._low_gas_fuse(avg_green, avg_yellow, avg_brown, color_signal)
        elif gas_status in ("HIGH", "CRITICAL"):
            verdict, confidence, reasoning = self._high_gas_fuse(avg_green, avg_yellow, avg_brown, color_signal, gas_status)
        else:
            verdict, confidence, reasoning = self._moderate_gas_fuse(avg_green, avg_yellow, avg_brown)

        return {
            "bunch_status": verdict, "gas_signal": gas_status, "gas_ppm": gas_ppm,
            "color_signal": color_signal,
            "dominant_problem": self._dominant_problem(avg_green, avg_yellow, avg_brown),
            "confidence": confidence, "reasoning": reasoning,
            "gas_reading": gas_reading,
            "color_analysis": {"avg_green_pct": round(avg_green, 1), "avg_yellow_pct": round(avg_yellow, 1), "avg_brown_pct": round(avg_brown, 1), "banana_count": n},
        }

    def _color_signal(self, green, yellow, brown):
        if brown > 20:
            return "overripe"
        if green > 15 and yellow > 30:
            return "semi-ripe"
        return max([("unripe", green), ("ripe", yellow), ("overripe", brown)], key=lambda x: x[1])[0]

    def _low_gas_fuse(self, green, yellow, brown, signal):
        if signal == "overripe":
            return "ACCEPTABLE", "HIGH", "Low ethylene - overripe is stable, not actively ripening further."
        elif signal == "semi-ripe":
            return "SEMI-RIPE", "MEDIUM", "Low ethylene - mid-ripening, stable at semi-ripe stage."
        elif signal == "ripe":
            return "ACCEPTABLE", "HIGH", "Low ethylene - bananas are ripe and stable."
        else:
            return "ACCEPTABLE", "HIGH", "Low ethylene - bananas are unripe but not actively ripening."

    def _high_gas_fuse(self, green, yellow, brown, signal, gas_status):
        if brown >= self.brown_risk_threshold:
            return "HIGH RISK", "HIGH", f"{gas_status} ethylene with {brown:.1f}% brown - ripening fast, near spoilage."
        elif signal == "semi-ripe":
            return "SEMI-RIPE", "MEDIUM", f"{gas_status} ethylene - semi-ripe bananas ripening quickly, monitor closely."
        elif yellow >= self.yellow_ok_threshold:
            return "HIGH RISK", "MEDIUM", f"{gas_status} ethylene with {yellow:.1f}% yellow - will overripen soon."
        else:
            return "HIGH RISK", "LOW", f"{gas_status} ethylene - surface may look {signal}, but internal ripening is active."

    def _moderate_gas_fuse(self, green, yellow, brown):
        if brown > 20:
            return "HIGH RISK", "MEDIUM", f"Moderate ethylene with {brown:.1f}% brown - approaching overripe."
        elif abs(green - yellow) <= 15 and max(green, yellow) > 30:
            return "SEMI-RIPE", "MEDIUM", "Moderate ethylene - mid-ripening process active."
        elif yellow > 50:
            return "ACCEPTABLE", "MEDIUM", f"Moderate ethylene - normal ripening ({yellow:.1f}% yellow)."
        elif green > 50:
            return "ACCEPTABLE", "MEDIUM", f"Moderate ethylene - still mostly unripe ({green:.1f}% green)."
        else:
            return "ACCEPTABLE", "LOW", "Mixed colors with moderate ethylene."

    def _dominant_problem(self, green, yellow, brown):
        if brown > 20: return f"High brown ({brown:.1f}%)"
        if green > 15 and yellow > 30: return "Balanced green/yellow - mid-ripening"
        if yellow > 50 and green > 30: return "Mixed ripeness - uneven"
        if green > 70: return "Mostly unripe"
        if yellow > 70: return "Mostly ripe"
        if brown > 10: return f"Some browning ({brown:.1f}%)"
        return "No significant problems"

    def _no_data_result(self, gas_reading):
        return {
            "bunch_status": "UNKNOWN", "gas_signal": gas_reading["status"],
            "gas_ppm": gas_reading["ethylene_ppm"], "color_signal": "unknown",
            "dominant_problem": "No color data available", "confidence": "LOW",
            "reasoning": "No bananas detected.", "gas_reading": gas_reading,
            "color_analysis": {"avg_green_pct": 0, "avg_yellow_pct": 0, "avg_brown_pct": 0, "banana_count": 0},
        }
