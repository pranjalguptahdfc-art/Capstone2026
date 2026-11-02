"""
lcd_line.py
-----------
Build the two short lines the ESP32's 16×2 I2C LCD flashes for a verdict.

Pure function — no I/O, deterministic, fully unit-tested in
``main/tests/test_lcd_line.py``. The firmware displays whatever it receives
from the ``POST /verdict`` endpoint (see ``docs/esp32-hardware-integration.md``);
truncation happens *here* so the LCD can never show a clipped word.
"""

import re

# The I2C LCD (PCF8574-based, LiquidCrystal_I2C 16×2 in Config.h) fits
# exactly this many characters per line.
LCD_LINE_MAX = 16

_ASCII_FALLBACK_LINE = "No data"


def truncate(line: str, max_len: int = LCD_LINE_MAX) -> str:
    """Cap a line at ``max_len`` characters (hard truncation, no ellipsis)."""
    return line[:max_len]


def _time_to_ripen_line(time_to_ripen) -> str | None:
    """Map the firmware's time_to_ripen string to an LCD-short line.

    Returns None when it can't be expressed in 16 chars / is unrecognized,
    in which case the caller falls back to the color-only estimate.
    """
    if not time_to_ripen or not isinstance(time_to_ripen, str):
        return None
    t = time_to_ripen.strip()
    low = t.lower()
    if "overripe" in low or "immediately" in low:
        return "Discard now"
    if "just ripe" in low:
        return "Ripe - eat now"
    m = re.search(r"(\d+)\s*to\s*(\d+)", t)
    if m:
        return f"Ripe in {m.group(1)}-{m.group(2)} days"
    m = re.search(r"(\d+)\s*day", t, re.IGNORECASE)
    if m:
        return f"Ripe in {m.group(1)} days"
    return None


def build_lcd_lines(bunch: dict, telemetry: dict | None = None) -> tuple[str, str]:
    """Turn a pipeline bunch result (+ optional ESP32 telemetry) into
    ``(line1, line2)`` — each ≤ :data:`LCD_LINE_MAX` chars, ASCII-only.

    ``bunch`` is the dict the app pipeline already produces
    (``bunch_status``, ``color_signal``). ``telemetry`` is the parsed
    ``/download_csv`` row (or None for the no-sensor path); its
    ``time_to_ripen`` refines the line for unripe bunches.
    """
    status = (bunch or {}).get("bunch_status", "UNKNOWN")
    signal = (bunch or {}).get("color_signal")

    # ── Line 1: the verdict, at a glance ───────────────────────────────────
    if status == "HIGH RISK":
        line1 = "HIGH RISK"
    elif status == "SEMI-RIPE":
        line1 = "SEMI-RIPE"
    elif status == "ACCEPTABLE":
        line1 = "ACCEPTABLE"
    else:
        line1 = "NO BANANAS"

    # ── Line 2: the action ────────────────────────────────────────────────
    if status == "HIGH RISK":
        line2 = "Discard now" if signal == "overripe" else "High risk, act!"
    elif status == "SEMI-RIPE":
        line2 = "Ripe in ~1 day"
    elif status == "ACCEPTABLE":
        if signal == "ripe":
            line2 = "Ripe - eat now"
        elif signal == "overripe":
            line2 = "Discard now"
        else:
            # unripe / mixed: prefer the hardware's time estimate, else the
            # color-only default (a cold-ish, stable bunch).
            timed = _time_to_ripen_line((telemetry or {}).get("time_to_ripen"))
            line2 = timed or "Ripe in ~3 days"
    else:
        line2 = "No bananas seen"

    return truncate(line1), truncate(line2)
