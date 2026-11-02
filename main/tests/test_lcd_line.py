"""
test_lcd_line.py
----------------
Unit tests for hardware.lcd_line.build_lcd_lines — the pure function that
turns a pipeline bunch result (+ optional ESP32 telemetry) into the two
≤16-char ASCII lines the ESP32's 16×2 I2C LCD will flash for 3 seconds.

Run:
    python main/tests/test_lcd_line.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.lcd_line import LCD_LINE_MAX, build_lcd_lines, truncate  # noqa: E402

assert LCD_LINE_MAX == 16


def _bunch(status, signal=None):
    b = {"bunch_status": status}
    if signal is not None:
        b["color_signal"] = signal
    return b


def test_high_risk_overripe_line():
    l1, l2 = build_lcd_lines(_bunch("HIGH RISK", "overripe"))
    assert l1 == "HIGH RISK"
    assert l2 == "Discard now"


def test_high_risk_non_overripe_line():
    l1, l2 = build_lcd_lines(_bunch("HIGH RISK", "ripe"))
    assert l1 == "HIGH RISK"
    assert l2 == "High risk, act!"


def test_semi_ripe_line():
    l1, l2 = build_lcd_lines(_bunch("SEMI-RIPE", "semi-ripe"))
    assert l1 == "SEMI-RIPE"
    assert l2 == "Ripe in ~1 day"


def test_acceptable_ripe_line():
    l1, l2 = build_lcd_lines(_bunch("ACCEPTABLE", "ripe"))
    assert l1 == "ACCEPTABLE"
    assert l2 == "Ripe - eat now"


def test_acceptable_unripe_default_line():
    l1, l2 = build_lcd_lines(_bunch("ACCEPTABLE", "unripe"))
    assert l1 == "ACCEPTABLE"
    assert l2 == "Ripe in ~3 days"


def test_acceptable_unripe_uses_telemetry_time_to_ripen():
    telemetry = {"time_to_ripen": "3 to 5 Days"}
    l1, l2 = build_lcd_lines(_bunch("ACCEPTABLE", "unripe"), telemetry)
    assert l1 == "ACCEPTABLE"
    assert l2 == "Ripe in 3-5 days"


def test_acceptable_unripe_telemetry_just_ripe():
    _, l2 = build_lcd_lines(
        _bunch("ACCEPTABLE", "unripe"), {"time_to_ripen": "0 Days (Just Ripe)"}
    )
    assert l2 == "Ripe - eat now"


def test_acceptable_unripe_telemetry_overripe():
    _, l2 = build_lcd_lines(
        _bunch("ACCEPTABLE", "unripe"), {"time_to_ripen": "Overripe (Consume Immediately)"}
    )
    assert l2 == "Discard now"


def test_unknown_status_line():
    l1, _ = build_lcd_lines(_bunch("UNKNOWN"))
    assert l1 == "NO BANANAS"


def test_missing_status_defaults_to_no_bananas():
    l1, _ = build_lcd_lines({})
    assert l1 == "NO BANANAS"


def test_lines_are_ascii_and_within_16_chars():
    """Even a long, odd telemetry value can never break the 16×2 LCD."""
    bunch = _bunch("ACCEPTABLE", "unripe")
    telemetry = {"time_to_ripen": "Sehr lange Beschreibung des Ripenings 123456789"}
    l1, l2 = build_lcd_lines(bunch, telemetry)
    for line in (l1, l2):
        assert isinstance(line, str)
        assert len(line) <= LCD_LINE_MAX, f"line too long for LCD: {line!r}"
        assert all(ord(c) < 128 for c in line), f"non-ASCII on LCD: {line!r}"


def test_truncate_caps_length_at_16():
    assert len(truncate("X" * 30)) == LCD_LINE_MAX
    assert truncate("ok") == "ok"
    assert truncate("") == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
