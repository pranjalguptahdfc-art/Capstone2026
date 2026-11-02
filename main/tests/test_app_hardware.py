"""
test_app_hardware.py
-------------------
App-level tests for the Hardware (ESP32) block in app.py.

- test_app_hardware_block_renders : the sidebar hardware controls exist
- test_app_fetch_offline_fails_soft : fetching from an unreachable ESP32 shows
  a friendly "offline" warning instead of crashing the app

The full fetch→analyze→verdict happy path is covered end-to-end by
``main/tests/verify_hardware_loop.py`` (against the real simulator), not here.

Run:
    python main/tests/test_app_hardware.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = PROJECT_ROOT / "app.py"


def _run_app_with_hardware(address: str, click_fetch: bool = True):
    at = AppTest.from_file(str(APP_FILE), default_timeout=300)
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]
    # Enable the hardware block and point it at an address we control.
    at.checkbox(key="use_hardware").set_value(True)
    at.text_input(key="hw_address").set_value(address)
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]
    if click_fetch:
        at.button(key="btn_fetch_hardware").set_value(True)
        at.run()
        assert not at.exception, [repr(e.value) for e in at.exception]
    return at


def test_app_hardware_block_renders():
    at = AppTest.from_file(str(APP_FILE), default_timeout=300)
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]
    assert at.checkbox(key="use_hardware"), "hardware enable checkbox missing"
    assert at.text_input(key="hw_address"), "hardware address input missing"
    assert at.button(key="btn_fetch_hardware"), "fetch-from-hardware button missing"


def test_app_fetch_offline_fails_soft():
    """An unreachable ESP32 must not raise — the app shows an offline note."""
    at = _run_app_with_hardware("127.0.0.1:1", click_fetch=True)
    text = "\n".join(str(m.value) for m in at.warning)
    assert "offline" in text.lower(), f"expected an offline warning, got: {text!r}"


def _main():
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    _main()
