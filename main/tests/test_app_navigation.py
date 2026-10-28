"""
test_app_navigation.py
---------------------
Regression tests for the "View Analytics" button crash
(`StreamlitWidgetAlreadyInstantiatedError` — a widget's session_state key
was written *after* that widget was instantiated in the same run) and for
the navigation fix (button on_click callback, which runs before the next
script pass re-instantiates the nav radio).

Run:
    python main/tests/test_app_navigation.py
    (pytest also collects these; the end-to-end test loads the YOLO
    detector, so it's the slow one — ~30s cold.)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # == main/
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest
from streamlit.errors import StreamlitWidgetAlreadyInstantiatedError

APP_FILE = PROJECT_ROOT / "app.py"


def test_post_widget_session_state_write_raises():
    """Documents the crash mechanism: writing a widget's session_state key
    after the widget was instantiated in the same run raises
    `StreamlitWidgetAlreadyInstantiatedError`."""
    code = """
import streamlit as st
st.radio("Navigation", ["A", "B"], key="app_nav")
if st.button("go", key="btn_go"):
    st.session_state["app_nav"] = "B"
    st.rerun()
"""
    at = AppTest.from_string(code)
    at.run()
    at.button(key="btn_go").set_value(True)
    at.run()
    assert len(at.exception) == 1
    # In 1.63 the testing API surfaces the error message string.
    msg = at.exception[0].value
    if isinstance(msg, str):
        needle = "cannot be modified after the widget with key `app_nav`"
    else:
        needle = "cannot be modified after the widget with key `app_nav`"
        msg = str(msg)
    assert needle in msg, f"expected the widget-already-instantiated error, got: {msg}"


def test_on_click_callback_nav_pattern_is_safe():
    """The fix pattern: an `on_click` callback runs *before* the next
    script pass re-instantiates the radio, so writing the radio's key
    there is legal and the navigation completes in one run."""
    code = """
import streamlit as st

def _select_analytics():
    st.session_state["app_nav"] = "B"

st.radio("Navigation", ["A", "B"], key="app_nav")
st.button("go", key="btn_goto", on_click=_select_analytics)
"""
    at = AppTest.from_string(code)
    at.run()
    assert at.radio(key="app_nav").value == "A"
    at.button(key="btn_goto").set_value(True)
    at.run()
    assert not at.exception
    assert at.radio(key="app_nav").value == "B"


def test_app_view_analytics_button_end_to_end():
    """Real app: upload the far-bunch image, click 'View Analytics'.

    Before the fix this raised `StreamlitWidgetAlreadyInstantiatedError`
    (app.py wrote `st.session_state["app_nav"]` after the radio had been
    instantiated). After: no exception and the radio lands on Analytics.
    """
    at = AppTest.from_file(str(APP_FILE), default_timeout=300)
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]

    image_bytes = (PROJECT_ROOT.parent / "images.jfif").read_bytes()
    at.file_uploader[0].set_value(("images.jfif", image_bytes, "image/jpeg"))
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]

    btn = at.button(key="btn_goto_analytics")
    assert btn, "View Analytics button was not rendered"
    btn.set_value(True)
    at.run()
    assert not at.exception, [repr(e.value) for e in at.exception]
    assert at.radio(key="app_nav").value == "📊 Analytics"


def _main():
    failed = []
    for name, fn in [
        ("test_post_widget_session_state_write_raises",
         test_post_widget_session_state_write_raises),
        ("test_on_click_callback_nav_pattern_is_safe",
         test_on_click_callback_nav_pattern_is_safe),
    ]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}")
    # The end-to-end test loads the YOLO detector; run it standalone.
    import pytest
    sys.exit(pytest.main([__file__, "::test_app_view_analytics_button_end_to_end", "-v"]))


if __name__ == "__main__":
    _main()
