import sys
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# ── Windows c10.dll preload (must run BEFORE any pandas import) ─────────────
# On this machine (Windows, torch 2.14.0+cpu) the OpenMP runtimes clash:
# if pandas loads first, the later ultralytics→torch import crashes with
# "OSError [WinError 1114] ... c10.dll". Below, analytics_view (line ~20)
# imports pandas at module top; the detector loads torch later, so torch
# must be registered first. Loading torch here (before pandas) makes the
# app's import order the safe one. Same fix as main/tests/conftest.py.
try:
    import torch  # noqa: F401
except Exception:
    pass  # torch absent: detector will surface the real import error

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import BananaDetector
from vision.color_analyzer import analyze_batch, annotate_colors
from fusion.fusion_engine import FusionEngine
from sensor.gas_reader import GasReader, ETHYLENE_LOW, ETHYLENE_HIGH, ETHYLENE_CRITICAL
from utils.image_utils import annotate_image, cv2_to_pil, pil_to_cv2
from analytics_view import render_analytics_page
from utils.analytics_store import log_inspection, compute_ripeness_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Banana Ripeness & Quality Analytics", page_icon="🍌", layout="wide")


CANONICAL_DISPLAY = {
    "unripe":     (" UNRIPE",   "#27ae60"),
    "ripe":       (" RIPE",     "#f39c12"),
    "semi-ripe":  (" SEMI-RIPE","#2980b9"),
    "overripe":   (" OVERRIPE", "#e74c3c"),
    "unknown":    (" UNKNOWN",  "#7f8c8d"),
}

RISK_COLORS = {
    "ACCEPTABLE":  "#27ae60",
    "SEMI-RIPE":   "#2980b9",
    "HIGH RISK":   "#e74c3c",
    "UNKNOWN":     "#7f8c8d",
}


@st.cache_resource(show_spinner="Loading YOLOv8n banana detector...")
def load_detector():
    return BananaDetector(confidence_threshold=0.25)


@st.cache_resource
def load_fusion_engine():
    return FusionEngine()


@st.cache_resource
def load_gas_reader():
    return GasReader(demo=True)


def uploaded_file_to_bgr(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    return pil_to_cv2(pil_img)


def _goto_analytics():
    """Button on_click callback (Streamlit-safe navigation).

    A callback runs in the *pre* phase, before the next script pass
    re-instantiates the nav radio — so writing the radio's session_state key
    here is legal. Writing `st.session_state["app_nav"]` directly from inside
    the button's `if st.button(...)` block (i.e. after the radio was already
    instantiated in the same run) raises
    `StreamlitWidgetAlreadyInstantiatedError`.
    """
    st.session_state["app_nav"] = "📊 Analytics"


def main():
    with st.sidebar:
        st.markdown("## 🍌 Banana Vision AI")
        page = st.radio(
            "Navigation",
            ["🍌 Ripeness Detector", "📊 Analytics"],
            key="app_nav",
        )
        st.divider()

    if page == "📊 Analytics":
        render_analytics_page()
        return

    st.title("🍌 Banana Bunch Ripeness Detector")
    st.caption(
        "Detect a banana bunch, classify ripeness via color analysis, "
        "and compute the overall bunch status. No models to train."
    )

    with st.sidebar:
        st.header("Settings")
        det_mode = st.radio(
            "Detection mode", ["Auto", "Manual"], index=0,
            help="Auto adapts to each image (recommended). Manual uses your threshold as a floor.",
        )
        if det_mode == "Auto":
            conf_threshold, _floor = 0.10, False
            st.caption("Adaptive: detects all bananas, then removes low-confidence duplicates.")
        else:
            conf_threshold = st.slider(
                "Confidence floor", min_value=0.05, max_value=0.90,
                value=0.20, step=0.05,
            )
            _floor = True

        st.divider()
        st.subheader("Gas Sensor")
        use_gas = st.checkbox("Enable gas sensor", value=False)
        gas_mode = "demo"
        serial_port = None
        if use_gas:
            gas_mode = st.radio("Gas source", ["Demo (simulated)", "ESP32 serial"])
            if "ESP32" in gas_mode:
                serial_port = st.text_input("Serial port", value="COM3")

        st.divider()
        st.subheader("Model Status")
        try:
            detector = load_detector()
            if detector.is_fallback:
                st.warning(
                    " Detector: YOLOv8n COCO (pre-built)\n\n"
                    "Using generic COCO model (class 46 = banana).\n"
                    "No custom training needed."
                )
            else:
                st.success(f" Detector: {detector.model_label}")
        except Exception as e:
            st.error(f"Detector failed: {e}")
            st.stop()

        gas_reader = load_gas_reader()
        if not gas_reader.demo:
            gas_reader = GasReader(demo=False, serial_port=serial_port)

        st.divider()
        st.caption("How it works:")
        st.caption("1. YOLOv8 detects the bunch (adaptive: finds all, removes duplicates)")
        st.caption("2. Banana-mask isolates the fruit, ignores the background")
        st.caption("3. HSV color classifies each banana")
        st.caption("4. Fusion engine computes bunch verdict")

        st.divider()
        st.caption("Image tips:")
        st.caption("Clear, well-lit photos work best.")
        st.caption("Full bunch in frame = better accuracy.")

    st.subheader("Upload a banana bunch image")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "jfif", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Please upload a banana bunch image (JPG / JPEG / JFIF / PNG).")
        return

    try:
        bgr_image = uploaded_file_to_bgr(uploaded)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    img_h, img_w = bgr_image.shape[:2]

    with st.spinner("Detecting bananas..."):
        try:
            detections = detector.detect(
                bgr_image,
                confidence_threshold=conf_threshold,
                floor=_floor,
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return

    if not detections:
        st.warning("**No bananas detected.**\n\nTry another image or lower confidence.")
        return

    with st.spinner("Analyzing ripeness via color..."):
        color_analyses = analyze_batch(bgr_image, detections)

    gas_reading = None
    if use_gas:
        with st.spinner("Reading gas sensor..."):
            gas_reading = gas_reader.read()

    engine = load_fusion_engine()
    if gas_reading:
        bunch = engine.evaluate(gas_reading, color_analyses)
    else:
        bunch = _color_only_result(color_analyses)

    annotated_bgr = annotate_colors(bgr_image, detections, color_analyses)

    # ── Bunch-level result (primary view) ──────────────────────────────────
    st.subheader("Bunch Result")
    b_status = bunch["bunch_status"]
    b_color = RISK_COLORS.get(b_status, "#7f8c8d")
    n_bananas = len(detections)

    n_colors = len(color_analyses)
    total_green = sum(a["green_pct"] for a in color_analyses) / n_colors if n_colors > 0 else 0
    total_yellow = sum(a["yellow_pct"] for a in color_analyses) / n_colors if n_colors > 0 else 0
    total_brown = sum(a["brown_pct"] for a in color_analyses) / n_colors if n_colors > 0 else 0

    stage, avg_ripeness_pct, decision = compute_ripeness_metrics(
        avg_green=total_green,
        avg_yellow=total_yellow,
        avg_brown=total_brown,
        bunch_status=b_status,
        color_signal=bunch.get("color_signal"),
    )

    dec_color = "#27ae60" if decision == "Approved" else "#e74c3c"
    dec_icon = "✅" if decision == "Approved" else "❌"

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    bcol1.markdown(
        f"<div style='text-align:center; padding:16px; border-radius:8px; "
        f"background:{dec_color}15; border:2px solid {dec_color};'>"
        f"<div style='font-size:12px; color:#666;'>DECISION</div>"
        f"<div style='font-size:24px; font-weight:bold; color:{dec_color};'>{dec_icon} {decision.upper()}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    bcol2.markdown(
        f"<div style='text-align:center; padding:16px; border-radius:8px; "
        f"background:{b_color}15; border:2px solid {b_color};'>"
        f"<div style='font-size:12px; color:#666;'>STATUS</div>"
        f"<div style='font-size:24px; font-weight:bold; color:{b_color};'>{b_status}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    bcol3.markdown(
        f"<div style='text-align:center; padding:16px; border-radius:8px; "
        f"background:#f8f9fa; border:2px solid #ddd;'>"
        f"<div style='font-size:12px; color:#666;'>BANANAS DETECTED</div>"
        f"<div style='font-size:24px; font-weight:bold; color:#333;'>{n_bananas}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    bcol4.markdown(
        f"<div style='text-align:center; padding:16px; border-radius:8px; "
        f"background:#fffbeb; border:2px solid #f59e0b;'>"
        f"<div style='font-size:12px; color:#666;'>AVG RIPENESS</div>"
        f"<div style='font-size:24px; font-weight:bold; color:#b45309;'>{avg_ripeness_pct}%</div>"
        f"<div style='font-size:11px; color:#92400e; font-weight:600;'>{stage.upper()}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Automated logging to 30-day analytics store ────────────────────────
    log_key = f"{uploaded.name}_{uploaded.size}_{n_bananas}_{b_status}_{avg_ripeness_pct}"
    if st.session_state.get("last_logged_key") != log_key:
        gas_ppm = gas_reading["ethylene_ppm"] if gas_reading else 0.0
        log_inspection(
            bunch_id=uploaded.name,
            banana_count=n_bananas,
            avg_green=total_green,
            avg_yellow=total_yellow,
            avg_brown=total_brown,
            bunch_status=b_status,
            confidence=bunch["confidence"],
            dominant_problem=bunch["dominant_problem"],
            reasoning=bunch["reasoning"],
            gas_ppm=gas_ppm,
            color_signal=bunch.get("color_signal"),
        )
        st.session_state["last_logged_key"] = log_key

    log_c1, _ = st.columns([3, 1])
    with log_c1:
        st.success(
            f"✓ Inspection record timestamped & logged to 30-day analytics database! "
            f"(Decision: **{decision}**, Stage: **{stage}**, Ripeness: **{avg_ripeness_pct}%**)"
        )

    # "View Analytics" navigation, centered under the stat cards. Uses an
    # on_click callback (the Streamlit-safe way to switch pages — see
    # _goto_analytics) so the radio's session_state key is written in the
    # pre-rerun phase, not after the radio is already instantiated.
    _cl, _cm, _cr = st.columns([3, 2, 3])
    with _cm:
        st.button(
            "📊 View Analytics",
            key="btn_goto_analytics",
            on_click=_goto_analytics,
            width="stretch",
        )

    st.info(f"**Reasoning:** {bunch['reasoning']}")
    st.markdown(f"**Dominant Problem:** {bunch['dominant_problem']}")

    # ── Images side by side ────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2_to_pil(bgr_image), width="stretch")
        st.caption(f"Size: {img_w} x {img_h} px")

    with col2:
        st.subheader("Annotated Result")
        st.image(cv2_to_pil(annotated_bgr), width="stretch")
        if detector.is_fallback:
            st.caption("Using pre-built YOLOv8n COCO model (no custom training).")

    # ── Per-banana breakdown (expandable) ──────────────────────────────────
    st.markdown("---")
    with st.expander(f"Individual Banana Breakdown ({n_bananas} bananas)", expanded=True):
        rows = []
        for det, analysis in zip(detections, color_analyses):
            canonical = analysis.get("ripeness_class", "unknown")
            rows.append({
                "Banana": f"#{det['banana_id']}",
                "Ripeness": canonical.upper(),
                "Green": f"{analysis['green_pct']}%",
                "Yellow": f"{analysis['yellow_pct']}%",
                "Brown": f"{analysis['brown_pct']}%",
            })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), width="content", hide_index=True)

    # ── Bunch-level metrics ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Bunch-Level Analysis")
    n = len(color_analyses)
    total_green = sum(a["green_pct"] for a in color_analyses) / n
    total_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
    total_brown = sum(a["brown_pct"] for a in color_analyses) / n

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Bananas", n)
    m2.metric("Avg Green", f"{total_green:.1f}%")
    m3.metric("Avg Yellow", f"{total_yellow:.1f}%")
    m4.metric("Avg Brown", f"{total_brown:.1f}%")

    st.subheader("Color Breakdown")
    _render_color_bar(total_green, total_yellow, total_brown)

    # ── Gas sensor reading ─────────────────────────────────────────────────
    if gas_reading:
        st.markdown("---")
        st.subheader("Gas Sensor Reading")
        g1, g2, g3 = st.columns(3)
        g1.metric("Ethylene", f"{gas_reading['ethylene_ppm']} ppm")
        g2.metric("Raw ADC", gas_reading["raw_adc"])
        g3.metric("Status", gas_reading["status"])
        st.caption(
            f"Thresholds: LOW < {ETHYLENE_LOW} ppm | "
            f"MODERATE < {ETHYLENE_HIGH} ppm | "
            f"HIGH < {ETHYLENE_CRITICAL} ppm | "
            f"CRITICAL >= {ETHYLENE_CRITICAL} ppm"
        )

    # ── Download ───────────────────────────────────────────────────────────
    annotated_pil = cv2_to_pil(annotated_bgr)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="Download annotated image",
        data=buf.getvalue(),
        file_name=f"annotated_{uploaded.name}",
        mime="image/png",
    )


def _color_only_result(color_analyses):
    n = len(color_analyses)
    if n == 0:
        return {
            "bunch_status": "UNKNOWN", "gas_signal": "N/A", "gas_ppm": 0.0,
            "color_signal": "unknown", "dominant_problem": "No bananas detected",
            "confidence": "LOW", "reasoning": "No bananas detected.",
            "gas_reading": {},
            "color_analysis": {"avg_green_pct": 0, "avg_yellow_pct": 0, "avg_brown_pct": 0, "banana_count": 0},
        }
    avg_green = sum(a["green_pct"] for a in color_analyses) / n
    avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
    avg_brown = sum(a["brown_pct"] for a in color_analyses) / n

    # If any individual bananas are overripe, the bunch is at risk
    overripe_count = sum(1 for a in color_analyses if a["ripeness_class"] == "overripe")
    semi_ripe_count = sum(1 for a in color_analyses if a["ripeness_class"] == "semi-ripe")

    if overripe_count > 0:
        status, confidence, reasoning = "HIGH RISK", "MEDIUM", f"{overripe_count}/{n} bananas overripe - approaching spoilage."
    elif semi_ripe_count / n >= 0.20:
        status, confidence, reasoning = "SEMI-RIPE", "MEDIUM", f"{semi_ripe_count}/{n} bananas mid-ripening - still maturing."
    elif avg_yellow > 50:
        status, confidence, reasoning = "ACCEPTABLE", "MEDIUM", f"Mostly yellow ({avg_yellow:.1f}%) - ripe."
    elif avg_green > 50:
        status, confidence, reasoning = "ACCEPTABLE", "MEDIUM", f"Mostly green ({avg_green:.1f}%) - unripe."
    else:
        status, confidence, reasoning = "ACCEPTABLE", "LOW", "Mixed ripeness levels."

    return {
        "bunch_status": status, "gas_signal": "N/A", "gas_ppm": 0.0,
        "color_signal": _bunch_color_signal(color_analyses),
        "dominant_problem": _color_problem(avg_green, avg_yellow, avg_brown),
        "confidence": confidence,
        "reasoning": reasoning + " (color only, no gas sensor).",
        "gas_reading": {},
        "color_analysis": {
            "avg_green_pct": round(avg_green, 1),
            "avg_yellow_pct": round(avg_yellow, 1),
            "avg_brown_pct": round(avg_brown, 1),
            "banana_count": n,
        },
    }


def _color_problem(green, yellow, brown):
    if brown > 20: return f"High brown ({brown:.1f}%)"
    if green > 15 and yellow > 30: return "Mid-ripening - green fading to yellow"
    if yellow > 50 and green > 30: return "Mixed ripeness - uneven"
    if green > 70: return "Mostly unripe"
    if yellow > 70: return "Mostly ripe"
    if brown > 10: return f"Some browning ({brown:.1f}%)"
    return "No significant problems"


def _bunch_color_signal(color_analyses):
    """Signal based on per-banana classification counts, not averages."""
    n = len(color_analyses)
    if n == 0: return "unknown"
    overripe = sum(1 for a in color_analyses if a["ripeness_class"] == "overripe")
    semi_ripe = sum(1 for a in color_analyses if a["ripeness_class"] == "semi-ripe")
    unripe = sum(1 for a in color_analyses if a["ripeness_class"] == "unripe")
    if overripe > 0: return "overripe"
    if semi_ripe / n >= 0.20: return "semi-ripe"
    if semi_ripe / n >= 0.10: return "semi-ripe"
    avg_yellow = sum(a["yellow_pct"] for a in color_analyses) / n
    if avg_yellow > 50: return "ripe"
    return "unripe"


def _render_color_bar(green, yellow, brown):
    bar_html = '<div style="display:flex; height:30px; border-radius:6px; overflow:hidden; border:1px solid #ccc;">'
    bar_html += f'<div style="width:{green}%; background:#27ae60;" title="Green: {green}%"></div>'
    bar_html += f'<div style="width:{yellow}%; background:#f1c40f;" title="Yellow: {yellow}%"></div>'
    bar_html += f'<div style="width:{brown}%; background:#e74c3c;" title="Brown: {brown}%"></div>'
    bar_html += '</div>'
    bar_html += '<div style="display:flex; justify-content:space-between; margin-top:4px; font-size:12px;">'
    bar_html += f'<span style="color:#27ae60;"> Green {green:.1f}%</span>'
    bar_html += f'<span style="color:#f1c40f;"> Yellow {yellow:.1f}%</span>'
    bar_html += f'<span style="color:#e74c3c;"> Brown {brown:.1f}%</span>'
    bar_html += '</div>'
    st.markdown(bar_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
