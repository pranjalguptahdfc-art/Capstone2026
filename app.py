"""
app.py
------
🍌 Banana Bunch Ripeness Detector — Streamlit UI

Run:
    streamlit run app.py

from inside:
    D:\\Pranjal Files\\Capstone2026\\Banana-Ripeness-Bunch\\
with the .venv activated.
"""

import sys
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Path setup (ensure project root is on PYTHONPATH)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.banana_detector import BananaDetector
from classification.ripeness_classifier import RipenessClassifier
from processing.bunch_analyzer import BunchAnalyzer
from utils.image_utils import annotate_image, crop_banana, cv2_to_pil, pil_to_cv2

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍌 Banana Bunch Ripeness Detector",
    page_icon="🍌",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.40
PADDING = 0.05

CANONICAL_DISPLAY: dict = {
    "unripe":   ("🟢 UNRIPE",   "#27ae60"),
    "ripe":     ("🟡 RIPE",     "#f39c12"),
    "overripe": ("🔴 OVERRIPE", "#e74c3c"),
    "unknown":  ("❓ UNKNOWN",  "#7f8c8d"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Cached model loading — runs ONCE per session
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLOv8 banana detector …")
def load_detector(confidence: float = DEFAULT_CONFIDENCE) -> BananaDetector:
    """Load the YOLOv8 banana detector (once per Streamlit session)."""
    detector = BananaDetector(confidence_threshold=confidence)
    return detector


@st.cache_resource(show_spinner="Loading ripeness classifier …")
def load_classifier() -> RipenessClassifier:
    """Load the EfficientNetB0 ripeness classifier (once per session)."""
    classifier = RipenessClassifier()
    if classifier.is_available():
        try:
            classifier.load()
        except Exception as e:
            st.warning(
                f"⚠️ Ripeness classifier failed to load:\n{e}\n\n"
                "The detector will still run. Provide `models/banana_ripeness.h5` to enable classification."
            )
    return classifier


@st.cache_resource
def load_analyzer() -> BunchAnalyzer:
    """Load the bunch analyzer (once per session)."""
    return BunchAnalyzer()


# ──────────────────────────────────────────────────────────────────────────────
# Helper — convert uploaded file to OpenCV BGR array
# ──────────────────────────────────────────────────────────────────────────────

def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Convert a Streamlit uploaded file object to a BGR numpy array."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    bgr = pil_to_cv2(pil_img)
    return bgr


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🍌 Banana Bunch Ripeness Detector")
    st.caption(
        "Detect individual bananas in a bunch image, classify each banana's ripeness, "
        "and compute the overall bunch ripeness."
    )

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        conf_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            help="Minimum YOLO confidence to accept a banana detection.",
        )
        st.divider()
        st.subheader("📋 Model Status")

        # Load models
        try:
            detector = load_detector(DEFAULT_CONFIDENCE)
            if detector.is_fallback:
                st.warning(
                    "🟡 **Detector:** YOLOv8n COCO FALLBACK\n\n"
                    "Using generic model (class 46 = banana).\n"
                    "Place `banana_detector.pt` in `models/` for the banana-specific detector."
                )
            else:
                st.success(f"🟢 **Detector:** {detector.model_label}")
        except Exception as e:
            st.error(f"❌ Detector failed to load:\n{e}")
            st.stop()

        try:
            classifier = load_classifier()
            if classifier.is_available() and classifier._loaded:
                st.success("🟢 **Classifier:** EfficientNetB0 loaded")
            elif classifier.is_available():
                st.warning("🟡 **Classifier:** File found, not yet loaded")
            else:
                st.error(
                    "❌ **Classifier:** `models/banana_ripeness.h5` not found\n\n"
                    "Download from the vgry5 repository and place it in `models/`."
                )
        except Exception as e:
            st.error(f"❌ Classifier error:\n{e}")

        analyzer = load_analyzer()

        st.divider()
        st.caption("**Class mapping (confirmed from repository):**")
        st.code("0 → Over Ripe\n1 → Ripe\n2 → Unripe", language=None)

    # ── Upload ───────────────────────────────────────────────────────────────
    st.subheader("📤 Upload a banana bunch image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Please upload a banana bunch image (JPG / JPEG / PNG).")
        return

    # ── Load image ───────────────────────────────────────────────────────────
    try:
        bgr_image = uploaded_file_to_bgr(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read the uploaded image: {e}")
        return

    img_h, img_w = bgr_image.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(cv2_to_pil(bgr_image), use_container_width=True)
        st.caption(f"Size: {img_w} × {img_h} px")

    # ── Detection ────────────────────────────────────────────────────────────
    with st.spinner("🔍 Detecting bananas …"):
        try:
            detections = detector.detect(bgr_image, confidence_threshold=conf_threshold)
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")
            return

    if not detections:
        st.warning(
            "⚠️ **No bananas detected.**\n\n"
            "Try another image or lower the detection confidence threshold in the sidebar."
        )
        return

    st.success(f"✅ **{len(detections)} banana(s) detected.**")

    # ── Classification ───────────────────────────────────────────────────────
    banana_results = []
    classifier_available = classifier.is_available() and classifier._loaded

    with st.spinner("🧠 Classifying banana ripeness …"):
        for det in detections:
            result_entry = {
                "banana_id":      det["banana_id"],
                "bbox":           det["bbox"],
                "confidence":     det["confidence"],
                "ripeness_class": "N/A",
                "canonical":      None,
                "ripe_confidence": 0.0,
            }

            if classifier_available:
                try:
                    crop = crop_banana(bgr_image, det["bbox"], padding=PADDING)
                    pred = classifier.predict(crop)
                    result_entry["ripeness_class"]  = pred["ripeness_class"]
                    result_entry["canonical"]        = pred["canonical"]
                    result_entry["ripe_confidence"]  = pred["confidence"]
                except Exception as e:
                    result_entry["ripeness_class"] = "Error"
                    logger.warning(f"Classification error for banana {det['banana_id']}: {e}")

            banana_results.append(result_entry)

    # ── Annotated image ──────────────────────────────────────────────────────
    with col2:
        st.subheader("🎯 Detection Result")
        preds_for_annotation = banana_results if classifier_available else None
        annotated_bgr = annotate_image(bgr_image, detections, preds_for_annotation)
        st.image(cv2_to_pil(annotated_bgr), use_container_width=True)
        if detector.is_fallback:
            st.caption("⚠️ Using generic COCO YOLOv8n fallback detector.")

    # ── Per-banana results table ─────────────────────────────────────────────
    st.subheader("🍌 Individual Banana Results")

    if classifier_available:
        import pandas as pd
        rows = []
        for b in banana_results:
            canonical = b.get("canonical")
            emoji = CANONICAL_DISPLAY.get(canonical or "unknown", ("❓", ""))[0].split()[0]
            rows.append({
                "Banana #":             b["banana_id"],
                "Det. Confidence":      f"{b['confidence'] * 100:.1f}%",
                "Ripeness":             b["ripeness_class"],
                "Ripe. Confidence":     f"{b['ripe_confidence'] * 100:.1f}%" if b["ripe_confidence"] else "—",
                "Status":               emoji,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "ℹ️ Ripeness classifier not available. "
            "Provide `models/banana_ripeness.h5` to enable per-banana classification."
        )
        import pandas as pd
        rows = [
            {
                "Banana #":         b["banana_id"],
                "Det. Confidence":  f"{b['confidence'] * 100:.1f}%",
                "BBox":             str(b["bbox"]),
            }
            for b in banana_results
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Bunch analysis ───────────────────────────────────────────────────────
    if classifier_available:
        st.subheader("📊 Bunch-Level Analysis")

        bunch = analyzer.analyze(banana_results)

        # ── Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Bananas", bunch["total_bananas"])
        m2.metric("🟢 Unripe",   f"{bunch['percentages']['unripe']:.1f}%",
                  f"{bunch['counts']['unripe']} bananas")
        m3.metric("🟡 Ripe",     f"{bunch['percentages']['ripe']:.1f}%",
                  f"{bunch['counts']['ripe']} bananas")
        m4.metric("🔴 Overripe", f"{bunch['percentages']['overripe']:.1f}%",
                  f"{bunch['counts']['overripe']} bananas")

        # ── Problematic Bananas Analysis (PRIMARY RESULT)
        st.markdown("---")
        st.subheader("⚠️ Bunch Risk Assessment")

        # Problematic bananas metrics
        p1, p2 = st.columns(2)
        p1.metric("Problematic Bananas",
                  f"{bunch['problematic_count']} ({bunch['problematic_percentage']:.1f}%)")

        # Bunch status with color coding
        status = bunch["bunch_status"]
        if status == "HIGH RISK":
            status_color = "#e74c3c"  # Red
            status_emoji = "🚨"
        else:
            status_color = "#27ae60"  # Green
            status_emoji = "✅"

        p2.metric("Bunch Status", f"{status_emoji} {status}")

        # Dominant problem
        st.info(f"🔍 **Dominant Problem**: {bunch['dominant_problem']}")

        # ── Score & result (Existing methods as supporting information)
        st.markdown("---")
        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Aggregation Methods (Supporting Info)**")
            st.write(f"• Majority Vote: **{bunch['majority_result'].title()}**")
            st.write(f"• Average Ripeness Score: **{bunch['avg_ripeness_score']:.3f}** (Unripe=0, Ripe=1, Overripe=2)")
            st.write(f"  → Score classification: **{bunch['score_result'].title()}**")
            st.caption("Note: Official bunch status is now based on problematic banana percentage.")

        with a2:
            final = bunch["final_result"]
            display_label, color = CANONICAL_DISPLAY.get(
                final, ("❓ UNKNOWN", "#7f8c8d")
            )
            st.markdown(
                f"""
                <div style="
                    background: {color}22;
                    border: 3px solid {color};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                ">
                    <p style="font-size:14px; color: #aaa; margin:0;">CLASSIFICATION (Majority Vote)</p>
                    <h1 style="color: {color}; margin: 8px 0 0 0; font-size: 2.5rem;">
                        {display_label}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif not classifier_available:
        st.info(
            "📊 Bunch analysis is disabled until the ripeness classifier model is loaded.\n\n"
            "**To enable:**\n"
            "1. Download `EfficientnetBo.h5` from the vgry5 GitHub repository.\n"
            "2. Place it at `models/banana_ripeness.h5`.\n"
            "3. Restart the Streamlit app."
        )

    # ── Save annotated image (optional) ──────────────────────────────────────
    annotated_pil = cv2_to_pil(annotated_bgr)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name=f"annotated_{uploaded.name}",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
