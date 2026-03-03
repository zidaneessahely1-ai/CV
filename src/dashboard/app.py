"""Streamlit dashboard for the YOLO Vision Agent.

Launch with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.agent.vision_agent import VisionAgent
from src.detection.detector import YOLODetector
from src.utils.config import load_config
from src.utils.image_processing import draw_detections

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="YOLO Vision Agent", page_icon="🔍", layout="wide")
st.title("🔍 YOLO Vision Agent Dashboard")
st.caption("Powered by YOLOv11-L · Real-time object detection & analysis")


# ── Cached singletons ───────────────────────────────────────────────
@st.cache_resource
def get_detector():
    cfg = load_config()
    return YOLODetector(config=cfg.model)


@st.cache_resource
def get_agent():
    cfg = load_config()
    return VisionAgent(config=cfg.agent)


detector = get_detector()
agent = get_agent()


# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    show_boxes = st.checkbox("Show bounding boxes", value=True)
    st.divider()

    st.header("📊 Agent Summary")
    if st.button("🔄 Refresh summary"):
        st.rerun()
    summary = agent.summary()
    st.json(summary)


# ── Main area ────────────────────────────────────────────────────────
tab_upload, tab_webcam, tab_about = st.tabs(
    ["📤 Upload Image", "📷 Webcam", "ℹ️ About"]
)

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        img_array = np.array(pil_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Running YOLOv11-L detection…"):
            result = detector.detect(bgr, conf=conf_thresh)
            alerts = agent.process(result)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.subheader(f"Detections ({result.count})")
            if show_boxes and result.detections:
                annotated = draw_detections(
                    bgr, [d.to_dict() for d in result.detections]
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            else:
                st.image(pil_img, use_container_width=True)

        st.metric("Inference time", f"{result.inference_time_ms:.1f} ms")
        st.subheader("Class counts")
        st.json(result.class_counts())

        if alerts:
            st.subheader("🚨 Alerts")
            for a in alerts:
                st.warning(a.message)

with tab_webcam:
    st.info(
        "Webcam support requires a running browser environment. "
        "Use the **API** endpoint ``POST /detect`` for programmatic access."
    )

with tab_about:
    st.markdown(
        """
        ### YOLO Vision Agent

        This dashboard is powered by **YOLOv11-L** (Large) from
        [Ultralytics](https://docs.ultralytics.com/).

        **Features**
        - Real-time object detection with 80 COCO classes
        - AI agent with alert rules, zone monitoring, and trend tracking
        - REST API for integration
        - Dockerised deployment

        **Architecture**
        ```
        Image → YOLOv11-L Detector → DetectionResult → VisionAgent → Alerts / Summary
        ```
        """
    )
