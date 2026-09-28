import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.detector import CrowdDetector, DetectionConfig

st.set_page_config(
    page_title="Crowd Detection & Counting System",
    page_icon="👥",
    layout="wide",
)

st.markdown("""
<style>
.main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
.subtitle {color: #666; margin-bottom: 1.2rem;}
.metric-card {padding: 12px; border-radius: 12px; border: 1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">👥 Crowd Detection & Counting System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLO-powered person detection, counting, density analysis and video analytics.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Detection Settings")
    model_name = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt"], index=0)
    confidence = st.slider("Confidence", 0.10, 0.95, 0.40, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.95, 0.45, 0.05)
    show_boxes = st.checkbox("Show bounding boxes", True)
    show_labels = st.checkbox("Show labels", True)
    show_density = st.checkbox("Show density zones", True)
    st.divider()
    st.caption("Tip: yolo11n is faster; yolo11s is generally more accurate but heavier.")

@st.cache_resource
def load_detector(model_name, confidence, iou):
    return CrowdDetector(DetectionConfig(
        model_name=model_name,
        confidence=confidence,
        iou=iou,
    ))

detector = load_detector(model_name, confidence, iou)

mode = st.radio(
    "Input mode",
    ["📷 Image", "🎥 Video File", "📹 Webcam"],
    horizontal=True,
)

if mode == "📷 Image":
    uploaded = st.file_uploader(
        "Upload a crowd image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name

        image = cv2.imread(temp_path)
        result = detector.process_frame(
            image,
            draw=show_boxes,
            labels=show_labels,
            density=show_density,
        )
        st.image(cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("People detected", result.count)
        c2.metric("Density level", result.density_level)
        c3.metric("Average confidence", f"{result.avg_confidence:.1%}")

        if result.count > 0:
            st.dataframe(
                pd.DataFrame(result.records),
                use_container_width=True,
                hide_index=True,
            )
        os.unlink(temp_path)

elif mode == "🎥 Video File":
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".mp4"
        input_path = Path(tempfile.gettempdir()) / f"crowd_input{suffix}"
        input_path.write_bytes(uploaded.getbuffer())

        st.info("Processing video. Larger videos can take several minutes.")
        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()

        output_path = Path(tempfile.gettempdir()) / "crowd_annotated_output.mp4"
        csv_path = Path(tempfile.gettempdir()) / "crowd_analytics.csv"

        summary = detector.process_video(
            str(input_path),
            str(output_path),
            str(csv_path),
            draw=show_boxes,
            labels=show_labels,
            density=show_density,
            progress_callback=lambda p, msg: (
                progress.progress(min(100, int(p))),
                status.write(msg),
            ),
            preview_callback=lambda frame: preview.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            ),
        )

        st.success("Video processing completed.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max people", summary["max_count"])
        c2.metric("Average people", f'{summary["avg_count"]:.1f}')
        c3.metric("Peak density", summary["peak_density"])
        c4.metric("Frames processed", summary["frames_processed"])

        if output_path.exists():
            st.video(output_path.read_bytes())
            st.download_button(
                "⬇️ Download annotated video",
                output_path.read_bytes(),
                file_name="crowd_annotated_output.mp4",
                mime="video/mp4",
            )
        if csv_path.exists():
            st.download_button(
                "⬇️ Download analytics CSV",
                csv_path.read_bytes(),
                file_name="crowd_analytics.csv",
                mime="text/csv",
            )

else:
    st.warning(
        "Webcam mode requires camera permission. For best performance, use a "
        "local browser session rather than a remote Streamlit deployment."
    )
    run_camera = st.button("▶ Start Webcam")
    stop_camera = st.button("⏹ Stop")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if run_camera:
        st.session_state.camera_running = True
    if stop_camera:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        frame_slot = st.empty()
        metric_slot = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam. Check camera permissions and device index.")
        else:
            try:
                while st.session_state.camera_running:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("Camera frame could not be read.")
                        break

                    result = detector.process_frame(
                        frame,
                        draw=show_boxes,
                        labels=show_labels,
                        density=show_density,
                    )
                    frame_slot.image(
                        cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    metric_slot.metric(
                        "People detected",
                        result.count,
                        delta=f"{result.density_level} density",
                    )
            finally:
                cap.release()

st.divider()
st.caption(
    "Built with Python, OpenCV, Ultralytics YOLO and Streamlit. "
    "Use responsibly and follow local privacy laws when processing people."
)
