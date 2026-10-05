"""
Object Counting System
Count people, cars and other objects in images or videos using YOLOv8 + Streamlit.

Run:  streamlit run app.py
"""

import tempfile
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Object Counting System", page_icon="🔢", layout="wide")


@st.cache_resource
def load_model(name: str) -> YOLO:
    # Weights download automatically on first run
    return YOLO(name)


def count_objects(result, allowed_ids, names):
    """Return a Counter of class name -> count for the allowed classes."""
    counts = Counter()
    if result.boxes is None:
        return counts
    for cls_id in result.boxes.cls.cpu().numpy().astype(int):
        if cls_id in allowed_ids:
            counts[names[cls_id]] += 1
    return counts


def annotate(model, frame_bgr, conf, allowed_ids):
    """Run detection on one BGR frame, return (annotated RGB image, Counter)."""
    result = model.predict(frame_bgr, conf=conf, classes=allowed_ids, verbose=False)[0]
    counts = count_objects(result, set(allowed_ids), model.names)
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return annotated, counts


def show_counts(counts: Counter):
    total = sum(counts.values())
    st.metric("Total objects", total)
    if counts:
        df = pd.DataFrame(counts.items(), columns=["Class", "Count"]).sort_values(
            "Count", ascending=False
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Class"))
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Model (n = fastest, x = most accurate)",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

model = load_model(model_name)
all_classes = model.names  # {id: name}
default_names = [n for n in ("person", "car") if n in all_classes.values()]
selected_names = st.sidebar.multiselect(
    "Objects to count", list(all_classes.values()), default=default_names
)
allowed_ids = [i for i, n in all_classes.items() if n in selected_names]

# ---------------- Main ----------------
st.title("🔢 Object Counting System")
st.caption("Automatically count people, cars and other items in images or videos.")

if not allowed_ids:
    st.warning("Select at least one object type in the sidebar.")
    st.stop()

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Camera"])

# ----- Image tab -----
with tab_img:
    files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    for f in files or []:
        img = np.array(Image.open(f).convert("RGB"))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        annotated, counts = annotate(model, frame_bgr, conf, allowed_ids)
        st.subheader(f.name)
        c1, c2 = st.columns([2, 1])
        c1.image(annotated, use_container_width=True)
        with c2:
            show_counts(counts)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            if ok:
                st.download_button(
                    "Download annotated image",
                    buf.tobytes(),
                    file_name=f"counted_{f.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    key=f"dl_{f.name}",
                )
        st.divider()

# ----- Video tab -----
with tab_vid:
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    frame_skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 3)
    if vid and st.button("Start counting", type="primary"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        placeholder = st.empty()
        progress = st.progress(0.0)
        metric = st.empty()
        history, idx = [], 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_skip == 0:
                annotated, counts = annotate(model, frame, conf, allowed_ids)
                n = sum(counts.values())
                history.append({"frame": idx, "count": n})
                placeholder.image(annotated, use_container_width=True)
                metric.metric("Objects in current frame", n)
            idx += 1
            progress.progress(min(idx / total_frames, 1.0))
        cap.release()

        if history:
            df = pd.DataFrame(history).set_index("frame")
            st.subheader("Count over time")
            st.line_chart(df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Max in a frame", int(df["count"].max()))
            c2.metric("Average", f"{df['count'].mean():.1f}")
            c3.metric("Frames analysed", len(df))
            st.download_button(
                "Download counts (CSV)", df.to_csv().encode(), "video_counts.csv", "text/csv"
            )

# ----- Camera tab -----
with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        img = np.array(Image.open(shot).convert("RGB"))
        annotated, counts = annotate(
            model, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf, allowed_ids
        )
        c1, c2 = st.columns([2, 1])
        c1.image(annotated, use_container_width=True)
        with c2:
            show_counts(counts)