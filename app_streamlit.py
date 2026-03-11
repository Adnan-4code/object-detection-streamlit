import streamlit as st
import torch

# ---- FIX FOR PYTORCH 2.6 SAFE LOADING ----
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv

from torch.nn.modules.container import Sequential

torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Conv
])

import numpy as np
from PIL import Image
import cv2, tempfile, os, glob
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Object Detection Model", layout="wide")

background_url = "https://images.unsplash.com/photo-1498050108023-c5249f4df085"

st.markdown(
   """
   <style>
   .stApp {
       background-color: #f5f7fa;
   }

   section[data-testid="stSidebar"] {
       background: #ffffff;
       border-radius: 12px;
       padding: 10px;
       border: 1px solid #e6e6e6;
   }

   div[data-testid="stFileUploader"] {
       background: #ffffff;
       border-radius: 12px;
       padding: 15px;
       border: 1px solid #e6e6e6;
   }

   .stDataFrame {
       background: #ffffff;
       border-radius: 12px;
       padding: 10px;
       border: 1px solid #e6e6e6;
   }

   h1, h2, h3, h4, h5, h6 {
       color: #2c3e50 !important;
       font-weight: 600;
   }

   p, label, span, div {
       color: #4a4a4a !important;
       font-weight: 400;
   }
   </style>
   """,
   unsafe_allow_html=True
)

st.title("Object Detection")


uploaded_weights = st.sidebar.file_uploader(
    "Upload custom .pt weights (optional)", type=["pt"]
)

conf = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.25, 0.01
)

img_size = st.sidebar.selectbox(
    "Inference image size (px)", [320, 416, 640, 1280], index=2
)


# ✅ FIXED MODEL LOADING
@st.cache_resource
def load_model(weights_path):
    model = YOLO(weights_path)
    return model


def save_uploaded_file(uploaded_file, suffix=""):
    suffix = suffix if suffix else Path(uploaded_file.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    return tf.name


def annotate_and_table(results, model):
    res = results[0]

    try:
        plotted = res.plot()
        annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    except Exception:
        annotated = res.orig_img if hasattr(res, "orig_img") else None

    detections = []

    try:
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for c, cf, box in zip(
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.xyxy.cpu().numpy(),
            ):
                name = model.names[int(c)]
                detections.append(
                    {
                        "class": name,
                        "conf": float(cf),
                        "bbox": [float(x) for x in box],
                    }
                )
    except:
        detections = []

    return annotated, pd.DataFrame(detections)


# default model
weights_to_load = "yolov8n.pt"

if uploaded_weights:
    weights_to_load = save_uploaded_file(uploaded_weights, suffix=".pt")
    st.sidebar.success("Using uploaded weights")


model = load_model(weights_to_load)


mode = st.radio("Select input", ["Image upload", "Video upload"])


if mode == "Image upload":

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        st.image(img, caption="Input image")

        results = model.predict(
            np.array(img),
            conf=conf,
            imgsz=img_size
        )

        annotated, df = annotate_and_table(results, model)

        if annotated is not None:
            st.image(annotated, caption="Annotated")

        if not df.empty:
            st.dataframe(df)


elif mode == "Video upload":

    uploaded_vid = st.file_uploader(
        "Upload video", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_vid:

        tmp = save_uploaded_file(uploaded_vid)

        st.video(tmp)

        project_dir = tempfile.mkdtemp()

        results = model.predict(
            source=tmp,
            conf=conf,
            imgsz=img_size,
            project=project_dir,
            name="run",
            save=True,
        )

        try:
            out_dir = str(results[0].save_dir)

            vids = glob.glob(os.path.join(out_dir, "*"))

            vids = [
                v
                for v in vids
                if Path(v).suffix.lower()
                in [".mp4", ".avi", ".mov", ".mkv"]
            ]

            if vids:
                st.success("Annotated video")
                st.video(vids[0])

        except:
            st.warning("Could not display annotated video")



