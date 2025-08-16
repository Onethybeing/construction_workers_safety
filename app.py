import streamlit as st
import cv2
import numpy as np
from model import load_model, predict
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from dataset_registration import register_datasets

st.title("Custom Object Detection with Detectron2")

# Register datasets + metadata
register_datasets()
metadata = MetadataCatalog.get("hardhat_val")

# Load model
predictor = load_model("best.pth", "config.yaml")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    outputs = predict(image, predictor)

    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    st.image(out.get_image()[:, :, ::-1], caption="Detected Objects")
