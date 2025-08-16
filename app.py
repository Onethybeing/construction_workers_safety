import streamlit as st
import cv2
import numpy as np
from model import load_model, predict
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

st.title("Custom Object Detection with Detectron2")

# Load model
predictor = load_model("best.pth", "config.yaml")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    outputs = predict(image, predictor)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("__unused__"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    st.image(out.get_image()[:, :, ::-1], caption="Detected Objects")
