import streamlit as st
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from model import load_model, predict

# Load model
predictor = load_model()

# Register dummy dataset metadata (important for visualization)
MetadataCatalog.get("helmet_dataset").set(thing_classes=["Helmet", "No Helmet"])
metadata = MetadataCatalog.get("helmet_dataset")

st.title("👷 Construction Worker Helmet Detection")
st.write("Upload an image to check if workers are wearing helmets.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to numpy array (OpenCV format)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Run prediction
    outputs = predict(image, predictor)

    # Visualize results
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    st.image(out.get_image()[:, :, ::-1], caption="Prediction", use_column_width=True)
