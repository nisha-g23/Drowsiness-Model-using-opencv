import streamlit as st
import torch
from PIL import Image
import numpy as np

# Title
st.title("YOLOv5 Object Detection App")

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(image)

    # Show results
    st.subheader("Detection Results")
    results.print()

    # Render and display the image
    rendered_img = np.squeeze(results.render())  # Returns a list
    st.image(rendered_img, caption="Detected Image", use_column_width=True)
