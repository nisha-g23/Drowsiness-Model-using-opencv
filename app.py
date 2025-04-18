import streamlit as st
import torch
from PIL import Image
import numpy as np

# Set page config (optional but recommended)
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# Title
st.title("Drowsiness Detection App")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
with st.spinner("Loading YOLOv5 model..."):
    model = load_model()

if model is None:
    st.stop()  # Stop execution if model failed to load

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Run detection with progress indicator
        with st.spinner("Detecting objects..."):
            results = model(image)
        
        # Show detection summary
        st.subheader("Detection Results")
        
        # Extract and display detection information
        df = results.pandas().xyxy[0]  # Results as DataFrame
        if len(df) > 0:
            st.dataframe(df[['name', 'confidence']])
        else:
            st.write("No objects detected")
        
        # Render and display the image with bounding boxes
        rendered_img = np.squeeze(results.render())
        st.image(rendered_img, caption="Detected Objects", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
