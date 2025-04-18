import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Title and page config
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection App")

# Display a warning during model loading
st.info("⚠️ The model will take a moment to load on first run")

# Use a safer import approach for torch and YOLOv5
@st.cache_resource
def load_model():
    try:
        # Import inside function to avoid torch import errors during Streamlit initialization
        import torch
        
        # Alter the import path approach
        model = None
        try:
            # Try loading the model with the standard approach
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, 
                              trust_repo=True, force_reload=False)
        except Exception as e1:
            st.warning(f"Standard model loading failed: {e1}")
            try:
                # Alternative loading method
                from ultralytics import YOLO
                model = YOLO('yolov5s.pt')
            except Exception as e2:
                st.error(f"Alternative model loading also failed: {e2}")
                return None
                
        return model
    except Exception as e:
        st.error(f"Error importing required libraries: {e}")
        return None

# System information (helps with debugging)
with st.expander("System Information"):
    st.write(f"Python version: {sys.version}")
    st.write(f"Current working directory: {os.getcwd()}")
    try:
        import torch
        st.write(f"PyTorch version: {torch.__version__}")
        st.write(f"CUDA available: {torch.cuda.is_available()}")
    except:
        st.write("PyTorch info not available")

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the YOLOv5 model. Please check the system information.")
    st.stop()
else:
    st.success("Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Run detection with progress indicator
        with st.spinner("Detecting objects..."):
            # Check if model is YOLO or torch.hub model and run accordingly
            import inspect
            if 'ultralytics.engine.results.Results' in str(inspect.getmro(type(model))):
                # Using YOLO from ultralytics
                results = model(image)
                # Extract detection information
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.subheader("Detection Results")
                    # Convert to DataFrame
                    df = boxes.pandas().xyxy[0]
                    st.dataframe(df)
                else:
                    st.write("No objects detected")
                
                # Display the image with bounding boxes
                rendered_img = results[0].plot()
                st.image(rendered_img, caption="Detected Objects", use_column_width=True)
            else:
                # Using torch.hub model
                results = model(image)
                
                # Show detection summary
                st.subheader("Detection Results")
                
                # Extract and display detection information
                df = results.pandas().xyxy[0]
                if len(df) > 0:
                    st.dataframe(df)
                    st.write(f"Detected {len(df)} objects")
                else:
                    st.write("No objects detected")
                
                # Render and display the image with bounding boxes
                rendered_img = np.squeeze(results.render())
                st.image(rendered_img, caption="Detected Objects", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.write("Try uploading a different image or check the logs for more details.")
