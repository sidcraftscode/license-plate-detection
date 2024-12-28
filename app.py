import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
from paddleocr import PaddleOCR
import re
import os

# Initialize models
@st.cache_resource
def load_models():
    try:
        # Using YOLOv8n for vehicle detection
        vehicle_model = YOLO('yolov8n.pt')
        # Using custom YOLO model for license plate detection
        plate_model = YOLO('license_plate_detector.pt')
        
        # Initialize PaddleOCR with modified parameters
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=True,  # Enable logging for debugging
            use_mp=False,
            enable_mkldnn=False,
            # Add specific model parameters
            rec_algorithm='SVTR_LCNet',  # Use a more stable recognition algorithm
            det_algorithm='DB',          # Standard detection algorithm
            rec_batch_num=1,            # Process one image at a time
            det_db_box_thresh=0.5,      # Adjust detection threshold
            rec_model_dir=None,         # Let PaddleOCR download models automatically
            det_model_dir=None,
            cls_model_dir=None,
            download_link=True          # Enable automatic model download
        )
        return vehicle_model, plate_model, ocr
    
    except Exception as e:
        # Get more detailed error information
        error_msg = f"Model loading error: {str(e)}\n"
        if hasattr(e, 'args'):
            error_msg += f"Additional error info: {e.args}\n"
        
        # Check if model files exist
        home_dir = os.path.expanduser('~')
        paddle_dir = os.path.join(home_dir, '.paddleocr')
        if not os.path.exists(paddle_dir):
            error_msg += f"PaddleOCR directory not found at {paddle_dir}\n"
        
        raise Exception(error_msg)

# Rest of your code remains the same...

def main():
    st.title("License Plate Detection System")

    st.markdown("""
    This system uses:
    - YOLOv8n for vehicle detection
    - Custom YOLO model for license plate detection
    - PaddleOCR for license plate recognition
    
    First time initialization may take a few minutes to download required models.
    """)

    try:
        with st.spinner("Loading models (this might take a few minutes on first run)..."):
            vehicle_model, plate_model, ocr = load_models()
            st.success("Models loaded successfully!")
    except Exception as e:
        st.error(str(e))
        st.error("""
        Troubleshooting steps:
        1. Make sure you have enough disk space
        2. Check your internet connection
        3. Try clearing your browser cache
        4. Restart the application
        """)
        return

    # Rest of your main() function remains the same...

if __name__ == "__main__":
    main()
