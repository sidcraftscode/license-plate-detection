import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
from paddleocr import PaddleOCR

# Initialize models
@st.cache_resource
def load_models():
    # Using YOLOv8n for faster inference while maintaining good accuracy
    vehicle_model = YOLO('yolov8n.pt')
    # Initialize PaddleOCR with license plate detection and recognition
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return vehicle_model, ocr

def process_image(image, vehicle_model, ocr):
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Detect vehicles
    vehicle_results = vehicle_model(img_cv2, classes=[2, 7])  # 2: car, 7: truck
    detections = []
    
    for vehicle_box in vehicle_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, vehicle_box[:4])
        vehicle_crop = img_cv2[y1:y2, x1:x2]
        
        # Use PaddleOCR to detect and recognize license plates
        ocr_results = ocr.ocr(vehicle_crop, cls=True)
        
        if ocr_results[0]:
            for line in ocr_results[0]:
                text = line[1][0]
                confidence = line[1][1]
                
                # Basic license plate validation
                # Remove spaces and special characters
                plate_text = ''.join(c for c in text if c.isalnum())
                
                if len(plate_text) >= 5:  # Minimum valid plate length
                    # Save the vehicle crop as bytes for display
                    vehicle_pil = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
                    img_byte_arr = io.BytesIO()
                    vehicle_pil.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    detections.append({
                        'vehicle_image': img_byte_arr,
                        'license_plate': plate_text,
                        'confidence': f"{confidence:.2f}"
                    })
    
    return detections

def main():
    st.title("License Plate Detection System")
    
    # Add some helpful information
    st.markdown("""
    This system uses:
    - YOLOv8n for vehicle detection
    - PaddleOCR for license plate recognition
    
    Upload one or more images to begin detection.
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        vehicle_model, ocr = load_models()
    
    # File uploader
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        progress_bar = st.progress(0)
        all_detections = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Process each image
            image = Image.open(uploaded_file)
            with st.spinner(f"Processing {uploaded_file.name}..."):
                detections = process_image(image, vehicle_model, ocr)
                all_detections.extend(detections)
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if all_detections:
            # Create DataFrame
            df = pd.DataFrame(all_detections)
            
            # Display results in a table
            st.subheader("Detection Results")
            
            # Custom table display with improved styling
            for idx, row in df.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.image(row['vehicle_image'], caption="Vehicle", use_column_width=True)
                
                with col2:
                    st.markdown(f"""
                    **License Plate:**  
                    {row['license_plate']}
                    """)
                
                with col3:
                    st.markdown(f"""
                    **Confidence:**  
                    {row['confidence']}
                    """)
                
                st.divider()
            
            # Export options
            st.subheader("Export Results")
            csv = df[['license_plate', 'confidence']].to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="license_plate_detections.csv",
                mime="text/csv"
            )
        else:
            st.warning("No license plates detected in the uploaded images.")

if __name__ == "__main__":
    main()
