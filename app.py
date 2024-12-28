import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
from paddleocr import PaddleOCR
import re

# Initialize models
@st.cache_resource
def load_models():
    # Using YOLOv8n for vehicle detection
    vehicle_model = YOLO('yolov8n.pt')
    # Using custom YOLO model for license plate detection
    plate_model = YOLO('license_plate_detector.pt')
    # Initialize PaddleOCR for license plate recognition
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return vehicle_model, plate_model, ocr

def is_valid_license_plate(text):
    # Define a regex pattern to match common license plate formats
    pattern = r'^[A-Z0-9]{5,8}$'  # Only alphanumeric, 5 to 8 characters
    return re.match(pattern, text) is not None

def process_image(image, vehicle_model, plate_model, ocr):
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect vehicles
    vehicle_results = vehicle_model(img_cv2, classes=[2, 7])  # 2: car, 7: truck
    detections = []

    for vehicle_box in vehicle_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, vehicle_box[:4])
        vehicle_crop = img_cv2[y1:y2, x1:x2]

        # Save the vehicle crop
        vehicle_pil = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
        vehicle_byte_arr = io.BytesIO()
        vehicle_pil.save(vehicle_byte_arr, format='PNG')
        vehicle_bytes = vehicle_byte_arr.getvalue()

        # Use the custom YOLO model to detect license plates within the vehicle crop
        plate_results = plate_model(vehicle_crop)

        for plate_box in plate_results[0].boxes.xyxy:
            px1, py1, px2, py2 = map(int, plate_box[:4])
            plate_crop = vehicle_crop[py1:py2, px1:px2]

            # Save the plate crop
            plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            plate_byte_arr = io.BytesIO()
            plate_pil.save(plate_byte_arr, format='PNG')
            plate_bytes = plate_byte_arr.getvalue()

            # Use PaddleOCR to detect and recognize license plates
            ocr_results = ocr.ocr(plate_crop, cls=True)

            if ocr_results[0]:
                for line in ocr_results[0]:
                    text = line[1][0]
                    confidence = line[1][1]

                    # Basic license plate validation
                    plate_text = ''.join(c for c in text if c.isalnum())

                    if len(plate_text) >= 5 and is_valid_license_plate(plate_text) and confidence >= 0.90:
                        detections.append({
                            'vehicle_image': vehicle_bytes,
                            'plate_image': plate_bytes,
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
    - Custom YOLO model for license plate detection
    - PaddleOCR for license plate recognition

    Upload one or more images to begin detection.
    """)

    # Load models
    with st.spinner("Loading models..."):
        vehicle_model, plate_model, ocr = load_models()

    # File uploader
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        all_detections = []

        for idx, uploaded_file in enumerate(uploaded_files):
            # Process each image
            image = Image.open(uploaded_file)
            with st.spinner(f"Processing {uploaded_file.name}..."):
                detections = process_image(image, vehicle_model, plate_model, ocr)
                all_detections.extend(detections)

            # Update progress bar
            progress_bar.progress((idx + 1) / len(uploaded_files))

        if all_detections:
            # Display results
            st.subheader("Detection Results")

            for detection in all_detections:
                col1, col2, col3 = st.columns([2, 2, 1])

                # Display vehicle image
                with col1:
                    st.image(detection['vehicle_image'], caption="Vehicle", use_container_width=True)

                # Display license plate image
                with col2:
                    st.image(detection['plate_image'], caption="License Plate", use_container_width=True)

                # Display detected text and confidence
                with col3:
                    st.markdown(f"""
                    **License Plate:**  
                    {detection['license_plate']}  

                    **Confidence:**  
                    {detection['confidence']}
                    """)

                st.divider()

        else:
            st.warning("No license plates detected in the uploaded images.")

if __name__ == "__main__":
    main()
