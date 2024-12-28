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
    # Initialize PaddleOCR with additional parameters
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='en', 
        show_log=False,
        # Add error handling parameters
        use_mp=False,  # Disable multiprocessing
        enable_mkldnn=False  # Disable MKL-DNN optimization
    )
    return vehicle_model, plate_model, ocr

def is_valid_license_plate(text):
    pattern = r'^[A-Z0-9]{5,8}$'
    return re.match(pattern, text) is not None

def process_image(image, vehicle_model, plate_model, ocr):
    try:
        # Convert PIL Image to cv2 format
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect vehicles
        vehicle_results = vehicle_model(img_cv2, classes=[2, 7])  # 2: car, 7: truck
        detections = []

        for vehicle_box in vehicle_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, vehicle_box[:4])
            if x1 >= x2 or y1 >= y2 or x2 > img_cv2.shape[1] or y2 > img_cv2.shape[0]:
                continue
                
            vehicle_crop = img_cv2[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            # Save the vehicle crop
            vehicle_pil = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
            vehicle_byte_arr = io.BytesIO()
            vehicle_pil.save(vehicle_byte_arr, format='PNG')
            vehicle_bytes = vehicle_byte_arr.getvalue()

            # Detect license plates
            plate_results = plate_model(vehicle_crop)

            for plate_box in plate_results[0].boxes.xyxy:
                px1, py1, px2, py2 = map(int, plate_box[:4])
                if px1 >= px2 or py1 >= py2 or px2 > vehicle_crop.shape[1] or py2 > vehicle_crop.shape[0]:
                    continue
                    
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue

                # Preprocess plate image
                plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2)  # Upscale
                plate_crop = cv2.convertScaleAbs(plate_crop, alpha=1.2, beta=10)  # Enhance contrast

                # Save the plate crop
                plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                plate_byte_arr = io.BytesIO()
                plate_pil.save(plate_byte_arr, format='PNG')
                plate_bytes = plate_byte_arr.getvalue()

                try:
                    # Use PaddleOCR with error handling
                    ocr_results = ocr.ocr(np.array(plate_pil), cls=True)

                    if ocr_results and ocr_results[0]:
                        for line in ocr_results[0]:
                            text = line[1][0]
                            confidence = line[1][1]

                            plate_text = ''.join(c for c in text if c.isalnum()).upper()

                            if len(plate_text) >= 5 and is_valid_license_plate(plate_text) and confidence >= 0.90:
                                detections.append({
                                    'vehicle_image': vehicle_bytes,
                                    'plate_image': plate_bytes,
                                    'license_plate': plate_text,
                                    'confidence': f"{confidence:.2f}"
                                })
                except Exception as e:
                    st.error(f"OCR error: {str(e)}")
                    continue

        return detections
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return []

def main():
    st.title("License Plate Detection System")

    st.markdown("""
    This system uses:
    - YOLOv8n for vehicle detection
    - Custom YOLO model for license plate detection
    - PaddleOCR for license plate recognition
    """)

    try:
        with st.spinner("Loading models..."):
            vehicle_model, plate_model, ocr = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        all_detections = []

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    detections = process_image(image, vehicle_model, plate_model, ocr)
                    if detections:
                        all_detections.extend(detections)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            finally:
                progress_bar.progress((idx + 1) / len(uploaded_files))

        if all_detections:
            st.subheader("Detection Results")
            for detection in all_detections:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.image(detection['vehicle_image'], caption="Vehicle", use_container_width=True)
                with col2:
                    st.image(detection['plate_image'], caption="License Plate", use_container_width=True)
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
