# License Plate Detection System

A powerful and efficient license plate detection system that utilizes YOLOv8 for vehicle detection, a custom YOLO model for license plate detection, and PaddleOCR for optical character recognition. This system allows users to upload images and extract vehicle license plates with high accuracy.

You can preview it [here](https://licenseplatedetector.streamlit.app).

## Getting Started
### Prerequisites
- Python 3.7+
- Required Python libraries:
  - streamlit
  - opencv-python
  - ultralytics (for YOLO models)
  - numpy
  - pillow
  - pandas
  - paddleocr
  - io
  - regex

### Installation
1. Clone the repo:
```
git clone https://github.com/yourusername/license-plate-detection cd license-plate-detection
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

4. Navigate to the displayed local URL to access the application.

### Models and Files
- The app uses pre-trained YOLOv8 models for vehicle and license plate detection.
- PaddleOCR is used for license plate recognition with high accuracy.

## Built With
* [Streamlit](https://streamlit.io)
* YOLOv8
* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* [OpenCV](https://opencv.org/)
* Pillow
* [Pandas](https://pandas.pydata.org/)

## Features
- [x] Upload and process multiple images at once
- [x] Detect vehicles in the uploaded images
- [x] Extract license plates from detected vehicles
- [x] Use OCR to recognize the text on license plates
- [x] Basic validation for detected license plates
- [x] Display the vehicle and license plate images with confidence score

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

