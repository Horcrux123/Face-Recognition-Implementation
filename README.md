# Face Detection using OpenCV

## Overview
This project implements real-time face detection using OpenCV's Haar Cascade Classifier. It captures video from a webcam, detects faces, and highlights them with bounding boxes. The detection model is based on a pre-trained Haar cascade XML file.

## Features
- Real-time face detection using OpenCV
- Uses Haar cascade classifier for frontal face detection
- Highlights detected faces with bounding boxes
- Works with any webcam
- Press 'f' to stop the program

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install opencv-python pathlib
```

## Installation and Usage
Clone this repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/face-detection-opencv.git
cd face-detection-opencv
```

Run the Python script:

```sh
python face_detection.py
```

## Code Explanation
### 1. Importing Libraries
```python
import cv2
import pathlib
```
- `cv2`: OpenCV library for image processing.
- `pathlib`: Handles file paths.

### 2. Loading the Haar Cascade Model
```python
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/ "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
```
- Loads the pre-trained face detection model.

### 3. Capturing Video from Webcam
```python
camera = cv2.VideoCapture(0)
```
- Opens the default camera.

### 4. Processing Video Frames
```python
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
- Captures and converts frames to grayscale for better performance.

### 5. Detecting Faces
```python
faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30,30))
```
- Uses the Haar cascade classifier to detect faces.

### 6. Drawing Rectangles Around Faces
```python
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 55), 2)
```
- Draws a bounding box around each detected face.

### 7. Displaying the Output
```python
cv2.imshow("Faces", frame)
if cv2.waitKey(1) == ord("f"):
    break
```
- Displays the frame and stops when 'f' is pressed.

### 8. Releasing Resources
```python
camera.release()
cv2.destroyAllWindows()
```
- Closes the webcam and OpenCV windows.

## Adjusting `scaleFactor`
- **Increasing (`>1.1`)**: Faster but may miss small faces.
- **Decreasing (`<1.1`)**: More accurate but slower.

## Contributing
Feel free to fork this repository and submit pull requests!

## License
This project is open-source and available under the MIT License.

