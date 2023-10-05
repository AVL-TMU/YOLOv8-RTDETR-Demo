# Object Detection with YOLOv8 & RT-DETR
This is a Python script that uses the YOLOv8 object detection model to detect objects in a video stream from a camera. The script uses the ultralytics library to load the YOLOv8 model and the opencv-python library to capture and display the video stream.

## Requirements
* Python 3.7 or later
* Pytorch 1.7 or later (for YOLO models) or Pytorch 1.9 or later (for RTDETR model)
* Ultralytics library
* OpenCV
* Numpy
* Supervision

```
pip install torch
pip install numpy
pip install opencv-python
pip install ultralytics
pip install supervision
```

## Usage
* Clone this repository to your local machine.
* Install the required libraries using the pip commands above.
* Open a terminal or command prompt and navigate to the directory containing the camera.py script.
* Run the script using the command python camera.py.
* The script will open a window showing the video stream from the camera with object detections overlaid on the video. Press the ESC key to exit the script.

Note: You may need to modify the capture_index parameter in the ObjectDetection class constructor to select the correct camera index for your system.
