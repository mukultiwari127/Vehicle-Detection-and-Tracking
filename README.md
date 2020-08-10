# Vehicle Detection And Tracking
This repository hosts the code for Vehicle Detection and Tracking using YoloV3 and YoloV4 with DeepSort

Yolo V3 abd Yolo V4 - https://github.com/AlexeyAB/darknet

# Dependencies
1. Python - 3.6 or higher
2. Tensorflow - 1.3
3. For additional python packages refer to requirements.txt

# Installation
1. Install python packages using requirements.txt ("pip install -r requirements.txt")
2. Download the model files for Yolov3 and Yolov4 from [Here](https://drive.google.com/file/d/1IpOpnniT1ZBMnT6b5mURQrBWO2FMQKor/view?usp=sharing)
3. Extract and copy the contents to VehicleTracker/model_data/

# Running Instructions
1. Navigate inside the folder VehicleTracker
2. For Starting the demo run demo.py (python demo.py)

Positional Arguments - 

1. choice - For choosing between YoloV3 and YoloV4 (Default - YoloV4)

"python demo.py --choice yolov3" or "python demo.py --choice yolov4"

2. video - Path to Input Video.

"python demo.py --video Input/test_video.mp4"

3. output - Path to save output video.

"python demo.py --output Output/result.avi"

4. show_detections - Visualize the Detections (Default - True)

"python demo.py --show_detections"

5. write_video - For writing the Output video (Default - True)

"python demo.py --write_video"
