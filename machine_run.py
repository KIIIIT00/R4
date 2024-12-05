import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from utils.eval import CNNEval
from utils.CNN_weed_classifier import WeedClassifierCNN
from utils.DXL_XM import DynamixelXM
from utils.CameraCalibration import CameraCalibration

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Made Dir Path:{dir_path}")
    else:
        print(f"Exist Dir Path:{dir_path}")

def split_frame_into_9(frame):
    height, width, _ = frame.shape
    h_step = height // 3
    w_step = width // 3
    segments = []
    positions = []
    for i in range(3):
        for j in range(3):
            segments.append(frame[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step])
            positions.append((j*w_step, i*h_step))  
    return segments, positions  

def draw_grid(frame):
    height, width, _ = frame.shape
    
    h_step = height // 3
    w_step = width // 3
    
    for i in range(1, 3):
        cv2.line(frame, (i * w_step, 0), (i * w_step, height), (0, 255, 0), 2)  # 緑色の線
    
    for i in range(1, 3):
        cv2.line(frame, (0, i * h_step), (width, i * h_step), (0, 255, 0), 2)
    return frame

# OUTPUT PATH SETTING
OUTPUT_VIDEO = './videos/weed_calssifier'
VIDEO_NAME = 'weed_calssifier.mp4'
make_dir(OUTPUT_VIDEO)
VIDEO_FILE_PATH = os.path.join(OUTPUT_VIDEO, VIDEO_NAME)

LOGFILE_DIR = 'results/processing_time'
LOGFILE_NAME = "processing_times.log"
make_dir(LOGFILE_DIR)
LOGFILE_PATH = os.path.join(LOGFILE_DIR, LOGFILE_NAME)
with open(LOGFILE_PATH, 'w') as f:
    f.write("Processing Times Log\n\n")


# Setting Model's Parameter 
input_size = (522, 318)
num_classes = 3  # 3classes
# Load Model
EP = 20
model_path = f"./models/weed_classifier_ep{EP}.pth"
model = CNNEval(model_path=model_path, image_size=input_size, num_classes=num_classes)

# Matrix of Internal Parameters 
mtx = [[1.53602652e+03,0.00000000e+00,9.70163963e+02],[0.00000000e+00,1.53797480e+03,4.82325767e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]
# Distortion Factor
dist = [[-0.41024555,0.10730146,-0.00048076,0.00142299,0.19989097]]

# Calibration
camera_calibration = CameraCalibration(mtx, dist)

# video capture
cap = cv2.VideoCapture(0)
# get each properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of frames
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height of frames
fps = cap.get(cv2.CAP_PROP_FPS)

detection_frames_num = 6 # detection segments frames number(select 6 or 9)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    undisorted_frame = camera_calibration.undistort(frame) # undistort
    undistorted_frame = cv2.resize(undistorted_frame, (1566, 954)) # resize

    segments, positions = split_frame_into_9(undistorted_frame) # 9 split
    frame_with_grid = draw_grid(undistorted_frame) # add grid on Undistorted_Frame
    
    cv2.imshow("Original Frame", frame_with_grid) # display Original Frame

    if detection_frames_num == 6:
        cnn_segments = segments[3:]
        cnn_positions = positions[3:]
    else:
        cnn_segments = segments
        cnn_positions = positions
    frame_start_time = time.time()

    segments_times = []
    for idx, 


    