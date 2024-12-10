import cv2
import os
import time
from threading import Thread
from utils.eval import CNNEval
from utils.DXL_XM import DynamixelXM
from utils.ddsm115 import DDSM115
from utils.CameraCalibration import CameraCalibration
from utils.frame_display import FrameDisplay

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Made Dir Path:{dir_path}")
    else:
        print(f"Exist Dir Path:{dir_path}")

def detection_weed_count(idx, label, upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt, num_detection_num):
    if label == 'many_weed':
        if num_detection_num == 9:
            if 0 <= idx and idx < 3:
                upper_weed_cnt += 1
            elif 3 <= idx and idx < 6:
                middle_weed_cnt += 1
            else:
                bottom_weed_cnt += 1
        else:
            if 0 <= idx and idx < 3:
                middle_weed_cnt += 1
            else:
                bottom_weed_cnt += 1
    
    return upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt
    
def decision_state(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt):
    state_list = []
    if upper_weed_cnt >= 2:
        state_list.append(3)
    
    if middle_weed_cnt >= 2:
        state_list.append(2)
    
    if bottom_weed_cnt >= 2:
        state_list.append(1)
        
    return state_list

def thread_task(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt):
    state_list = decision_state(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt)
    dxl_xm.move_by_state(state_list)
    time.sleep(5)

def join_thread(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt):
    state = decision_state(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt)
    return dxl_xm.is_current_same_as_destination(state)

# Matrix of Internal Parameters 
mtx = [[1.53602652e+03,0.00000000e+00,9.70163963e+02],[0.00000000e+00,1.53797480e+03,4.82325767e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]
# Distortion Factor
dist = [[-0.41024555,0.10730146,-0.00048076,0.00142299,0.19989097]]
# Calibration
camera_calibration = CameraCalibration(mtx, dist)

# video capture
VIDEO_PATH = './videos/we.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
# get each properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of frames
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height of frames
fps = cap.get(cv2.CAP_PROP_FPS)

##################### <OUTPUT PATH SETTING> #####################
OUTPUT_VIDEO = './videos/weed_calssifier'
VIDEO_NAME = 'weed_calssifier.mp4'
make_dir(OUTPUT_VIDEO)
VIDEO_FILE_PATH = os.path.join(OUTPUT_VIDEO, VIDEO_NAME)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを指定
out = cv2.VideoWriter(VIDEO_FILE_PATH, fourcc, fps, (frame_width, frame_height), True)

LOGFILE_DIR = 'results/processing_time'
LOGFILE_NAME = "processing_times.log"
make_dir(LOGFILE_DIR)
LOGFILE_PATH = os.path.join(LOGFILE_DIR, LOGFILE_NAME)
with open(LOGFILE_PATH, 'w') as f:
    f.write("Processing Times Log\n\n")
#################################################################

######################## <MOTOR SETTING> ########################
port_name = "/dev/cu.usbserial-FT4TCJC1"
serial_port_name = "/dev/cu.usbserial-FT4TCJC1"
baudrate = 115200
dxl_id = 4
dxl_xm = DynamixelXM(port_name, baudrate, dxl_id)
ddsm = DDSM115(serial_port_name, baudrate)
#################################################################

# Setting Model's Parameter 
input_size = (522, 318)
num_classes = 2  # 3classes or 2classes
# Load Model
EP = 20
model_path = f"./models/weed_classifier_ep{EP}.pth"
model = CNNEval(model_path=model_path, image_size=input_size, num_classes=num_classes)
    
detection_frames_num = 6 # detection segments frames number(select 6 or 9)
frame_display = FrameDisplay(detection_frames_num, num_classes)

state = -1
dxl_xm.move_by_state(state)
while cap.isOpened():
    # キーボード入力で車輪を操作
    input_key = cv2.waitKey(1)
    ddsm.move(input_key)
    
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = camera_calibration.undistort(frame) # undistort
    undistorted_frame = cv2.resize(undistorted_frame, (1566, 954)) # resize

    segments, positions = frame_display.split_frame_into_9(undistorted_frame) # 9 split
    frame_with_grid = frame_display.draw_grid(undistorted_frame) # add grid on Undistorted_Frame
    
    cv2.imshow("Original Frame", frame_with_grid) # display Original Frame

    segments, positions = frame_display.detection_segments(segments, positions)
    frame_start_time = time.time()

    segments_times = []
    
    # ラベルの出現回数をカウントする変数
    upper_weed_cnt = 0
    middle_weed_cnt = 0
    bottom_weed_cnt = 0
    for idx, (segment, position) in enumerate(zip(segments, positions)):
        predicted_label, segments_times = model.eval(segment, segments_times)
        
        upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt = detection_weed_count(idx, predicted_label, upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt, detection_frames_num)
        
        frame_display.frame_put_label(frame_with_grid, predicted_label, position)
        
        out.write(undistorted_frame)
    
    # 吸引しに行く
    # state = decision_state(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt)
    # dxl_xm.move_by_state(state)
    thread = Thread(target=thread_task, 
                    args=(upper_weed_cnt,
                          middle_weed_cnt,
                          bottom_weed_cnt,
                          )
                    )
    thread.start()
    if join_thread(upper_weed_cnt, middle_weed_cnt, bottom_weed_cnt):
        thread.join()
    
    "-------------------------"
    # TODO: 吸引して初期位置に戻るまでの時間をここで調整
    "-------------------------"
    
    # 初期位置に戻す
    # state = -1
    # dxl_xm.move_by_state(state)
    
    cv2.imshow("Weed Detection", frame_with_grid)
    
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time  # フレーム全体の処理時間
    
    print(f"Segments Eval Processing Time: {frame_end_time - frame_start_time:.4f} seconds")
     # 処理時間をログファイルに保存
    with open(LOGFILE_PATH, 'a') as f:
        f.write(f"Frame Time: {frame_time:.4f} seconds\n")
        for idx, t in enumerate(segments_times):
            f.write(f"  Segment {idx + 1} Time: {t:.4f} seconds\n")
        f.write("\n")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 初期状態に戻す
state = -1
dxl_xm.move_by_state(state)

# リソースを解放
# ddsm.close()
cap.release()
out.release()
cv2.destroyAllWindows()

