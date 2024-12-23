"""
動画を撮影しながら，雑草のクラスを判別
"""

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from utils.CNN_weed_classifier import WeedClassifierCNN

def split_frame_into_9(frame):
    height, width, _ = frame.shape
    h_step = height // 3
    w_step = width // 3
    segments = []
    positions = []
    for i in range(3):
        for j in range(3):
            segments.append(frame[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step])
            positions.append((j*w_step, i*h_step))  # 座標情報を追加
    return segments, positions  # 2つのリストを返す

def draw_grid(frame):
    height, width, _ = frame.shape
    
    # 3等分の線を引く位置を計算
    h_step = height // 3
    w_step = width // 3
    
    # 縦線を描画
    for i in range(1, 3):
        cv2.line(frame, (i * w_step, 0), (i * w_step, height), (0, 255, 0), 2)  # 緑色の線
    
    # 横線を描画
    for i in range(1, 3):
        cv2.line(frame, (0, i * h_step), (width, i * h_step), (0, 255, 0), 2)
    
    return frame


# パラメータ設定
batch_size = 16
learning_rate = 0.001
num_epochs = 20
input_size = (522, 318)
num_classes = 3  # 雑草の有無を3クラス分類
# モデルの読み込み
EP = 20
model_path = f"./models/weed_classifier_ep{EP}_160_4layers.pth"
model = WeedClassifierCNN()
model.load_state_dict(torch.load(model_path, weights_only=True, map_location={'cuda:0': 'cpu'}))
model.eval()  # 推論モードに設定


# カメラ内部パラメータ行列
mtx = [[1.53602652e+03,0.00000000e+00,9.70163963e+02],[0.00000000e+00,1.53797480e+03,4.82325767e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]
# 歪み係数
dist = [[-0.41024555,0.10730146,-0.00048076,0.00142299,0.19989097]]

# 行列をNumPy配列に変換
import numpy as np
mtx = np.array(mtx)
dist = np.array(dist)

VIDEO_PATH = './videos/original/we2.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
# 各種プロパティーを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレームの幅
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレームの高さ
fps = cap.get(cv2.CAP_PROP_FPS)

################# アウトプットビデオパス #################################
OUTPUT_VIDEO = './videos/weed_calssifier'
VIDEO_NAME = 'weed_calssifier.mp4'
if not os.path.exists(OUTPUT_VIDEO):
    os.makedirs(OUTPUT_VIDEO)
VIDEO_FILE_PATH = os.path.join(OUTPUT_VIDEO, VIDEO_NAME)
#######################################################################

# ビデオライターの設定（保存用のファイル名とエンコーディング、フレームレートなど）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを指定
out = cv2.VideoWriter(VIDEO_FILE_PATH, fourcc, fps, (frame_width, frame_height), True)
################ ログファイル #################
LOGFILE_NAME = "processing_times.log"
LOGFILE_DIR = 'results/processing_time'
LOGFILE_PATH = os.path.join(LOGFILE_DIR, LOGFILE_NAME)

if not os.path.exists(LOGFILE_DIR):
    os.makedirs(LOGFILE_DIR)

with open(LOGFILE_PATH, 'w') as f:
    f.write("Processing Times Log\n\n")
################################################################
image_size = (522, 318)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
])

# クラスのラベル（適宜変更）
class_labels = ["exist_weed", "no_weed"]


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 歪み補正
    undistorted_frame = cv2.undistort(frame, mtx, dist, None)
    # reisize
    undistorted_frame = cv2.resize(undistorted_frame, (1566, 954))
    
    
    # フレームを9分割
    segments, positions = split_frame_into_9(undistorted_frame)
    
    # 境界線を追加
    frame_with_grid = draw_grid(undistorted_frame)
    
    # オリジナル画像の表示
    cv2.imshow("Original Frame", frame_with_grid)
    
    frame_start_time = time.time()  # フレーム全体の処理時間計測
    
    # 各セグメントに対して推論開始
    segment_times = []
    for idx, (segment, position) in enumerate(zip(segments, positions)):
        pil_image = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_image).unsqueeze(0)
        
        segment_start_time = time.time()  # 各セグメントの処理時間計測
        
        # 推論の実行
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        segment_time = time.time() - segment_start_time
        segment_times.append(segment_time)
        
        print(f"Eval Processing Time: {segment_time:.4f} seconds")
        # 結果を元のフレームにオーバーレイ
        label = class_labels[predicted.item()]
        x, y = position
        if label == class_labels[0]:# Case of "little weed"
            text_color = (255, 0, 0)
        elif label == class_labels[1]:# Case of "many weed"
            text_color = (0, 255, 0)
        else:
            text_color = (0, 0, 255)
        cv2.putText(frame_with_grid, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        #　分類結果の画像を保存
        out.write(undistorted_frame)
        
    # 結果の表示
    cv2.imshow("Weed Detection", frame_with_grid)
    
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time  # フレーム全体の処理時間
    
    print(f"Segments Eval Processing Time: {frame_end_time - frame_start_time:.4f} seconds")
     # 処理時間をログファイルに保存
    with open(LOGFILE_PATH, 'a') as f:
        f.write(f"Frame Time: {frame_time:.4f} seconds\n")
        for idx, t in enumerate(segment_times):
            f.write(f"  Segment {idx + 1} Time: {t:.4f} seconds\n")
        f.write("\n")
    
    # 'q' キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()