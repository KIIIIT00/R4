from tkinter import Frame
import cv2

# キャリブレーションデータ（例）
# カメラ内部パラメータ行列
mtx = [[1.53602652e+03,0.00000000e+00,9.70163963e+02],[0.00000000e+00,1.53797480e+03,4.82325767e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]]
# 歪み係数
dist = [[-0.41024555,0.10730146,-0.00048076,0.00142299,0.19989097]]

# 行列をNumPy配列に変換
import numpy as np
mtx = np.array(mtx)
dist = np.array(dist)

# Webカメラのビデオキャプチャ
cap = cv2.VideoCapture(0)
# 各種プロパティーを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレームの幅
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレームの高さ
fps = cap.get(cv2.CAP_PROP_FPS)

# ビデオライターの設定（保存用のファイル名とエンコーディング、フレームレートなど）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを指定
out = cv2.VideoWriter('./videos/output_video_2.mp4', fourcc, fps, (frame_width, frame_height), True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 歪み補正
    # undistorted_frame = cv2.undistort(frame, mtx, dist, None)

    # 補正後のフレームを表示
    cv2.imshow('Undistorted Video', frame)

    # 補正後のフレームを保存
    out.write(frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
