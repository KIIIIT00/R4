"""
指定した動画をフレームに分割し，オリジナルとゆがみ補正したフレームでそれぞれ保存する
"""
import cv2
import os
import numpy as np

def save_frames(video_name):
    video_path = os.path.join('./videos', video_name)
    video_file_name = os.path.splitext(os.path.basename(video_name))[0]
    
    # 出力フォルダのパス
    output_folder = os.path.join('./images/', video_file_name)

    output_folder_original = os.path.join(output_folder, 'original/')
    output_folder_undistort = os.path.join(output_folder, 'undistorted/')

    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder_original):
        os.makedirs(output_folder_original)
    
    if not os.path.exists(output_folder_undistort):
        os.makedirs(output_folder_undistort)
    
    cap = cv2.VideoCapture(video_path)
    saved_count = 0

    # 動画が正しく開けるか確認
    if not cap.isOpened():
        print("動画ファイルを開けませんでした")
        return
    
    # フレームを読み込んで保存
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        frame_undistorted = undistort(frame)

        frame_name = os.path.join(output_folder_original, f"frame_{saved_count:06d}.jpg")
        frame_undistorted_name = os.path.join(output_folder_undistort,  f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_name, frame)
        cv2.imwrite(frame_undistorted_name, frame_undistorted)

        print(f"{frame_name} を保存しました")
        print(f"{frame_undistorted_name} を保存しました")
        saved_count += 1
    
    cap.release()
    print("すべてのフレームを保存しました")
    
def undistort(frame):
    """-------- 変更しないこと！！--------"""
    # カメラ内部パラメータ行列
    mtx = np.array([1.53602652e+03,0.00000000e+00,9.70163963e+02,
                    0.00000000e+00,1.53797480e+03,4.82325767e+02, 
                    0.00000000e+00,0.00000000e+00,1.00000000e+00]).reshape(3,3)
    # 歪み係数
    dist = np.array([-0.41024555,0.10730146,-0.00048076,0.00142299,0.19989097])
    """ ---------------------------------"""
    h, w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # トリミング
    x, y, w, h = roi
    undistort_frame = undistorted_frame[y:y+h, x:x+w]
    return undistort_frame

if __name__ == '__main__':
    VIDEO_NAME ='no2.mp4' 
    save_frames(VIDEO_NAME)