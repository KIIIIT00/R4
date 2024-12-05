import cv2
import numpy as np

class CameraCalibration:
    def __init__(self, mtx, dist):
        """
        parameter:
            mtx: 内部パラメータ
            dist: 歪み係数
        """
        self.mxt = np.array(mtx)
        self.dist = np.array(dist)
    
    def undistort(self, frame):
        undistorted_frame = cv2.undistort(frame, self.mtx, self.dist, None)
        return undistorted_frame
    
        
