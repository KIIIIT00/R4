import cv2

class FrameDisplay:
    def __init__(self, detection_frames_num=6, num_classes=3):
        self.detection_frames_num = detection_frames_num
        self.num_classes = num_classes
        
    def split_frame_into_9(self, frame):
        """
        Parameter:
            frame : 入力フレーム
        
        Return:
            segmetns : 9分割したときの画像配列
            position : 9分割の画像の画像座標を格納した配列
        """
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
    
    def draw_grid(welf, frame):
        """
        Parameter:
            frame : 入力フレーム
        
        Return:
            frame : 9分割に境界線を描いたframe
        """
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
    
    def get_label_color(self, label):
        """
        Parameter:
            label : 雑草クラスのラベル
        
        Returns:
            label_color : ラベルのテキストカラー
        """
        if label == 'many_weed':
                label_color = (0, 255, 0)
        elif label == 'no_weed':
                label_color = (0, 0, 255)
        else:
            label_color = (255, 0, 0)
        return label_color
    
    def detection_segments(self, segments, positions):
        """
        self.detection_frames_num分割したときの画像やそのときの画像座標を格納した配列を返す
        
        Parameters:
            segments : 9分割したときの画像を格納した配列
            positions : 9分割した画像の画像座標を格納した配列
        
        Return:
            segments : 9 or 6分割したときの画像を格納した配列
            positions : 9 or 6分割した画像の画像座標を格納した配列
        """
        if self.detection_frames_num == 6:
            segments = segments[3:]
            positions = positions[3:]
            return segments, positions
        elif self.detection_frames_num == 9:
            return segments, positions
        else:
            raise SegmentsNumError("[Error] Choose detection_frame_num other than 6 or 9")
    
    def frame_put_label(self, frame, label, position):
        """
        Parameters:
            frame : 入力フレーム
            label : 雑草のクラスラベル
            position : 分割画像の画像座標
        """
        x, y = position
        label_color = self.get_label_color(label)
        cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    
class SegmentsNumError(Exception):
    pass
        
        