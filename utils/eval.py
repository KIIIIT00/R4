import cv2
import torch
import time
from torchvision import transforms
from PIL import Image
from utils.CNN_weed_classifier import WeedClassifierCNN

class CNNEval:
    def __init__(self, model_path, image_size=(522, 318), num_classes=3):
        self.model_path = model_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_labels = ["little_weed", "many_weed", "no_weed"]
        self.model = WeedClassifierCNN()
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location={'cuda:0': 'cpu'}))
        self.model.eval() # Setting Eval Mode
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize
        ])
    
    def get_label(self, predicted_item):
        """
        Parameter: 
            predicted_item : モデルの出力クラス(0, 1, 2)
        
        Return:
            predicted_label : 出力ラベル
        """
        if self.num_classes == len(self.class_labels):
            predicted_label = self.class_labels[predicted_item]
        elif self.num_classes < len(self.class_labels):
            if predicted_item == 0:
                predicted_label = 'no_weed'
            else:
                predicted_label = self.class_labels[predicted_item]
        else:
            raise LabelsError("[Error]Number of classes exceeds number of labels")
        
        return predicted_label

    def eval(self, segment_frame, segments_times_list):
        """
        Parameter:
            segment_frame : 推論する画像
            segments_times_list : 画像において，処理時間を格納する配列
            
        Return:
            predicted_label : 予測ラベル
            segments_times_list : 処理時間を格納した配列
        """
        pil_image = Image.fromarray(cv2.cvtColor(segment_frame, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(pil_image).unsqueeze(0)

        segment_start_time = time.time()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        segment_time = time.time() - segment_start_time
        segments_times_list.append(segment_time)
        print(f"Eval Processing Time: {segment_time:.4f} seconds")
        predicted_label =  self.get_label(predicted_item=predicted.item())
        return predicted_label, segments_times_list
    

class LabelsError(Exception):
    pass
