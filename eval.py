import torch
from utils.CNN_weed_classifier import WeedClassifierCNN

def eval(model_path):
    model = WeedClassifierCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location={'cuda:0': 'cpu'}))
    model.eval()  # 推論モードに設定
    return model