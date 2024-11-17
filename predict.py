import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time

# パラメータ設定
batch_size = 16
learning_rate = 0.001
num_epochs = 10
input_size = (522, 318)
num_classes = 3  # 雑草の有無を3クラス分類

# CNNモデル定義
class WeedClassifierCNN(nn.Module):
    def __init__(self):
        super(WeedClassifierCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 畳み込み
            nn.ReLU(),  # 活性化
            nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 畳み込み
            nn.ReLU(),  # 活性化
            nn.MaxPool2d(kernel_size=2, stride=2)  # プーリング
        )

        # 全結合層
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * (input_size[0] // 4) * (input_size[1] // 4), 128),  # フラット化→全結合
            nn.ReLU(),
            nn.Linear(128, num_classes)  # クラス数に合わせた出力
        )

    def forward(self, x):
        x = self.conv_layer(x)  # 畳み込み層
        x = x.view(x.size(0), -1)  # フラット化
        x = self.fc_layer(x)  # 全結合層
        return x
    
# モデルの読み込み
model_path = "./models/weed_classifier.pth"
model = WeedClassifierCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()  # 推論モードに設定

# 推論対象画像の準備
image_path = "./datasets/val/many_weed/frame_0007_1_2_2_factor_1.0.jpg"  # 推論したい画像のパス
input_size = (522, 318)

transform = transforms.Compose([
    transforms.Resize(input_size),  # モデルの入力サイズにリサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
])

image = Image.open(image_path).convert("RGB")  # RGB形式に変換
image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

# 処理時間の計測開始
start_time = time.time()

# 推論
with torch.no_grad():
    outputs = model(image_tensor)  # モデルに入力
    _, predicted = torch.max(outputs, 1)  # 最も高いスコアのクラスを取得

# 処理時間の計測終了
end_time = time.time()

# クラスのラベル（適宜変更）
class_labels = ["No Weed", "Little Weeds", "Many Weeds"]

# 推論結果の表示
print(f"Predicted Class: {class_labels[predicted.item()]}")
print(f"Processing Time: {end_time - start_time:.4f} seconds")