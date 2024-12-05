import torch.nn as nn
 
class WeedClassifierCNN(nn.Module):
    def __init__(self, input_size = (522, 318), num_classes = 3):
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