import torch.nn as nn
 
class WeedClassifierCNN(nn.Module):
    def __init__(self, input_size=(522, 318), num_classes=3):
        super(WeedClassifierCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 1層目
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 2層目
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 3層目 (追加)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 4層目 (追加)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全結合層を追加
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * (input_size[0] // 16) * (input_size[1] // 16), 256),  # フラット化→全結合1
            nn.ReLU(),
            nn.Dropout(0.5),  # 過学習防止のためのDropout

            nn.Linear(256, 128),  # 全結合2 (追加)
            nn.ReLU(),

            nn.Linear(128, num_classes)  # 出力層
        )

    def forward(self, x):
        x = self.conv_layer(x)  # 畳み込み層
        x = x.view(x.size(0), -1)  # フラット化
        x = self.fc_layer(x)  # 全結合層
        return x