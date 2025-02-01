import torch.nn as nn
 
class WeedClassifierCNN(nn.Module):
    def __init__(self, input_size = (522, 318), num_classes = 3, num_conv_layers=4):
        super(WeedClassifierCNN, self).__init__()
        
        if num_conv_layers == 2:
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
        
        if num_conv_layers == 4:
            # 畳み込み層（4層）
            self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 畳み込み1
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング1
            
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 畳み込み2
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング2
            
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 畳み込み3
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング3
            
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 畳み込み4
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # プーリング4
            )
        
            # 入力サイズをプーリングの影響で計算（4層分のプーリングで1/16に縮小）
            reduced_size = (input_size[0] // (2 ** 4), input_size[1] // (2 ** 4))
            flattened_size = reduced_size[0] * reduced_size[1] * 128  # 128は最終の出力チャンネル数
        
            # 全結合層
            self.fc_layer = nn.Sequential(
                nn.Linear(flattened_size, 128),  # フラット化→全結合
                nn.ReLU(),
                nn.Linear(128, num_classes)  # クラス数に合わせた出力
            )
        
        
        
    def forward(self, x):
        x = self.conv_layer(x)  # 畳み込み層
        x = x.view(x.size(0), -1)  # フラット化
        x = self.fc_layer(x)  # 全結合層
        return x