"""
CNNでクラス分類をする
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json

from utils.CNN_weed_classifier import WeedClassifierCNN

# パラメータ設定
batch_size = 16
learning_rate = 0.001
num_epochs = 30
input_size = (522, 318)
num_classes = 3  # 雑草の有無を3クラス分類

# フォルダの設定
data_dir = "./datasets/"
train_dir = os.path.join(data_dir, "train")  # 訓練データフォルダ
val_dir = os.path.join(data_dir, "val")      # 検証データフォルダ

# データ前処理
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 標準化
])

# データセットとデータローダー
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# モデル、損失関数、最適化手法
model = WeedClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_accuracies = []
val_accuracies = []

# 損失関数を保存するリスト
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        running_loss += loss.item()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss/len(train_loader))
    train_accuracies.append(epoch_train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 検証
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss /len(val_loader)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(100*correct/total)
    
    print(f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

metrics = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}
with open(f"./models/weed_classifier_metrics_ep{num_epochs}.json", "w") as f:
    json.dump(metrics, f)
print("損失と正解率を保存しました")
# モデルの保存
torch.save(model.state_dict(), f"./models/weed_classifier_ep{num_epochs}.pth")
print("モデルを保存しました")
