import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from utils.CNN_weed_classifier import WeedClassifierCNN

# パラメータ設定
batch_size = 16
epochs = 100
learning_rate = 0.001
input_size = (522, 318)
num_classes = 2
k_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# フォルダ設定
data_dir = "./datasets/"
train_dir = os.path.join(data_dir, "train")
model_dir = "./models/"
results_dir = "./results/"

# 保存フォルダ作成
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# データ変換
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# データセットの読み込み
full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
class_names = full_dataset.classes  # ["no_weed", "exist_weed"]
labels = np.array(full_dataset.targets)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Foldごとの結果を記録
best_fold = None
best_accuracy = 0.0
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n### Fold {fold+1}/{k_folds} 開始 ###")

    # Foldごとにデータを分割
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    model = WeedClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # --- 学習 ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # --- 検証 ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        print(f"Fold {fold+1}, Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    # Foldごとの評価
    if epoch_val_acc > best_accuracy:
        best_accuracy = epoch_val_acc
        best_fold = fold + 1
        best_model_path = os.path.join(model_dir, f"best_model.pth")
        torch.save(model.state_dict(), best_model_path)

    fold_results.append((fold + 1, epoch_val_acc))

    # Foldごとの評価保存
    fold_report_path = os.path.join(results_dir, f"classification_report_fold{fold+1}.txt")
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(fold_report_path, "w") as f:
        f.write(report)

    # 混同行列を保存
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    conf_matrix_path = os.path.join(results_dir, f"confusion_matrix_fold{fold+1}.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # 損失のグラフを保存
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_loss_history, label="Train Loss", marker="o")
    plt.plot(range(1, epochs+1), val_loss_history, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.legend()
    plt.grid()
    loss_curve_path = os.path.join(results_dir, f"loss_curve_fold{fold+1}.png")
    plt.savefig(loss_curve_path)
    plt.close()

    # 正解率のグラフを保存
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_acc_history, label="Train Accuracy", marker="o")
    plt.plot(range(1, epochs+1), val_acc_history, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve - Fold {fold+1}")
    plt.legend()
    plt.grid()
    acc_curve_path = os.path.join(results_dir, f"accuracy_curve_fold{fold+1}.png")
    plt.savefig(acc_curve_path)
    plt.close()

    print(f"損失曲線と正解率曲線を保存しました: {loss_curve_path}, {acc_curve_path}")

    # 学習履歴をJSONで保存
    history_data = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history
    }

    history_json_path = os.path.join(results_dir, f"training_history_fold{fold+1}.json")
    with open(history_json_path, "w") as json_file:
        json.dump(history_data, json_file, indent=4)

    print(f"学習履歴を保存しました: {history_json_path}")
    
    print(f"Fold {fold+1} の結果を保存しました")

# K-Fold 結果を保存
results_path = os.path.join(results_dir, "kfold_results.txt")
with open(results_path, "w") as f:
    for fold, acc in fold_results:
        f.write(f"Fold {fold}: Validation Accuracy: {acc:.2f}%\n")
    f.write(f"\nBest Model from Fold {best_fold} with Accuracy: {best_accuracy:.2f}%\n")

print(f"\n### K-Fold の結果を {results_path} に保存しました ###")

