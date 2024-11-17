"""
損失と正解率をグラフ表示する
"""
import json
import matplotlib.pyplot as plt
EP = 30
JSON_FILE_PATH = f"./models/weed_classifier_metrics_ep{EP}.json"
# JSONファイルの読み込み
with open(JSON_FILE_PATH, "r") as f:
    metrics = json.load(f)

train_losses = metrics["train_losses"]
val_losses = metrics["val_losses"]
train_accuracies = metrics["train_accuracies"]
val_accuracies = metrics["val_accuracies"]

# エポック数の取得
epochs = range(1, len(train_losses) + 1)

# 損失関数のグラフ
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss", marker="o")
plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"./figures/loss_curve_ep{EP}.png")  # グラフをファイルに保存
plt.show()

# 正解率のグラフ
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig(f"./figures/accuracy_curve_ep{EP}.png")  # グラフをファイルに保存
plt.show()