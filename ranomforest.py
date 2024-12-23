"""
雑草のあるなしをランダムフォレストで判定する
"""
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data_path = "./datasets/train/" # 各画像ファイルが「no_weed」，「exist_weed」に分けとく
val_path = "./datasets/val/" 
image_size = (522, 318)

# 画像とラベルの読み込み
def load_images_and_labels(data_path):
    images = []
    labels = []
    label_folders = {'no_weed': 0, 'little_weed': 1, 'many_weed':2}
    for label_name, label in label_folders.items():
        folder_path = os.path.join(data_path, label_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img.flatten())  # 1次元ベクトルに変換
                labels.append(label)
    return np.array(images), np.array(labels)

# データの読み込み
X, y = load_images_and_labels(data_path)

# 訓練データとテストデータに分割
train_images, train_labels  = load_images_and_labels(data_path)
test_images, test_labels = load_images_and_labels(val_path)

# ランダムフォレストのモデルを作成
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_images, train_labels)

# 学習済みモデルを保存
model_name = "random_forest_weed_3classes.joblib"
model_path = os.path.join("./models", model_name)
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

# テストデータで予測
y_pred = rf_model.predict(test_images)

# 正解率を表示
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy:.2f}")