import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class WeedClassifier:
    def __init__(self, data_path, image_size=(640, 480), n_estimators = 300, model_name = "random_forest_model.joblib"):
        self.data_path = data_path
        self.image_size = image_size
        self.model_name = model_name
        self.model_path = os.path.join("./models", self.model_name)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def load_images_and_labels(self):
        images = []
        labels = []
        label_folders = {'no_weed': 0, 'exist_weed': 1}
        for label_name, label in label_folders.items():
            folder_path = os.path.join(self.data_path, label_name)
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    images.append(img.flatten())  # 1次元ベクトルに変換
                    labels.append(label)
        return np.array(images), np.array(labels)

    def train(self):
        X, y = self.load_images_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training accuracy: {accuracy:.2f}")
        return accuracy

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or could not be loaded.")
        img = cv2.resize(img, self.image_size)
        img_flat = img.flatten().reshape(1, -1)  # 1次元ベクトルに変換し、入力用に整形
        prediction = self.model.predict(img_flat)
        return prediction[0]

if __name__ == '__main__':
    # データセットのパス
    data_path = "path/to/dataset"

    # クラスのインスタンス化
    classifier = WeedClassifier(data_path)

    # モデルの学習
    classifier.train()

    # モデルの保存
    classifier.save_model()

    # モデルの読み込み
    classifier.load_model()

    # 新しい画像での予測
    image_path = "path/to/image.jpg"
    prediction = classifier.predict(image_path)
    print("Prediction:", "exist_weed" if prediction == 1 else "no_weed")