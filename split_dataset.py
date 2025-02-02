"""
指定したフォルダを訓練用データとテスト用データに分割し，それぞれ保存する
"""
import os
import shutil
import random

def split_images(input_folder, train_folder, val_folder, train_ratio=0.8, seed=42):
    """
    フォルダ内の画像を訓練用と評価用に分割する。

    Args:
        input_folder (str): 元画像フォルダ。
        train_folder (str): 訓練用データの出力フォルダ。
        val_folder (str): 評価用データの出力フォルダ。
        train_ratio (float): 訓練データの割合（デフォルトは80%）。
    """
    # 入力フォルダ内の画像ファイルを取得
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(image_files) == 0:
        print("Error: No image files found in the input folder.")
        return

    # 出力フォルダを作成
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # シャッフル分割
    random.seed(seed)
    random.shuffle(image_files)
    train_count = int(len(image_files) * train_ratio)
    train_images = image_files[:train_count]
    val_images = image_files[train_count:]

    # ファイルを移動
    for image in train_images:
        shutil.copy(os.path.join(input_folder, image), os.path.join(train_folder, image))

    for image in val_images:
        shutil.copy(os.path.join(input_folder, image), os.path.join(val_folder, image))

    print(f"Training data: {len(train_images)} images")
    print(f"Validation data: {len(val_images)} images")


# 使用例
input_folder = "./datasets/reduce/exist_weed"  # 画像が格納されたフォルダ
train_folder = "./datasets/train/exist_weed"   # 訓練用データ出力フォルダ
val_folder = "./datasets/test/exist_weed"       # 評価用データ出力フォルダ

split_images(input_folder, train_folder, val_folder, train_ratio=0.5)

input_folder = "./datasets/reduce/no_weed"  # 画像が格納されたフォルダ
train_folder = "./datasets/train/no_weed"   # 訓練用データ出力フォルダ
val_folder = "./datasets/test/no_weed"       # 評価用データ出力フォルダ
split_images(input_folder, train_folder, val_folder, train_ratio=0.5)

# input_folder = "./flash_imgs/many_weed"  # 画像が格納されたフォルダ
# train_folder = "./datasets/train/2classes/many_weed"   # 訓練用データ出力フォルダ
# val_folder = "./datasets/val/2classes/many_weed"       # 評価用データ出力フォルダ
# split_images(input_folder, train_folder, val_folder, train_ratio=0.6)