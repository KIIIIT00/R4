"""
データセットの画像のデータ数をそろえる
"""
import os
import random
from collections import defaultdict


def balance_image_count(folders):
    """
    3つのフォルダ内の画像枚数を揃える。

    Args:
        folders (list): 画像が保存されているフォルダのパスのリスト（3つ）。

    Returns:
        None
    """
    # 各フォルダ内の画像リストを取得
    image_lists = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 対応する拡張子

    for folder in folders:
        images = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
        image_lists.append(images)
        print(f"{folder} 内の画像枚数: {len(images)}")

    # 最小の画像枚数を取得
    # min_count = min(len(images) for images in image_lists)
    min_count = 4220
    print(f"最小の画像枚数: {min_count}")

    # 各フォルダの画像を最小枚数に揃える
    for folder, images in zip(folders, image_lists):
        if len(images) > min_count:
            # ランダムに画像を削除
            to_remove = random.sample(images, len(images) - min_count)
            for image in to_remove:
                image_path = os.path.join(folder, image)
                os.remove(image_path)
                print(f"削除しました: {image_path}")

    print("すべてのフォルダの画像枚数を揃えました。")

if __name__ == "__main__":
    folder = ['./datasets/train/exist_weed',
              './datasets/train/no_weed',
              ]
    balance_image_count(folder)
    