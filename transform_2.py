"""
上下左右盲フォルダー（画像サイズ維持、拡大処理修正）
"""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms


# 必要なフォルダーのパスを指定
input_folder = "./datasets/train/exist_weed"  # 元画像が保存されているフォルダー
output_folder = "./datasets/train/exist_weed"  # 加工後の画像を保存するフォルダー

# 出力フォルダーを作成（存在しない場合）
os.makedirs(output_folder, exist_ok=True)

# データ拡張操作の定義
# 1. 上下反転
transform_vertical_flip = transforms.RandomVerticalFlip(p=1.0)

# 2. 左右反転
transform_horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

# 3. 上下左右反転
def vertical_horizontal_flip(image):
    vertical_flipped = transforms.RandomVerticalFlip(p=1.0)(image)
    both_flipped = transforms.RandomHorizontalFlip(p=1.0)(vertical_flipped)
    return both_flipped

# 4. 拡大（画像サイズ維持）
def transform_scale_up(image):
    scale_factor = 1.1 + 0.4 * torch.rand(1).item()  # 拡大率を1.1~1.5の範囲でランダムに設定
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 一時的に拡大した画像を生成
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 拡大した画像を元のサイズにクロップ（中心を維持）
    left = (new_width - original_width) // 2
    top = (new_height - original_height) // 2
    right = left + original_width
    bottom = top + original_height
    return scaled_image.crop((left, top, right, bottom))

# 5. 縮小（縦横比を維持）
def transform_scale_down(image):
    scale_factor = 0.8  # 縮小率を固定 (80%)
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 縮小した画像を元のサイズにリサイズ
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return scaled_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

# フォルダー内の全ての画像に対して変換を適用
def process_images(input_folder, output_folder):
    # サポートされる画像形式のみをフィルタリングし、隠しファイルを除外
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg')) and not f.startswith('.')]

    for i, image_file in enumerate(image_files):
        # 元画像を開く
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        # オリジナル画像を保存
        # image.save(os.path.join(output_folder, f"image_{i+1}_original.jpg"))

        # 各変換を適用
        image_vflip = transform_vertical_flip(image)  # 上下反転
        image_hflip = transform_horizontal_flip(image)  # 左右反転
        image_vhflip = vertical_horizontal_flip(image)  # 上下左右反転
        image_scaled_up = transform_scale_up(image)  # 拡大（画像サイズ維持）
        image_scaled_down = transform_scale_down(image)  # 縮小（画像サイズ維持）

        # 保存
        image_vflip.save(os.path.join(output_folder, f"image_{i+1}_vflip.jpg"))
        image_hflip.save(os.path.join(output_folder, f"image_{i+1}_hflip.jpg"))
        image_vhflip.save(os.path.join(output_folder, f"image_{i+1}_vhflip.jpg"))
        image_scaled_up.save(os.path.join(output_folder, f"image_{i+1}_scaled_up.jpg"))
        image_scaled_down.save(os.path.join(output_folder, f"image_{i+1}_scaled_down.jpg"))

# 実行
process_images(input_folder, output_folder)
print(f"Processing complete. Transformed images are saved in '{output_folder}'.")
