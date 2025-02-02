"""
明るさを3段階に調整する
"""

from PIL import Image, ImageEnhance
import os

def adjusts_brightness_from_multiple_folders(folders, save_folder, brightness_factors=[0.5, 1.0, 1.5]):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for folder_path in folders:
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            adjust_brightness(image_path, save_folder, brightness_factors)

def get_filename_without_extension(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename

def adjust_brightness(image_path, save_folder, brightness_factors):
    # 画像を読み込む
    image = Image.open(image_path)
    file_name = get_filename_without_extension(image_path)

    # 各明るさ調整後の画像を保存
    for i, factor in enumerate(brightness_factors):
        enhancer = ImageEnhance.Brightness(image)
        bright_image = enhancer.enhance(factor)
        
        # 調整した画像を保存
        bright_image.save(os.path.join(save_folder, f"{file_name}_factor_{factor}.jpg"))

# 使用例
input_folders = [
    "./datasets/train/no_weed"
]
output_folder = "./datasets/train/no_weed"

# 複数フォルダの画像に対して明るさを調整
adjusts_brightness_from_multiple_folders(input_folders, output_folder)