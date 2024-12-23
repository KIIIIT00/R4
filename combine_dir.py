"""
指定した複数のフォルダの中にあるファイルをあるフォルダに保存する
"""
import os
import cv2
import shutil

def combine(soruce_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    saved_count = 0
    for folder in soruce_folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1]
                new_filename = f"frame_{saved_count:06d}{file_extension}"
                new_file_path = os.path.join(destination_folder, new_filename)

                shutil.copy(file_path, new_file_path)
                print(f"Rename: {filename} to {new_filename}")

                saved_count += 1
    
    print("Save all")

if __name__ == '__main__':
    soruce_folders = ['./datasets/train/2classes/little_weed', './datasets/train/2classes/many_weed']
    destination_folder = './datasets/train/2classes/exist_weed'
    combine(soruce_folders, destination_folder)
