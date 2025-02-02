import os

def delete_flash_imgs(dir):
    for filename in os.listdir(dir):
        if filename.endswith("1.0.jpg"):
            file_path = os.path.join(dir, filename)
            if  os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {filename}")

if __name__ == "__main__":
    delete_flash_imgs("./datasets/train/no_weed")