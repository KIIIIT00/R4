import os

def delete_flash_imgs(dir):
    for filename in os.listdir(dir):
        if filename.endswith(".5.jpg"):
            file_path = os.path.join(dir, filename)
            if  os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {filename}")

if __name__ == "__main__":
    delete_flash_imgs("./flash_imgs/little_weed")
    delete_flash_imgs("./flash_imgs/many_weed")
    delete_flash_imgs("./flash_imgs/no_weed")