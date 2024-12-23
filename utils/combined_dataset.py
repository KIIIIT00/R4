import torch
from torchvision import datasets, transforms

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, root, original_transform, augmentation_transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        original_img = self.original_transform(img)  # 元画像
        augmented_img = self.augmentation_transform(img)  # 水増し画像
        return torch.stack([original_img, augmented_img]), label  # 2つの画像をまとめる