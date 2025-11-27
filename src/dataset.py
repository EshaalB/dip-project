import torch
from torch.utils.data import Dataset
import os
import glob
from src.utils import loadImage, rgbtoLab

class ImageDataset(Dataset):
    def __init__(self, data_folder="data", img_size=(256, 256)):
        self.img_size = img_size
        self.data_folder = data_folder
        
        self.imageFiles = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.imageFiles.extend(glob.glob(os.path.join(data_folder, ext)))
            self.imageFiles.extend(glob.glob(os.path.join(data_folder, ext.upper())))

        if len(self.imageFiles) == 0:
            raise ValueError(f"No images in {data_folder}/")

        print(f"Found {len(self.imageFiles)} images")

    def __getitem__(self, idx):
        file = self.imageFiles[idx]
        rgb = loadImage(file, self.img_size)
        L, a, b = rgbtoLab(rgb)


        L_normalized = L / 100.0

        L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).float()
        a_tensor = torch.from_numpy(a).float()
        b_tensor = torch.from_numpy(b).float()
        ab_tensor = torch.stack([a_tensor, b_tensor], dim=0)

        return L_tensor, ab_tensor

    def __len__(self):
        return len(self.imageFiles)
