import torch
from torch.utils.data import Dataset
import os
import glob
from src.utils import load_image, rgb_to_lab

class ImageDataset(Dataset):
    def __init__(self, data_folder="data", img_size=(256, 256)):
        self.img_size = img_size
        self.data_folder = data_folder
        
        # Find all images (store paths only, not loaded data)
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(os.path.join(data_folder, ext)))
            self.image_files.extend(glob.glob(os.path.join(data_folder, ext.upper())))

        if len(self.image_files) == 0:
            raise ValueError(f"No images in {data_folder}/")

        print(f"Found {len(self.image_files)} images (will load on-demand to save RAM)")

    def __getitem__(self, idx):
        # Load image on-demand
        file = self.image_files[idx]
        rgb = load_image(file, self.img_size)
        L, a, b = rgb_to_lab(rgb)

        # Normalize L to [0, 1] for model input
        # Keep a, b in LAB units (no normalization) since model outputs LAB units
        L_normalized = L / 100.0

        # Convert to tensors
        L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).float()
        a_tensor = torch.from_numpy(a).float()
        b_tensor = torch.from_numpy(b).float()
        ab_tensor = torch.stack([a_tensor, b_tensor], dim=0)

        return L_tensor, ab_tensor

    def __len__(self):
        return len(self.image_files)
