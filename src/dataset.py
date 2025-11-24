import torch
from torch.utils.data import Dataset
import os
import glob
from src.utils import load_image, rgb_to_lab

class ImageDataset(Dataset):
    def __init__(self, data_folder="data", img_size=(256, 256)):
        self.img_size = img_size
        self.images = []

        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(data_folder, ext)))
            image_files.extend(glob.glob(os.path.join(data_folder, ext.upper())))

        if len(image_files) == 0:
            raise ValueError(f"No images in {data_folder}/")

        print(f"Found {len(image_files)} images, loading into memory...")

        # Preload all images
        for i, file in enumerate(image_files, 1):
            rgb = load_image(file, self.img_size)
            L, a, b = rgb_to_lab(rgb)

            # Normalize for model
            L_normalized = L / 100.0
            a_normalized = a / 127.0
            b_normalized = b / 127.0

            # Convert to tensors
            L_tensor = torch.from_numpy(L_normalized).unsqueeze(0).float()
            a_tensor = torch.from_numpy(a_normalized).float()
            b_tensor = torch.from_numpy(b_normalized).float()
            ab_tensor = torch.stack([a_tensor, b_tensor], dim=0)

            self.images.append((L_tensor, ab_tensor))

            if i % 100 == 0 or i == len(image_files):
                print(f"Loaded {i}/{len(image_files)} images into memory")

        print("All images loaded into memory!")

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)
