import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from src.utils import load_image, rgb_to_lab

class ImageDataset(Dataset):
    """Dataset that preloads all images into memory"""
    def __init__(self, data_folder="data", img_size=(256, 256), augment=True):
        self.img_size = img_size
        self.augment = augment
        self.images = []

        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(data_folder, ext)))
            image_files.extend(glob.glob(os.path.join(data_folder, ext.upper())))

        if len(image_files) == 0:
            raise ValueError(f"No images found in {data_folder}/")

        print(f"Loading {len(image_files)} images...")

        # Preload images (ok for small datasets, won't scale to thousands)
        # TODO: Implement lazy loading for larger datasets
        failed_files = []
        for i, file_path in enumerate(image_files, 1):
            try:
                rgb = load_image(file_path, self.img_size)
                L, a, b = rgb_to_lab(rgb)

                # Normalize to [-1, 1] or [0, 1]
                L_norm = L / 100.0
                a_norm = a / 127.0
                b_norm = b / 127.0

                # To tensors
                L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
                ab_tensor = torch.stack([
                    torch.from_numpy(a_norm).float(),
                    torch.from_numpy(b_norm).float()
                ], dim=0)

                self.images.append((L_tensor, ab_tensor))

                if i % 50 == 0 or i == len(image_files):
                    print(f"  {i}/{len(image_files)} loaded")
            except Exception as e:
                failed_files.append((os.path.basename(file_path), str(e)))
                print(f"  Warning: Skipped {os.path.basename(file_path)} - {e}")

        if failed_files:
            print(f"\nWarning: {len(failed_files)} files failed to load")

        if len(self.images) == 0:
            raise ValueError(f"No valid images loaded from {data_folder}/")

        print("Done loading dataset")

    def __getitem__(self, idx):
        L_tensor, ab_tensor = self.images[idx]

        # Simple augmentation: random horizontal flip
        if self.augment and np.random.rand() > 0.5:
            L_tensor = torch.flip(L_tensor, [2])  # Flip width dimension
            ab_tensor = torch.flip(ab_tensor, [2])

        return L_tensor, ab_tensor

    def __len__(self):
        return len(self.images)
