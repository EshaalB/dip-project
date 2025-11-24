# Utility functions for pseudo-colorization project

import numpy as np
import cv2
import os
import glob
import torch
import torch.nn as nn


def rgb_to_lab(rgb_image):
    # Convert RGB to Lab color space
    rgb_float = rgb_image.astype(np.float32) / 255.0
    bgr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    return L, a, b


def lab_to_rgb(L, a, b):
    # Convert Lab to RGB color space with proper range handling
    # Keep as float32 throughout to preserve precision until final conversion
    L_clipped = np.clip(L, 0, 100).astype(np.float32)
    a_clipped = np.clip(a, -127, 127).astype(np.float32)
    b_clipped = np.clip(b, -127, 127).astype(np.float32)

    # Shift a and b to [0, 255] range for OpenCV LAB format
    # OpenCV LAB: L in [0, 100], a/b in [0, 255] (representing -127 to 127)
    a_shifted = a_clipped + 128.0
    b_shifted = b_clipped + 128.0

    # Stack LAB channels - convert to uint8 only for OpenCV cvtColor
    # This preserves precision better than converting earlier
    L_uint8 = np.clip(L_clipped, 0, 100).astype(np.uint8)
    a_uint8 = np.clip(a_shifted, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(b_shifted, 0, 255).astype(np.uint8)

    lab = np.stack([L_uint8, a_uint8, b_uint8], axis=2)

    # Convert LAB to BGR, then BGR to RGB
    # OpenCV cvtColor works with uint8 and handles the conversion precisely
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb

def lab_to_rgb_tensor(ab, L):
    # L: (B,1,H,W)  [0..100], ab: [-1..1] scaled to real LAB
    L_scaled = L * 100.0

    a = ab[:, 0:1, :, :] * 127.0
    b = ab[:, 1:1+1, :, :] * 127.0

    lab = torch.cat([L_scaled, a + 128.0, b + 128.0], dim=1)

    # Convert using OpenCV-like transformation via kornia
    import kornia
    return kornia.color.lab_to_rgb(lab)



def load_image(image_path, target_size=(256, 256)):
    # Load and resize image with validation
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image (invalid format?): {image_path}")

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must be RGB (3 channels), got shape: {img.shape}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size[::-1], interpolation=cv2.INTER_AREA)

    return img_resized


def save_image(image, output_path):
    # Save RGB image to file
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)


def color_balance_histogram_remap(a, b, target_mean_a=0.0, target_mean_b=0.0):
    # ORIGINAL CONTRIBUTION 2: Color Balance Correction
    # Fixes global color tint by adjusting mean of a and b channels
    # This prevents yellow tint and keeps colors natural
    a_balanced = np.clip(a - np.mean(a) + target_mean_a, -127, 127)
    b_balanced = np.clip(b - np.mean(b) + target_mean_b, -127, 127)
    return a_balanced, b_balanced


def verify_dataset(data_dir="data"):
    # Verify dataset is set up correctly
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))

    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}/")
        return False

    print(f"Found {len(image_files)} images in {data_dir}/")

    for img_path in image_files[:3]:
        try:
            img = load_image(img_path)
            L, a, b = rgb_to_lab(img)
            print(f"  [OK] {os.path.basename(img_path)}: {img.shape}")
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: Error - {e}")
            return False

    print("Dataset verification passed!")
    return True
