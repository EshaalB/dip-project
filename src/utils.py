import numpy as np
import cv2
import os
import glob
import torch
import kornia


def rgbtoLab(rgb_image):
    rgb_float = rgb_image.astype(np.float32) / 255.0
    bgr = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    return L, a, b


def labtoRGB(L, a, b):
    L_clipped = np.clip(L, 0, 100).astype(np.float32)
    a_clipped = np.clip(a, -127, 127).astype(np.float32)
    b_clipped = np.clip(b, -127, 127).astype(np.float32)

    #change range from -127-127 to 0-255
    a_shifted = a_clipped + 128.0
    b_shifted = b_clipped + 128.0

    #conver to int
    L_uint8 = np.clip(L_clipped * 2.55, 0, 255).astype(np.uint8)
    a_uint8 = np.clip(a_shifted, 0, 255).astype(np.uint8)
    b_uint8 = np.clip(b_shifted, 0, 255).astype(np.uint8)

    lab = np.stack([L_uint8, a_uint8, b_uint8], axis=2)

    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb

def labABtoRGB(ab, L):
    L_scaled = L * 100.0

    a = ab[:, 0:1, :, :] * 127.0
    b = ab[:, 1:1+1, :, :] * 127.0


    lab = torch.cat([L_scaled, a, b], dim=1)

    return kornia.color.lab_to_rgb(lab)

def loadImage(image_path, target_size=(256, 256)):
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


def saveImg(image, output_path):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)

def verifyDataset(dataFolder="data"):
    # Verify dataset is set up correctly
    imageExtensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imageFiles = []

    for ext in imageExtensions:
        imageFiles.extend(glob.glob(os.path.join(dataFolder, ext)))
        imageFiles.extend(glob.glob(os.path.join(dataFolder, ext.upper())))

    if len(imageFiles) == 0:
        print(f"ERROR: No images found in {dataFolder}/")
        return False

    print(f"Found {len(imageFiles)} images in {dataFolder}/")

    for imgPath in imageFiles[:3]:
        try:
            img = loadImage(imgPath)
            L, a, b = rgbtoLab(img)
            print(f"Found: {os.path.basename(imgPath)}: {img.shape}")
        except Exception as e:
            print(f" {os.path.basename(imgPath)}: Error - {e}")
            return False

    print("Dataset verification passed!")
    return True
