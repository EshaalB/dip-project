
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import rgb_to_lab, lab_to_rgb

def test_conversion():
    print("Testing Color Conversion...")
    
    # Test Case 1: Pure White
    # RGB: [255, 255, 255] -> Lab: [100, 0, 0] (approx)
    rgb_white = np.ones((10, 10, 3), dtype=np.uint8) * 255
    L, a, b = rgb_to_lab(rgb_white)
    
    print(f"White RGB [255, 255, 255] -> L mean: {np.mean(L):.2f} (Expected ~100)")
    
    # Convert back
    rgb_back = lab_to_rgb(L, a, b)
    print(f"White Lab -> RGB mean: {np.mean(rgb_back):.2f} (Expected ~255)")
    
    if np.mean(rgb_back) < 200:
        print("FAIL: White converted back to RGB is too dark!")
    else:
        print("PASS: White conversion looks correct.")

    # Test Case 2: Pure Red
    # RGB: [255, 0, 0]
    rgb_red = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_red[:, :, 0] = 255
    
    L, a, b = rgb_to_lab(rgb_red)
    print(f"Red RGB [255, 0, 0] -> L mean: {np.mean(L):.2f}")
    
    rgb_back = lab_to_rgb(L, a, b)
    r_mean = np.mean(rgb_back[:, :, 0])
    print(f"Red Lab -> RGB R-channel mean: {r_mean:.2f} (Expected ~255)")

if __name__ == "__main__":
    test_conversion()
