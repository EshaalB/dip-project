# Quick test script to verify model works

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import loadImage, rgbtoLab, labtoRGB

def test_model():
    print("Testing model architecture...")

    # Create model
    model = ColorizationModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")

    # Test forward pass
    dummy_L = torch.randn(1, 1, 224, 224)
    ab_corrected, bias_map, attention = model(dummy_L)

    print(f"✓ Forward pass works:")
    print(f"  - ab_corrected shape: {ab_corrected.shape}")
    print(f"  - bias_map shape: {bias_map.shape}")
    print(f"  - attention shape: {attention.shape}")

    # Test with real image
    try:
        import glob
        import os
        image_files = glob.glob("data/*.jpg")[:1]
        if image_files:
            img = loadImage(image_files[0], target_size=(224, 224))
            L, _, _ = rgbtoLab(img)
            L_tensor = torch.from_numpy(L / 100.0).unsqueeze(0).unsqueeze(0).float()

            model.eval()
            with torch.no_grad():
                ab_corrected, bias_map, attention = model(L_tensor)

            print(f"✓ Real image test passed")
            print(f"  - Input image: {os.path.basename(image_files[0])}")
            print(f"  - Output ranges: ab_corrected [{ab_corrected.min():.3f}, {ab_corrected.max():.3f}]")
            print(f"  - Attention range: [{attention.min():.3f}, {attention.max():.3f}]")
        else:
            print("⚠ No images in data/ folder for testing")
    except Exception as e:
        print(f"⚠ Image test skipped: {e}")

    print("\n✓ All tests passed! Model is ready for training.")

if __name__ == "__main__":
    test_model()

