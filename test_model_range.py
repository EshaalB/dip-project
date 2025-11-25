"""Quick test to verify the model now outputs in the correct range"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel

def test_fresh_model():
    print("="*60)
    print("Testing Fresh (Untrained) Model Output Range")
    print("="*60)
    
    # Create a new untrained model
    model = ColorizationModel()
    model.eval()
    
    # Create dummy input (grayscale)
    dummy_input = torch.randn(1, 1, 224, 224) * 0.5 + 0.5  # [0, 1] range
    
    with torch.no_grad():
        ab_pred, bias_map, attention = model(dummy_input)
    
    ab_np = ab_pred[0].cpu().numpy()
    
    print(f"\nModel output shape: {ab_np.shape}")
    print(f"Model output range: [{ab_np.min():.2f}, {ab_np.max():.2f}]")
    print(f"Model output mean: a={ab_np[0].mean():.2f}, b={ab_np[1].mean():.2f}")
    print(f"Model output std: a={ab_np[0].std():.2f}, b={ab_np[1].std():.2f}")
    
    print(f"\nExpected range for LAB ab channels: [-110, 110]")
    print(f"Typical std for color images: 10-30")
    
    if -110 <= ab_np.min() <= 110 and -110 <= ab_np.max() <= 110:
        print("\n✓ Output range is valid!")
    else:
        print("\n✗ Output range is INVALID!")
    
    if 5 <= ab_np[0].std() <= 60 and 5 <= ab_np[1].std() <= 60:
        print("✓ Output variance looks reasonable for training!")
    else:
        print("✗ Output variance may be too low/high")
    
    print("="*60)

if __name__ == "__main__":
    test_fresh_model()
