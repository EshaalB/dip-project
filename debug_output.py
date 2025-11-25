import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb

def debug_model_output():
    """Debug what the model is actually outputting"""
    
    # Load model
    model_path = "models/colorization_best.pth"
    device = "cpu"
    
    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load a sample image
    img = load_image("data/sample.jpg", target_size=(224, 224))
    L, a_gt, b_gt = rgb_to_lab(img)
    
    # Prepare input
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float()
    
    # Get model output
    with torch.no_grad():
        ab_pred, bias_map, attention = model(L_tensor)
    
    ab_np = ab_pred[0].cpu().numpy()
    
    print("=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    print(f"\nModel output shape: {ab_np.shape}")
    print(f"Model output range: [{ab_np.min():.4f}, {ab_np.max():.4f}]")
    print(f"Model output mean: a={ab_np[0].mean():.4f}, b={ab_np[1].mean():.4f}")
    print(f"Model output std: a={ab_np[0].std():.4f}, b={ab_np[1].std():.4f}")
    
    print(f"\n Ground Truth (Lab units [-127, 127]):")
    print(f"GT range: a=[{a_gt.min():.2f}, {a_gt.max():.2f}], b=[{b_gt.min():.2f}, {b_gt.max():.2f}]")
    print(f"GT mean: a={a_gt.mean():.2f}, b={b_gt.mean():.2f}")
    print(f"GT std: a={a_gt.std():.2f}, b={b_gt.std():.2f}")
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    # Test different scaling factors
    scalings = [1.0, 10.0, 50.0, 100.0, 110.0, 127.0]
    
    for scale in scalings:
        a_scaled = ab_np[0] * scale
        b_scaled = ab_np[1] * scale
        
        # Compare variance with ground truth
        a_var_ratio = np.var(a_scaled) / (np.var(a_gt) + 1e-6)
        b_var_ratio = np.var(b_scaled) / (np.var(b_gt) + 1e-6)
        
        print(f"\nScale {scale:6.1f}x:")
        print(f"  Range: a=[{a_scaled.min():7.2f}, {a_scaled.max():7.2f}], b=[{b_scaled.min():7.2f}, {b_scaled.max():7.2f}]")
        print(f"  Variance ratio to GT: a={a_var_ratio:.3f}, b={b_var_ratio:.3f}")
        
        if 0.5 <= a_var_ratio <= 2.0 and 0.5 <= b_var_ratio <= 2.0:
            print(f"  *** GOOD MATCH! ***")

if __name__ == "__main__":
    # Find first image in data folder
    import glob
    images = glob.glob("data/*.jpg") + glob.glob("data/*.png")
    if images:
        os.system(f"cp {images[0]} data/sample.jpg")
    
    debug_model_output()
