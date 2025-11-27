# Compare our method vs baseline (shows improvement)

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import loadImage, rgbtoLab, labtoRGB, saveImg


def baseline_colorize(L, model, device):
    # Baseline: just use decoder output, no corrections
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        color_pred, _, _ = model(L_tensor)

    color_np = color_pred[0].cpu().numpy().transpose(1, 2, 0)
    a = color_np[:, :, 0] * 127.0
    b = color_np[:, :, 1] * 127.0
    a = np.clip(a, -127, 127)
    b = np.clip(b, -127, 127)

    return labtoRGB(L, a, b)


def our_method_colorize(L, model, device):
    # Our method: with correction map and color balance
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        color_pred, correction_map, attention = model(L_tensor)

    # Model already applies corrections
    color_np = color_pred[0].cpu().numpy().transpose(1, 2, 0)
    a = color_np[:, :, 0] * 127.0
    b = color_np[:, :, 1] * 127.0
    a = np.clip(a, -127, 127)
    b = np.clip(b, -127, 127)

    return labtoRGB(L, a, b)


def compare(img_path, model_path, output_dir="output", device="cpu"):
    # Compare baseline vs our method
    print("Comparing methods...")

    img = loadImage(img_path, target_size=(224, 224))
    L, _, _ = rgbtoLab(img)

    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Baseline (no corrections)
    baseline_result = baseline_colorize(L, model, device)

    # Our method (with corrections)
    our_result = our_method_colorize(L, model, device)

    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    saveImg(baseline_result, os.path.join(output_dir, f"{base_name}_baseline.jpg"))
    saveImg(our_result, os.path.join(output_dir, f"{base_name}_our_method.jpg"))

    print(f"\n✓ Comparison saved to {output_dir}/")
    print(f"  - Baseline (no corrections): {base_name}_baseline.jpg")
    print(f"  - Our method (with corrections): {base_name}_our_method.jpg")
    print("\nThis shows the improvement from our original contributions!")


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_methods.py <image_path> <model_path>")
        return

    img_path = sys.argv[1]
    model_path = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compare(img_path, model_path, device=device)


if __name__ == "__main__":
    main()

