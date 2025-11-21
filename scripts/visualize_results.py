# Visualize original contributions: bias map and attention
# This shows what makes our method unique!

import torch
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb, save_image


def load_model(model_path, device="cpu"):
    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def visualize_contributions(img_path, model_path, output_dir="output", device="cpu"):
    # Show our original contributions visually
    print(f"Loading image: {img_path}")
    img = load_image(img_path, target_size=(256, 256))
    L, _, _ = rgb_to_lab(img)

    # Load model
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    # Run model
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        color_pred, correction_map, attention = model(L_tensor)

    # Convert to numpy
    color_np = color_pred[0].cpu().numpy().transpose(1, 2, 0)
    correction_np = correction_map[0].cpu().numpy().transpose(1, 2, 0)
    attention_np = attention[0].cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    a_pred = color_np[:, :, 0] * 127.0
    b_pred = color_np[:, :, 1] * 127.0

    # Apply our original: color balance
    from src.utils import color_balance_histogram_remap
    a_balanced, b_balanced = color_balance_histogram_remap(a_pred, b_pred)

    # Final result
    rgb_result = lab_to_rgb(L, a_balanced, b_balanced)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Save main result
    save_image(rgb_result, os.path.join(output_dir, f"{base_name}_colorized.jpg"))
    print(f"✓ Saved colorized image")

    # Visualize correction map (our original contribution 1)
    correction_vis = ((correction_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    correction_vis = cv2.applyColorMap(correction_vis[:, :, 0], cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_correction_map.jpg"),
                cv2.cvtColor(correction_vis, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved correction map (our original contribution)")

    # Visualize attention map (shows where corrections are applied)
    attention_vis = (attention_np[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    attention_vis = cv2.applyColorMap(attention_vis, cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_attention_map.jpg"),
                cv2.cvtColor(attention_vis, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved attention map")

    # Compare: with and without color balance (our original contribution 2)
    rgb_no_balance = lab_to_rgb(L, a_pred, b_pred)
    save_image(rgb_no_balance, os.path.join(output_dir, f"{base_name}_no_balance.jpg"))
    print(f"✓ Saved comparison (without color balance)")

    print(f"\nAll results saved to {output_dir}/")
    print("This demonstrates our original contributions:")
    print("1. Correction Map - fixes local color mistakes")
    print("2. Color Balance - fixes global color tint")


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/visualize_results.py <image_path> <model_path>")
        print("Example: python scripts/visualize_results.py data/portrait_man.jpg models/colorization_final.pth")
        return

    img_path = sys.argv[1]
    model_path = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visualize_contributions(img_path, model_path, output_dir="output", device=device)


if __name__ == "__main__":
    main()

