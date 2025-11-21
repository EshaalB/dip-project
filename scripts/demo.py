# Complete demo script - shows everything for presentation

import torch
import numpy as np
import sys
import os
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb, save_image, color_balance_histogram_remap


def demo(img_path, model_path, output_dir="output", device="cpu"):
    print("=" * 60)
    print("DEMO: Our Pseudo-Colorization Method")
    print("=" * 60)

    # Load image
    print(f"\n1. Loading image: {img_path}")
    img = load_image(img_path, target_size=(256, 256))
    L, a_original, b_original = rgb_to_lab(img)
    print("   ✓ Image loaded")

    # Load model
    print(f"\n2. Loading model: {model_path}")
    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("   ✓ Model loaded")

    # Step 1: Neural network prediction
    print("\n3. Step 1: Neural Network Predicts Colors")
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        color_pred, correction_map, attention = model(L_tensor)

    color_np = color_pred[0].cpu().numpy().transpose(1, 2, 0)
    a_pred = color_np[:, :, 0] * 127.0
    b_pred = color_np[:, :, 1] * 127.0
    print("   ✓ Colors predicted")

    # Step 2: Our original - Correction Map
    print("\n4. Step 2: Apply Correction Map (OUR ORIGINAL)")
    correction_np = correction_map[0].cpu().numpy().transpose(1, 2, 0)
    attention_np = attention[0].cpu().numpy().transpose(1, 2, 0)

    # Model already applies correction via fusion, but show the map
    print("   ✓ Correction map applied (fixes local mistakes)")
    print(f"   - Correction map range: [{correction_np.min():.3f}, {correction_np.max():.3f}]")
    print(f"   - Attention map range: [{attention_np.min():.3f}, {attention_np.max():.3f}]")

    # Step 3: Our original - Color Balance
    print("\n5. Step 3: Apply Color Balance (OUR ORIGINAL)")
    a_before = np.mean(a_pred)
    b_before = np.mean(b_pred)
    a_balanced, b_balanced = color_balance_histogram_remap(
        np.clip(a_pred, -127, 127),
        np.clip(b_pred, -127, 127)
    )
    a_after = np.mean(a_balanced)
    b_after = np.mean(b_balanced)
    print(f"   ✓ Color balance applied (fixes global tint)")
    print(f"   - Before: a_mean={a_before:.2f}, b_mean={b_before:.2f}")
    print(f"   - After:  a_mean={a_after:.2f}, b_mean={b_after:.2f}")

    # Final result
    print("\n6. Final Result")
    rgb_result = lab_to_rgb(L, a_balanced, b_balanced)

    # Save all outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Save grayscale input
    gray_vis = (L / 100.0 * 255.0).clip(0, 255).astype(np.uint8)
    import cv2
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_input_grayscale.jpg"), gray_vis)

    # Save result
    save_image(rgb_result, os.path.join(output_dir, f"{base_name}_output_colorized.jpg"))

    # Save correction map visualization
    correction_vis = ((correction_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    correction_vis = cv2.applyColorMap(correction_vis[:, :, 0], cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_correction_map.jpg"),
                cv2.cvtColor(correction_vis, cv2.COLOR_RGB2BGR))

    # Save attention map
    attention_vis = (attention_np[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    attention_vis = cv2.applyColorMap(attention_vis, cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_attention_map.jpg"),
                cv2.cvtColor(attention_vis, cv2.COLOR_RGB2BGR))

    print(f"\n✓ All files saved to {output_dir}/")
    print(f"  - Input grayscale: {base_name}_input_grayscale.jpg")
    print(f"  - Output colorized: {base_name}_output_colorized.jpg")
    print(f"  - Correction map: {base_name}_correction_map.jpg (OUR CONTRIBUTION 1)")
    print(f"  - Attention map: {base_name}_attention_map.jpg")

    print("\n" + "=" * 60)
    print("OUR ORIGINAL CONTRIBUTIONS:")
    print("1. Correction Map - fixes local color mistakes")
    print("2. Color Balance - fixes global color tint")
    print("3. Attention - guides where corrections are needed")
    print("4. Learned Fusion - intelligently combines everything")
    print("=" * 60)


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/demo.py <image_path> <model_path>")
        print("Example: python scripts/demo.py data/portrait_man.jpg models/colorization_final.pth")
        return

    img_path = sys.argv[1]
    model_path = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    demo(img_path, model_path, device=device)


if __name__ == "__main__":
    main()

