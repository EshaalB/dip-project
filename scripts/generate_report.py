import torch
import numpy as np
import sys
import os
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import loadImage, rgbtoLab, labtoRGB, saveImg


def generate_report(model_path, data_folder="data", output_dir="output", device="cpu", num_images=5):
    print("=" * 60)
    print("GENERATING RESULTS REPORT")
    print("=" * 60)

    # Load model
    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    import glob
    image_files = glob.glob(os.path.join(data_folder, "*.jpg"))[:num_images]

    if len(image_files) == 0:
        print("No images found")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)

    results = []

    for img_path in image_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")

        # Load original
        rgb_original = loadImage(img_path, target_size=(224, 224))
        L, a_target, b_target = rgbtoLab(rgb_original)

        # Colorize
        L_norm = L / 100.0
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            color_pred, correction_map, attention = model(L_tensor)

        color_np = color_pred[0].cpu().numpy().transpose(1, 2, 0)
        a_pred = color_np[:, :, 0] * 127.0
        b_pred = color_np[:, :, 1] * 127.0
        a_balanced = np.clip(a_pred, -127, 127)
        b_balanced = np.clip(b_pred, -127, 127)

        # Final result
        rgb_result = labtoRGB(L, a_balanced, b_balanced)

        # Compute metrics
        original_lab = cv2.cvtColor(rgb_original.astype(np.float32), cv2.COLOR_RGB2LAB)
        result_lab = cv2.cvtColor(rgb_result.astype(np.float32), cv2.COLOR_RGB2LAB)

        # Brightness preservation (L channel difference)
        brightness_diff = np.mean(np.abs(original_lab[:, :, 0] - result_lab[:, :, 0]))

        # Color accuracy (a, b channel difference)
        color_diff = np.mean(np.abs(original_lab[:, :, 1:] - result_lab[:, :, 1:]))

        # Save images
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        saveImg(rgb_original, os.path.join(output_dir, "report", f"{base_name}_original.jpg"))
        saveImg(rgb_result, os.path.join(output_dir, "report", f"{base_name}_result.jpg"))

        # Save grayscale
        gray = (L / 100.0 * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "report", f"{base_name}_grayscale.jpg"), gray)

        results.append({
            'name': base_name,
            'brightness_error': brightness_diff,
            'color_error': color_diff
        })

        print(f"  Brightness Error: {brightness_diff:.2f} (lower is better)")
        print(f"  Color Error: {color_diff:.2f} (lower is better)")

    # Generate summary
    avg_brightness = np.mean([r['brightness_error'] for r in results])
    avg_color = np.mean([r['color_error'] for r in results])

    report_text = f"""
RESULTS REPORT
==============

Our Method: Pseudo-Colorization with Correction Map and Color Balance

Dataset: {len(image_files)} images
Model: {os.path.basename(model_path)}

AVERAGE RESULTS:
- Brightness Preservation Error: {avg_brightness:.2f} (lower is better)
- Color Prediction Error: {avg_color:.2f} (lower is better)

PER IMAGE RESULTS:
"""
    for r in results:
        report_text += f"- {r['name']}: Brightness={r['brightness_error']:.2f}, Color={r['color_error']:.2f}\n"

    report_text += f"""
OUR ORIGINAL CONTRIBUTIONS:
1. Correction Map - Fixes local color mistakes
2. Color Balance - Fixes global color tint
3. Attention Mechanism - Guides corrections
4. Learned Fusion - Intelligently combines predictions

KEY ACHIEVEMENT:
- L channel (brightness) is kept CONSTANT
- Only a and b channels (color) are predicted
- This preserves image brightness while adding color
"""

    report_path = os.path.join(output_dir, "report", "results_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    print("\n" + "=" * 60)
    print("REPORT GENERATED")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    print(f"Images saved to: {output_dir}/report/")
    print("\n" + report_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_report.py <model_path> [num_images]")
        print("Example: python scripts/generate_report.py models/colorization_final.pth 5")
        return

    model_path = sys.argv[1]
    num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_report(model_path, device=device, num_images=num_images)


if __name__ == "__main__":
    main()

