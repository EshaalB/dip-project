import torch
import numpy as np
import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    # Fallback to cv2 if scipy not available
    def gaussian_filter(img, sigma):
        ksize = int(sigma * 4) | 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def load_model(model_path, device="cpu"):
    model = ColorizationModel()
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Try common keys
        state_dict = None
        for key in ('model_state_dict', 'model_state', 'state_dict'):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            # Might be raw state_dict wrapped in dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # print(f"[DEBUG] Loading model from {model_path}")

    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: {e}")
        print("Attempting to load with non-strict mode...")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model

def colorize_image(L, model, device="cpu", bias_strength=1.0, use_color_balance=True, temperature=0.0):
    """
    Run inference and apply post-processing

    Args:
        temperature: Color temperature shift (-1.0 to 1.0)
                    Negative = cooler (more blue), Positive = warmer (more yellow)
    """
    # Normalize and prepare input
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        ab_pred, bias_map, attention = model(L_tensor)

    # Convert to numpy
    ab_np = ab_pred[0].cpu().numpy().transpose(1, 2, 0)
    a_pred = ab_np[:, :, 0] * 127.0
    b_pred = ab_np[:, :, 1] * 127.0

    # print(f"[DEBUG] Prediction range: a=[{a_pred.min():.1f}, {a_pred.max():.1f}], b=[{b_pred.min():.1f}, {b_pred.max():.1f}]")

    # Apply bias adjustment if needed (for GUI control)
    if bias_strength != 1.0:
        bias_np = bias_map[0].cpu().numpy().transpose(1, 2, 0)
        a_pred += bias_np[:, :, 0] * 10.0 * (bias_strength - 1.0)
        b_pred += bias_np[:, :, 1] * 10.0 * (bias_strength - 1.0)

    # ORIGINAL: Color temperature adjustment
    # Shifts b channel: positive = warmer (yellow), negative = cooler (blue)
    if temperature != 0.0:
        b_pred += temperature * 30.0  # Scale for noticeable effect

    # Adaptive color stretching - expand compressed ranges
    def stretch_channel(ch):
        p5, p95 = np.percentile(ch, [5, 95])
        span = p95 - p5
        if span < 20 and span > 0:  # Too compressed
            center = (p5 + p95) / 2
            return (ch - center) * (20 / span) + center
        elif span > 80:  # Too wide
            center = (p5 + p95) / 2
            return (ch - center) * (80 / span) + center
        return ch

    a_stretched = stretch_channel(a_pred)
    b_stretched = stretch_channel(b_pred)

    # Luminance-based saturation adjustment
    L_norm_2d = L / 100.0
    sat_factor = 1.0 - 0.4 * ((L_norm_2d - 0.5) ** 2) * 4.0
    sat_factor = np.clip(sat_factor, 0.6, 1.2)

    a_final = a_stretched * sat_factor
    b_final = b_stretched * sat_factor

    # Color balance correction
    if use_color_balance:
        a_mean = np.mean(a_final)
        b_mean = np.mean(b_final)
        # print(f"[DEBUG] Color means before balance: a={a_mean:.2f}, b={b_mean:.2f}")
        a_final -= a_mean * 0.5  # Reduce color cast
        b_final -= b_mean * 0.5

    # Clip and convert
    a_final = np.clip(a_final, -127, 127)
    b_final = np.clip(b_final, -127, 127)

    return lab_to_rgb(L, a_final, b_final)



def colorize_from_grayscale(grayscale_path, model_path, output_path=None, device="cpu"):
    """Full pipeline - load, colorize, save"""
    model = load_model(model_path, device)
    img = load_image(grayscale_path, target_size=(224, 224))

    # Extract L channel from image
    if len(img.shape) == 3:
        L, _, _ = rgb_to_lab(img)
    else:
        # Already grayscale
        L = (img.astype(np.float32) / 255.0) * 100.0

    # Colorize
    rgb_output = colorize_image(L, model, device)

    # Save result
    if output_path:
        from src.utils import save_image
        save_image(rgb_output, output_path)
        print(f"Saved to {output_path}")

    return rgb_output

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <input> <model> [output]")
        print("Example: python inference.py input/test.jpg models/colorization_best.pth output.jpg")
        return

    input_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # TODO: Add batch processing support
    # TODO: Add option to adjust color intensity

    colorize_from_grayscale(input_path, model_path, output_path, device)

if __name__ == "__main__":
    main()
