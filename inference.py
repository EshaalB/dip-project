import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.model import ColorizationModel
from src.utils import loadImage, labtoRGB, rgbtoLab, saveImg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Try to import scipy for advanced filtering, fallback to cv2
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    import cv2
    HAS_SCIPY = False
    def gaussian_filter(image, sigma):
        """Fallback gaussian filter using cv2"""
        ksize = int(sigma * 4) | 1  # Ensure odd kernel size
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def load_model(model_path, device="cpu"):
    # Load trained model with robust error handling
    try:
        model = ColorizationModel()
        checkpoint = torch.load(model_path, map_location=device)

        # Support multiple checkpoint formats saved by different scripts/tools.
        # Common formats:
        # 1) raw state_dict (saved with torch.save(model.state_dict(), path))
        # 2) dict with key 'model_state_dict' or 'model_state' or 'state_dict'
        # 3) dict wrapping many fields (epoch, optimizer_state, loss, ...)

        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ('model_state_dict', 'model_state', 'state_dict', 'model'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

            # If no known key was found, assume the checkpoint might already be a state_dict
            if state_dict is None:
                # Heuristic: if all values are tensors, treat checkpoint as state_dict
                if all(hasattr(v, 'size') or isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    raise ValueError(f"Unrecognized checkpoint format with keys: {list(checkpoint.keys())}")
        else:
            # checkpoint is likely a raw state_dict
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Attempt to remap older/newer checkpoint key names to current model keys.
            # Common renames between versions are handled here.
            key_map = {
                'attention_upsample': 'upsample_attention',
                'bias_head.feature_adapt': 'bias_head.adapt_features',
                'bias_head.bias_decoder': 'bias_head.decode',
                'fusion.fusion_conv': 'fusion.network',
            }

            remapped = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in key_map.items():
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                remapped[new_k] = v

            try:
                # Try loading remapped keys (non-strict to allow minor mismatches)
                model.load_state_dict(remapped, strict=False)
            except Exception:
                # Re-raise original error if remapping didn't help
                raise e

        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

def colorize_image(L, model, device="cpu", bias_strength=1.0, use_color_balance=True):
    """
    ORIGINAL CONTRIBUTION: Adaptive Multi-Scale Color Enhancement

    1. Dual-path processing (model internally)
    2. Luminance-weighted saturation (Step 4)
    3. Anisotropic color stretching (Step 3)
    4. Attention-guided local enhancement (Step 5)
    """
    # Normalize luminance to [0,1]
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Model outputs predicted ab already in LAB units [-110, 110], bias_map, and attention map
        ab_corrected, bias_map, attention = model(L_tensor)

    ab_corrected_np = ab_corrected[0].cpu().numpy().transpose(1, 2, 0)
    attention_np = attention[0, 0].cpu().numpy()

    # Model now outputs in LAB units directly (no need to scale by 127)
    a_pred = ab_corrected_np[:, :, 0]
    b_pred = ab_corrected_np[:, :, 1]

    # Step 2: Apply bias map scaled by bias_strength (for subtle control)
    if bias_strength != 1.0:
        bias_map_np = bias_map[0].cpu().numpy().transpose(1, 2, 0)
        bias_a = bias_map_np[:, :, 0] * (bias_strength - 1.0)
        bias_b = bias_map_np[:, :, 1] * (bias_strength - 1.0)
        a_pred += bias_a
        b_pred += bias_b

    # Step 3: Adaptive Stretching (reduced aggression)
    def adaptive_stretch(channel):
        p5, p95 = np.percentile(channel, [5, 95])
        span = p95 - p5
        center = (p5 + p95) / 2.0
        min_span, max_span = 10.0, 100.0
        if span < min_span and span > 0:
            factor = min(min_span / span, 1.5)  # Cap at 1.5x
        elif span > max_span:
            factor = max_span / span
        else:
            factor = 1.0

        return (channel - center) * factor + center

    a_stretched = adaptive_stretch(a_pred)
    b_stretched = adaptive_stretch(b_pred)

    # Step 4: Luminance-weighted saturation modulation
    saturation_curve = 1.0 - 0.3 * (4.0 * ((L_norm - 0.5) ** 2))
    saturation_curve = np.clip(saturation_curve, 0.7, 1.2)

    a_modulated = a_stretched * saturation_curve
    b_modulated = b_stretched * saturation_curve

    # Step 5: Attention-guided local enhancement (subtle)
    attention_normalized = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
    attention_multiplier = 1.0 + attention_normalized * 0.2  # max ~20% boost

    a_attended = a_modulated * attention_multiplier
    b_attended = b_modulated * attention_multiplier

    # Step 6: Gentle smoothing only if very noisy
    a_var, b_var = np.var(a_attended), np.var(b_attended)
    if a_var > 500 or b_var > 500:
        a_smooth = gaussian_filter(a_attended, sigma=0.5)
        b_smooth = gaussian_filter(b_attended, sigma=0.5)
        # Edge-preserving blend
        L_edges = gaussian_filter(L, sigma=1.0)
        edge_map = np.abs(L - L_edges)
        edge_weight = np.clip(edge_map / 10.0, 0, 1)
        a_final = a_attended * edge_weight + a_smooth * (1 - edge_weight)
        b_final = b_attended * edge_weight + b_smooth * (1 - edge_weight)
    else:
        a_final = a_attended
        b_final = b_attended

    # Step 7: Gentle color balance (reduce color cast, retain warmth)
    if use_color_balance:
        a_mean, b_mean = np.mean(a_final), np.mean(b_final)
        a_final = a_final - a_mean * 0.15
        b_final = b_final - b_mean * 0.15

    # Step 8: Clip to valid LAB range and convert to RGB
    a_final = np.clip(a_final, -127, 127)
    b_final = np.clip(b_final, -127, 127)

    rgb_output = labtoRGB(L, a_final, b_final)
    return rgb_output




def colorize_from_grayscale(grayscale_path, model_path, output_path=None,
                            bias_strength=1.0, use_color_balance=True, device="cpu"):
    # Complete pipeline: load image, colorize, save
    model = load_model(model_path, device)
    img = loadImage(grayscale_path, target_size=(224, 224))

    # Extract L channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        L, _, _ = rgbtoLab(img)
    else:
        L = (img.astype(np.float32) / 255.0) * 100.0
        if len(L.shape) != 2:
            L = L[:, :, 0]

    # Colorize
    rgb_output = colorize_image(L, model, device, bias_strength, use_color_balance)

    # Save if path provided
    if output_path:

        saveImg(rgb_output, output_path)
        print(f"Colorized image saved to {output_path}")

    return rgb_output


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <input_image> <model_path> [output_path]")
        print("Example: python inference.py data/sample.jpg models/colorization_final.pth output.jpg")
        return

    input_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_colorized.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    colorize_from_grayscale(input_path, model_path, output_path,
                            bias_strength=1.0, use_color_balance=True, device=device)


if __name__ == "__main__":
    main()
