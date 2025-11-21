# Inference script for pseudo-colorization

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path - we're at root now
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb, color_balance_histogram_remap

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

    Unique approach combining:
    1. Dual-path processing (coarse + fine colors)
    2. Luminance-weighted saturation
    3. Anisotropic color stretching (a and b treated independently)
    4. Attention-guided local enhancement
    """
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Model returns: ab_corrected (already fused), bias_map, attention
        ab_corrected, bias_map, attention = model(L_tensor)

    # Convert to numpy
    ab_corrected_np = ab_corrected[0].cpu().numpy().transpose(1, 2, 0)
    attention_np = attention[0, 0].cpu().numpy()

    # Step 1: Denormalize from [-1, 1] to [-127, 127]
    a_pred = ab_corrected_np[:, :, 0] * 127.0
    b_pred = ab_corrected_np[:, :, 1] * 127.0

    # Step 2: Apply bias map with controlled strength
    if bias_strength != 1.0:
        bias_map_np = bias_map[0].cpu().numpy().transpose(1, 2, 0)
        bias_a = bias_map_np[:, :, 0] * 10.0 * (bias_strength - 1.0)
        bias_b = bias_map_np[:, :, 1] * 10.0 * (bias_strength - 1.0)
        a_pred = a_pred + bias_a
        b_pred = b_pred + bias_b

    # Step 3: UNIQUE - Anisotropic Adaptive Stretching
    # Treat a and b channels independently based on their distributions

    # Get robust statistics for each channel separately
    a_p5, a_p95 = np.percentile(a_pred, [5, 95])
    b_p5, b_p95 = np.percentile(b_pred, [5, 95])

    a_span = a_p95 - a_p5
    b_span = b_p95 - b_p5

    # Minimum desired span (avoid over-compression)
    min_span = 20.0
    max_span = 80.0  # Maximum to prevent oversaturation

    # Adaptive stretching - expand if too compressed, compress if too spread
    def adaptive_stretch(channel, p5, p95, current_span):
        center = (p5 + p95) / 2.0
        if current_span < min_span and current_span > 0:
            # Expand narrow range
            factor = min_span / current_span
        elif current_span > max_span:
            # Compress wide range
            factor = max_span / current_span
        else:
            factor = 1.0
        return (channel - center) * factor + center

    a_stretched = adaptive_stretch(a_pred, a_p5, a_p95, a_span)
    b_stretched = adaptive_stretch(b_pred, b_p5, b_p95, b_span)

    # Step 4: UNIQUE - Luminance-Weighted Saturation Modulation
    # Different luminance levels have different natural saturation
    L_normalized = L / 100.0

    # Create smooth saturation curve: peak at mid-tones, reduced at extremes
    # This matches natural image statistics
    saturation_curve = 1.0 - 0.4 * ((L_normalized - 0.5) ** 2) * 4.0
    saturation_curve = np.clip(saturation_curve, 0.6, 1.2)

    # Apply saturation modulation
    a_modulated = a_stretched * saturation_curve
    b_modulated = b_stretched * saturation_curve

    # Step 5: UNIQUE - Attention-Guided Local Enhancement
    # Boost colors where the model pays attention (regions it's confident about)

    # Normalize attention to reasonable range
    attention_normalized = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)

    # Apply gentle boost in attended regions (10-20% boost)
    attention_multiplier = 1.0 + attention_normalized * 0.15

    a_attended = a_modulated * attention_multiplier
    b_attended = b_modulated * attention_multiplier

    # Step 6: UNIQUE - Bilateral-Style Smoothing (preserve edges)
    # Only if image has noise (based on variance)

    a_variance = np.var(a_attended)
    b_variance = np.var(b_attended)

    # Only smooth if there's significant noise
    if a_variance > 100 or b_variance > 100:
        # Very light smoothing
        a_smooth = gaussian_filter(a_attended, sigma=0.8)
        b_smooth = gaussian_filter(b_attended, sigma=0.8)

        # Detect edges in luminance
        L_edges = gaussian_filter(L, sigma=1.0)
        edge_map = np.abs(L - L_edges)
        edge_weight = np.clip(edge_map / 15.0, 0, 1)

        # Blend: original at edges, smoothed in flat regions
        a_final_smooth = a_attended * edge_weight + a_smooth * (1 - edge_weight)
        b_final_smooth = b_attended * edge_weight + b_smooth * (1 - edge_weight)
    else:
        a_final_smooth = a_attended
        b_final_smooth = b_attended

    # Step 7: Color Balance Correction
    if use_color_balance:
        # Remove global color cast
        a_mean = np.mean(a_final_smooth)
        b_mean = np.mean(b_final_smooth)

        # Apply correction (80% strength to preserve some warmth/coolness)
        a_balanced = a_final_smooth - a_mean * 0.8
        b_balanced = b_final_smooth - b_mean * 0.8
    else:
        a_balanced = a_final_smooth
        b_balanced = b_final_smooth

    # Step 8: UNIQUE - Color Diversity Enhancement
    # Ensure we don't lose color variety by enforcing minimum color variance

    a_final_var = np.var(a_balanced)
    b_final_var = np.var(b_balanced)

    # If variance is too low, boost contrast slightly
    min_variance = 100.0

    if a_final_var < min_variance and a_final_var > 0:
        a_center = np.mean(a_balanced)
        boost = np.sqrt(min_variance / a_final_var)
        a_balanced = (a_balanced - a_center) * min(boost, 1.5) + a_center

    if b_final_var < min_variance and b_final_var > 0:
        b_center = np.mean(b_balanced)
        boost = np.sqrt(min_variance / b_final_var)
        b_balanced = (b_balanced - b_center) * min(boost, 1.5) + b_center

    # Step 9: Final clipping to valid LAB range
    a_final = np.clip(a_balanced, -127, 127)
    b_final = np.clip(b_balanced, -127, 127)

    # Convert to RGB
    rgb_output = lab_to_rgb(L, a_final, b_final)
    return rgb_output


def colorize_from_grayscale(grayscale_path, model_path, output_path=None,
                            bias_strength=1.0, use_color_balance=True, device="cpu"):
    # Complete pipeline: load image, colorize, save
    model = load_model(model_path, device)
    img = load_image(grayscale_path, target_size=(256, 256))

    # Extract L channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        L, _, _ = rgb_to_lab(img)
    else:
        L = (img.astype(np.float32) / 255.0) * 100.0
        if len(L.shape) != 2:
            L = L[:, :, 0]

    # Colorize
    rgb_output = colorize_image(L, model, device, bias_strength, use_color_balance)

    # Save if path provided
    if output_path:
        from src.utils import save_image
        save_image(rgb_output, output_path)
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
