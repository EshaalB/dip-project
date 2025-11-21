# Inference script for pseudo-colorization

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import ColorizationModel
from src.utils import load_image, rgb_to_lab, lab_to_rgb, color_balance_histogram_remap


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
                model.load_state_dict(remapped, strict=False)
            except Exception:
                raise e

        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")


def colorize_image(L, model, device="cpu", bias_strength=1.0, use_color_balance=True):
    # Colorize grayscale L channel with learned fusion and attention
    L_norm = L / 100.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Model returns: ab_corrected (already fused), bias_map, attention
        ab_corrected, bias_map, attention = model(L_tensor)

    # Convert to numpy
    ab_corrected_np = ab_corrected[0].cpu().numpy().transpose(1, 2, 0)

    # Denormalize from [-1, 1] to [-127, 127]
    a_corrected = ab_corrected_np[:, :, 0] * 127.0
    b_corrected = ab_corrected_np[:, :, 1] * 127.0

    # Optional: apply additional bias strength if needed (for GUI control)
    if bias_strength != 1.0:
        bias_map_np = bias_map[0].cpu().numpy().transpose(1, 2, 0)
        bias_a = bias_map_np[:, :, 0] * 10.0 * (bias_strength - 1.0)
        bias_b = bias_map_np[:, :, 1] * 10.0 * (bias_strength - 1.0)
        a_corrected = np.clip(a_corrected + bias_a, -127, 127)
        b_corrected = np.clip(b_corrected + bias_b, -127, 127)
    else:
        a_corrected = np.clip(a_corrected, -127, 127)
        b_corrected = np.clip(b_corrected, -127, 127)

    # Apply color balance
    if use_color_balance:
        a_corrected, b_corrected = color_balance_histogram_remap(a_corrected, b_corrected)

    # Convert to RGB
    rgb_output = lab_to_rgb(L, a_corrected, b_corrected)
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
