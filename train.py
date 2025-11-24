# Training script - Enhanced with perceptual and multi-scale losses

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.dataset import ImageDataset
from src.utils import verify_dataset as check_dataset
from src.utils import lab_to_rgb_tensor

class PerceptualFeatureExtractor(nn.Module):
    """
    REAL perceptual loss using VGG16 features on reconstructed RGB.
    Compares semantic content, removing blob artifacts.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential()

        # Use layers up to conv3_3 (good balance between detail & speed)
        for x in range(16):
            self.slice.add_module(str(x), vgg[x])

        # Freeze VGG
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, rgb):
        return self.slice(rgb)


def compute_loss(prediction, target, attention, feature_extractor, L_source):
    l1_loss = nn.L1Loss(reduction='none')
    pixel_error = l1_loss(prediction, target)
    weighted_pixel = pixel_error * (1.0 + attention)
    loss_pixel = weighted_pixel.mean()

    # Use true L channel to reconstruct RGB for perceptual loss
    pred_rgb = lab_to_rgb_tensor(L_source, prediction)
    target_rgb = lab_to_rgb_tensor(L_source, target)

    pred_features = feature_extractor(pred_rgb)
    target_features = feature_extractor(target_rgb)
    loss_perceptual = F.l1_loss(pred_features, target_features)

    chroma_magnitude = torch.sqrt(prediction[:, 0:1]**2 + prediction[:, 1:2]**2)
    loss_chroma = torch.mean(F.relu(chroma_magnitude - 1.0))

    smooth_h = torch.mean(torch.abs(attention[:, :, 1:] - attention[:, :, :-1]))
    smooth_w = torch.mean(torch.abs(attention[:, :, :, 1:] - attention[:, :, :, :-1]))
    loss_smooth = smooth_h + smooth_w

    pred_grad_h = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_grad_h = target[:, :, 1:] - target[:, :, :-1]
    pred_grad_w = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]
    loss_gradient = F.l1_loss(pred_grad_h, target_grad_h) + F.l1_loss(pred_grad_w, target_grad_w)

    pred_mean_a = prediction[:, 0].mean()
    pred_mean_b = prediction[:, 1].mean()
    target_mean_a = target[:, 0].mean()
    target_mean_b = target[:, 1].mean()
    loss_mean_bias = F.l1_loss(pred_mean_a, target_mean_a) + F.l1_loss(pred_mean_b, target_mean_b)

    total_loss = (
        0.7 * loss_pixel +
        0.7 * loss_perceptual +
        0.1 * loss_chroma +
        0.05 * loss_smooth +
        0.1 * loss_gradient +
        0.5 * loss_mean_bias
    )

    return total_loss, {
        'pixel': loss_pixel.item(),
        'perceptual': loss_perceptual.item(),
        'chroma': loss_chroma.item(),
        'smooth': loss_smooth.item(),
        'gradient': loss_gradient.item(),
        'mean_bias': loss_mean_bias.item()
    }


def train_model(model, data_loader, epochs=100, device="cpu", save_folder="models", save_every=10):
    """
    ORIGINAL CONTRIBUTION: Multi-objective training with adaptive learning
    Uses perceptual features, gradient preservation, and attention weighting
    """
    # Initialize perceptual feature extractor
    feature_extractor = PerceptualFeatureExtractor().to(device)
    feature_extractor.eval()

    # Optimizer with lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Scheduler: Reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    os.makedirs(save_folder, exist_ok=True)

    model.train()
    print(f"\n{'='*60}")
    print(f"  ENHANCED TRAINING WITH PERCEPTUAL LOSS")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | Device: {device}")
    print(f"Components: Pixel + Perceptual + Chroma + Gradient + Smooth + MeanBias")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        loss_components = {'pixel': 0.0, 'perceptual': 0.0, 'chroma': 0.0, 'smooth': 0.0, 'gradient': 0.0, 'mean_bias': 0.0}
        batch_count = 0

        img_number = 0

        for grayscale, color_target in data_loader:
            batch_size_actual = grayscale.size(0)
            for _ in range(batch_size_actual):
                img_number += 1
                print(f"Image loaded: {img_number}")  
            grayscale = grayscale.to(device)
            color_target = color_target.to(device)

            # Get model output
            color_pred, correction_map, attention = model(grayscale)

            # Compute multi-component loss
            loss, components = compute_loss(color_pred, color_target, attention, feature_extractor, grayscale)

            # Update model
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components.get(key, 0.0)
            batch_count += 1

        # Average losses
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        loss_history.append(avg_loss)

        for key in loss_components:
            loss_components[key] /= batch_count

        # Update scheduler
        scheduler.step(avg_loss)

        # Progress output
        print(f"\n\nEpoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  > Pixel: {loss_components['pixel']:.4f} | Percept: {loss_components['perceptual']:.4f} | "
                f"Chroma: {loss_components['chroma']:.4f} | Grad: {loss_components['gradient']:.4f} | "
                f"MeanBias: {loss_components['mean_bias']:.4f}\n\n")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_folder, "colorization_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'loss_history': loss_history
            }, best_path)

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_folder, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history
            }, checkpoint_path)

    # Save final model
    final_path = os.path.join(save_folder, "colorization_final.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Loss: {best_loss:.5f} | Final Loss: {avg_loss:.5f}")
    print(f"Improvement: {((loss_history[0] - best_loss) / loss_history[0] * 100):.1f}%")
    print(f"Best model: {os.path.join(save_folder, 'colorization_best.pth')}")
    print(f"{'='*60}\n")


def main():
    # Check dataset
    if not check_dataset("data"):
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Create dataset and loader (larger batch for faster training)
    dataset = ImageDataset("data", img_size=(256, 256))
    batch_size = 32 if len(dataset) >= 4 else 2
    num_workers = min(12, os.cpu_count())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create model
    model = ColorizationModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    print(f"Batch size: {batch_size} (for faster training)\n")

    # Train (100 epochs for real results)
    # This will take longer (maybe 20-30 mins on CPU) but is necessary for quality
    train_model(model, loader, epochs=15, device=device, save_folder="models", save_every=1  )
    print("\nDone!")


if __name__ == "__main__":
    main()
