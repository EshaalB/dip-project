import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import VGG16_Weights
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.dataset import ImageDataset
from src.utils import lab_to_rgb_tensor

class PerceptualFeatureExtractor(nn.Module):
    """VGG16-based feature extractor for perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential()

        # Using first 16 layers (up to conv3_3)
        for x in range(16):
            self.slice.add_module(str(x), vgg[x])

        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, rgb):
        return self.slice(rgb)


def compute_loss(prediction, target, attention, feature_extractor, L_source):
    # Pixel loss weighted by attention map
    pixel_error = F.l1_loss(prediction, target, reduction='none')
    loss_pixel = (pixel_error * (1.0 + attention)).mean()

    # Perceptual loss - compare VGG features
    pred_rgb = lab_to_rgb_tensor(L_source, prediction)
    target_rgb = lab_to_rgb_tensor(L_source, target)
    pred_feat = feature_extractor(pred_rgb)
    target_feat = feature_extractor(target_rgb)
    loss_perceptual = F.l1_loss(pred_feat, target_feat)

    # print(f"[DEBUG] Pixel loss: {loss_pixel.item():.4f}, Perceptual: {loss_perceptual.item():.4f}")

    # Chroma constraint - prevent oversaturation
    # Predictions are in [-1,1], so max magnitude is sqrt(2) ≈ 1.414
    chroma_mag = torch.sqrt(prediction[:, 0:1]**2 + prediction[:, 1:2]**2)
    loss_chroma = torch.mean(F.relu(chroma_mag - 1.3))  # Allow up to ~1.3, penalize beyond

    # Attention smoothness
    loss_smooth = (torch.mean(torch.abs(attention[:, :, 1:] - attention[:, :, :-1])) +
                   torch.mean(torch.abs(attention[:, :, :, 1:] - attention[:, :, :, :-1])))

    # Gradient preservation for edges
    grad_h_pred = prediction[:, :, 1:] - prediction[:, :, :-1]
    grad_h_target = target[:, :, 1:] - target[:, :, :-1]
    grad_w_pred = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    grad_w_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    loss_gradient = F.l1_loss(grad_h_pred, grad_h_target) + F.l1_loss(grad_w_pred, grad_w_target)

    # Mean color bias correction
    loss_mean = (F.l1_loss(prediction[:, 0].mean(), target[:, 0].mean()) +
                 F.l1_loss(prediction[:, 1].mean(), target[:, 1].mean()))

    # Combine losses (tuned weights)
    total = (0.7 * loss_pixel +
             0.7 * loss_perceptual +
             0.1 * loss_chroma +
             0.05 * loss_smooth +
             0.1 * loss_gradient +
             0.5 * loss_mean)

    return total, {
        'pixel': loss_pixel.item(),
        'perceptual': loss_perceptual.item(),
        'chroma': loss_chroma.item(),
        'smooth': loss_smooth.item(),
        'gradient': loss_gradient.item(),
        'mean': loss_mean.item()
    }


def train_model(model, train_loader, val_loader, epochs, device, save_folder="models", save_every=10):
    os.makedirs(save_folder, exist_ok=True)

    # Setup perceptual loss
    feature_extractor = PerceptualFeatureExtractor().to(device)
    feature_extractor.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                      patience=15, min_lr=1e-6)

    print(f"\nTraining on {device} for {epochs} epochs...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        losses = {'pixel': 0, 'perceptual': 0, 'chroma': 0, 'smooth': 0, 'gradient': 0, 'mean': 0}
        n_batches = 0

        for grayscale, color_target in train_loader:
            grayscale = grayscale.to(device)
            color_target = color_target.to(device)

            color_pred, correction_map, attention = model(grayscale)

            # print(f"[DEBUG] Batch {n_batches}: pred range [{color_pred.min():.3f}, {color_pred.max():.3f}]")

            loss, components = compute_loss(color_pred, color_target, attention,
                                           feature_extractor, grayscale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k in losses:
                losses[k] += components[k]
            n_batches += 1

        avg_train_loss = train_loss / n_batches
        for k in losses:
            losses[k] /= n_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for grayscale, color_target in val_loader:
                grayscale = grayscale.to(device)
                color_target = color_target.to(device)
                color_pred, _, attention = model(grayscale)
                loss, _ = compute_loss(color_pred, color_target, attention,
                                      feature_extractor, grayscale)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        scheduler.step(avg_val_loss)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Pixel: {losses['pixel']:.4f} | Perceptual: {losses['perceptual']:.4f} | Gradient: {losses['gradient']:.4f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(save_folder, "colorization_best.pth"))
            # print(f"[DEBUG] Saved best model at epoch {epoch+1}")

        # Periodic checkpoints
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(save_folder, f"model_epoch_{epoch+1}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_folder, "colorization_final.pth"))

    print("\n" + "="*60)
    print(f"Training done! Best val loss: {best_val_loss:.4f}")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("  Image Colorization Training")
    print("="*60)

    # Device selection
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        choice = input("Use GPU? (y/n, default y): ").strip().lower()
        device = torch.device("cuda" if choice != 'n' else "cpu")
    else:
        print("\nNo GPU detected, using CPU")
        device = torch.device("cpu")
        input("Press Enter to continue...")

    print(f"Device: {device}")

    # Load dataset
    dataset_path = "data"
    if not os.path.exists(dataset_path):
        print(f"\nError: '{dataset_path}' folder not found")
        input("Press Enter to exit...")
        return

    # print(f"[DEBUG] Loading images from {dataset_path}")
    dataset = ImageDataset(dataset_path, img_size=(256, 256), augment=True)

    if len(dataset) == 0:
        print(f"\nError: No images found in '{dataset_path}'")
        input("Press Enter to exit...")
        return

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nDataset split: {train_size} train, {val_size} val")

    # Determine batch size based on dataset size
    if len(dataset) < 8:
        batch_size = 2
    elif len(dataset) < 32:
        batch_size = 4
    else:
        batch_size = 8

    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model
    model = ColorizationModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Model: {param_count:,} params | Batch: {batch_size}")

    # Get epochs from user
    epochs_input = input("\nEpochs (default 15): ").strip()
    epochs = int(epochs_input) if epochs_input and epochs_input.isdigit() else 15

    # TODO: Add option to resume from checkpoint
    # TODO: Add early stopping

    print(f"\nStarting training for {epochs} epochs...")
    input("Press Enter to begin...")

    # Train with validation
    train_model(model, train_loader, val_loader, epochs, device,
                "models", max(1, epochs//10))

    print(f"\nModels saved in 'models/' folder")
    print("  - colorization_best.pth")
    print("  - colorization_final.pth")
    print("\nRun gui.py to test the model")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
