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

# ---------------------------
# TinyVGG perceptual extractor
# ---------------------------
class TinyVGG(nn.Module):
    """
    Very small VGG-like feature extractor for perceptual loss.
    Not pretrained (lightweight). Captures low-mid level features.
    Returns features at a single layer for L1 perceptual loss.
    """
    def __init__(self, in_channels=3, out_layer=3):
        super().__init__()
        # small stack of convs
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # we stop here (out_layer ~3 gives these mid-level features)
        )

    def forward(self, x):
        # expects RGB input normalized to [0,1] that we will normalize consistently in train loop
        return self.features(x)

# ---------------------------
# Loss function
# ---------------------------
def compute_loss(pred_ab, target_ab, attention, perceptual_net, L_source):
    """
    Stable loss: pixel + perceptual + attention smooth + L-guided chrominance smooth
    - pred_ab: (B,2,H,W) predicted ab scaled in real LAB units (we keep model returning [-110,110])
    - target_ab: (B,2,H,W) ground-truth ab in same scale
    - attention: (B,1 or 2,H',W') upsampled to (B,2,H,W) in model; if zeros it's fine.
    - perceptual_net: tiny VGG; expects RGB in [0,1] normalized using ImageNet mean/std
    - L_source: (B,1,H,W) L normalized [0,1]
    """
    device = pred_ab.device

    # 1) Pixel loss (smooth L1 on ab in LAB range)
    loss_pixel = F.smooth_l1_loss(pred_ab, target_ab)

    # 2) Perceptual loss: convert lab->rgb via your util (lab_to_rgb_tensor expects L in same scale as used in your utils)
    # We need to prepare RGB tensors normalized for perceptual network.
    with torch.no_grad():
        # create RGB tensors in [0,1] using lab_to_rgb_tensor (it should output 0..1)
        # Note: src/utils.py defines lab_to_rgb_tensor(ab, L)
        pred_rgb = lab_to_rgb_tensor(pred_ab, L_source)
        target_rgb = lab_to_rgb_tensor(target_ab, L_source)
    
    # normalize for perceptual extractor (ImageNet-like mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    pred_in = (pred_rgb - mean) / std
    targ_in = (target_rgb - mean) / std

    feat_pred = perceptual_net(pred_in)
    feat_targ = perceptual_net(targ_in)
    loss_perc = F.l1_loss(feat_pred, feat_targ)

    # 3) Attention smoothness (encourage spatial coherence of attention)
    # attention may have 1 or 2 channels; compute along spatial dims
    if attention is None or attention.sum() == 0:
        loss_att = torch.tensor(0.0, device=device)
    else:
        # make sure attention is same size as prediction (upsample if needed)
        if attention.shape[2:] != pred_ab.shape[2:]:
            attention_resized = F.interpolate(attention, size=pred_ab.shape[2:], mode='bilinear', align_corners=False)
        else:
            attention_resized = attention
        # L1 on gradient of attention
        loss_att = torch.mean(torch.abs(attention_resized[:, :, :, 1:] - attention_resized[:, :, :, :-1])) + \
                   torch.mean(torch.abs(attention_resized[:, :, 1:, :] - attention_resized[:, :, :-1, :]))
        loss_att = loss_att * 0.01

    # 4) L-guided chrominance smoothness: encourage similar ab where L is similar
    L = L_source
    # horizontal
    L_dx = L[:, :, :, 1:] - L[:, :, :, :-1]
    ab_dx = pred_ab[:, :, :, 1:] - pred_ab[:, :, :, :-1]
    w_x = torch.exp(-torch.abs(L_dx))
    loss_lx = (w_x * torch.abs(ab_dx)).mean()
    # vertical
    L_dy = L[:, :, 1:, :] - L[:, :, :-1, :]
    ab_dy = pred_ab[:, :, 1:, :] - pred_ab[:, :, :-1, :]
    w_y = torch.exp(-torch.abs(L_dy))
    loss_ly = (w_y * torch.abs(ab_dy)).mean()
    loss_l_smooth = 0.03 * (loss_lx + loss_ly)

    # total
    total = (1.0 * loss_pixel) + (0.2 * loss_perc) + loss_att + loss_l_smooth

    components = {
        "pixel": loss_pixel.item(),
        "perceptual": loss_perc.item(),
        "attention_smooth": loss_att.item() if isinstance(loss_att, torch.Tensor) else float(loss_att),
        "l_smooth": loss_l_smooth.item()
    }
    return total, components

def train_model(model, data_loader, epochs=100, device="cpu", save_folder="models", save_every=10):
    """
    Enhanced training with multi-objective loss:
    Pixel + Perceptual + Attention Smooth + L-guided Smooth
    """
    # Initialize perceptual feature extractor
    feature_extractor = TinyVGG().to(device)
    feature_extractor.eval()

    # Reduced learning rate for more stable training
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    os.makedirs(save_folder, exist_ok=True)

    model.train()
    print(f"\n{'='*60}")
    print(f"  ENHANCED TRAINING WITH UPDATED LOSS")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | Device: {device}")
    print(f"Components: Pixel + Perceptual + Attention Smooth + L-guided Smooth")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        loss_components = {k: 0.0 for k in ['pixel', 'perceptual', 'attention_smooth', 'l_smooth']}
        batch_count = 0

        img_number = 0

        for grayscale, color_target in data_loader:
            batch_size_actual = grayscale.size(0)
            for _ in range(batch_size_actual):
                img_number += 1
                # print(f"Image loaded: {img_number}")  

            grayscale = grayscale.to(device)
            color_target = color_target.to(device)

            # Forward pass
            color_pred, correction_map, attention = model(grayscale)

            # Compute loss using updated compute_loss
            loss, components = compute_loss(
                pred_ab=color_pred,
                target_ab=color_target,
                attention=attention,
                perceptual_net=feature_extractor,
                L_source=grayscale
            )

            # Backprop
            optimizer.zero_grad()
            loss.backward()
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

        scheduler.step(avg_loss)

        # Progress output
        print(f"\nEpoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("  > " + " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in loss_components.items()]) + "\n")

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

        # Periodic checkpoint
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
    if len(loss_history) > 0:
        print(f"Improvement: {((loss_history[0] - best_loss) / loss_history[0] * 100):.1f}%")
    print(f"Best model: {best_path}")
    print(f"{'='*60}\n")


def main():
    # Check dataset
    if not check_dataset("data"):
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Create dataset and loader with reduced RAM footprint
    dataset = ImageDataset("data", img_size=(224, 224))
    batch_size = 8  # Reduced for lower RAM usage
    num_workers = 0  # No multiprocessing to save RAM
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create model
    model = ColorizationModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    print(f"Batch size: {batch_size} (optimized for low RAM usage)\n")

    # Train (100 epochs for real results)
    # This will take longer (maybe 20-30 mins on CPU) but is necessary for quality
    train_model(model, loader, epochs=30, device=device, save_folder="models", save_every=1  )
    print("\nDone!")


if __name__ == "__main__":
    main()
