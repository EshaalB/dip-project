# Training script - run this to train the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.dataset import ImageDataset
from src.utils import verify_dataset as check_dataset


def compute_loss(prediction, target, attention):
    # ORIGINAL CONTRIBUTION: Spatial-aware loss
    # Focuses training on areas needing correction (guided by attention)
    l1 = nn.L1Loss(reduction='none')
    pixel_error = l1(prediction, target)

    # Weight by attention (more weight where correction needed)
    weighted_error = pixel_error * (1.0 + attention)

    # Smoothness term for attention map (encourages smooth corrections)
    smooth = torch.mean(torch.abs(attention[:, :, 1:, :] - attention[:, :, :-1, :]))

    return weighted_error.mean() + 0.01 * smooth


def train_model(model, data_loader, epochs=10, device="cpu", save_folder="models", save_every=5):
    # Train the model (optimized for speed)
    standard_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    os.makedirs(save_folder, exist_ok=True)

    model.train()
    print(f"\nTraining for {epochs} epochs on {device}...")
    print("(Saving checkpoints every {} epochs to save time)".format(save_every))
    print("-" * 50)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        for grayscale, color_target in data_loader:
            grayscale = grayscale.to(device)
            color_target = color_target.to(device)

            # Get model output
            color_pred, correction_map, attention = model(grayscale)

            # Compute loss (simplified for speed)
            loss = compute_loss(color_pred, color_target, attention)
            loss += 0.5 * standard_loss(color_pred, color_target)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Save checkpoint only every N epochs (faster)
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_folder, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    # Save final model
    final_path = os.path.join(save_folder, "colorization_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining done! Model saved to {final_path}")


def main():
    # Check dataset
    if not check_dataset("data"):
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Create dataset and loader (larger batch for faster training)
    dataset = ImageDataset("data", img_size=(256, 256))
    batch_size = 4 if len(dataset) >= 4 else 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = ColorizationModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    print(f"Batch size: {batch_size} (for faster training)\n")

    # Train (5 epochs for quick demo ~5-10 min, 10+ for better results)
    # For presentation: 5 epochs is enough to show it works
    # For better quality: change epochs=5 to epochs=10 or 20
    train_model(model, loader, epochs=5, device=device, save_folder="models", save_every=5)
    print("\nDone!")


if __name__ == "__main__":
    main()
