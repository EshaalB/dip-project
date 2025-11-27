import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ColorizationModel
from src.dataset import ImageDataset
from src.utils import verifyDataset as check_dataset
from src.utils import labABtoRGB


class TinyVGG(nn.Module):
    def __init__(self, in_channels=3, out_layer=3):
        super().__init__()
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
        )

    def forward(self, x):
        return self.features(x)


def compute_loss(pred_ab, target_ab, attention, perceptual_net, L_source):
    device = pred_ab.device

    lossPixel = F.smooth_l1_loss(pred_ab, target_ab)

    with torch.no_grad():
        predRgb = labABtoRGB(pred_ab, L_source)
        targetRgb = labABtoRGB(target_ab, L_source)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    predIn = (predRgb - mean) / std
    targIn = (targetRgb - mean) / std

    featPred = perceptual_net(predIn)
    featTarg = perceptual_net(targIn)
    lossPerc = F.l1_loss(featPred, featTarg)

    if attention is None or attention.sum() == 0:
        lossAtt = torch.tensor(0.0, device=device)
    else:
        if attention.shape[2:] != pred_ab.shape[2:]:
            attentionResized = F.interpolate(attention, size=pred_ab.shape[2:], mode='bilinear', align_corners=False)
        else:
            attentionResized = attention
        lossAtt = torch.mean(torch.abs(attentionResized[:, :, :, 1:] - attentionResized[:, :, :, :-1])) + \
                  torch.mean(torch.abs(attentionResized[:, :, 1:, :] - attentionResized[:, :, :-1, :]))
        lossAtt = lossAtt * 0.01

    L = L_source
    L_dx = L[:, :, :, 1:] - L[:, :, :, :-1]
    ab_dx = pred_ab[:, :, :, 1:] - pred_ab[:, :, :, :-1]
    w_x = torch.exp(-torch.abs(L_dx))
    loss_lx = (w_x * torch.abs(ab_dx)).mean()
    
    L_dy = L[:, :, 1:, :] - L[:, :, :-1, :]
    ab_dy = pred_ab[:, :, 1:, :] - pred_ab[:, :, :-1, :]
    w_y = torch.exp(-torch.abs(L_dy))
    loss_ly = (w_y * torch.abs(ab_dy)).mean()
    lossLSmooth = 0.03 * (loss_lx + loss_ly)

    total = (1.0 * lossPixel) + (0.2 * lossPerc) + lossAtt + lossLSmooth

    components = {
        "pixel": lossPixel.item(),
        "perceptual": lossPerc.item(),
        "attention_smooth": lossAtt.item() if isinstance(lossAtt, torch.Tensor) else float(lossAtt),
        "l_smooth": lossLSmooth.item()
    }
    return total, components


def train_model(model, data_loader, epochs=100, device="cpu", save_folder="models", save_every=10):
    featureExtractor = TinyVGG().to(device)
    featureExtractor.eval()

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
    print(f"Batches per epoch: {len(data_loader)}")
    print(f"{'='*60}\n")

    bestLoss = float('inf')
    lossHistory = []

    for epoch in range(epochs):
        startTime = time.time()
        totalLoss = 0.0
        lossComponents = {k: 0.0 for k in ['pixel', 'perceptual', 'attention_smooth', 'l_smooth']}
        batchCount = 0

        imgNumber = 0

        for grayscale, colorTarget in data_loader:
            batchSizeActual = grayscale.size(0)
            imgNumber += batchSizeActual

            grayscale = grayscale.to(device)
            colorTarget = colorTarget.to(device)

            colorPred, correctionMap, attention = model(grayscale)

            loss, components = compute_loss(
                pred_ab=colorPred,
                target_ab=colorTarget,
                attention=attention,
                perceptual_net=featureExtractor,
                L_source=grayscale
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totalLoss += loss.item()
            for key in lossComponents:
                lossComponents[key] += components.get(key, 0.0)

            print(f"  Batch {batchCount + 1}/{len(data_loader)} done", end="\r")

            batchCount += 1


        avgLoss = totalLoss / batchCount if batchCount > 0 else 0.0
        lossHistory.append(avgLoss)

        for key in lossComponents:
            lossComponents[key] /= batchCount

        scheduler.step(avgLoss)

        elapsed = time.time() - startTime
        print(f"\nEpoch {epoch+1:3d}/{epochs} | Loss: {avgLoss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s")
        print("  > " + " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in lossComponents.items()]) + "\n")

        if avgLoss < bestLoss:
            bestLoss = avgLoss
            bestPath = os.path.join(save_folder, "colorization_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avgLoss,
                'loss_history': lossHistory
            }, bestPath)

        if (epoch + 1) % save_every == 0:
            checkpointPath = os.path.join(save_folder, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avgLoss,
                'loss_history': lossHistory
            }, checkpointPath)

    finalPath = os.path.join(save_folder, "colorization_final.pth")
    torch.save(model.state_dict(), finalPath)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Loss: {bestLoss:.5f} | Final Loss: {avgLoss:.5f}")
    if len(lossHistory) > 0:
        print(f"Improvement: {((lossHistory[0] - bestLoss) / lossHistory[0] * 100):.1f}%")
    print(f"Best model: {bestPath}")
    print(f"{'='*60}\n")


def main():
    if not check_dataset("data"):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    dataset = ImageDataset("data", img_size=(224, 224))
    batchSize = 16
    numWorkers = min(8, os.cpu_count())
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)

    model = ColorizationModel().to(device)
    paramCount = sum(p.numel() for p in model.parameters())
    print(f"Model has {paramCount:,} parameters")
    print(f"Batch size: {batchSize} (optimized for low RAM usage)\n")

    train_model(model, loader, epochs=30, device=device, save_folder="models", save_every=1)
    print("\nDone!")


if __name__ == "__main__":
    main()
