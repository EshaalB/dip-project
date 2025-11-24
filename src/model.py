import torch
import torch.nn as nn

class Attention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class BiasHead(nn.Module):
    """Correction map generator"""
    def __init__(self, feature_size=128):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(feature_size, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        # Using upsample+conv instead of transposed conv to avoid checkerboard artifacts
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        x = self.adapt(features)
        return self.decode(x)

class Fusion(nn.Module):
    """Fuses main prediction with correction map"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),
        )

    def forward(self, prediction, correction, attention):
        # Weight correction by attention
        weighted_corr = correction * attention
        combined = torch.cat([prediction, weighted_corr], dim=1)
        return self.conv(combined)


class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder - downsamples to extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # Decoder - predicts initial colors
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )

        self.bias_head = BiasHead(feature_size=128)
        self.attention = Attention(channels=128)
        self.fusion = Fusion()

        # Upsample attention to match output resolution
        self.upsample_attn = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, grayscale):
        # print(f"[DEBUG] Input shape: {grayscale.shape}")

        features = self.encoder(grayscale)
        color_pred = self.decoder(features)
        correction = self.bias_head(features)
        attention = self.attention(features)
        attention_full = self.upsample_attn(attention)

        # Fuse prediction with correction
        final = self.fusion(color_pred, correction, attention_full)

        return final, correction, attention_full
