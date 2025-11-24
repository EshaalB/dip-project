import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        attention_map = self.sigmoid(self.conv(features))
        return attention_map

class BiasHead(nn.Module):
    def __init__(self, feature_size=128):
        super().__init__()
        self.adapt_features = nn.Sequential(
            nn.Conv2d(feature_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        # UPGRADE: Use upsample + conv instead of conv transpose for artifact-free decoding
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        adapted = self.adapt_features(features)
        correction_map = self.decode(adapted)
        return correction_map

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, prediction, correction, attention):
        weighted_correction = correction * attention
        combined = torch.cat([prediction, weighted_correction], dim=1)
        final = self.network(combined)
        return final


class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        # UPGRADE: Decoder now uses upsample+conv blocks for artifact-free color prediction
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.bias_head = BiasHead(feature_size=128)
        self.attention = Attention(channels=128)
        self.fusion = Fusion()
        self.upsample_attention = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, grayscale):
        features = self.encoder(grayscale)
        color_prediction = self.decoder(features)
        correction_map = self.bias_head(features)
        attention_small = self.attention(features)
        attention_full = self.upsample_attention(attention_small)
        final_output = self.fusion(color_prediction, correction_map, attention_full)
        return final_output, correction_map, attention_full
