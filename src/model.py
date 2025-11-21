# Neural network model for colorization

import torch
import torch.nn as nn


class Attention(nn.Module):
    # ORIGINAL CONTRIBUTION: Finds where color corrections are needed
    # This guides our correction map to focus on problem areas
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        attention_map = self.sigmoid(self.conv(features))
        return attention_map


class BiasHead(nn.Module):
    # ORIGINAL CONTRIBUTION 1: Correction Map
    # Creates adaptive correction map based on local image features
    # This fixes local color mistakes that the main network misses
    def __init__(self, feature_size=128):
        super().__init__()
        self.adapt_features = nn.Sequential(
            nn.Conv2d(feature_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        adapted = self.adapt_features(features)
        correction_map = self.decode(adapted)
        return correction_map


class Fusion(nn.Module):
    # ORIGINAL CONTRIBUTION: Learned Fusion
    # Learns how to intelligently combine prediction and correction
    # Instead of fixed weight, network learns optimal combination
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

        # Extract features from grayscale
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

        # Predict colors
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Our original contributions
        self.bias_head = BiasHead(feature_size=128)
        self.attention = Attention(channels=128)
        self.fusion = Fusion()

        # Upsample attention to full size
        self.upsample_attention = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, grayscale):
        features = self.encoder(grayscale)
        color_prediction = self.decoder(features)
        correction_map = self.bias_head(features)
        attention_small = self.attention(features)
        attention_full = self.upsample_attention(attention_small)
        final_output = self.fusion(color_prediction, correction_map, attention_full)

        return final_output, correction_map, attention_full

