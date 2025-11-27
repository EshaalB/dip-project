import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

    def forward(self, x):
        x = self.gray_to_rgb(x)
        features = self.model.forward_features(x)
        
        if isinstance(features, dict):
            tokens = features["x_norm_patchtokens"]
        else:
            tokens = features[:, 1:, :].transpose(1, 2)
            
        B, C, N = tokens.shape
        H = W = int(N ** 0.5)
        return tokens.view(B, C, H, W)


class ColorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        def upsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True)
            )

        self.decode = nn.Sequential(
            upsample_block(384, 256),
            upsample_block(256, 128),
            upsample_block(128, 64),
            upsample_block(64, 32)
        )

    def forward(self, x):
        return self.decode(x)


class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.decoder = ColorDecoder()
        
        self.base_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        
        self.bias_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        
        self.attention_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, grayscale):
        features = self.encoder(grayscale)
        decFeatures = self.decoder(features)
        
        baseAb = self.base_head(decFeatures)
        biasMap = self.bias_head(decFeatures)
        attention = self.attention_head(decFeatures)
        
        finalAb = baseAb + (biasMap * attention)
        finalAb = torch.clamp(finalAb, -1.0, 1.0)
        
        finalAb = finalAb * 110.0
        biasMap = biasMap * 110.0
        
        return finalAb, biasMap, attention
