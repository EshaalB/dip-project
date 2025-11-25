import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------
# DINOv2 Encoder (semantic)
# -----------------------------
class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained semantic vision model
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )
        # convert grayscale to 3-channel input for ViT
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

    def forward(self, x):
        x = self.gray_to_rgb(x)
        features = self.model.forward_features(x)
        
        # Check if output is dict (DINOv2) or Tensor (Standard ViT)
        if isinstance(features, dict):
            tokens = features["x_norm_patchtokens"]
        else:
            # Standard ViT returns (B, N_patches+1, C)
            # Remove CLS token (index 0)
            tokens = features[:, 1:, :]
            # Permute to (B, C, N)
            tokens = tokens.transpose(1, 2)
            
        B, C, N = tokens.shape
        H = W = int(N ** 0.5)
        return tokens.view(B, C, H, W)  # reshape into spatial map



# -----------------------------
# Simple Semantic Decoder (Modified)
# -----------------------------
# -----------------------------
# Simple Semantic Decoder (Modified)
# -----------------------------
class ColorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        def upsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
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


# -----------------------------
# Final Colorization Model
# -----------------------------
class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.decoder = ColorDecoder()
        
        # Heads attached to the decoder output (32 channels)
        
        # 1. Base Color Prediction
        self.base_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # 2. Bias/Correction Map
        self.bias_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # 3. Attention Map
        self.attention_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, grayscale):
        # 1. Encode
        features = self.encoder(grayscale)
        
        # 2. Decode to high-res features
        dec_features = self.decoder(features)
        
        # 3. Generate components
        # Outputs are in [-1, 1] range
        base_ab = self.base_head(dec_features)
        bias_map = self.bias_head(dec_features)
        attention = self.attention_head(dec_features)
        
        # 4. Learned Fusion: Base + (Bias * Attention)
        # Attention gates the correction
        final_ab = base_ab + (bias_map * attention)
        
        # Clamp to valid range
        final_ab = torch.clamp(final_ab, -1.0, 1.0)
        
        return final_ab, bias_map, attention
