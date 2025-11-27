# Original Contributions

This document outlines the novel architectural and methodological contributions of this colorization system.

## Architecture Overview

The system uses a three-stage pipeline: **Semantic Encoding** → **Multi-Head Decoding** → **Adaptive Enhancement**.

### Model Architecture (`src/model.py`)

**DinoEncoder**
- Uses Vision Transformer (ViT-Small) for semantic feature extraction
- Pretrained on ImageNet for rich semantic understanding
- Converts grayscale input to 3-channel via learned 1x1 convolution
- Outputs spatial feature maps (14×14) from patch tokens

**ColorDecoder**
- Simple upsampling decoder with 4 blocks (384→256→128→64→32 channels)
- Uses bilinear upsampling + convolution + ReLU

**Multi-Head Architecture**
- Three parallel prediction heads from decoder features:
  - `base_head`: Predicts primary ab color channels
  - `bias_head`: Predicts correction/refinement map
  - `attention_head`: Predicts spatial confidence map
- Fusion: `output = base + (bias × attention)`
- Model output scaled to LAB color range [-110, 110]

## Novel Contributions

### 1. Attention-Gated Multi-Head Architecture
**Location:** `src/model.py` (lines 59-88)

Instead of single-path prediction, the decoder splits into three heads that are learned jointly:
- Base color prediction provides the foundation
- Bias map adds refinements
- Attention map gates where corrections apply

This allows the model to separate coarse and fine predictions, focusing refinement capacity selectively.

### 2. L-Guided Chrominance Smoothness Loss
**Location:** `train.py` (lines 100-112)

Novel loss term that enforces color smoothness only where luminance is smooth:
```
L_smooth = exp(-|∇L|) × |∇ab|
```

This prevents color bleeding across edges while allowing smooth color in uniform regions. Sharp edges in grayscale (L) permit sharp color transitions.

### 3. Adaptive Post-Processing Pipeline
**Location:** `inference.py` (lines 122-180)

Multi-stage enhancement applied per-image:
- **Adaptive Stretching**: Expands color range based on percentiles
- **Luminance-Weighted Saturation**: Modulates saturation by brightness (reduces in shadows/highlights)
- **Attention-Guided Boosting**: Uses model's confidence map to enhance color
- **Edge-Preserving Smoothing**: Conditional smoothing based on variance

### 4. Multi-Objective Training Loss
**Location:** `train.py` (lines 44-97)

Combined loss function:
- **Pixel Loss**: Smooth L1 on ab channels (1.0×)
- **Perceptual Loss**: L1 on TinyVGG features (0.2×)
- **Attention Smoothness**: L1 gradient penalty on attention (0.01×)
- **L-Guided Smoothness**: Edge-aware chrominance regularization (0.03×)

## Technical Details

**Training:**
- Optimizer: Adam (lr=0.0001, β=(0.5, 0.999))
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Batch size: 16
- Input size: 224×224

**Model Size:**
- Parameters: ~22.9M
- Architecture: Transformer encoder + CNN decoder

**Color Space:**
- L channel: [0, 100] (normalized to [0, 1] for model input)
- ab channels: [-127, 127] (model outputs [-110, 110] then post-processed)
