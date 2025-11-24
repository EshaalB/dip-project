# Image Colorization Project

Converts grayscale images to color using deep learning. Based on encoder-decoder CNN with our custom improvements.

# Contributers: Eshaal Rehmatullah(23L-0648),Atika Hussain(23L-0570),Ibrahim Faisal(23L-0759)

**Setup:**

```bash
pip install -r requirements.txt
```

**Training:**

```bash
python train.py
```

- Prompts for GPU/CPU selection and epochs (default: 15)
- Uses 80/20 train/val split with data augmentation
- Takes 5-10 mins on GPU, 20-30 mins on CPU
- Saves models to `models/colorization_best.pth`

**GUI:**

```bash
python gui.py
```

1. Load Image → Load Model → Run Colorization → Save Output
2. Adjustable controls: Bias strength, Color balance, Color temperature (warm/cool)

**Command Line:**

```bash
python inference.py input/test.jpg models/colorization_best.pth output.jpg
```

## What We Built

**Standard Parts (from research papers):**

- Encoder-decoder CNN architecture
- LAB color space
- VGG16 for perceptual loss

**Our Contributions:**

1. **Correction Head** - Extra network branch that fixes color mistakes
2. **Attention Fusion** - Smart way to combine predictions and corrections
3. **Better Loss Function** - Uses 6 different loss terms instead of just 1-2
4. **Edge Preservation** - Keeps sharp edges from getting blurry
5. **Post-Processing** - Adjusts colors after prediction for better results
6. **Temperature Control** - GUI slider to make images warmer or cooler

## Project Structure

```
├── train.py           # Training with validation split
├── inference.py       # Inference with post-processing
├── gui.py            # PyQt5 GUI application
├── src/
│   ├── model.py      # Model architecture
│   ├── dataset.py    # Data loader with augmentation
│   └── utils.py      # LAB conversion utilities
├── data/             # Training images (32 RGB images)
├── models/           # Saved models
└── scripts/          # Helper scripts
```

## How We Improved It

Most colorization methods just use encoder-decoder + simple loss. We added:

- Two-path system (main prediction + correction branch)
- Better training with multiple loss types
- Post-processing that adapts to each image
- User controls in the GUI

## References

**Base Paper:** Zhang et al. (2016) - "Colorful Image Colorization" (encoder-decoder + LAB color space)

**Perceptual Loss:** Johnson et al. (2016) - "Perceptual Losses" (VGG16 features)

**Our Work:** Everything else (correction head, fusion, loss function, post-processing, GUI controls) is our original code.

## Troubleshooting

**No images found:** Place RGB .jpg images in `data/` folder
**Model loading error:** Train first with `python train.py`
**GUI not opening:** Install PyQt5 with `pip install PyQt5`
