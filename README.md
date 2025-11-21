# Pseudo-Colorization Project

Turn grayscale images into color images using deep learning.

## What This Does

1. **Neural Network** - Predicts colors from grayscale
2. **Correction Map** - Fixes local color mistakes (our original)
3. **Color Balance** - Fixes global color tint (our original)
4. **Output** - Final colorized image

## Quick Start

### 1. Install Packages

```bash
pip install -r requirements.txt
```

### 2. Check Setup

```bash
python scripts/test_model.py
```

Or use the main scripts directly:

```bash
python train.py    # Train model
python gui.py      # Open GUI
python inference.py <image> <model>  # Colorize image
```

### 3. Train Model

```bash
python train.py
```

This takes 5-10 minutes. Models save in `models/` folder.

### 4. Test with GUI

```bash
python gui.py
```

- Click "Load Image" → select image
- Click "Load Model" → select `models/model_final.pth`
- Click "Colorize"
- Click "Save Result" to save

## Folder Structure

```
project/
├── src/              # Core code (model, utils, dataset)
├── scripts/          # Run these (train, inference, gui)
├── tests/            # Test files
├── data/             # Training images (put RGB images here)
├── input/            # Test grayscale images
├── models/           # Saved models (created after training)
└── output/           # Results (created when you save)
```

## How to Modify

### Change Training Settings

Edit `scripts/train.py`:

- `epochs=10` → change number of training rounds
- `batch_size=2` → change batch size
- `lr=0.001` → change learning rate

### Change Model Architecture

Edit `src/model.py`:

- Modify `encoder` or `decoder` layers
- Adjust `BiasHead` or `Attention` modules

### Add New Features

- Add functions to `src/utils.py`
- Import and use in `scripts/`

## Our Original Contributions

1. **Correction Map** - `BiasHead` in `src/model.py` learns local corrections
2. **Color Balance** - `fix_color_balance()` in `src/utils.py` fixes global tint
3. **Attention** - Guides where corrections are needed
4. **Learned Fusion** - Learns how to combine prediction and correction

### Evaluate Quality

```bash
python scripts/evaluate.py models/colorization_final.pth
```

Shows metrics: color accuracy, brightness preservation, PSNR

### Generate Report

```bash
python scripts/generate_report.py models/colorization_final.pth
```

Creates comprehensive results report with all metrics

### Validate Setup

```bash
python scripts/validate.py
```

Checks everything is working correctly

### Compare Methods

```bash
python scripts/compare_methods.py data/portrait_woman.jpg models/colorization_final.pth
```

Shows baseline vs our method (demonstrates improvement)

## Troubleshooting

**Error loading model?**

- Make sure you trained first: `python train.py`
- Check `models/` folder has `colorization_final.pth`

**No images found?**

- Put RGB images in `data/` folder
- Run `python scripts/test_model.py` to check

**GUI not opening?**

- Install PyQt5: `pip install PyQt5`
- Check Python version (3.7+)

## For Team Members

- **To train**: Run `python train.py` (main file at root)
- **To test**: Run `python gui.py` (main file at root)
- **To colorize**: Run `python inference.py <image> <model>`
- **To modify**: Edit files in `src/` folder
- **To add features**: Add to `src/utils.py` or create new files

**Main files are at root level for easy access:**

- `train.py` - Train the model
- `gui.py` - GUI interface
- `inference.py` - Command-line colorization

Everything is organized and easy to understand. Start with `train.py` and `gui.py`!
