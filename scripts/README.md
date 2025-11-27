# Scripts Directory

This directory contains utility scripts for testing, validating, and demonstrating the colorization model.

## Prerequisites

- Trained model in `models/` directory (run `train.py` from root first)
- Test images in `data/` directory
- All dependencies installed (`pip install -r requirements.txt`)

## Scripts Overview

### 1. test_model.py
**Purpose:** Quick test to verify the model architecture works correctly

**Usage:**
```bash
python scripts/test_model.py
```

**What it does:**
- Creates a model instance and counts parameters
- Tests forward pass with dummy data
- Tests with a real image from `data/` directory (if available)
- Validates output shapes and ranges

### 2. validate.py
**Purpose:** Comprehensive validation of dataset, model, and color conversions

**Usage:**
```bash
python scripts/validate.py
```

**What it checks:**
- Dataset integrity and image loading
- Model creation and forward pass
- Color space conversions (RGB ↔ LAB)
- Original contributions (bias head, attention)
- Brightness preservation

### 3. compare_methods.py
**Purpose:** Compare baseline vs enhanced method to show improvements

**Usage:**
```bash
python scripts/compare_methods.py <image_path> <model_path>
```

**Example:**
```bash
python scripts/compare_methods.py data/image0001.jpg models/colorization_best.pth
```

**Output:**
- `output/<name>_baseline.jpg` - Without corrections
- `output/<name>_our_method.jpg` - With corrections

### 4. evaluate.py
**Purpose:** Quantitative evaluation with metrics

**Usage:**
```bash
python scripts/evaluate.py <model_path> [num_images]
```

**Example:**
```bash
python scripts/evaluate.py models/colorization_best.pth 5
```

**Metrics computed:**
- Brightness error (L channel preservation)
- Color error (a, b channel accuracy)
- PSNR (Peak Signal-to-Noise Ratio)

### 5. demo.py
**Purpose:** Complete demonstration showing all processing steps

**Usage:**
```bash
python scripts/demo.py <image_path> <model_path>
```

**Example:**
```bash
python scripts/demo.py data/image0001.jpg models/colorization_best.pth
```

**Output:**
- Input grayscale
- Output colorized
- Correction map visualization
- Attention map visualization

### 6. visualize_results.py
**Purpose:** Visualize the model's internal components and contributions

**Usage:**
```bash
python scripts/visualize_results.py <image_path> <model_path>
```

**Example:**
```bash
python scripts/visualize_results.py data/image0001.jpg models/colorization_best.pth
```

**Output:**
- Colorized image
- Correction map (shows where model applies corrections)
- Attention map (shows where model focuses)
- Comparison without enhancements

### 7. generate_report.py
**Purpose:** Generate comprehensive evaluation report with multiple images

**Usage:**
```bash
python scripts/generate_report.py <model_path> [num_images]
```

**Example:**
```bash
python scripts/generate_report.py models/colorization_best.pth 5
```

**Output:**
- `output/report/` directory with all results
- `output/report/results_report.txt` with metrics summary
- Original, colorized, and grayscale versions for each image