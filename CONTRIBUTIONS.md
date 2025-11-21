# Our Original Contributions

## Summary

We designed a pseudo-colorization method that:

1. Uses neural network to predict colors (Ca, Cr channels)
2. Adds correction map to fix local mistakes (OUR ORIGINAL)
3. Adds color balance to fix global tint (OUR ORIGINAL)
4. Keeps brightness (L channel) constant

## Contribution 1: Correction Map

**Location:** `src/model.py` - `BiasHead` class

**What it does:**

- Learns local color corrections based on image features
- Adapts to different regions of the image
- Fixes mistakes the main network makes

**Why it's original:**

- Not just post-processing - it's learned during training
- Adapts to local image features
- Works with attention to focus corrections

## Contribution 2: Color Balance

**Location:** `src/utils.py` - `color_balance_histogram_remap()` function

**What it does:**

- Fixes global color tint (prevents yellow tint)
- Adjusts mean of a and b channels to neutral
- Keeps colors natural

**Why it's original:**

- Our specific design for histogram remapping
- Prevents common colorization problems
- Simple but effective

## Contribution 3: Attention Mechanism

**Location:** `src/model.py` - `Attention` class

**What it does:**

- Finds where corrections are needed
- Guides correction map application
- Focuses on problem areas

## Contribution 4: Learned Fusion

**Location:** `src/model.py` - `Fusion` class

**What it does:**

- Learns how to combine prediction and correction
- Instead of fixed weight, network learns optimal combination
- More intelligent than simple addition

## Key Achievement: Brightness Preservation

- **L channel is kept CONSTANT** (not modified)
- Only a and b channels are predicted
- This preserves image brightness while adding color
- Verified in validation scripts

## How to Verify

1. Run: `python scripts/validate.py` - checks brightness preservation
2. Run: `python scripts/evaluate.py` - shows brightness error (should be low)
3. Run: `python scripts/demo.py` - shows step-by-step process
