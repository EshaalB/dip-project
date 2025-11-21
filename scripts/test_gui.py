# Quick test to check if GUI can run

import sys
import os

print("Testing GUI setup...")
print("=" * 60)

# Check PyQt5
try:
    from PyQt5.QtWidgets import QApplication
    print("✓ PyQt5 installed")
except ImportError as e:
    print(f"✗ PyQt5 NOT installed: {e}")
    print("  Install with: pip install PyQt5")
    sys.exit(1)

# Check torch
try:
    import torch
    print(f"✓ PyTorch installed (version: {torch.__version__})")
except OSError as e:
    if "DLL" in str(e) or "c10.dll" in str(e):
        print(f"✗ PyTorch DLL ERROR: {e}")
        print("\n  FIX: Run fix_pytorch.bat or:")
        print("  pip uninstall torch torchvision -y")
        print("  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu")
    else:
        print(f"✗ PyTorch error: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"✗ PyTorch NOT installed: {e}")
    sys.exit(1)

# Check our modules
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.utils import load_image, rgb_to_lab, save_image
    from src.model import ColorizationModel
    from scripts.inference import load_model, colorize_image
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test QApplication creation
try:
    app = QApplication(sys.argv)
    print("✓ QApplication created successfully")
    print("\n" + "=" * 60)
    print("GUI setup is OK! You can run: python scripts/gui.py")
    print("=" * 60)
except Exception as e:
    print(f"✗ QApplication creation failed: {e}")
    sys.exit(1)

