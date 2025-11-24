#!/usr/bin/env python3
"""
Download the image colorization dataset from Kaggle using kagglehub.
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download and setup the dataset in the data folder."""

    # Download latest version
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("aayush9753/image-colorization-dataset")

    print(f"Path to dataset files: {path}")

    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)

    # Copy files from downloaded path to data directory
    downloaded_path = Path(path)
    if downloaded_path.exists():
        print(f"\nCopying files to {data_dir}...")

        # Check if there are subdirectories or files directly
        items = list(downloaded_path.iterdir())

        for item in items:
            if item.is_file():
                dest = data_dir / item.name
                print(f"Copying {item.name}...")
                shutil.copy2(item, dest)
            elif item.is_dir():
                dest = data_dir / item.name
                print(f"Copying directory {item.name}...")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)

        print(f"\n[SUCCESS] Dataset successfully downloaded and extracted to: {data_dir}")
        print(f"\nContents of data directory:")
        for item in sorted(data_dir.iterdir()):
            if item.is_file():
                size = item.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"  - {item.name} ({size:.2f} MB)")
            else:
                print(f"  - {item.name}/ (directory)")
    else:
        print(f"Error: Downloaded path does not exist: {path}")

    return path

if __name__ == "__main__":
    download_dataset()

