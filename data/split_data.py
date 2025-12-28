#!/usr/bin/env python3
"""
Split CSV files into train/test directories for local training.
Usage: python split_data.py
"""

import os
import shutil
from pathlib import Path

def split_data(csv_dir="data/csv", train_dir="data/train", test_dir="data/test"):
    """
    Split CSV files into 80/20 train/test split.

    Args:
        csv_dir: Source directory containing CSV files
        train_dir: Destination directory for training data
        test_dir: Destination directory for test data
    """
    # Convert to Path objects
    csv_path = Path(csv_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # Create directories if they don't exist
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Get all CSV files and sort them
    csv_files = sorted(csv_path.glob("*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_dir}")
        return

    # Calculate split point (80% train, 20% test)
    n_files = len(csv_files)
    split_idx = int(n_files * 0.8)

    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]

    print(f"Found {n_files} CSV files")
    print(f"Splitting: {len(train_files)} train, {len(test_files)} test\n")

    # Copy train files
    print("Copying train files...")
    for f in train_files:
        dest = train_path / f.name
        shutil.copy2(f, dest)
        print(f"  ✓ {f.name}")

    # Copy test files
    print("\nCopying test files...")
    for f in test_files:
        dest = test_path / f.name
        shutil.copy2(f, dest)
        print(f"  ✓ {f.name}")

    print(f"\n{'='*60}")
    print("Split complete!")
    print(f"Train files: {len(train_files)} in {train_path.absolute()}")
    print(f"Test files: {len(test_files)} in {test_path.absolute()}")
    print(f"{'='*60}\n")

    # Print training command
    print("To train the model, run:")
    print(f"\npython src/py/train_rl.py \\")
    print(f"  --train-dir {train_path.absolute()} \\")
    print(f"  --test-dir {test_path.absolute()} \\")
    print(f"  --total-timesteps 100000 \\")
    print(f"  --device cpu\n")

if __name__ == "__main__":
    split_data()
