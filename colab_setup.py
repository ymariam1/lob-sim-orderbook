#!/usr/bin/env python3
"""
Quick setup script for Google Colab GPU training.
Run this first in Colab to verify everything works.
"""

import subprocess
import sys
import os

def check_gpu():
    """Check if GPU is available."""
    print("=" * 60)
    print("Checking GPU availability...")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("⚠️  No GPU detected! Make sure Runtime > Change runtime type > GPU is selected")
        return False

def install_dependencies():
    """Install required packages."""
    print("\n" + "=" * 60)
    print("Installing dependencies...")
    print("=" * 60)

    # Install the package
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)

    # Install RL dependencies
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        'stable-baselines3[extra]', 'wandb', 'tensorboard'
    ], check=True)

    print("✅ Dependencies installed successfully")

def verify_imports():
    """Verify all imports work."""
    print("\n" + "=" * 60)
    print("Verifying imports...")
    print("=" * 60)

    try:
        import lob_sim as ob
        print("✅ lob_sim imported")

        import stable_baselines3
        print("✅ stable-baselines3 imported")

        from src.py.gym import LOBEnv
        print("✅ LOBEnv imported")

        from src.py.train_rl import train
        print("✅ train_rl imported")

        import torch
        print(f"✅ PyTorch imported (CUDA available: {torch.cuda.is_available()})")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_data():
    """Check if training data exists."""
    print("\n" + "=" * 60)
    print("Checking for training data...")
    print("=" * 60)

    data_dir = "data/csv"
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files in {data_dir}")
            print(f"   First file: {csv_files[0]}")
            return True
        else:
            print(f"⚠️  {data_dir} exists but no CSV files found")
    else:
        print(f"⚠️  {data_dir} does not exist")

    print("\nTo download data, run:")
    print("  !python data/fetch_l3.py --hours 1")
    return False

def main():
    """Run all setup checks."""
    print("🚀 LOB-SIM Colab Setup")
    print("=" * 60)

    has_gpu = check_gpu()

    # Only install if not already installed
    try:
        import lob_sim
        print("\n✅ lob_sim already installed, skipping installation")
    except ImportError:
        install_dependencies()

    imports_ok = verify_imports()
    data_ok = check_data()

    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    print(f"GPU Available: {'✅' if has_gpu else '⚠️ '}")
    print(f"Imports OK: {'✅' if imports_ok else '❌'}")
    print(f"Training Data: {'✅' if data_ok else '⚠️ '}")

    if has_gpu and imports_ok:
        print("\n✅ Ready to train! Run:")
        print("  !python src/py/train_rl.py --train-data data/csv --timesteps 50000")

    print("=" * 60)

if __name__ == "__main__":
    main()
