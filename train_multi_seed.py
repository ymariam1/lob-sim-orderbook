#!/usr/bin/env python3
"""
Multi-seed training for statistical significance.

Trains multiple RL agents with different random seeds to enable proper
statistical testing with mean ± std across independent runs.

Usage:
    python train_multi_seed.py --n-seeds 5 --timesteps 100000
    python train_multi_seed.py --n-seeds 3 --timesteps 500000 --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_multiple_seeds(
    train_data: str,
    test_data: str,
    n_seeds: int = 5,
    timesteps: int = 100000,
    agent_type: str = "institutional",
    target_qty: int = None,
    lr: float = 3e-4,
    save_dir: str = "models",
    device: str = "cpu",
    model_size: str = None,
):
    """
    Train multiple models with different random seeds.

    Args:
        train_data: Path to training data directory
        test_data: Path to test data directory
        n_seeds: Number of independent runs
        timesteps: Training timesteps per run
        agent_type: Latency profile
        target_qty: Target quantity per episode
        lr: Learning rate
        save_dir: Directory to save models
        device: cpu or cuda
    """
    print("=" * 70)
    print("MULTI-SEED RL TRAINING FOR STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    print(f"Number of seeds: {n_seeds}")
    print(f"Timesteps per run: {timesteps:,}")
    print(f"Training data: {train_data}")
    print(f"Test data: {test_data}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    # Create base save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"TRAINING RUN {seed + 1}/{n_seeds} (seed={seed})")
        print(f"{'='*70}\n")

        # Create seed-specific save directory
        seed_save_dir = save_path / f"seed_{seed}"
        seed_save_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            "src/py/train_rl.py",
            "--train-data", train_data,
            "--test-data", test_data,
            "--timesteps", str(timesteps),
            "--agent-type", agent_type,
            "--lr", str(lr),
            "--save-dir", str(seed_save_dir),
            "--log-dir", f"logs/seed_{seed}",
        ]

        # Add optional parameters
        if target_qty is not None:
            cmd.extend(["--target-qty", str(target_qty)])

        if model_size is not None:
            cmd.extend(["--model-size", model_size])

        # Add device if not default
        # Note: train_rl.py doesn't have --device flag, but keeping this for future
        # if device != "cpu":
        #     cmd.extend(["--device", device])

        # Run training
        try:
            result = subprocess.run(cmd, check=True)
            results.append({
                'seed': seed,
                'status': 'success',
                'best_model_path': seed_save_dir / "best" / "best_model.zip",
                'latest_model_path': seed_save_dir / "ppo_lob_latest.zip"
            })
            print(f"\n✓ Seed {seed} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Seed {seed} failed with error code {e.returncode}")
            results.append({
                'seed': seed,
                'status': 'failed'
            })
            # Continue with remaining seeds
            continue

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-SEED TRAINING SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"Successful runs: {len(successful)}/{n_seeds}")
    print(f"Failed runs: {len(failed)}/{n_seeds}")

    if successful:
        print("\nBest models (use these for evaluation):")
        for r in successful:
            print(f"  - Seed {r['seed']}: {r['best_model_path']}")

        print("\nLatest models (for resuming training):")
        for r in successful:
            print(f"  - Seed {r['seed']}: {r['latest_model_path']}")

    if failed:
        print("\nFailed seeds:")
        for r in failed:
            print(f"  - Seed {r['seed']}")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. Evaluate each BEST model on test data:")
    print("   python src/py/train_rl.py --eval-only \\")
    print(f"     --model {save_path}/seed_0/best/best_model \\")
    print(f"     --train-data {test_data} \\")
    print("     --n-eval-episodes 20")
    print()
    print("2. Run statistical comparison (create evaluate_multi_seed.py):")
    print(f"   python evaluate_multi_seed.py --save-dir {save_path}")
    print()
    print("3. Report: mean ± std across all seeds")
    print()
    print("NOTE: Use BEST models (models/seed_X/best/best_model.zip) for evaluation,")
    print("      not the latest checkpoints.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple RL agents with different seeds for statistical significance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--train-data",
        default="data/train",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--test-data",
        default="data/test",
        help="Path to test data directory"
    )

    # Training
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of independent runs with different seeds (default: 5)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Training timesteps per run (default: 100k)"
    )
    parser.add_argument(
        "--agent-type",
        default="institutional",
        help="Latency profile (default: institutional)"
    )
    parser.add_argument(
        "--target-qty",
        type=int,
        default=None,
        help="Target quantity per episode (default: calculated from volume)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large", "xlarge"],
        default="base",
        help="Model size: small=[64,64], base=[128,128], large=[256,256], xlarge=[512,512] (default: base)"
    )
    parser.add_argument(
        "--save-dir",
        default="models_multi_seed",
        help="Base directory for saving models (default: models_multi_seed)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default: cpu)"
    )

    args = parser.parse_args()

    train_multiple_seeds(
        train_data=args.train_data,
        test_data=args.test_data,
        n_seeds=args.n_seeds,
        timesteps=args.timesteps,
        agent_type=args.agent_type,
        target_qty=args.target_qty,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
