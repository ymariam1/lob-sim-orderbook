#!/usr/bin/env python3
"""
Grid search over reward hyperparameters (inventory penalty, execution bonus).

This script trains multiple models with different combinations of:
- inventory_penalty_coef: Controls urgency to complete execution
- execution_bonus: Bonus multiplier for making trading progress

Usage:
    # Quick sweep (3 models, 200k steps each)
    python sweep_rewards.py --quick

    # Full sweep (15 models, 500k steps each)
    python sweep_rewards.py --full

    # Custom sweep
    python sweep_rewards.py \
        --inventory-penalties 0.5 1.0 2.0 5.0 \
        --execution-bonuses 0.5 1.0 2.0 \
        --timesteps 500000 \
        --train-data data/train \
        --test-data data/test

    # Resume from specific run (if a run failed)
    python sweep_rewards.py --resume-from 3
"""

import os
import sys
import json
import argparse
import subprocess
import glob as glob_module
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import itertools


def edit_gym_execution_bonus(bonus_value: float):
    """
    Temporarily edit gym.py to change execution bonus.
    Returns the original value so it can be restored.
    """
    gym_path = Path("src/py/gym.py")

    # Read current file
    with open(gym_path, 'r') as f:
        lines = f.readlines()

    # Find and modify the execution bonus line
    original_value = None
    for i, line in enumerate(lines):
        if "execution_bonus =" in line and "execution_progress" in line:
            # Extract current value
            parts = line.split("=")[1].split("*")[0].strip()
            try:
                original_value = float(parts)
            except:
                original_value = 1.0  # Default from our recent change

            # Replace with new value
            indent = len(line) - len(line.lstrip())
            lines[i] = f"{' ' * indent}execution_bonus = {bonus_value} * execution_progress  # Sweep parameter\n"
            break

    # Write modified file
    with open(gym_path, 'w') as f:
        f.writelines(lines)

    # Clear Python cache
    subprocess.run(["rm", "-rf", "src/py/__pycache__"], check=False)

    return original_value


def train_model(
    inventory_penalty: float,
    execution_bonus: float,
    train_data: str,
    test_data: str,
    timesteps: int,
    model_size: str,
    run_id: int,
    skip_eval: bool = False,
) -> Dict:
    """Train a single model with given hyperparameters."""

    # Edit gym.py to set execution bonus
    print(f"\n{'='*70}")
    print(f"RUN {run_id}: inventory_penalty={inventory_penalty}, execution_bonus={execution_bonus}")
    print(f"{'='*70}")

    original_bonus = edit_gym_execution_bonus(execution_bonus)

    # Create unique save directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sweep_run_{run_id}_inv{inventory_penalty}_exec{execution_bonus}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Train command
    cmd = [
        sys.executable,
        "src/py/train_rl.py",
        "--train-data", train_data,
        "--test-data", test_data,
        "--timesteps", str(timesteps),
        "--model-size", model_size,
        "--inventory-penalty-coef", str(inventory_penalty),
        "--save-dir", save_dir,
        "--no-wandb",  # Disable wandb to avoid interactive prompts
    ]

    try:
        print(f"Training with command: {' '.join(cmd)}")
        print(f"Training in progress (this may take ~50 minutes)...")

        # Run with real-time output instead of capturing
        result_train = subprocess.run(cmd)

        if result_train.returncode != 0:
            print(f"\nTRAINING FAILED with exit code {result_train.returncode}")
            raise subprocess.CalledProcessError(result_train.returncode, cmd)

        print(f"✓ Training completed successfully")

        # Skip evaluation if requested
        if skip_eval:
            print(f"Skipping evaluation (--skip-eval flag set)")
            result = {
                "run_id": run_id,
                "inventory_penalty": inventory_penalty,
                "execution_bonus": execution_bonus,
                "save_dir": save_dir,
                "status": "trained_only",
                "rl_slippage": None,
                "twap_slippage": None,
            }
            return result

        # Evaluate the best model
        print(f"\n{'='*70}")
        print(f"Evaluating run {run_id}...")
        print(f"{'='*70}")

        # Expand glob pattern to actual files
        if os.path.isdir(test_data):
            test_files = glob_module.glob(os.path.join(test_data, "*.csv"))
            if not test_files:
                print(f"WARNING: No CSV files found in {test_data}")
                raise FileNotFoundError(f"No CSV files in {test_data}")
        else:
            test_files = [test_data]

        eval_cmd = [
            sys.executable,
            "src/py/evaluate.py",
            "--model", f"{save_dir}/best/best_model",
            "--test-data", *test_files,  # Unpack list of files
            "--num-runs", "10",
            "--horizons", "1800", "3600",
            "--data-duration", "24.0",
            "--output-dir", f"{save_dir}/eval_results",
        ]

        print(f"Evaluating on {len(test_files)} test files...")
        result_eval = subprocess.run(eval_cmd)

        if result_eval.returncode != 0:
            print(f"\nEVALUATION FAILED with exit code {result_eval.returncode}")
            raise subprocess.CalledProcessError(result_eval.returncode, eval_cmd)

        # Parse evaluation results
        eval_results = parse_latest_eval_results(f"{save_dir}/eval_results")

        # Check if evaluation actually worked
        if eval_results.get("rl_slippage") is None:
            status = "eval_failed"
            print(f"WARNING: Evaluation produced no results for run {run_id}")
        else:
            status = "success"

        result = {
            "run_id": run_id,
            "inventory_penalty": inventory_penalty,
            "execution_bonus": execution_bonus,
            "save_dir": save_dir,
            "status": status,
            **eval_results,
        }

    except subprocess.CalledProcessError as e:
        print(f"ERROR in run {run_id}: {e}")
        result = {
            "run_id": run_id,
            "inventory_penalty": inventory_penalty,
            "execution_bonus": execution_bonus,
            "save_dir": save_dir,
            "status": "failed",
            "error": str(e),
        }
    except Exception as e:
        print(f"UNEXPECTED ERROR in run {run_id}: {e}")
        result = {
            "run_id": run_id,
            "inventory_penalty": inventory_penalty,
            "execution_bonus": execution_bonus,
            "save_dir": save_dir,
            "status": "failed",
            "error": str(e),
        }
    finally:
        # Restore original execution bonus
        edit_gym_execution_bonus(original_bonus)

    return result


def parse_latest_eval_results(eval_dir: str) -> Dict:
    """Parse the latest evaluation results JSON."""
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        return {"rl_slippage": None, "twap_slippage": None, "error": "No eval results"}

    # Find latest results file
    json_files = list(eval_path.glob("eval_results_*.json"))
    if not json_files:
        return {"rl_slippage": None, "twap_slippage": None, "error": "No JSON found"}

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return {
        "rl_slippage": data.get("results", {}).get("rl_mean"),
        "twap_slippage": data.get("results", {}).get("twap_mean"),
        "ac_slippage": data.get("results", {}).get("ac_mean"),
    }


def print_results_table(results: List[Dict]):
    """Print a nice table of results."""
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Run':<4} {'Inv Penalty':<12} {'Exec Bonus':<12} {'RL Slippage':<14} {'TWAP':<10} {'vs TWAP':<10} {'Status':<10}")
    print("-"*80)

    for r in results:
        if r["status"] in ["failed", "eval_failed"]:
            status_str = "EVAL FAIL" if r["status"] == "eval_failed" else "FAILED"
            print(f"{r['run_id']:<4} {r['inventory_penalty']:<12.1f} {r['execution_bonus']:<12.1f} {'N/A':<14} {'N/A':<10} {'N/A':<10} {status_str:<10}")
            continue

        rl = r.get("rl_slippage")
        twap = r.get("twap_slippage")

        # Format values, handling None
        rl_str = f"{rl:.2f}" if rl is not None else "N/A"
        twap_str = f"{twap:.2f}" if twap is not None else "N/A"

        if rl is not None and twap is not None and twap > 0:
            vs_twap = ((rl - twap) / twap * 100)
            vs_str = f"{vs_twap:+.1f}%"
        else:
            vs_str = "N/A"

        print(f"{r['run_id']:<4} {r['inventory_penalty']:<12.1f} {r['execution_bonus']:<12.1f} "
              f"{rl_str:<14} {twap_str:<10} {vs_str:<10} {r['status']:<10}")

    print("="*80)

    # Find best model
    valid_results = [r for r in results if r["status"] == "success" and r.get("rl_slippage") is not None]
    if valid_results:
        best = min(valid_results, key=lambda r: r["rl_slippage"])
        print(f"\n🏆 BEST MODEL:")
        print(f"   Run {best['run_id']}: inventory_penalty={best['inventory_penalty']}, "
              f"execution_bonus={best['execution_bonus']}")
        print(f"   RL Slippage: {best['rl_slippage']:.2f} bps")
        print(f"   TWAP: {best.get('twap_slippage', 0):.2f} bps")
        print(f"   Saved in: {best['save_dir']}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over reward hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset sweeps
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep: 3 configs, 200k steps (for testing)")
    parser.add_argument("--full", action="store_true",
                        help="Full sweep: 15 configs, 500k steps (comprehensive)")

    # Custom sweep
    parser.add_argument("--inventory-penalties", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0, 5.0],
                        help="Inventory penalty coefficients to try (NEW: 0.1-5.0 range)")
    parser.add_argument("--execution-bonuses", type=float, nargs="+",
                        default=[5.0, 10.0, 20.0],
                        help="Execution bonus multipliers to try (NEW: 5.0-20.0 range, increased from 0.5-2.0)")

    # Training params
    parser.add_argument("--train-data", default="data/train",
                        help="Training data directory")
    parser.add_argument("--test-data", default="data/test",
                        help="Test data directory")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Training timesteps per run")
    parser.add_argument("--model-size", default="base",
                        choices=["small", "base", "large", "xlarge"],
                        help="Model architecture size")

    # Resume
    parser.add_argument("--resume-from", type=int,
                        help="Resume from specific run ID (skip earlier runs)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only train models (faster, evaluate manually later)")

    args = parser.parse_args()

    # Handle presets
    if args.quick:
        inventory_penalties = [0.1, 0.5, 1.0]
        execution_bonuses = [5.0, 10.0, 20.0]
        timesteps = 200000
        print("QUICK SWEEP MODE: 9 runs, 200k steps each")
    elif args.full:
        inventory_penalties = [0.1, 0.5, 1.0, 2.0, 5.0]
        execution_bonuses = [5.0, 10.0, 20.0]
        timesteps = 500000
        print("FULL SWEEP MODE: 15 runs, 500k steps each")
    else:
        inventory_penalties = args.inventory_penalties
        execution_bonuses = args.execution_bonuses
        timesteps = args.timesteps

    # Generate all combinations
    configs = list(itertools.product(inventory_penalties, execution_bonuses))

    print(f"\nSWEEP CONFIGURATION:")
    print(f"  Inventory penalties: {inventory_penalties}")
    print(f"  Execution bonuses: {execution_bonuses}")
    print(f"  Total runs: {len(configs)}")
    print(f"  Timesteps per run: {timesteps:,}")
    print(f"  Model size: {args.model_size}")
    print(f"  Estimated time: ~{len(configs) * timesteps / 100000 * 10:.0f} minutes")
    print()

    input("Press Enter to start sweep (or Ctrl+C to cancel)...")

    # Run sweep
    results = []
    start_run = args.resume_from if args.resume_from else 1

    for run_id, (inv_penalty, exec_bonus) in enumerate(configs, start=1):
        if run_id < start_run:
            print(f"Skipping run {run_id} (resume from {start_run})")
            continue

        result = train_model(
            inventory_penalty=inv_penalty,
            execution_bonus=exec_bonus,
            train_data=args.train_data,
            test_data=args.test_data,
            timesteps=timesteps,
            model_size=args.model_size,
            run_id=run_id,
            skip_eval=args.skip_eval,
        )
        results.append(result)

        # Save intermediate results
        results_file = f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nIntermediate results saved to: {results_file}")

    # Print final results table
    print_results_table(results)

    # Save final results
    final_results_file = f"sweep_results_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal results saved to: {final_results_file}")


if __name__ == "__main__":
    main()
