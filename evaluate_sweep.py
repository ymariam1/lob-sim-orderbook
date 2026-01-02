#!/usr/bin/env python3
"""
Evaluate all models from a completed sweep.

Use this if:
- You ran sweep with --skip-eval
- Evaluation failed during sweep
- You want to re-evaluate with different parameters

Usage:
    python evaluate_sweep.py sweep_results_final_20260102_130647.json
    python evaluate_sweep.py sweep_results_final_20260102_130647.json --num-runs 20
"""

import sys
import json
import argparse
import subprocess
import glob as glob_module
import os
from pathlib import Path


def evaluate_model(
    model_dir: str,
    test_data: str,
    num_runs: int = 10,
    horizons: list = None,
    data_duration: float = 24.0,
):
    """Evaluate a single model."""
    if horizons is None:
        horizons = [1800, 3600]

    # Expand glob if needed
    if os.path.isdir(test_data):
        test_files = glob_module.glob(os.path.join(test_data, "*.csv"))
    else:
        test_files = [test_data]

    eval_cmd = [
        sys.executable,
        "src/py/evaluate.py",
        "--model", f"{model_dir}/best/best_model",
        "--test-data", *test_files,
        "--num-runs", str(num_runs),
        "--horizons", *[str(h) for h in horizons],
        "--data-duration", str(data_duration),
        "--output-dir", f"{model_dir}/eval_results",
    ]

    print(f"Evaluating: {model_dir}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ FAILED")
        print(f"  Error: {result.stderr[-500:]}")
        return None

    print(f"  ✓ Success")

    # Parse results
    eval_dir = Path(f"{model_dir}/eval_results")
    json_files = list(eval_dir.glob("eval_results_*.json"))
    if not json_files:
        print(f"  ⚠ No results JSON found")
        return None

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r') as f:
        data = json.load(f)

    return {
        "rl_slippage": data.get("results", {}).get("rl_mean"),
        "twap_slippage": data.get("results", {}).get("twap_mean"),
        "ac_slippage": data.get("results", {}).get("ac_mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models from sweep results")
    parser.add_argument("results_file", help="Sweep results JSON file")
    parser.add_argument("--test-data", default="data/test",
                        help="Test data directory")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of evaluation runs per horizon")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1800, 3600],
                        help="Time horizons in seconds")
    parser.add_argument("--data-duration", type=float, default=24.0,
                        help="Expected data duration in hours")

    args = parser.parse_args()

    # Load sweep results
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    print(f"Found {len(results)} runs in {args.results_file}")
    print()

    # Evaluate each model
    updated_results = []
    for run in results:
        print(f"Run {run['run_id']}: inv={run['inventory_penalty']}, exec={run['execution_bonus']}")

        if run['status'] == 'failed':
            print(f"  Skipping (training failed)")
            updated_results.append(run)
            continue

        if not os.path.exists(run['save_dir']):
            print(f"  ✗ Model directory not found: {run['save_dir']}")
            run['status'] = 'model_not_found'
            updated_results.append(run)
            continue

        # Evaluate
        eval_results = evaluate_model(
            run['save_dir'],
            args.test_data,
            args.num_runs,
            args.horizons,
            args.data_duration,
        )

        if eval_results:
            run.update(eval_results)
            run['status'] = 'success'
        else:
            run['status'] = 'eval_failed'

        updated_results.append(run)
        print()

    # Save updated results
    output_file = args.results_file.replace('.json', '_evaluated.json')
    with open(output_file, 'w') as f:
        json.dump(updated_results, f, indent=2)

    print(f"Updated results saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Run':<4} {'Inv':<6} {'Exec':<6} {'RL Slip':<10} {'TWAP':<10} {'vs TWAP':<10} {'Status':<12}")
    print("-"*80)

    for r in updated_results:
        rl = r.get('rl_slippage')
        twap = r.get('twap_slippage')

        rl_str = f"{rl:.2f}" if rl is not None else "N/A"
        twap_str = f"{twap:.2f}" if twap is not None else "N/A"

        if rl and twap and twap > 0:
            vs_twap = ((rl - twap) / twap * 100)
            vs_str = f"{vs_twap:+.1f}%"
        else:
            vs_str = "N/A"

        print(f"{r['run_id']:<4} {r['inventory_penalty']:<6.1f} {r['execution_bonus']:<6.1f} "
              f"{rl_str:<10} {twap_str:<10} {vs_str:<10} {r['status']:<12}")

    # Find best
    valid = [r for r in updated_results if r['status'] == 'success' and r.get('rl_slippage')]
    if valid:
        best = min(valid, key=lambda r: r['rl_slippage'])
        print("\n🏆 BEST MODEL:")
        print(f"   Run {best['run_id']}: inv={best['inventory_penalty']}, exec={best['execution_bonus']}")
        print(f"   RL: {best['rl_slippage']:.2f} bps, TWAP: {best['twap_slippage']:.2f} bps")
        print(f"   Saved in: {best['save_dir']}")


if __name__ == "__main__":
    main()
