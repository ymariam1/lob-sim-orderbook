#!/usr/bin/env python3
"""
Run WandB hyperparameter sweep for RL training.

Usage:
    # Initialize sweep
    python scripts/run_sweep.py --init

    # Run agent (called by wandb automatically)
    python scripts/run_sweep.py --agent

    # Resume existing sweep
    python scripts/run_sweep.py --resume SWEEP_ID
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def init_sweep(config_path: str = "configs/sweep.yaml"):
    """Initialize a new WandB sweep."""
    sweep_id = wandb.sweep(
        sweep=config_path,
        project="lob-rl",
    )
    print(f"Sweep initialized: {sweep_id}")
    print(f"Run agents with: wandb agent {sweep_id}")
    return sweep_id


def run_agent(sweep_id: str = None, count: int = None):
    """Run a WandB agent (for hyperparameter search)."""
    if sweep_id is None:
        # Get from environment (set by wandb)
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        if not sweep_id:
            print("ERROR: No sweep ID provided. Use --sweep-id or set WANDB_SWEEP_ID")
            sys.exit(1)
    
    wandb.agent(
        sweep_id,
        function=main_training_function,
        count=count,
    )


def main_training_function():
    """Main training function called by wandb agent."""
    import subprocess
    
    # Get config from wandb
    config = wandb.config
    
    # Build command
    cmd = [
        sys.executable,
        "src/py/train_rl.py",
        "--data", config.get("data_path", "data/csv/blockchain_l3_2023-03-01.csv"),
        "--timesteps", str(config.get("timesteps", 1000000)),
        "--agent-type", config.get("agent_type", "institutional"),
        "--target-qty", str(config.get("target_qty", 1000)),
        "--side", config.get("execution_side", "SELL"),
        "--lr", str(config["learning_rate"]),
        "--n-steps", str(config["n_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--n-epochs", str(config.get("n_epochs", 10)),
        "--gamma", str(config["gamma"]),
        "--net-arch", *[str(x) for x in config.get("net_arch", [64, 64])],
        "--eval-freq", str(config.get("eval_freq", 50000)),
    ]
    
    # Add optional parameters
    if "gae_lambda" in config:
        cmd.extend(["--gae-lambda", str(config["gae_lambda"])])
    if "ent_coef" in config:
        cmd.extend(["--ent-coef", str(config["ent_coef"])])
    if "clip_range" in config:
        cmd.extend(["--clip-range", str(config["clip_range"])])
    
    # Run training
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run WandB hyperparameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--init", action="store_true",
                        help="Initialize a new sweep")
    parser.add_argument("--agent", action="store_true",
                        help="Run as wandb agent")
    parser.add_argument("--sweep-id", help="Sweep ID for agent mode")
    parser.add_argument("--count", type=int, help="Number of runs for agent")
    parser.add_argument("--config", default="configs/sweep.yaml",
                        help="Sweep config file")
    
    args = parser.parse_args()
    
    if args.init:
        init_sweep(args.config)
    elif args.agent:
        run_agent(args.sweep_id, args.count)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

