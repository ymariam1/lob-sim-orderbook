#!/usr/bin/env python3
"""
Diagnostic script to debug RL model behavior.

This script helps identify why a model isn't learning by examining:
1. Action distribution (is the agent stuck on "Hold"?)
2. Reward patterns (are rewards too small/large?)
3. Execution completion (is the agent trading at all?)
4. Environment/model compatibility

Usage:
    python diagnose_model.py --model models/best/best_model --data data/test/blockchain_l3_2023-03-01.csv
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    print("  pip install stable-baselines3")
    sys.exit(1)

from src.py.gym import LOBEnv


def diagnose_model(
    model_path: str,
    data_path: str,
    n_steps: int = 1000,
    n_episodes: int = 5,
    target_qty: int = 100,
    execution_side: str = "SELL",
):
    """
    Run diagnostic checks on a trained model.

    Args:
        model_path: Path to trained model (without .zip extension)
        data_path: Path to test data CSV
        n_steps: Max steps per episode
        n_episodes: Number of episodes to run
        target_qty: Target quantity per episode
        execution_side: "BUY" or "SELL"
    """
    print("=" * 70)
    print("RL MODEL DIAGNOSTICS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {n_steps}")
    print("=" * 70)

    # Load model
    try:
        model = PPO.load(model_path)
        print("\n✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        return

    # Create environment with SAME params as training
    try:
        env = LOBEnv(
            data_path=data_path,
            target_qty=target_qty,
            execution_side=execution_side,
            timestamp_unit_ns=1000,  # Tardis/Blockchain microseconds
            inventory_penalty_coef=1.0,  # New default
        )
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return

    # Check compatibility
    print("\n" + "-" * 70)
    print("COMPATIBILITY CHECK")
    print("-" * 70)
    print(f"Model observation space: {model.observation_space}")
    print(f"Env observation space:   {env.observation_space}")
    print(f"Model action space:      {model.action_space}")
    print(f"Env action space:        {env.action_space}")

    if model.observation_space != env.observation_space:
        print("\n⚠️  WARNING: Observation spaces don't match!")
        print("   The model was trained with a different environment.")
        print("   Results may be unreliable.")

    if model.action_space != env.action_space:
        print("\n⚠️  WARNING: Action spaces don't match!")
        print("   The model was trained with a different action space.")
        print("   The agent will behave randomly!")
        return

    # Run episodes
    print("\n" + "-" * 70)
    print("RUNNING EPISODES")
    print("-" * 70)

    all_actions = []
    all_rewards = []
    all_completions = []
    all_slippages = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        actions = []
        rewards = []
        steps = 0

        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))

            obs, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            steps += 1

            if term or trunc:
                break

        all_actions.extend(actions)
        all_rewards.extend(rewards)

        executed = info.get('executed_qty', 0)
        completion = (executed / target_qty) * 100 if target_qty > 0 else 0
        all_completions.append(completion)

        if 'slippage_bps' in info:
            all_slippages.append(info['slippage_bps'])

        print(f"Episode {ep+1}/{n_episodes}:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {sum(rewards):.3f}")
        print(f"  Mean reward: {np.mean(rewards):.4f}")
        print(f"  Executed: {executed}/{target_qty} ({completion:.1f}%)")
        if 'slippage_bps' in info:
            print(f"  Slippage: {info['slippage_bps']:.2f} bps")
        print()

    # Aggregate results
    print("=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)

    # Action distribution
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)

    print("\n1. ACTION DISTRIBUTION")
    print("-" * 70)
    action_names = {
        0: "Hold",
        1: "Limit @best",
        2: "Limit +1 tick",
        3: "Limit +2 ticks",
        4: "Limit +3 ticks",
        5: "Limit +4 ticks",
        6: "Market",
        7: "Cancel",
    }

    for action in range(8):
        count = action_counts.get(action, 0)
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        name = action_names.get(action, f"Action {action}")
        print(f"  {action}: {name:<20} {count:>6} ({pct:>5.1f}%)")

    # Diagnosis
    print("\nDIAGNOSIS:")
    hold_pct = (action_counts.get(0, 0) / total_actions * 100) if total_actions > 0 else 0

    if hold_pct > 90:
        print("  ❌ CRITICAL: Agent is stuck on HOLD (>90%)")
        print("     → The agent hasn't learned to trade")
        print("     → Increase inventory_penalty_coef or add execution bonus")
    elif hold_pct > 50:
        print("  ⚠️  WARNING: Agent holds too much (>50%)")
        print("     → May need stronger trading incentives")
    else:
        print("  ✓ Agent is actively trading")

    # Reward analysis
    print("\n2. REWARD ANALYSIS")
    print("-" * 70)
    print(f"  Mean reward per step: {np.mean(all_rewards):.4f}")
    print(f"  Std dev:              {np.std(all_rewards):.4f}")
    print(f"  Min reward:           {np.min(all_rewards):.4f}")
    print(f"  Max reward:           {np.max(all_rewards):.4f}")
    print(f"  Total reward:         {np.sum(all_rewards):.3f}")

    if abs(np.mean(all_rewards)) < 1e-4:
        print("\n  ⚠️  WARNING: Rewards are very small")
        print("     → Agent may have trouble learning signal")
        print("     → Consider scaling rewards or increasing penalties")

    # Completion analysis
    print("\n3. EXECUTION COMPLETION")
    print("-" * 70)
    mean_completion = np.mean(all_completions)
    print(f"  Mean completion: {mean_completion:.1f}%")
    print(f"  Min completion:  {np.min(all_completions):.1f}%")
    print(f"  Max completion:  {np.max(all_completions):.1f}%")

    if mean_completion < 50:
        print("\n  ❌ CRITICAL: Agent fails to complete execution (<50%)")
        print("     → Terminal penalty may be too weak")
        print("     → Inventory penalty may be too weak")
    elif mean_completion < 90:
        print("\n  ⚠️  WARNING: Agent completes inconsistently (<90%)")
        print("     → May need tuning")
    else:
        print("\n  ✓ Agent completes execution reliably")

    # Slippage analysis
    if all_slippages:
        print("\n4. SLIPPAGE ANALYSIS")
        print("-" * 70)
        print(f"  Mean slippage: {np.mean(all_slippages):.2f} bps")
        print(f"  Std dev:       {np.std(all_slippages):.2f} bps")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if hold_pct > 90:
        print("1. Agent isn't trading - Try:")
        print("   - Increase --inventory-penalty-coef to 5.0 or 10.0")
        print("   - Verify execution bonus is enabled in gym.py")
        print("   - Check terminal penalty strength")
    elif mean_completion < 50:
        print("1. Agent doesn't complete - Try:")
        print("   - Increase terminal penalty in gym.py")
        print("   - Increase --inventory-penalty-coef")
    elif mean_completion > 95 and np.mean(all_slippages) < 10:
        print("1. Agent is performing well!")
        print("   - Consider longer training for further improvement")
        print("   - Try different model sizes (--model-size)")
    else:
        print("1. Agent needs more training:")
        print("   - Increase --timesteps (try 500k or 1M)")
        print("   - Try different learning rates (--lr)")

    print("\n" + "=" * 70)
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose RL model behavior and training issues"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (without .zip extension)"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)"
    )
    parser.add_argument(
        "--target-qty",
        type=int,
        default=100,
        help="Target quantity per episode (default: 100)"
    )
    parser.add_argument(
        "--side",
        choices=["BUY", "SELL"],
        default="SELL",
        help="Execution side (default: SELL)"
    )

    args = parser.parse_args()

    diagnose_model(
        model_path=args.model,
        data_path=args.data,
        n_steps=args.n_steps,
        n_episodes=args.n_episodes,
        target_qty=args.target_qty,
        execution_side=args.side,
    )


if __name__ == "__main__":
    main()
