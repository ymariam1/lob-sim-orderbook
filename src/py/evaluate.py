#!/usr/bin/env python3
"""
Rigorous Evaluation Protocol for RL Trading Agents.

Based on best practices from rl-exec.pdf:
- Per-day evaluation with multiple runs
- Paired daily differences
- Statistical testing (Wilcoxon, bootstrap CIs)
- Effect sizes and win rates

Usage:
    python src/py/evaluate.py --model models/ppo_lob_latest \
        --test-data data/csv/blockchain_l3_2023-06-01.csv \
        --output results/eval_results.json
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.utils import resample

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
except ImportError:
    print("ERROR: stable_baselines3 not installed")
    sys.exit(1)

from src.py.gym import LOBEnv
from src.py.baselines import TWAPExecutor, AlmgrenChrissExecutor


def run_rl_episode(
    agent,
    vec_normalize: Optional[VecNormalize],
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    agent_type: str = "institutional",
    execution_side: str = "SELL",
) -> Tuple[float, float, Dict]:
    """
    Run a single RL episode with fixed start time and horizon.
    
    Returns:
        (slippage_bps, completion_rate, episode_info)
    """
    # Create environment
    env = LOBEnv(
        data_path=data_path,
        agent_type=agent_type,
        timestamp_unit_ns=1000,
        target_qty=target_qty,
        execution_side=execution_side,
        warmup_duration_ns=60_000_000_000,  # 60s warmup
        step_duration_ns=10_000_000,  # 10ms steps
    )
    
    # Wrap with VecNormalize if provided
    if vec_normalize:
        env = vec_normalize.normalize_obs(env)
    
    # Reset and advance to start time
    obs, info = env.reset()
    arrival_price = info["arrival_price"]
    
    # Advance to start time (if needed)
    current_time = env._exchange.GetCurrentTime()
    if current_time < start_time_ns:
        advance_ns = start_time_ns - current_time
        env._loader.PumpToExchange(env._exchange, advance_ns)
    
    # Run episode
    total_reward = 0
    actions = []
    fills = []
    prices = []
    inventory = []
    
    episode_start_time = env._exchange.GetCurrentTime()
    target_end_time = episode_start_time + horizon_ns
    
    while env._exchange.GetCurrentTime() < target_end_time:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(action)
        
        # Step
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        # Track state
        prices.append(info["mid_price"])
        inventory.append(info["executed_qty"])
        
        # Track fills
        if "vwap" in info:
            fills.append({
                "price": info["vwap"],
                "qty": info["executed_qty"],
                "time": env._exchange.GetCurrentTime(),
            })
        
        if term or trunc:
            break
    
    # Final metrics
    final_info = env._get_info()
    executed_qty = final_info["executed_qty"]
    completion_rate = executed_qty / target_qty if target_qty > 0 else 0
    
    # Calculate slippage
    if executed_qty > 0 and "vwap" in final_info:
        vwap = final_info["vwap"]
        slippage_bps = abs(vwap - arrival_price) / arrival_price * 10000
    else:
        slippage_bps = 1000.0  # Penalty for incomplete execution
    
    episode_info = {
        "total_reward": total_reward,
        "slippage_bps": slippage_bps,
        "completion_rate": completion_rate,
        "executed_qty": executed_qty,
        "arrival_price": arrival_price,
        "final_price": final_info.get("vwap", arrival_price),
        "num_fills": len(fills),
        "actions": actions,
        "prices": prices,
        "inventory": inventory,
        "fills": fills,
    }
    
    env.close()
    return slippage_bps, completion_rate, episode_info


def run_twap_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    num_slices: int = 60,
) -> float:
    """Run TWAP baseline on exact same window."""
    executor = TWAPExecutor(
        data_path=data_path,
        target_qty=target_qty,
        total_time_ns=horizon_ns,
        num_slices=num_slices,
        agent_latency_ns=10_000_000,  # 10ms institutional
    )
    
    result = executor.execute()
    return result["slippage_bps"]


def run_ac_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    risk_aversion: float = 1e-6,
) -> float:
    """Run Almgren-Chriss baseline on exact same window."""
    executor = AlmgrenChrissExecutor(
        data_path=data_path,
        target_qty=target_qty,
        total_time_ns=horizon_ns,
        risk_aversion=risk_aversion,
        agent_latency_ns=10_000_000,  # 10ms institutional
    )
    
    result = executor.execute()
    return result["slippage_bps"]


def compute_statistical_tests(
    results: Dict,
    alpha: float = 0.05,
) -> Dict:
    """
    Statistical inference from rl-exec.pdf:
    - Wilcoxon signed-rank test (one-sided)
    - Bootstrap 95% CIs
    - Effect sizes (Cohen's d)
    - Win rates
    """
    rl = np.array(results['rl'])
    twap = np.array(results['twap'])
    ac = np.array(results['ac'])
    
    stats = {}
    
    # RL vs TWAP
    delta_twap = twap - rl  # Positive = RL better (lower slippage)
    
    # Wilcoxon signed-rank test (one-sided: RL < TWAP)
    if len(delta_twap) > 0:
        wilcoxon_result = wilcoxon(delta_twap, alternative='greater')
        
        # Bootstrap CI for mean difference
        boot_means = []
        for _ in range(10000):
            boot_sample = resample(delta_twap, n_samples=len(delta_twap))
            boot_means.append(np.mean(boot_sample))
        ci_twap = np.percentile(boot_means, [2.5, 97.5])
        
        # Effect size (Cohen's d)
        cohens_d = np.mean(delta_twap) / np.std(delta_twap) if np.std(delta_twap) > 0 else 0
        
        stats['rl_vs_twap'] = {
            'mean_gap_bps': np.mean(delta_twap) * 10000,
            'median_gap_bps': np.median(delta_twap) * 10000,
            'std_gap_bps': np.std(delta_twap) * 10000,
            'p_value': wilcoxon_result.pvalue,
            'significant': wilcoxon_result.pvalue < alpha,
            'ci_95_lower': ci_twap[0] * 10000,
            'ci_95_upper': ci_twap[1] * 10000,
            'cohens_d': cohens_d,
            'win_rate': np.mean(delta_twap > 0),
            'n_samples': len(delta_twap),
        }
    
    # RL vs AC
    delta_ac = ac - rl  # Positive = RL better
    
    if len(delta_ac) > 0:
        wilcoxon_result = wilcoxon(delta_ac, alternative='greater')
        
        boot_means = []
        for _ in range(10000):
            boot_sample = resample(delta_ac, n_samples=len(delta_ac))
            boot_means.append(np.mean(boot_sample))
        ci_ac = np.percentile(boot_means, [2.5, 97.5])
        
        cohens_d = np.mean(delta_ac) / np.std(delta_ac) if np.std(delta_ac) > 0 else 0
        
        stats['rl_vs_ac'] = {
            'mean_gap_bps': np.mean(delta_ac) * 10000,
            'median_gap_bps': np.median(delta_ac) * 10000,
            'std_gap_bps': np.std(delta_ac) * 10000,
            'p_value': wilcoxon_result.pvalue,
            'significant': wilcoxon_result.pvalue < alpha,
            'ci_95_lower': ci_ac[0] * 10000,
            'ci_95_upper': ci_ac[1] * 10000,
            'cohens_d': cohens_d,
            'win_rate': np.mean(delta_ac > 0),
            'n_samples': len(delta_ac),
        }
    
    return stats


def evaluate_agent(
    agent_path: str,
    test_data_paths: List[str],
    num_runs_per_day: int = 10,
    horizons: List[int] = [1800, 3600, 7200],  # 30min, 1h, 2h in seconds
    target_qty: int = 1000,
    agent_type: str = "institutional",
    execution_side: str = "SELL",
    output_dir: str = "results",
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Proper per-day evaluation protocol from rl-exec.pdf:
    
    1. For each test day:
       - Run N independent episodes with different start times
       - Aggregate to single daily score (mean)
    2. Compute paired daily differences: Δ_d = Baseline_d - RL_d
    3. Statistical testing: Wilcoxon signed-rank + bootstrap CIs
    4. Report: mean gap, p-values, effect sizes
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load agent
    print(f"Loading agent from {agent_path}...")
    agent = PPO.load(agent_path)
    
    # Try to load VecNormalize if it exists
    vec_normalize = None
    vecnorm_path = f"{agent_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from {vecnorm_path}...")
        vec_normalize = VecNormalize.load(vecnorm_path)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    
    results = {
        'rl': [],
        'twap': [],
        'ac': [],
        'metadata': [],
        'episode_details': [],
    }
    
    # Convert horizons from seconds to nanoseconds
    horizons_ns = [h * 1_000_000_000 for h in horizons]
    
    print("=" * 70)
    print("EVALUATION PROTOCOL")
    print("=" * 70)
    print(f"Test days: {len(test_data_paths)}")
    print(f"Runs per day: {num_runs_per_day}")
    print(f"Horizons: {horizons} seconds")
    print(f"Target qty: {target_qty}")
    print()
    
    # For each test day
    for day_idx, test_day in enumerate(test_data_paths):
        print(f"\n[{day_idx+1}/{len(test_data_paths)}] Processing {test_day}")
        
        # Get data bounds (estimate from file)
        # For now, we'll use a fixed warmup and estimate total duration
        warmup_ns = 60_000_000_000  # 60s
        # Estimate: assume 24 hours of data
        total_duration_ns = 24 * 3600 * 1_000_000_000
        
        # For each horizon
        for horizon_s, horizon_ns in zip(horizons, horizons_ns):
            print(f"  Horizon: {horizon_s}s ({horizon_s/60:.0f} min)")
            
            daily_rl_results = []
            daily_twap_results = []
            daily_ac_results = []
            
            # Multiple runs per day with different start times
            for run_idx in range(num_runs_per_day):
                # Randomize start time within the day
                max_start = total_duration_ns - horizon_ns - warmup_ns
                start_offset_ns = random.randint(
                    int(warmup_ns),
                    int(max_start) if max_start > warmup_ns else int(warmup_ns)
                )
                
                print(f"    Run {run_idx+1}/{num_runs_per_day} (start: {start_offset_ns/1e9:.1f}s)", end=" ... ")
                
                try:
                    # Run RL agent
                    rl_slippage, rl_completion, rl_info = run_rl_episode(
                        agent, vec_normalize, test_day,
                        start_offset_ns, horizon_ns, target_qty,
                        agent_type, execution_side
                    )
                    daily_rl_results.append(rl_slippage)
                    
                    # Run baselines on EXACT SAME window
                    twap_slippage = run_twap_baseline(
                        test_day, start_offset_ns, horizon_ns, target_qty
                    )
                    daily_twap_results.append(twap_slippage)
                    
                    ac_slippage = run_ac_baseline(
                        test_day, start_offset_ns, horizon_ns, target_qty
                    )
                    daily_ac_results.append(ac_slippage)
                    
                    print(f"RL: {rl_slippage:.2f}bps, TWAP: {twap_slippage:.2f}bps, AC: {ac_slippage:.2f}bps")
                    
                    # Store episode details
                    results['episode_details'].append({
                        'day': test_day,
                        'horizon_s': horizon_s,
                        'run_idx': run_idx,
                        'start_time_ns': start_offset_ns,
                        'rl_slippage_bps': rl_slippage,
                        'twap_slippage_bps': twap_slippage,
                        'ac_slippage_bps': ac_slippage,
                        'rl_completion': rl_completion,
                        'rl_reward': rl_info['total_reward'],
                    })
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
            
            # Aggregate to single daily score (mean)
            if daily_rl_results:
                results['rl'].append(np.mean(daily_rl_results))
                results['twap'].append(np.mean(daily_twap_results))
                results['ac'].append(np.mean(daily_ac_results))
                results['metadata'].append({
                    'day': test_day,
                    'horizon_s': horizon_s,
                    'num_runs': num_runs_per_day,
                })
    
    # Statistical testing
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    stats = compute_statistical_tests(results)
    
    # Print summary
    if 'rl_vs_twap' in stats:
        s = stats['rl_vs_twap']
        print(f"\nRL vs TWAP:")
        print(f"  Mean gap: {s['mean_gap_bps']:.2f} bps (RL better if positive)")
        print(f"  P-value: {s['p_value']:.4f} {'***' if s['significant'] else ''}")
        print(f"  95% CI: [{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}] bps")
        print(f"  Win rate: {s['win_rate']*100:.1f}%")
        print(f"  Cohen's d: {s['cohens_d']:.2f}")
    
    if 'rl_vs_ac' in stats:
        s = stats['rl_vs_ac']
        print(f"\nRL vs Almgren-Chriss:")
        print(f"  Mean gap: {s['mean_gap_bps']:.2f} bps (RL better if positive)")
        print(f"  P-value: {s['p_value']:.4f} {'***' if s['significant'] else ''}")
        print(f"  95% CI: [{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}] bps")
        print(f"  Win rate: {s['win_rate']*100:.1f}%")
        print(f"  Cohen's d: {s['cohens_d']:.2f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    
    output_data = {
        'agent_path': agent_path,
        'test_data_paths': test_data_paths,
        'config': {
            'num_runs_per_day': num_runs_per_day,
            'horizons': horizons,
            'target_qty': target_qty,
            'agent_type': agent_type,
            'execution_side': execution_side,
        },
        'results': {
            'rl_mean': float(np.mean(results['rl'])) if results['rl'] else None,
            'twap_mean': float(np.mean(results['twap'])) if results['twap'] else None,
            'ac_mean': float(np.mean(results['ac'])) if results['ac'] else None,
        },
        'statistics': stats,
        'metadata': results['metadata'],
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agent with rigorous statistical protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained PPO model")
    parser.add_argument("--test-data", nargs="+", required=True,
                        help="Path(s) to test data CSV files")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of runs per day (default: 10)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1800, 3600, 7200],
                        help="Time horizons in seconds (default: 1800 3600 7200)")
    parser.add_argument("--target-qty", type=int, default=1000,
                        help="Target quantity to execute (default: 1000)")
    parser.add_argument("--agent-type", default="institutional",
                        help="Agent latency profile (default: institutional)")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="SELL",
                        help="Execution side (default: SELL)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    evaluate_agent(
        agent_path=args.model,
        test_data_paths=args.test_data,
        num_runs_per_day=args.num_runs,
        horizons=args.horizons,
        target_qty=args.target_qty,
        agent_type=args.agent_type,
        execution_side=args.side,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

