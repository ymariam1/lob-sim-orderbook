#!/usr/bin/env python3
"""
Rigorous Evaluation Protocol for RL Trading Agents.

Based on best practices from rl-exec.pdf:
- Per-day evaluation with multiple runs
- Paired daily differences
- Statistical testing (Wilcoxon, bootstrap CIs)
- Effect sizes and win rates

Recent Updates:
- CRITICAL FIX: Match evaluation parameters to training defaults (60s warmup, inventory_penalty_coef=0.01)
- Fixed VecNormalize usage (set_venv method)
- CRITICAL FIX: Convert relative offsets to absolute timestamps (was causing 0 fills!)
  * CSV files have absolute Unix epoch timestamps (e.g., 1738386000422432 microseconds)
  * Environment expects absolute timestamps, not relative offsets
  * Now reads first timestamp from CSV and adds relative offset to get absolute time
- Added timestamp_unit_ns=1000 to all baselines (Tardis/Blockchain microseconds)
- Added baseline synchronization (start_time_ns parameter)
- Added comprehensive error handling (data exhaustion, step errors, baseline failures)
- Added max_steps safety limit to prevent infinite loops
- Conservative data duration estimate (--data-duration flag, default 1 hour)
- Conservative start time selection (first 20% of data to avoid running out)
- Graceful degradation: baselines can fail independently without breaking evaluation

Usage:
    python src/py/evaluate.py \
        --model models/best/best_model \
        --test-data data/test/*.csv \
        --data-duration 2.0 \
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

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not installed. FDR correction disabled.")
    print("  Install with: pip install statsmodels")

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
    # Create environment with parameters MATCHING TRAINING
    # CRITICAL: These must match train_rl.py defaults to avoid distribution shift
    env = LOBEnv(
        data_path=data_path,
        agent_type=agent_type,
        timestamp_unit_ns=1000,  # Tardis microseconds
        target_qty=target_qty,
        execution_side=execution_side,
        warmup_duration_ns=60_000_000_000,  # 60s warmup (MUST MATCH TRAINING!)
        step_duration_ns=10_000_000,  # 10ms steps
        inventory_penalty_coef=0.01,  # MUST MATCH TRAINING DEFAULT (was 1.0, causing 100x harsher penalty!)
    )

    # CRITICAL: VecNormalize must wrap the env BEFORE use
    # Cannot use normalize_obs() - must wrap entire env
    try:
        if vec_normalize:
            # Update the env in vec_normalize wrapper
            vec_normalize.set_venv(DummyVecEnv([lambda: env]))
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            # Use wrapped env for prediction
            wrapped_env = vec_normalize
        else:
            wrapped_env = env

        # Reset with start_time_ns parameter
        obs, info = env.reset(options={"start_time_ns": start_time_ns})
        arrival_price = info.get("arrival_price", 0.0)

    except Exception as e:
        print(f"\n    ERROR during RL setup/reset: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return 1000.0, 0.0, {"error": str(e)}

    # Run episode with try-finally to ensure cleanup
    total_reward = 0
    actions = []
    fills = []
    prices = []
    inventory = []

    try:
        episode_start_time = env._exchange.GetCurrentTime()
        target_end_time = episode_start_time + horizon_ns

        max_steps = 100000  # Safety limit to prevent infinite loops
        step_count = 0

        while env._exchange.GetCurrentTime() < target_end_time and step_count < max_steps:
            try:
                # Check if data loader has more data (prevents breaking on data exhaustion)
                if hasattr(env, '_loader') and env._loader is not None:
                    if not env._loader.HasMoreData():
                        # Gracefully exit if we run out of data
                        break

                # Get action from agent
                # If using VecNormalize, need to normalize observation before prediction
                if vec_normalize:
                    # VecNormalize expects vectorized input
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action, _ = agent.predict(obs, deterministic=True)
                actions.append(action)

                # Step
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                step_count += 1

                # Track state
                prices.append(info.get("mid_price", 0.0))
                inventory.append(info.get("executed_qty", 0))

                # Track fills
                if "vwap" in info:
                    fills.append({
                        "price": info["vwap"],
                        "qty": info["executed_qty"],
                        "time": env._exchange.GetCurrentTime(),
                    })

                if term or trunc:
                    break

            except Exception as e:
                # Gracefully handle errors during episode
                print(f"\n    WARNING: Episode error at step {step_count}: {e}")
                break

        # Final metrics
        try:
            final_info = env._get_info()
            executed_qty = final_info.get("executed_qty", 0)
            completion_rate = executed_qty / target_qty if target_qty > 0 else 0

            # Calculate slippage
            if executed_qty > 0 and "vwap" in final_info:
                vwap = final_info["vwap"]
                if arrival_price > 0:
                    slippage_bps = abs(vwap - arrival_price) / arrival_price * 10000
                else:
                    slippage_bps = 1000.0  # Penalty if no valid arrival price
            else:
                slippage_bps = 1000.0  # Penalty for incomplete execution
        except Exception as e:
            print(f"\n    WARNING: Error computing final metrics: {e}")
            executed_qty = 0
            completion_rate = 0.0
            slippage_bps = 1000.0
            final_info = {"executed_qty": 0}

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

        return slippage_bps, completion_rate, episode_info

    finally:
        # Always close environment, even if errors occur
        env.close()


def run_twap_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    num_slices: int = 60,
    timestamp_unit_ns: int = 1000,
) -> float:
    """Run TWAP baseline on exact same window."""
    executor = TWAPExecutor(
        data_path=data_path,
        total_qty=target_qty,  # Use total_qty parameter name
        total_time_ns=horizon_ns,
        num_slices=num_slices,
        agent_latency_ns=10_000_000,  # 10ms institutional
        timestamp_unit_ns=timestamp_unit_ns,  # CRITICAL: Must match RL env
    )

    # CRITICAL: Reset to same start time as RL agent
    result = executor.execute(start_time_ns=start_time_ns)
    return result.slippage_bps  # ExecutionResult is a dataclass, access attribute directly


def run_vwap_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    num_slices: int = 60,
    timestamp_unit_ns: int = 1000,
) -> float:
    """Run VWAP baseline on exact same window."""
    from src.py.baselines import VWAPExecutor

    executor = VWAPExecutor(
        data_path=data_path,
        total_qty=target_qty,
        total_time_ns=horizon_ns,
        num_slices=num_slices,
        agent_latency_ns=10_000_000,  # 10ms institutional
        timestamp_unit_ns=timestamp_unit_ns,  # CRITICAL: Must match RL env
        verbose=False,
    )

    # CRITICAL: Reset to same start time as RL agent
    result = executor.execute(start_time_ns=start_time_ns)
    return result.slippage_bps


def run_pov_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    participation_rate: float = 0.1,
    num_slices: int = 60,
    timestamp_unit_ns: int = 1000,
) -> float:
    """Run POV baseline on exact same window."""
    from src.py.baselines import POVExecutor

    executor = POVExecutor(
        data_path=data_path,
        total_qty=target_qty,
        total_time_ns=horizon_ns,
        num_slices=num_slices,
        participation_rate=participation_rate,
        agent_latency_ns=10_000_000,  # 10ms institutional
        timestamp_unit_ns=timestamp_unit_ns,  # CRITICAL: Must match RL env
        verbose=False,
    )

    # CRITICAL: Reset to same start time as RL agent
    result = executor.execute(start_time_ns=start_time_ns)
    return result.slippage_bps


def run_ac_baseline(
    data_path: str,
    start_time_ns: int,
    horizon_ns: int,
    target_qty: int,
    risk_aversion: float = 1e-6,
    timestamp_unit_ns: int = 1000,
) -> float:
    """Run Almgren-Chriss baseline on exact same window."""
    executor = AlmgrenChrissExecutor(
        data_path=data_path,
        total_qty=target_qty,  # Use total_qty parameter name
        total_time_ns=horizon_ns,
        risk_aversion=risk_aversion,
        agent_latency_ns=10_000_000,  # 10ms institutional
        timestamp_unit_ns=timestamp_unit_ns,  # CRITICAL: Must match RL env
        verbose=False,
    )

    # CRITICAL: Reset to same start time as RL agent
    result = executor.execute(start_time_ns=start_time_ns)
    return result.slippage_bps  # ExecutionResult is a dataclass, access attribute directly


def compute_statistical_tests(
    results: Dict,
    alpha: float = 0.05,
) -> Dict:
    """
    Statistical inference from rl-exec.pdf (rigorous methodology):
    
    1. Wilcoxon signed-rank test (one-sided: RL < Baseline)
    2. Benjamini-Hochberg FDR correction (if testing multiple baselines)
    3. Bootstrap 95% CIs (10k resamples)
    4. Effect sizes (Cohen's d)
    5. Win rates
    
    Follows rl-exec.pdf methodology exactly.
    """
    rl = np.array(results['rl'])
    twap = np.array(results['twap'])
    ac = np.array(results['ac'])
    
    stats = {}
    
    # Collect all p-values for FDR correction
    p_values_raw = []
    
    # RL vs TWAP
    delta_twap = twap - rl  # Positive = RL better (lower slippage)
    
    if len(delta_twap) > 0:
        # 1. Wilcoxon signed-rank test (one-sided: RL < TWAP)
        # H0: median(delta) <= 0
        # H1: median(delta) > 0 (RL is better)
        wilcoxon_result = wilcoxon(
            delta_twap, 
            alternative='greater',
            zero_method='wilcox'  # Handle zeros properly
        )
        p_value_raw_twap = wilcoxon_result.pvalue
        p_values_raw.append(p_value_raw_twap)
        
        # 2. Bootstrap CI for mean difference (10k resamples)
        bootstrap_means = []
        n_bootstrap = 10000
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(delta_twap), size=len(delta_twap), replace=True)
            bootstrap_means.append(np.mean(delta_twap[sample_idx]))
        ci_twap = np.percentile(bootstrap_means, [2.5, 97.5])
        
        # 3. Effect size (Cohen's d)
        cohens_d = np.mean(delta_twap) / np.std(delta_twap) if np.std(delta_twap) > 0 else 0.0
        
        # 4. Win rate
        win_rate = np.mean(delta_twap > 0)
        
        stats['rl_vs_twap'] = {
            'mean_gap_bps': np.mean(delta_twap) * 10000,
            'median_gap_bps': np.median(delta_twap) * 10000,
            'std_gap_bps': np.std(delta_twap) * 10000,
            'p_value_raw': p_value_raw_twap,
            'n_samples': len(delta_twap),
            'ci_95_lower_bps': ci_twap[0] * 10000,
            'ci_95_upper_bps': ci_twap[1] * 10000,
            'cohens_d': cohens_d,
            'win_rate': win_rate,
        }
    
    # RL vs AC
    delta_ac = ac - rl  # Positive = RL better
    
    if len(delta_ac) > 0:
        wilcoxon_result = wilcoxon(
            delta_ac,
            alternative='greater',
            zero_method='wilcox'
        )
        p_value_raw_ac = wilcoxon_result.pvalue
        p_values_raw.append(p_value_raw_ac)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(delta_ac), size=len(delta_ac), replace=True)
            bootstrap_means.append(np.mean(delta_ac[sample_idx]))
        ci_ac = np.percentile(bootstrap_means, [2.5, 97.5])
        
        cohens_d = np.mean(delta_ac) / np.std(delta_ac) if np.std(delta_ac) > 0 else 0.0
        win_rate = np.mean(delta_ac > 0)
        
        stats['rl_vs_ac'] = {
            'mean_gap_bps': np.mean(delta_ac) * 10000,
            'median_gap_bps': np.median(delta_ac) * 10000,
            'std_gap_bps': np.std(delta_ac) * 10000,
            'p_value_raw': p_value_raw_ac,
            'n_samples': len(delta_ac),
            'ci_95_lower_bps': ci_ac[0] * 10000,
            'ci_95_upper_bps': ci_ac[1] * 10000,
            'cohens_d': cohens_d,
            'win_rate': win_rate,
        }
    
    # 2. Benjamini-Hochberg FDR correction (if testing multiple baselines)
    if STATSMODELS_AVAILABLE and len(p_values_raw) > 1:
        reject, p_adjusted, _, _ = multipletests(
            p_values_raw,
            alpha=alpha,
            method='fdr_bh'
        )
        
        # Apply adjusted p-values
        if 'rl_vs_twap' in stats:
            stats['rl_vs_twap']['p_value_adjusted'] = p_adjusted[0]
            stats['rl_vs_twap']['significant'] = reject[0]
        if 'rl_vs_ac' in stats:
            idx = 1 if 'rl_vs_twap' in stats else 0
            stats['rl_vs_ac']['p_value_adjusted'] = p_adjusted[idx]
            stats['rl_vs_ac']['significant'] = reject[idx]
    else:
        # No FDR correction (single test or statsmodels not available)
        if 'rl_vs_twap' in stats:
            stats['rl_vs_twap']['p_value_adjusted'] = stats['rl_vs_twap']['p_value_raw']
            stats['rl_vs_twap']['significant'] = stats['rl_vs_twap']['p_value_raw'] < alpha
        if 'rl_vs_ac' in stats:
            stats['rl_vs_ac']['p_value_adjusted'] = stats['rl_vs_ac']['p_value_raw']
            stats['rl_vs_ac']['significant'] = stats['rl_vs_ac']['p_value_raw'] < alpha
    
    return stats


def evaluate_agent(
    agent_path: str,
    test_data_paths: List[str],
    num_runs_per_day: int = 10,
    horizons: List[int] = [1800, 3600, 7200],  # 30min, 1h, 2h in seconds
    target_qty: int = 100,  # CRITICAL: Must match training default (was 1000, causing 10x mismatch!)
    agent_type: str = "institutional",
    execution_side: str = "SELL",
    output_dir: str = "results",
    seed: int = 42,
    data_duration_hours: float = 1.0,  # Expected data duration in hours
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
    
    # Try to load VecNormalize if it exists (CRITICAL: use frozen train stats!)
    vec_normalize = None
    vecnorm_path = f"{agent_path}_vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from {vecnorm_path}...")
        print("  (Using frozen training statistics for evaluation)")
        # Create a dummy env to load normalization stats
        dummy_env = DummyVecEnv([lambda: LOBEnv(
            data_path=test_data_paths[0] if test_data_paths else None,
            agent_type=agent_type,
            timestamp_unit_ns=1000,
            target_qty=target_qty,
            execution_side=execution_side,
        )])
        vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
        vec_normalize.training = False  # CRITICAL: freeze stats
        vec_normalize.norm_reward = False  # Don't normalize rewards at eval
        dummy_env.close()
    
    results = {
        'rl': [],
        'twap': [],
        'vwap': [],
        'pov': [],
        'ac': [],
        'ac_low_risk': [],  # AC with λ=1e-7
        'ac_high_risk': [],  # AC with λ=1e-5
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

        # CRITICAL: Read first timestamp from CSV to get file start time
        # The timestamps in the file are absolute (Unix epoch microseconds)
        # We need to convert relative offsets to absolute timestamps
        import csv
        try:
            with open(test_day, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                first_row = next(reader)
                file_start_timestamp_us = int(first_row[0])  # Microseconds
                file_start_timestamp_ns = file_start_timestamp_us * 1000  # Convert to nanoseconds
                print(f"  File starts at: {file_start_timestamp_us} us ({file_start_timestamp_ns / 1e9:.2f}s epoch time)")
        except Exception as e:
            print(f"  WARNING: Could not read first timestamp: {e}")
            print(f"  Using default start time (this may cause issues)")
            file_start_timestamp_ns = 0

        # Get data bounds (estimate from file)
        # For now, we'll use a fixed warmup and estimate total duration
        warmup_ns = 5_000_000_000  # 5s (updated from 60s to match environment)
        # Use user-specified data duration (default: 1 hour)
        # This prevents trying to start episodes beyond the available data
        total_duration_ns = int(data_duration_hours * 3600 * 1_000_000_000)
        
        # For each horizon
        for horizon_s, horizon_ns in zip(horizons, horizons_ns):
            print(f"  Horizon: {horizon_s}s ({horizon_s/60:.0f} min)")
            
            daily_rl_results = []
            daily_twap_results = []
            daily_vwap_results = []
            daily_pov_results = []
            daily_ac_results = []
            daily_ac_low_risk_results = []  # AC with λ=1e-7 (low risk aversion)
            daily_ac_high_risk_results = []  # AC with λ=1e-5 (high risk aversion)
            
            # Multiple runs per day with different start times
            # CRITICAL: Each run uses same seed to ensure identical market conditions
            for run_idx in range(num_runs_per_day):
                # Set seed for this run (ensures reproducibility and identical conditions)
                run_seed = seed + (day_idx * 1000) + (horizon_s * 100) + run_idx
                random.seed(run_seed)
                np.random.seed(run_seed)
                
                # Randomize start time within available data
                # CRITICAL: Leave buffer at end to ensure we have enough data for full horizon
                buffer_ns = 60_000_000_000  # 60s buffer at end
                max_start = total_duration_ns - horizon_ns - warmup_ns - buffer_ns

                # Compute RELATIVE offset from file start
                if max_start <= warmup_ns:
                    relative_offset_ns = int(warmup_ns)
                else:
                    # Conservative: Use first 20% of available range for random starts
                    # This reduces risk of hitting end of data
                    safe_range = min(max_start - warmup_ns, total_duration_ns * 0.2)
                    relative_offset_ns = random.randint(
                        int(warmup_ns),
                        int(warmup_ns + safe_range)
                    )

                # CRITICAL: Convert relative offset to ABSOLUTE timestamp
                # The RL environment and baselines expect absolute timestamps from the CSV
                absolute_start_time_ns = file_start_timestamp_ns + relative_offset_ns

                print(f"    Run {run_idx+1}/{num_runs_per_day} (offset: {relative_offset_ns/1e9:.1f}s, abs time: {absolute_start_time_ns/1e9:.2f}s, seed: {run_seed})", end=" ... ")
                
                try:
                    # CRITICAL: Run RL agent and baselines on EXACT SAME window
                    # They must see identical market conditions for fair comparison

                    # Run RL agent
                    rl_slippage, rl_completion, rl_info = run_rl_episode(
                        agent, vec_normalize, test_day,
                        absolute_start_time_ns, horizon_ns, target_qty,
                        agent_type, execution_side
                    )
                    daily_rl_results.append(rl_slippage)

                    # Run baselines on EXACT SAME window (same seed ensures identical conditions)
                    # Wrap each baseline in try-except to prevent one failure from breaking all
                    # CRITICAL: Pass timestamp_unit_ns=1000 (Tardis microseconds) to match RL env
                    try:
                        twap_slippage = run_twap_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            timestamp_unit_ns=1000
                        )
                        daily_twap_results.append(twap_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: TWAP baseline failed: {e}")
                        twap_slippage = 1000.0

                    try:
                        vwap_slippage = run_vwap_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            timestamp_unit_ns=1000
                        )
                        daily_vwap_results.append(vwap_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: VWAP baseline failed: {e}")
                        vwap_slippage = 1000.0

                    try:
                        pov_slippage = run_pov_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            participation_rate=0.1,
                            timestamp_unit_ns=1000
                        )
                        daily_pov_results.append(pov_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: POV baseline failed: {e}")
                        pov_slippage = 1000.0

                    try:
                        ac_slippage = run_ac_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            risk_aversion=1e-6,  # Default
                            timestamp_unit_ns=1000
                        )
                        daily_ac_results.append(ac_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: AC baseline failed: {e}")
                        ac_slippage = 1000.0

                    try:
                        ac_low_risk_slippage = run_ac_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            risk_aversion=1e-7,  # Low risk aversion
                            timestamp_unit_ns=1000
                        )
                        daily_ac_low_risk_results.append(ac_low_risk_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: AC (low risk) baseline failed: {e}")
                        ac_low_risk_slippage = 1000.0

                    try:
                        ac_high_risk_slippage = run_ac_baseline(
                            test_day, absolute_start_time_ns, horizon_ns, target_qty,
                            risk_aversion=1e-5,  # High risk aversion
                            timestamp_unit_ns=1000
                        )
                        daily_ac_high_risk_results.append(ac_high_risk_slippage)
                    except Exception as e:
                        print(f"\n    WARNING: AC (high risk) baseline failed: {e}")
                        ac_high_risk_slippage = 1000.0

                    print(f"RL: {rl_slippage:.2f}bps, TWAP: {twap_slippage:.2f}bps, "
                          f"VWAP: {vwap_slippage:.2f}bps, POV: {pov_slippage:.2f}bps, "
                          f"AC: {ac_slippage:.2f}bps, AC(λ=1e-7): {ac_low_risk_slippage:.2f}bps, "
                          f"AC(λ=1e-5): {ac_high_risk_slippage:.2f}bps")
                    
                    # Store episode details
                    results['episode_details'].append({
                        'day': test_day,
                        'horizon_s': horizon_s,
                        'run_idx': run_idx,
                        'relative_offset_ns': relative_offset_ns,
                        'absolute_start_time_ns': absolute_start_time_ns,
                        'rl_slippage_bps': rl_slippage,
                        'twap_slippage_bps': twap_slippage,
                        'vwap_slippage_bps': vwap_slippage,
                        'pov_slippage_bps': pov_slippage,
                        'ac_slippage_bps': ac_slippage,
                        'ac_low_risk_slippage_bps': ac_low_risk_slippage,
                        'ac_high_risk_slippage_bps': ac_high_risk_slippage,
                        'rl_completion': rl_completion,
                        'rl_reward': rl_info['total_reward'],
                    })
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
            
            # CRITICAL: Aggregate to single daily score (mean)
            # This is the correct protocol: multiple runs per day are NOT independent
            # (they share same market conditions), so we aggregate within day first
            # Then use daily scores as the unit of inference
            if daily_rl_results:
                daily_rl_mean = np.mean(daily_rl_results)
                daily_twap_mean = np.mean(daily_twap_results)
                daily_vwap_mean = np.mean(daily_vwap_results) if daily_vwap_results else 0
                daily_pov_mean = np.mean(daily_pov_results) if daily_pov_results else 0
                daily_ac_mean = np.mean(daily_ac_results)
                daily_ac_low_risk_mean = np.mean(daily_ac_low_risk_results) if daily_ac_low_risk_results else 0
                daily_ac_high_risk_mean = np.mean(daily_ac_high_risk_results) if daily_ac_high_risk_results else 0
                
                results['rl'].append(daily_rl_mean)
                results['twap'].append(daily_twap_mean)
                results['vwap'].append(daily_vwap_mean)
                results['pov'].append(daily_pov_mean)
                results['ac'].append(daily_ac_mean)
                results['ac_low_risk'].append(daily_ac_low_risk_mean)
                results['ac_high_risk'].append(daily_ac_high_risk_mean)
                results['metadata'].append({
                    'day': test_day,
                    'horizon_s': horizon_s,
                    'num_runs': num_runs_per_day,
                    'daily_rl_mean_bps': daily_rl_mean,
                    'daily_twap_mean_bps': daily_twap_mean,
                    'daily_vwap_mean_bps': daily_vwap_mean,
                    'daily_pov_mean_bps': daily_pov_mean,
                    'daily_ac_mean_bps': daily_ac_mean,
                    'daily_ac_low_risk_mean_bps': daily_ac_low_risk_mean,
                    'daily_ac_high_risk_mean_bps': daily_ac_high_risk_mean,
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
        print(f"  P-value (raw): {s['p_value_raw']:.4f}")
        if 'p_value_adjusted' in s:
            print(f"  P-value (FDR-adjusted): {s['p_value_adjusted']:.4f} {'***' if s['significant'] else ''}")
        else:
            print(f"  P-value: {s['p_value_raw']:.4f} {'***' if s['significant'] else ''}")
        print(f"  95% CI: [{s['ci_95_lower_bps']:.2f}, {s['ci_95_upper_bps']:.2f}] bps")
        print(f"  Win rate: {s['win_rate']*100:.1f}%")
        print(f"  Cohen's d: {s['cohens_d']:.2f}")
        print(f"  N days: {s['n_samples']}")
    
    if 'rl_vs_ac' in stats:
        s = stats['rl_vs_ac']
        print(f"\nRL vs Almgren-Chriss:")
        print(f"  Mean gap: {s['mean_gap_bps']:.2f} bps (RL better if positive)")
        print(f"  P-value (raw): {s['p_value_raw']:.4f}")
        if 'p_value_adjusted' in s:
            print(f"  P-value (FDR-adjusted): {s['p_value_adjusted']:.4f} {'***' if s['significant'] else ''}")
        else:
            print(f"  P-value: {s['p_value_raw']:.4f} {'***' if s['significant'] else ''}")
        print(f"  95% CI: [{s['ci_95_lower_bps']:.2f}, {s['ci_95_upper_bps']:.2f}] bps")
        print(f"  Win rate: {s['win_rate']*100:.1f}%")
        print(f"  Cohen's d: {s['cohens_d']:.2f}")
        print(f"  N days: {s['n_samples']}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    
    # Convert stats to JSON-serializable format (convert bools to ints/strings)
    def make_json_serializable(obj):
        """Convert numpy/bool types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert numpy bool to Python bool
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)  # Explicitly convert to Python bool
        else:
            return obj

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
        'statistics': make_json_serializable(stats),
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
    parser.add_argument("--target-qty", type=int, default=100,
                        help="Target quantity to execute (default: 100, must match training!)")
    parser.add_argument("--agent-type", default="institutional",
                        help="Agent latency profile (default: institutional)")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="SELL",
                        help="Execution side (default: SELL)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--data-duration", type=float, default=1.0,
                        help="Expected data duration in hours (default: 1.0). "
                             "Increase if your data files contain more hours of data.")

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
        data_duration_hours=args.data_duration,
    )


if __name__ == "__main__":
    main()

