#!/usr/bin/env python3
"""
Evaluate multiple RL models (different seeds) against baselines with statistical testing.

This script:
1. Evaluates each RL model on test episodes
2. Evaluates TWAP and POV baselines on the SAME episodes (synchronized start times)
3. Computes statistics across seeds: mean ± std
4. Performs statistical significance tests on Implementation Shortfall and Slippage

Usage:
    python evaluate_multi_seed.py --save-dir models_multi_seed --test-data data/test
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from scipy import stats
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode_id: int
    strategy: str
    seed: int = 0  # For RL runs

    # Episode metrics
    arrival_price: float = 0.0
    vwap: float = 0.0
    executed_qty: int = 0
    target_qty: int = 100
    completion_rate: float = 0.0

    # Performance metrics (signed: positive = bad, negative = good)
    implementation_shortfall: float = 0.0  # $ value
    slippage_bps: float = 0.0  # Basis points

    # Episode metadata
    start_time_ns: int = 0
    end_time_ns: int = 0


@dataclass
class AggregatedResults:
    """Aggregated results across episodes."""
    strategy: str
    n_seeds: int = 1
    n_episodes: int = 0

    # Mean ± std
    mean_is: float = 0.0
    std_is: float = 0.0
    mean_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0
    mean_completion: float = 0.0
    std_completion: float = 0.0

    # All raw results for statistical testing
    all_is: List[float] = field(default_factory=list)
    all_slippage: List[float] = field(default_factory=list)
    all_completion: List[float] = field(default_factory=list)


def load_rl_models(save_dir: str) -> List[Path]:
    """
    Find all BEST trained models from multi-seed training.

    Returns:
        List of paths to best_model.zip files (one per seed)
    """
    save_path = Path(save_dir)
    models = []

    for seed_dir in sorted(save_path.glob("seed_*")):
        # Use BEST model (based on eval performance), not latest checkpoint
        model_path = seed_dir / "best" / "best_model.zip"
        if model_path.exists():
            models.append(model_path)
        else:
            print(f"WARNING: Best model not found for {seed_dir.name}")

    return models


def evaluate_rl_model(model_path: Path, test_data: str, n_episodes: int = 20) -> List[EpisodeResult]:
    """
    Evaluate an RL model on test episodes.

    Returns list of episode results with start_time_ns for each episode.
    """
    # TODO: This would call train_rl.py --eval-only and parse results
    # For now, returning placeholder
    print(f"  Evaluating {model_path.name}...")

    results = []
    # Parse seed from path (e.g., "seed_0" -> 0)
    seed = int(model_path.parent.name.split("_")[1])

    # In production, you'd run actual evaluation and collect these metrics
    # For now, placeholder:
    for i in range(n_episodes):
        results.append(EpisodeResult(
            episode_id=i,
            strategy="RL",
            seed=seed,
            # These would come from actual evaluation
        ))

    return results


def evaluate_baseline(
    strategy: str,
    test_data: str,
    episode_start_times: List[int],
    **kwargs
) -> List[EpisodeResult]:
    """
    Evaluate a baseline strategy on specific episodes (synchronized start times).

    Args:
        strategy: "TWAP" or "POV"
        test_data: Path to test data
        episode_start_times: List of start_time_ns to synchronize with RL episodes

    Returns:
        List of episode results
    """
    from src.py.baselines import TWAPExecutor, POVExecutor

    results = []

    for i, start_time_ns in enumerate(episode_start_times):
        # Create executor
        if strategy == "TWAP":
            executor = TWAPExecutor(data_path=test_data, **kwargs)
        elif strategy == "POV":
            executor = POVExecutor(data_path=test_data, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # CRITICAL: Pass start_time_ns to synchronize with RL
        result = executor.execute(start_time_ns=start_time_ns)

        results.append(EpisodeResult(
            episode_id=i,
            strategy=strategy,
            arrival_price=result.arrival_price,
            vwap=result.vwap,
            executed_qty=len(result.fills) * 10,  # Approximate
            target_qty=executor.total_qty,
            completion_rate=len(result.fills) * 10 / executor.total_qty,
            implementation_shortfall=result.implementation_shortfall,
            slippage_bps=result.slippage_bps,
            start_time_ns=result.start_time_ns,
            end_time_ns=result.end_time_ns,
        ))

    return results


def aggregate_results(results: List[EpisodeResult]) -> AggregatedResults:
    """Compute mean ± std across episodes/seeds."""
    strategy = results[0].strategy if results else "Unknown"
    n_seeds = len(set(r.seed for r in results))

    all_is = [r.implementation_shortfall for r in results]
    all_slippage = [r.slippage_bps for r in results]
    all_completion = [r.completion_rate for r in results]

    return AggregatedResults(
        strategy=strategy,
        n_seeds=n_seeds,
        n_episodes=len(results),
        mean_is=np.mean(all_is),
        std_is=np.std(all_is, ddof=1) if len(all_is) > 1 else 0.0,
        mean_slippage_bps=np.mean(all_slippage),
        std_slippage_bps=np.std(all_slippage, ddof=1) if len(all_slippage) > 1 else 0.0,
        mean_completion=np.mean(all_completion),
        std_completion=np.std(all_completion, ddof=1) if len(all_completion) > 1 else 0.0,
        all_is=all_is,
        all_slippage=all_slippage,
        all_completion=all_completion,
    )


def statistical_test(
    rl_results: AggregatedResults,
    baseline_results: AggregatedResults,
    metric: str = "is"
) -> Dict:
    """
    Perform statistical significance test (Welch's t-test).

    Args:
        rl_results: RL aggregated results
        baseline_results: Baseline aggregated results
        metric: "is" (implementation shortfall) or "slippage"

    Returns:
        Dict with test results
    """
    if metric == "is":
        rl_data = rl_results.all_is
        baseline_data = baseline_results.all_is
        metric_name = "Implementation Shortfall ($)"
    elif metric == "slippage":
        rl_data = rl_results.all_slippage
        baseline_data = baseline_results.all_slippage
        metric_name = "Slippage (bps)"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(rl_data, baseline_data, equal_var=False)

    # Effect size (Cohen's d)
    mean_diff = np.mean(rl_data) - np.mean(baseline_data)
    pooled_std = np.sqrt((np.std(rl_data, ddof=1)**2 + np.std(baseline_data, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Determine significance
    if p_value < 0.001:
        sig_str = "***"
    elif p_value < 0.01:
        sig_str = "**"
    elif p_value < 0.05:
        sig_str = "*"
    else:
        sig_str = "n.s."

    # Determine winner (lower is better for both IS and slippage)
    rl_mean = np.mean(rl_data)
    baseline_mean = np.mean(baseline_data)

    if rl_mean < baseline_mean:
        winner = "RL"
        improvement_pct = (baseline_mean - rl_mean) / abs(baseline_mean) * 100 if baseline_mean != 0 else 0
    else:
        winner = baseline_results.strategy
        improvement_pct = (rl_mean - baseline_mean) / abs(rl_mean) * 100 if rl_mean != 0 else 0

    return {
        'metric': metric_name,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significance': sig_str,
        'winner': winner,
        'improvement_pct': improvement_pct,
        'rl_mean': rl_mean,
        'baseline_mean': baseline_mean,
    }


def print_results_table(all_results: Dict[str, AggregatedResults]):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS (Mean ± Std)")
    print("=" * 90)
    print(f"{'Strategy':<15} {'N':<8} {'IS ($)':<20} {'Slippage (bps)':<20} {'Completion':<15}")
    print("-" * 90)

    for strategy, results in all_results.items():
        print(f"{strategy:<15} "
              f"{results.n_episodes:<8} "
              f"{results.mean_is:>8.2f} ± {results.std_is:<8.2f} "
              f"{results.mean_slippage_bps:>8.2f} ± {results.std_slippage_bps:<8.2f} "
              f"{results.mean_completion:>6.1%} ± {results.std_completion:<6.1%}")

    print("=" * 90)


def print_statistical_tests(tests: List[Dict]):
    """Print statistical test results."""
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 90)
    print("(*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant)")
    print("-" * 90)

    for test in tests:
        print(f"\n{test['comparison']}:")
        print(f"  {test['metric']}:")
        print(f"    RL:       {test['rl_mean']:>10.2f}")
        print(f"    Baseline: {test['baseline_mean']:>10.2f}")
        print(f"    p-value:  {test['p_value']:<10.4f} {test['significance']}")
        print(f"    Cohen's d: {test['cohens_d']:<10.2f}")
        print(f"    Winner:   {test['winner']} (improvement: {test['improvement_pct']:.1f}%)")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-seed RL models against baselines with statistical tests"
    )

    parser.add_argument(
        "--save-dir",
        default="models_multi_seed",
        help="Directory containing seed_* subdirectories"
    )
    parser.add_argument(
        "--test-data",
        default="data/test",
        help="Test data directory"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Number of test episodes per seed"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("MULTI-SEED EVALUATION WITH STATISTICAL TESTING")
    print("=" * 90)
    print(f"Models directory: {args.save_dir}")
    print(f"Test data: {args.test_data}")
    print(f"Episodes per seed: {args.n_episodes}")
    print("=" * 90)

    # Load RL models
    models = load_rl_models(args.save_dir)
    print(f"\nFound {len(models)} trained models")

    if not models:
        print("ERROR: No models found. Train models first with train_multi_seed.py")
        sys.exit(1)

    # Evaluate all RL models
    print("\n" + "=" * 90)
    print("EVALUATING RL MODELS")
    print("=" * 90)

    all_rl_results = []
    episode_start_times = []  # Collect start times for baseline synchronization

    for model in models:
        results = evaluate_rl_model(model, args.test_data, args.n_episodes)
        all_rl_results.extend(results)

        # Collect start times from first seed for baseline synchronization
        if not episode_start_times:
            episode_start_times = [r.start_time_ns for r in results]

    # Aggregate RL results
    rl_agg = aggregate_results(all_rl_results)

    # Evaluate baselines (SYNCHRONIZED with RL start times)
    print("\n" + "=" * 90)
    print("EVALUATING BASELINES (SYNCHRONIZED)")
    print("=" * 90)

    # NOTE: This assumes episode_start_times are available from RL evaluation
    # In practice, you'd need to save these or regenerate them

    # For now, showing the structure - you'd need to implement actual baseline evaluation
    # with proper start_time_ns synchronization

    print("\nNOTE: Complete baseline evaluation requires actual episode start times from RL")
    print("      For fair comparison, baselines must execute at the SAME start times as RL")

    # Placeholder results
    all_results = {
        'RL': rl_agg,
        # 'TWAP': twap_agg,  # Would evaluate with episode_start_times
        # 'POV': pov_agg,    # Would evaluate with episode_start_times
    }

    # Print results
    print_results_table(all_results)

    # Statistical tests
    # tests = []
    # tests.append({
    #     'comparison': 'RL vs TWAP',
    #     **statistical_test(rl_agg, twap_agg, 'is')
    # })
    # print_statistical_tests(tests)

    print("\n✓ Evaluation complete")
    print(f"\nNext steps:")
    print("1. Implement full RL evaluation in this script")
    print("2. Run synchronized baseline evaluations")
    print("3. Report results with mean ± std and statistical tests")


if __name__ == "__main__":
    main()
