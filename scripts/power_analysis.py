#!/usr/bin/env python3
"""
Statistical Power Analysis for RL Trading Experiments

Computes required sample size to detect a given effect size with specified power.
This is critical for ensuring experiments have sufficient statistical power.

Usage:
    python scripts/power_analysis.py --effect-size 2.0 --std 10.0 --power 0.8
"""

import argparse
import numpy as np
from scipy import stats


def compute_required_sample_size(
    effect_size: float,
    std: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = "one-sided"
) -> int:
    """
    Compute required sample size for paired t-test.
    
    Args:
        effect_size: Minimum detectable difference (in bps or same units as std)
        std: Standard deviation of the difference (in same units as effect_size)
        power: Desired statistical power (default 0.8 = 80%)
        alpha: Significance level (default 0.05 = 5%)
        test_type: "one-sided" or "two-sided"
    
    Returns:
        Required number of samples (days/episodes)
    """
    # Standardized effect size (Cohen's d)
    cohens_d = effect_size / std
    
    # Adjust alpha for one-sided vs two-sided
    if test_type == "one-sided":
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula for paired t-test
    # n = (z_alpha + z_beta)^2 / (d^2) * (1 + rho) / (1 - rho)
    # For paired test, we assume rho = 0.5 (moderate correlation)
    # Simplified: n = 2 * (z_alpha + z_beta)^2 / d^2
    n = 2 * ((z_alpha + z_beta) ** 2) / (cohens_d ** 2)
    
    return int(np.ceil(n))


def analyze_power_requirements():
    """Analyze power requirements for typical trading experiments."""
    parser = argparse.ArgumentParser(
        description="Statistical Power Analysis for RL Trading Experiments"
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        default=2.0,
        help="Minimum detectable difference in basis points (default: 2.0 bps)"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=10.0,
        help="Standard deviation of slippage differences in basis points (default: 10.0 bps)"
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.8,
        help="Desired statistical power (default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--test-type",
        choices=["one-sided", "two-sided"],
        default="one-sided",
        help="Type of statistical test (default: one-sided)"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Run multiple scenarios with different effect sizes"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Effect Size: {args.effect_size:.2f} bps")
    print(f"  Standard Deviation: {args.std:.2f} bps")
    print(f"  Desired Power: {args.power:.1%}")
    print(f"  Significance Level (α): {args.alpha:.1%}")
    print(f"  Test Type: {args.test_type}")
    
    # Compute required sample size
    n_required = compute_required_sample_size(
        args.effect_size,
        args.std,
        args.power,
        args.alpha,
        args.test_type
    )
    
    print(f"\n{'='*70}")
    print(f"REQUIRED SAMPLE SIZE: {n_required} days/episodes")
    print(f"{'='*70}")
    
    # Interpretation
    cohens_d = args.effect_size / args.std
    print(f"\nInterpretation:")
    print(f"  To detect a {args.effect_size:.2f} bps difference with {args.power:.0%} power,")
    print(f"  you need at least {n_required} independent samples (days).")
    print(f"  Cohen's d (effect size): {cohens_d:.3f}")
    
    if cohens_d < 0.2:
        effect_interpretation = "negligible"
    elif cohens_d < 0.5:
        effect_interpretation = "small"
    elif cohens_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"  Effect size interpretation: {effect_interpretation}")
    
    # Run scenarios if requested
    if args.scenarios:
        print(f"\n{'='*70}")
        print("SCENARIO ANALYSIS")
        print(f"{'='*70}")
        print(f"\n{'Effect Size (bps)':<20} {'Required N':<15} {'Cohen's d':<15}")
        print("-" * 50)
        
        for effect in [1.0, 2.0, 3.0, 5.0, 10.0]:
            n = compute_required_sample_size(
                effect, args.std, args.power, args.alpha, args.test_type
            )
            d = effect / args.std
            print(f"{effect:<20.1f} {n:<15} {d:<15.3f}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"\n1. For your evaluation protocol:")
    print(f"   - Use at least {n_required} test days")
    print(f"   - Run multiple episodes per day (e.g., 10-20)")
    print(f"   - Aggregate to one score per day to avoid pseudo-replication")
    
    print(f"\n2. If you have fewer than {n_required} days:")
    print(f"   - Consider increasing effect size threshold")
    print(f"   - Or accept lower statistical power")
    print(f"   - Or use more conservative significance level")
    
    print(f"\n3. For publication (ICAIF):")
    print(f"   - Report power analysis in methods section")
    print(f"   - Document actual sample size achieved")
    print(f"   - Discuss limitations if underpowered")
    
    return n_required


if __name__ == "__main__":
    analyze_power_requirements()

