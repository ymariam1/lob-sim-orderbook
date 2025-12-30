#!/usr/bin/env python3
"""
Test script to verify TRUE VWAP implementation.

This script demonstrates that VWAP now executes proportionally to market volume,
not uniformly like TWAP.

Usage:
    python test_vwap.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.py.baselines import TWAPExecutor, VWAPExecutor


def test_vwap_vs_twap():
    """
    Compare VWAP vs TWAP execution patterns.

    Expected behavior:
    - TWAP: Uniform slices (e.g., 100/60 ≈ 1.67 per slice)
    - VWAP: Variable slices proportional to market volume
    """
    # Use first test file if available
    test_files = list(Path("data/test").glob("*.csv"))
    if not test_files:
        print("ERROR: No test data found in data/test/")
        print("Run split_data.py first")
        return

    data_path = str(test_files[0])
    print(f"Using data: {data_path}")
    print("=" * 70)

    # Test parameters
    total_qty = 100
    total_time_ns = 60 * 60 * int(1e9)  # 1 hour
    num_slices = 10  # Use fewer slices for clearer visualization

    # Execute TWAP
    print("\n1. TWAP EXECUTION (Uniform time slices)")
    print("-" * 70)
    twap = TWAPExecutor(
        data_path=data_path,
        total_qty=total_qty,
        total_time_ns=total_time_ns,
        num_slices=num_slices,
        verbose=True,
    )
    twap_result = twap.execute()

    # Execute VWAP
    print("\n2. VWAP EXECUTION (Volume-weighted slices)")
    print("-" * 70)
    vwap = VWAPExecutor(
        data_path=data_path,
        total_qty=total_qty,
        total_time_ns=total_time_ns,
        num_slices=num_slices,
        verbose=True,
    )
    vwap_result = vwap.execute()

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'TWAP':<20} {'VWAP':<20}")
    print("-" * 70)
    print(f"{'VWAP Price':<30} {twap_result.vwap:<20.2f} {vwap_result.vwap:<20.2f}")
    print(f"{'Slippage (bps)':<30} {twap_result.slippage_bps:<20.2f} {vwap_result.slippage_bps:<20.2f}")
    print(f"{'Implementation Shortfall':<30} {twap_result.implementation_shortfall:<20.2f} {vwap_result.implementation_shortfall:<20.2f}")
    print(f"{'Num Fills':<30} {len(twap_result.fills):<20} {len(vwap_result.fills):<20}")
    print("=" * 70)

    print("\n✓ Test complete!")
    print("\nNOTE: VWAP should show different slice allocations than TWAP")
    print("      due to volume-weighted distribution.")


if __name__ == "__main__":
    test_vwap_vs_twap()
