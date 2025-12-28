#!/usr/bin/env python3
"""
Execution Trace Visualization.

Visualizes agent execution strategies similar to Figure 2 in rl-exec.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_execution_trace(
    prices: np.ndarray,
    rl_fills: List[Dict],
    twap_fills: Optional[List[Dict]] = None,
    ac_fills: Optional[List[Dict]] = None,
    arrival_price: float = None,
    target_qty: int = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize execution strategies (like Figure 2 in rl-exec.pdf).
    
    Args:
        prices: Mid price evolution over time
        rl_fills: RL agent fills [{'time': int, 'price': float, 'qty': int}, ...]
        twap_fills: TWAP baseline fills (optional)
        ac_fills: Almgren-Chriss baseline fills (optional)
        arrival_price: Arrival price (reference line)
        target_qty: Target quantity to execute
        output_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Price evolution with fills
    time_axis = np.arange(len(prices))
    ax1.plot(time_axis, prices, 'k-', alpha=0.3, linewidth=1, label='Mid Price')
    
    if arrival_price is not None:
        ax1.axhline(arrival_price, color='gray', linestyle='--', linewidth=2, 
                   label='Arrival Price', alpha=0.7)
    
    # Plot RL fills
    if rl_fills:
        rl_times = [f['time'] for f in rl_fills]
        rl_prices = [f['price'] for f in rl_fills]
        rl_qtys = [f['qty'] for f in rl_fills]
        # Scale marker size by quantity
        rl_sizes = [q * 2 for q in rl_qtys] if rl_qtys else [20] * len(rl_times)
        ax1.scatter(rl_times, rl_prices, s=rl_sizes, c='blue', alpha=0.6, 
                   label='RL Fills', edgecolors='darkblue', linewidths=0.5)
    
    # Plot TWAP fills
    if twap_fills:
        twap_times = [f['time'] for f in twap_fills]
        twap_prices = [f['price'] for f in twap_fills]
        twap_qtys = [f['qty'] for f in twap_fills]
        twap_sizes = [q * 2 for q in twap_qtys] if twap_qtys else [20] * len(twap_times)
        ax1.scatter(twap_times, twap_prices, s=twap_sizes, c='red', alpha=0.6,
                   label='TWAP Fills', marker='s', edgecolors='darkred', linewidths=0.5)
    
    # Plot AC fills
    if ac_fills:
        ac_times = [f['time'] for f in ac_fills]
        ac_prices = [f['price'] for f in ac_fills]
        ac_qtys = [f['qty'] for f in ac_fills]
        ac_sizes = [q * 2 for q in ac_qtys] if ac_qtys else [20] * len(ac_times)
        ax1.scatter(ac_times, ac_prices, s=ac_sizes, c='green', alpha=0.6,
                   label='AC Fills', marker='^', edgecolors='darkgreen', linewidths=0.5)
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Execution Trace: Price Evolution with Fills', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Inventory over time
    if target_qty is not None:
        # Compute inventory curves
        rl_inventory = compute_inventory_curve(rl_fills, target_qty, len(prices))
        ax2.plot(time_axis, rl_inventory, 'b-', label='RL', linewidth=2, alpha=0.8)
        
        if twap_fills:
            twap_inventory = compute_inventory_curve(twap_fills, target_qty, len(prices))
            ax2.plot(time_axis, twap_inventory, 'r--', label='TWAP', linewidth=2, alpha=0.8)
        
        if ac_fills:
            ac_inventory = compute_inventory_curve(ac_fills, target_qty, len(prices))
            ax2.plot(time_axis, ac_inventory, 'g-.', label='Almgren-Chriss', linewidth=2, alpha=0.8)
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_ylabel('Remaining Inventory', fontsize=12)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_title('Inventory Over Time', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=-target_qty * 0.1)  # Small margin below zero
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    return fig


def compute_inventory_curve(
    fills: List[Dict],
    target_qty: int,
    num_steps: int,
) -> np.ndarray:
    """
    Compute remaining inventory over time from fills.
    
    Args:
        fills: List of fills [{'time': int, 'qty': int}, ...]
        target_qty: Target quantity to execute
        num_steps: Number of time steps
    
    Returns:
        Array of remaining inventory at each time step
    """
    inventory = np.full(num_steps, target_qty, dtype=float)
    
    cumulative_filled = 0
    for fill in sorted(fills, key=lambda x: x.get('time', 0)):
        time_idx = min(int(fill.get('time', 0)), num_steps - 1)
        cumulative_filled += fill.get('qty', 0)
        # Update from this time step onwards
        inventory[time_idx:] = target_qty - cumulative_filled
    
    return inventory


def plot_slippage_distribution(
    rl_slippages: np.ndarray,
    twap_slippages: Optional[np.ndarray] = None,
    ac_slippages: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of slippage values across episodes.
    
    Args:
        rl_slippages: RL slippage values (in bps)
        twap_slippages: TWAP slippage values (optional)
        ac_slippages: AC slippage values (optional)
        output_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(rl_slippages, bins=30, alpha=0.6, label='RL', color='blue', edgecolor='black')
    
    if twap_slippages is not None:
        ax.hist(twap_slippages, bins=30, alpha=0.6, label='TWAP', color='red', edgecolor='black')
    
    if ac_slippages is not None:
        ax.hist(ac_slippages, bins=30, alpha=0.6, label='Almgren-Chriss', color='green', edgecolor='black')
    
    ax.set_xlabel('Slippage (basis points)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Slippage Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Execution Trace Visualization")
    print("Use this module to visualize agent execution strategies")
    print("See plot_execution_trace() and plot_slippage_distribution()")

