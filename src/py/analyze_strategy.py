"""
Strategy Analysis and Interpretability Tools

Analyzes what the RL agent learns:
- Action distribution
- Order placement patterns
- Execution timing
- Comparison with baselines
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_action_distribution(
    actions: List[int],
    action_names: Optional[List[str]] = None
) -> Dict:
    """
    Analyze the distribution of actions taken by the agent.
    
    Args:
        actions: List of action indices taken during episode(s)
        action_names: Optional names for actions (for display)
    
    Returns:
        Dictionary with action counts, frequencies, and statistics
    """
    if action_names is None:
        action_names = [f"Action {i}" for i in range(14)]
        action_names.extend(["Cancel All"])
    
    action_counts = defaultdict(int)
    for action in actions:
        action_counts[action] += 1
    
    total_actions = len(actions)
    action_freqs = {k: v / total_actions for k, v in action_counts.items()}
    
    # Categorize actions
    market_orders = sum(action_counts.get(i, 0) for i in [0, 1])  # Market buy/sell
    limit_orders = sum(action_counts.get(i, 0) for i in range(2, 13))  # Limit orders
    cancels = action_counts.get(13, 0)  # Cancel all
    
    return {
        "action_counts": dict(action_counts),
        "action_frequencies": action_freqs,
        "total_actions": total_actions,
        "market_orders": market_orders,
        "limit_orders": limit_orders,
        "cancels": cancels,
        "market_order_pct": market_orders / total_actions if total_actions > 0 else 0,
        "limit_order_pct": limit_orders / total_actions if total_actions > 0 else 0,
        "cancel_pct": cancels / total_actions if total_actions > 0 else 0,
    }


def plot_action_distribution(
    action_stats: Dict,
    output_path: Optional[str] = None,
    title: str = "Agent Action Distribution"
):
    """
    Plot action distribution as bar chart.
    
    Args:
        action_stats: Output from analyze_action_distribution
        output_path: Path to save figure (optional)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Action frequency bar chart
    actions = sorted(action_stats["action_counts"].keys())
    counts = [action_stats["action_counts"].get(a, 0) for a in actions]
    freqs = [action_stats["action_frequencies"].get(a, 0) for a in actions]
    
    ax1.bar(actions, counts, alpha=0.7, color='steelblue')
    ax1.set_xlabel("Action Index")
    ax1.set_ylabel("Count")
    ax1.set_title("Action Counts")
    ax1.grid(True, alpha=0.3)
    
    # Right: Category pie chart
    categories = ["Market Orders", "Limit Orders", "Cancels"]
    sizes = [
        action_stats["market_orders"],
        action_stats["limit_orders"],
        action_stats["cancels"]
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax2.pie(sizes, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title("Action Categories")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Action distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_execution_timing(
    fills: List[Dict],
    total_time_ns: int
) -> Dict:
    """
    Analyze when executions occur during the episode.
    
    Args:
        fills: List of fill dictionaries with 'time' and 'qty' keys
        total_time_ns: Total episode duration in nanoseconds
    
    Returns:
        Dictionary with timing statistics
    """
    if not fills:
        return {
            "num_fills": 0,
            "avg_fill_interval_s": 0,
            "execution_rate": 0,
            "early_execution_pct": 0,
            "late_execution_pct": 0,
        }
    
    fill_times = sorted([f.get('time', 0) for f in fills])
    fill_qtys = [f.get('qty', 0) for f in fills]
    
    # Calculate intervals between fills
    intervals = []
    for i in range(1, len(fill_times)):
        intervals.append((fill_times[i] - fill_times[i-1]) / 1e9)  # Convert to seconds
    
    # Execution rate (quantity per second)
    total_qty = sum(fill_qtys)
    total_time_s = total_time_ns / 1e9
    execution_rate = total_qty / total_time_s if total_time_s > 0 else 0
    
    # Early vs late execution
    mid_time = total_time_ns / 2
    early_fills = sum(1 for t in fill_times if t < mid_time)
    late_fills = len(fill_times) - early_fills
    
    return {
        "num_fills": len(fills),
        "total_qty": total_qty,
        "avg_fill_interval_s": np.mean(intervals) if intervals else 0,
        "std_fill_interval_s": np.std(intervals) if intervals else 0,
        "execution_rate": execution_rate,
        "early_execution_pct": early_fills / len(fill_times) if fill_times else 0,
        "late_execution_pct": late_fills / len(fill_times) if fill_times else 0,
        "first_fill_time_s": fill_times[0] / 1e9 if fill_times else 0,
        "last_fill_time_s": fill_times[-1] / 1e9 if fill_times else 0,
    }


def compare_strategies(
    rl_stats: Dict,
    twap_stats: Dict,
    ac_stats: Dict,
    output_path: Optional[str] = None
):
    """
    Compare RL agent strategy with baselines.
    
    Args:
        rl_stats: Statistics from RL agent
        twap_stats: Statistics from TWAP baseline
        ac_stats: Statistics from Almgren-Chriss baseline
        output_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Action distribution comparison
    ax = axes[0, 0]
    strategies = ["RL Agent", "TWAP", "AC"]
    market_pcts = [
        rl_stats.get("market_order_pct", 0),
        1.0,  # TWAP is all market orders
        ac_stats.get("market_order_pct", 0.5),  # AC is mixed
    ]
    limit_pcts = [
        rl_stats.get("limit_order_pct", 0),
        0.0,  # TWAP has no limit orders
        ac_stats.get("limit_order_pct", 0.5),
    ]
    
    x = np.arange(len(strategies))
    width = 0.35
    ax.bar(x - width/2, market_pcts, width, label='Market Orders', alpha=0.7)
    ax.bar(x + width/2, limit_pcts, width, label='Limit Orders', alpha=0.7)
    ax.set_ylabel("Percentage")
    ax.set_title("Order Type Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Execution timing
    ax = axes[0, 1]
    rl_timing = rl_stats.get("timing", {})
    twap_timing = twap_stats.get("timing", {})
    ac_timing = ac_stats.get("timing", {})
    
    early_pcts = [
        rl_timing.get("early_execution_pct", 0),
        twap_timing.get("early_execution_pct", 0.5),
        ac_timing.get("early_execution_pct", 0.5),
    ]
    
    ax.bar(strategies, early_pcts, alpha=0.7, color='steelblue')
    ax.set_ylabel("Early Execution %")
    ax.set_title("Execution Timing (Early vs Late)")
    ax.grid(True, alpha=0.3)
    
    # 3. Fill frequency
    ax = axes[1, 0]
    fill_counts = [
        rl_stats.get("num_fills", 0),
        twap_stats.get("num_fills", 0),
        ac_stats.get("num_fills", 0),
    ]
    
    ax.bar(strategies, fill_counts, alpha=0.7, color='green')
    ax.set_ylabel("Number of Fills")
    ax.set_title("Fill Frequency")
    ax.grid(True, alpha=0.3)
    
    # 4. Execution rate
    ax = axes[1, 1]
    exec_rates = [
        rl_timing.get("execution_rate", 0),
        twap_timing.get("execution_rate", 0),
        ac_timing.get("execution_rate", 0),
    ]
    
    ax.bar(strategies, exec_rates, alpha=0.7, color='orange')
    ax.set_ylabel("Execution Rate (qty/s)")
    ax.set_title("Execution Rate")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Strategy Comparison: RL vs Baselines", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Strategy comparison plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_strategy_report(
    actions: List[int],
    fills: List[Dict],
    total_time_ns: int,
    output_dir: str = "results"
):
    """
    Generate a comprehensive strategy analysis report.
    
    Args:
        actions: List of actions taken
        fills: List of fill dictionaries
        total_time_ns: Total episode duration
        output_dir: Directory to save reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze actions
    action_stats = analyze_action_distribution(actions)
    
    # Analyze timing
    timing_stats = analyze_execution_timing(fills, total_time_ns)
    
    # Combine stats
    combined_stats = {**action_stats, "timing": timing_stats, "num_fills": len(fills)}
    
    # Generate plots
    plot_action_distribution(
        action_stats,
        output_path=str(output_dir / "action_distribution.png"),
        title="RL Agent Action Distribution"
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("STRATEGY ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nAction Distribution:")
    print(f"  Total Actions: {action_stats['total_actions']}")
    print(f"  Market Orders: {action_stats['market_orders']} ({action_stats['market_order_pct']:.1%})")
    print(f"  Limit Orders: {action_stats['limit_orders']} ({action_stats['limit_order_pct']:.1%})")
    print(f"  Cancels: {action_stats['cancels']} ({action_stats['cancel_pct']:.1%})")
    
    print(f"\nExecution Timing:")
    print(f"  Number of Fills: {timing_stats['num_fills']}")
    print(f"  Total Quantity: {timing_stats['total_qty']}")
    print(f"  Execution Rate: {timing_stats['execution_rate']:.2f} qty/s")
    print(f"  Avg Fill Interval: {timing_stats['avg_fill_interval_s']:.2f} s")
    print(f"  Early Execution: {timing_stats['early_execution_pct']:.1%}")
    print(f"  Late Execution: {timing_stats['late_execution_pct']:.1%}")
    
    print(f"\nInterpretation:")
    if action_stats['market_order_pct'] > 0.5:
        print("  → Agent prefers aggressive execution (market orders)")
    elif action_stats['limit_order_pct'] > 0.5:
        print("  → Agent prefers passive execution (limit orders)")
    else:
        print("  → Agent uses balanced strategy (mixed orders)")
    
    if timing_stats['early_execution_pct'] > 0.6:
        print("  → Agent executes early in episode (front-loaded)")
    elif timing_stats['late_execution_pct'] > 0.6:
        print("  → Agent executes late in episode (back-loaded)")
    else:
        print("  → Agent executes uniformly over time")
    
    print("=" * 70)
    
    return combined_stats

