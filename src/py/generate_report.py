#!/usr/bin/env python3
"""
Generate publication-ready results tables and reports.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_results_table(
    results_file: str,
    output_path: str = None,
    format: str = "markdown",
) -> str:
    """
    Generate publication-ready tables from evaluation results.
    
    Args:
        results_file: Path to eval_results JSON file
        output_path: Path to save report (optional)
        format: "markdown" or "latex"
    
    Returns:
        Report string
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    stats = data.get('statistics', {})
    config = data.get('config', {})
    metadata = data.get('metadata', [])
    
    # Group by horizon
    by_horizon = {}
    for meta in metadata:
        horizon = meta.get('horizon_s', 0)
        if horizon not in by_horizon:
            by_horizon[horizon] = {
                'rl': [],
                'twap': [],
                'ac': [],
            }
    
    # Collect results by horizon
    rl_results = data.get('results', {}).get('rl', [])
    twap_results = data.get('results', {}).get('twap', [])
    ac_results = data.get('results', {}).get('ac', [])
    
    for i, meta in enumerate(metadata):
        horizon = meta.get('horizon_s', 0)
        if i < len(rl_results):
            by_horizon[horizon]['rl'].append(rl_results[i])
        if i < len(twap_results):
            by_horizon[horizon]['twap'].append(twap_results[i])
        if i < len(ac_results):
            by_horizon[horizon]['ac'].append(ac_results[i])
    
    if format == "markdown":
        return generate_markdown_table(stats, by_horizon, config, results_file)
    else:
        return generate_latex_table(stats, by_horizon, config, results_file)


def generate_markdown_table(
    stats: Dict,
    by_horizon: Dict,
    config: Dict,
    results_file: str,
) -> str:
    """Generate markdown table."""
    import numpy as np
    
    markdown = f"""# Execution Results

Generated from: `{Path(results_file).name}`  
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

- **Agent Type**: {config.get('agent_type', 'N/A')}
- **Target Quantity**: {config.get('target_qty', 'N/A')}
- **Execution Side**: {config.get('execution_side', 'N/A')}
- **Runs per Day**: {config.get('num_runs_per_day', 'N/A')}

## Performance by Horizon

| Horizon | RL Mean (bps) | TWAP Mean (bps) | AC Mean (bps) | RL Advantage | p-value | 95% CI | Win Rate |
|---------|---------------|-----------------|---------------|--------------|---------|--------|----------|
"""
    
    for horizon in sorted(by_horizon.keys()):
        h_data = by_horizon[horizon]
        
        rl_mean = np.mean(h_data['rl']) if h_data['rl'] else 0
        rl_std = np.std(h_data['rl']) if h_data['rl'] else 0
        twap_mean = np.mean(h_data['twap']) if h_data['twap'] else 0
        twap_std = np.std(h_data['twap']) if h_data['twap'] else 0
        ac_mean = np.mean(h_data['ac']) if h_data['ac'] else 0
        ac_std = np.std(h_data['ac']) if h_data['ac'] else 0
        
        # Get stats for this horizon (if available)
        advantage_twap = twap_mean - rl_mean
        advantage_ac = ac_mean - rl_mean
        
        # Find matching stats (may need to filter by horizon)
        p_value_twap = stats.get('rl_vs_twap', {}).get('p_value_adjusted', 
                                                       stats.get('rl_vs_twap', {}).get('p_value_raw', 0))
        ci_lower = stats.get('rl_vs_twap', {}).get('ci_95_lower_bps', 0)
        ci_upper = stats.get('rl_vs_twap', {}).get('ci_95_upper_bps', 0)
        win_rate = stats.get('rl_vs_twap', {}).get('win_rate', 0) * 100
        
        markdown += f"| {horizon}s ({horizon/60:.0f} min) | "
        markdown += f"{rl_mean:.2f} ± {rl_std:.2f} | "
        markdown += f"{twap_mean:.2f} ± {twap_std:.2f} | "
        markdown += f"{ac_mean:.2f} ± {ac_std:.2f} | "
        markdown += f"+{advantage_twap:.2f} | "
        markdown += f"{p_value_twap:.4f} | "
        markdown += f"[{ci_lower:.2f}, {ci_upper:.2f}] | "
        markdown += f"{win_rate:.1f}% |\n"
    
    markdown += f"""

## Statistical Summary

### RL vs TWAP

"""
    
    if 'rl_vs_twap' in stats:
        s = stats['rl_vs_twap']
        markdown += f"""
- **Mean gap**: {s['mean_gap_bps']:.2f} bps (RL better if positive)
- **Median gap**: {s['median_gap_bps']:.2f} bps
- **P-value (raw)**: {s['p_value_raw']:.4f}
"""
        if 'p_value_adjusted' in s:
            markdown += f"- **P-value (FDR-adjusted)**: {s['p_value_adjusted']:.4f} {'***' if s['significant'] else ''}\n"
        markdown += f"""
- **95% CI**: [{s['ci_95_lower_bps']:.2f}, {s['ci_95_upper_bps']:.2f}] bps
- **Win rate**: {s['win_rate']*100:.1f}% of days
- **Cohen's d**: {s['cohens_d']:.2f}
- **N days**: {s['n_samples']}
"""
    
    markdown += f"""

### RL vs Almgren-Chriss

"""
    
    if 'rl_vs_ac' in stats:
        s = stats['rl_vs_ac']
        markdown += f"""
- **Mean gap**: {s['mean_gap_bps']:.2f} bps (RL better if positive)
- **Median gap**: {s['median_gap_bps']:.2f} bps
- **P-value (raw)**: {s['p_value_raw']:.4f}
"""
        if 'p_value_adjusted' in s:
            markdown += f"- **P-value (FDR-adjusted)**: {s['p_value_adjusted']:.4f} {'***' if s['significant'] else ''}\n"
        markdown += f"""
- **95% CI**: [{s['ci_95_lower_bps']:.2f}, {s['ci_95_upper_bps']:.2f}] bps
- **Win rate**: {s['win_rate']*100:.1f}% of days
- **Cohen's d**: {s['cohens_d']:.2f}
- **N days**: {s['n_samples']}
"""
    
    markdown += f"""

## Methodology

- **Wilcoxon signed-rank test** (one-sided, α=0.05)
"""
    
    if any('p_value_adjusted' in stats.get(k, {}) for k in ['rl_vs_twap', 'rl_vs_ac']):
        markdown += "- **Benjamini-Hochberg FDR correction** across baselines\n"
    
    markdown += """- **Bootstrap confidence intervals** (10,000 resamples)
- **Per-day aggregation**: Multiple runs per day aggregated to single daily score
- **Paired comparisons**: RL and baselines tested on identical market windows

## Notes

- Positive advantage means RL outperforms baseline (lower slippage)
- *** indicates statistical significance (p < 0.05, FDR-adjusted if applicable)
- Win rate: Percentage of test days where RL beats baseline
- Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)
"""
    
    return markdown


def generate_latex_table(
    stats: Dict,
    by_horizon: Dict,
    config: Dict,
    results_file: str,
) -> str:
    """Generate LaTeX table (for academic papers)."""
    # Similar structure but with LaTeX formatting
    # Implementation similar to markdown but with LaTeX syntax
    return "LaTeX format not yet implemented. Use markdown format."


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready results tables",
    )
    parser.add_argument("results_file", help="Path to eval_results JSON file")
    parser.add_argument("--output", help="Output file path (default: results_file.md)")
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown",
                       help="Output format")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.results_file.replace('.json', '.md')
    
    report = generate_results_table(args.results_file, args.output, args.format)
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

