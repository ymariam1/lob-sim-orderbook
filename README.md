# LOB Simulator

A high-performance Limit Order Book (LOB) simulator designed for backtesting execution algorithms and training reinforcement learning agents. Features realistic latency simulation, real L3 market data support, and Python bindings for easy integration.

## Quick Start

```bash
# Install the Python package (builds C++ automatically)
pip install .

# Run the example
python examples/example_usage.py

# Run baseline comparison (TWAP vs Almgren-Chriss)
python src/py/baselines.py --data data/blockchain_l3_2023-03-01.csv --qty 1000 --time 3600
```

## Project Structure

```
lob-sim-orderbook/
├── src/
│   ├── cpp/                      # C++ Core (the fast stuff)
│   │   ├── OrderBook.hpp         # Matching engine with price-time priority
│   │   ├── ExchangeSimulator.hpp # Latency simulation + fill tracking
│   │   ├── DataLoader.hpp        # CSV market data loader
│   │   ├── bindings.cpp          # Python bindings (pybind11)
│   │   └── main.cpp              # Standalone C++ entry point
│   │
│   └── py/                       # Python Layer
│       ├── baselines.py          # TWAP & Almgren-Chriss execution
│       └── gym.py                # Gymnasium RL environment
│
├── data/
│   ├── blockchain_l3_2023-03-01.csv  # Sample L3 data (1.7M events)
│   └── fetch_l3.py                   # Script to download more L3 data
│
├── examples/
│   └── example_usage.py          # Demonstrates all key features
│
└── results/
    └── baseline_comparison.json  # Benchmark results
```
