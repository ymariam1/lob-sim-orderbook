# LOB Simulator

A high-performance Limit Order Book (LOB) simulator designed for backtesting execution algorithms and training reinforcement learning agents. Features realistic latency simulation, real L3 market data support, and Python bindings for easy integration.

## Quick Start

```bash
# Install the Python package (builds C++ automatically)
pip install .

# Run the example
python examples/example_usage.py

# Run baseline comparison (VWAP vs POV)
python src/py/baselines.py --data data/blockchain_l3_2023-03-01.csv --qty 1000 --time 3600
```

## Training RL Agents

### 1. Prepare Data (Train/Test Split)

Place your CSV files in `data/csv/`, then split them 80/20:

```bash
python split_data.py
```

This copies the first 80% of files to `data/train/` and remaining 20% to `data/test/`.

### 2. Train a Model

```bash
python src/py/train_rl.py \
  --train-data data/train \
  --test-data data/test \
  --timesteps 100000
```

Key arguments:

- `--train-data`: Directory containing training CSV files
- `--test-data`: Directory containing test CSV files (separate from training)
- `--timesteps`: Total training timesteps (default: 100k)
- `--agent-type`: Latency profile - `hft`, `institutional`, or `retail` (default: institutional)
- `--target-qty`: Target quantity per episode (default: 100)
- `--lr`: Learning rate (default: 3e-4)
- `--net-arch`: Network architecture (default: 64 64)

### 3. Evaluate the Model

```bash
python src/py/train_rl.py \
  --eval-only \
  --model models/ppo_lob_latest \
  --train-data data/test \
  --n-eval-episodes 10
```

This runs the trained model on test data and compares against VWAP/POV baselines.

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
│       ├── train_rl.py           # PPO training script with proper train/test split
│       ├── gym.py                # Gymnasium RL environment (Implementation Shortfall reward)
│       ├── baselines.py          # VWAP, POV, and Almgren-Chriss baselines
│       └── latency.py            # Stochastic latency simulation
│
├── data/
│   ├── csv/                      # Raw CSV files (place your data here)
│   ├── train/                    # Training data (80%)
│   ├── test/                     # Test data (20%)
│   └── fetch_l3.py               # Script to download L3 data from Tardis
│
├── split_data.py                 # Script to split data into train/test
├── models/                       # Saved RL models
├── logs/                         # Training logs (tensorboard)
├── examples/
│   └── example_usage.py          # Demonstrates all key features
│
└── pyproject.toml                # Python package configuration
```

## RL Environment Details

### Action Space (Discrete 14)

- **0**: Hold (do nothing)
- **1-5**: Limit Buy at best bid - (0,1,2,3,4) ticks
- **6-10**: Limit Sell at best ask + (0,1,2,3,4) ticks
- **11**: Market Buy
- **12**: Market Sell
- **13**: Cancel all active orders

### Observation Space

- Bid/Ask prices and quantities (10 levels each)
- Current position
- Realized P&L (cash)
- Current market time
- Number of active orders
- Execution progress (0-1)
- Time remaining (if horizon specified)

### Reward Function (Implementation Shortfall)

The agent is rewarded based on minimizing slippage relative to the arrival price:

```
reward = -slippage * qty_filled
where slippage = (execution_price - arrival_price) / arrival_price
```

Additionally:

- **Inventory penalty**: Almgren-Chriss style quadratic penalty for holding inventory (encourages completion)
- **Terminal penalty**: Large penalty for incomplete execution at episode end

### Key Parameters

- `warmup_duration_ns`: 5 seconds (builds initial order book state)
- `step_duration_ns`: 10ms per step
- `max_episode_steps`: 10,000 (prevents infinite episodes)
- `inventory_penalty_coef`: 0.01 (tunable urgency parameter)
