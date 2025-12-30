# LOB Simulator

A high-performance Limit Order Book (LOB) simulator designed for backtesting execution algorithms and training reinforcement learning agents. Features realistic latency simulation, real L3 market data support, and Python bindings for easy integration.

## Quick Start

### Prerequisites

**System dependencies** (install via package manager):
- `cmake` (3.15+)
- `python3.12-dev` or `python3.12-devel` (Python 3.12 development headers)
- C++ compiler with C++20 support (GCC 10+ or Clang 10+)
- Python 3.12 or higher

For SSH/cluster environments, ensure Python 3.12 development headers are available:
- Ubuntu/Debian: `sudo apt-get install python3.12-dev`
- CentOS/RHEL: `sudo yum install python3.12-devel`
- Or load Python module: `module load python/3.12`

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

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
- `--model-size`: Model size preset - `small`, `base`, `large`, or `xlarge` (default: base)
  - `small`: [64, 64] - Fast training, lower capacity
  - `base`: [128, 128] - Balanced (default)
  - `large`: [256, 256] - Higher capacity, slower training
  - `xlarge`: [512, 512] - Maximum capacity, slowest training
- `--agent-type`: Latency profile - `hft`, `institutional`, or `retail` (default: institutional)
- `--target-qty`: Target quantity per episode (default: calculated from volume)
- `--lr`: Learning rate (default: 3e-4)
- `--net-arch`: Custom network architecture (overrides `--model-size`)

### 3. Train with Different Model Sizes

```bash
# Small model (fast, lower capacity)
python src/py/train_rl.py \
  --train-data data/train \
  --test-data data/test \
  --model-size small \
  --timesteps 100000

# Base model (balanced - default)
python src/py/train_rl.py \
  --train-data data/train \
  --test-data data/test \
  --model-size base \
  --timesteps 100000

# Large model (higher capacity)
python src/py/train_rl.py \
  --train-data data/train \
  --test-data data/test \
  --model-size large \
  --timesteps 200000

# XLarge model (maximum capacity)
python src/py/train_rl.py \
  --train-data data/train \
  --test-data data/test \
  --model-size xlarge \
  --timesteps 500000
```

### 4. Evaluate the Model

```bash
python src/py/train_rl.py \
  --eval-only \
  --model models/ppo_lob_latest \
  --train-data data/test \
  --n-eval-episodes 10
```

This runs the trained model on test data and compares against VWAP/POV baselines.

### 5. Multi-Seed Training (for academic rigor)

For statistically valid results, train multiple models with different seeds:

```bash
python train_multi_seed.py \
  --train-data data/train \
  --test-data data/test \
  --n-seeds 5 \
  --timesteps 100000 \
  --model-size base
```

This trains 5 independent models, enabling proper statistical testing with mean ± std.

### Model Checkpointing

To save disk space, only the **best model** (based on evaluation performance) is saved during training:
- **Best model**: `models/best/best_model.zip` - Use this for evaluation
- **Latest model**: `models/ppo_lob_latest.zip` - For resuming interrupted training

Intermediate checkpoints are NOT saved. If you need more frequent checkpoints, modify the `EvalCallback` frequency in `train_rl.py`.

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
│       ├── baselines.py          # VWAP (true volume-weighted), POV, TWAP, and Almgren-Chriss baselines
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
