#!/usr/bin/env python3
"""
Train a PPO agent on the LOB environment.

This script trains a small, fast PPO agent to learn optimal execution
in a limit order book environment with latency simulation.

Usage:
    # Basic training
    python src/py/train_rl.py

    # With custom parameters
    python src/py/train_rl.py --timesteps 500000 --latency 5000000

    # Resume from checkpoint
    python src/py/train_rl.py --resume models/ppo_lob_latest

    # Evaluate only
    python src/py/train_rl.py --eval-only --model models/ppo_lob_best
"""

import os
import sys
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fix gym/gymnasium compatibility for stable-baselines3
import gymnasium
sys.modules["gym"] = gymnasium

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Please install stable-baselines3:")
    print("  pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Enhanced logging disabled.")
    print("  Install with: pip install wandb")

from src.py.gym import LOBEnv


def find_csv_files(data_path: str) -> List[str]:
    """
    Find all CSV files in a directory or return single file.
    
    Args:
        data_path: Path to CSV file or directory containing CSV files
        
    Returns:
        List of CSV file paths
    """
    path = Path(data_path)
    
    if path.is_file():
        # Single file
        return [str(path)]
    elif path.is_dir():
        # Directory - find all CSV files
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_path}")
        return [str(f) for f in csv_files]
    else:
        raise ValueError(f"Path does not exist: {data_path}")


class ActionDistributionCallback(BaseCallback):
    """
    Diagnostic callback to log action distribution.
    
    Helps identify if the agent is stuck on a single action (e.g., "Hold").
    """
    
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_buffer = []
    
    def _on_step(self) -> bool:
        # Collect actions from locals
        actions = self.locals.get('actions', None)
        if actions is not None:
            # Handle different action formats (array, list, scalar)
            if isinstance(actions, np.ndarray):
                actions_flat = actions.flatten()
            elif isinstance(actions, (list, tuple)):
                actions_flat = np.array(actions).flatten()
            else:
                actions_flat = np.array([actions])
            
            self.action_buffer.extend(actions_flat.tolist())
        
        # Log distribution periodically
        if self.n_calls % self.log_freq == 0 and len(self.action_buffer) > 0:
            unique, counts = np.unique(self.action_buffer, return_counts=True)
            dist = dict(zip(unique.astype(int), counts))
            total = len(self.action_buffer)
            dist_pct = {k: (v / total * 100) for k, v in dist.items()}
            
            if self.verbose > 0:
                print(f"\nStep {self.n_calls} action distribution:")
                for action_id in sorted(dist_pct.keys()):
                    print(f"  Action {action_id}: {dist_pct[action_id]:.1f}% ({dist[action_id]})")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                for action_id, pct in dist_pct.items():
                    wandb.log({f"action_dist/action_{action_id}": pct}, commit=False)
            
            # Clear buffer
            self.action_buffer = []
        
        return True


class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback to log detailed episode diagnostics.
    
    Tracks:
    - Execution metrics (slippage, completion rate)
    - Action distribution
    - Latency statistics
    - Market conditions
    """
    
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_data = []
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            self._log_episode()
        return True
    
    def _log_episode(self):
        """Log detailed episode information."""
        self.episode_count += 1
        
        # Get info from the environment
        if hasattr(self.training_env, 'get_attr'):
            try:
                infos = self.training_env.get_attr('_get_info')
                if infos and len(infos) > 0:
                    info = infos[0]()
                    
                    episode_metrics = {
                        'episode/num': self.episode_count,
                        'episode/executed_qty': info.get('executed_qty', 0),
                        'episode/target_qty': info.get('target_qty', 0),
                        'episode/completion_rate': (
                            info.get('executed_qty', 0) / info.get('target_qty', 1)
                            if info.get('target_qty', 0) > 0 else 0
                        ),
                        'episode/arrival_price': info.get('arrival_price', 0),
                        'episode/mid_price': info.get('mid_price', 0),
                        'episode/active_orders': info.get('active_orders', 0),
                        'episode/pending_actions': info.get('pending_actions', 0),
                    }
                    
                    # Latency metrics
                    if 'last_latency_ms' in info:
                        episode_metrics['latency/last_ms'] = info['last_latency_ms']
                    if 'latency_mean_ms' in info:
                        episode_metrics['latency/mean_ms'] = info['latency_mean_ms']
                    if 'latency_p99_ms' in info:
                        episode_metrics['latency/p99_ms'] = info['latency_p99_ms']
                    if 'market_regime' in info:
                        episode_metrics['market/regime'] = 1 if info['market_regime'] == 'stressed' else 0
                    if 'global_congestion_ms' in info:
                        episode_metrics['market/congestion_ms'] = info['global_congestion_ms']
                    
                    # Calculate slippage if we have VWAP
                    if 'vwap' in info and 'arrival_price' in info:
                        vwap = info['vwap']
                        arrival = info['arrival_price']
                        if arrival > 0:
                            slippage_bps = abs(vwap - arrival) / arrival * 10000
                            episode_metrics['episode/slippage_bps'] = slippage_bps
                    
                    self.episode_data.append(episode_metrics)
                    
                    # Log to wandb if available
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log(episode_metrics)
                    
                    # Print summary every log_freq episodes
                    if self.episode_count % self.log_freq == 0:
                        if self.verbose > 0:
                            print(f"\nEpisode {self.episode_count} Summary:")
                            print(f"  Completion: {episode_metrics.get('episode/completion_rate', 0):.1%}")
                            if 'episode/slippage_bps' in episode_metrics:
                                print(f"  Slippage: {episode_metrics['episode/slippage_bps']:.2f} bps")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not log episode metrics: {e}")


def make_env(
    data_paths: List[str],  # Can be single file or list of files
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    max_position: int = 100,
    step_duration_ns: int = 10_000_000,
    warmup_duration_ns: int = 60_000_000_000,
    target_qty: Optional[int] = None,  # None = auto from daily volume
    execution_side: str = "SELL",
    latency_seed: int = None,
    rank: int = 0,
    log_dir: str = None,
    allow_random_selection: bool = True,
    inventory_penalty_coef: float = 0.01,
    target_qty_pct: float = 0.03,  # Percentage of daily volume (1-5% = 0.01-0.05)
):
    """
    Create a wrapped LOBEnv for training.
    
    If multiple data files are provided, randomly selects one for each episode reset.
    This provides diversity in training data.
    
    CRITICAL: For evaluation, set allow_random_selection=False to use a specific file.
    This prevents data leakage between train and test sets.
    
    Args:
        data_paths: List of CSV file paths (or single path as list)
        allow_random_selection: If True, randomly selects file per episode (training).
                               If False, uses first file (evaluation).
    """
    # Ensure data_paths is a list
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    if not data_paths:
        raise ValueError("data_paths cannot be empty")
    
    def _init():
        # For training: randomly select a data file for diversity
        # For evaluation: use first file for consistency (prevents data leakage)
        if allow_random_selection and len(data_paths) > 1:
            selected_data = random.choice(data_paths)
        else:
            selected_data = data_paths[0]
        
        env = LOBEnv(
            data_path=selected_data,
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            max_position=max_position,
            step_duration_ns=step_duration_ns,
            warmup_duration_ns=warmup_duration_ns,
            timestamp_unit_ns=1000,  # Tardis/Blockchain data uses microseconds
            target_qty=target_qty,
            execution_side=execution_side,
            latency_seed=latency_seed,
            inventory_penalty_coef=inventory_penalty_coef,
            target_qty_pct=target_qty_pct,
        )
        # Wrap with Monitor for logging
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def train(
    train_data_path: str,  # Can be file or directory - TRAINING DATA ONLY
    test_data_path: str = None,  # Optional: separate test data for evaluation
    total_timesteps: int = 100_000,
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    target_qty: Optional[int] = None,  # None = auto from daily volume (1-5%)
    execution_side: str = "SELL",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    net_arch: list = None,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    max_eval_episode_steps: int = 10000,  # Max steps per eval episode (prevents hangs)
    resume_path: str = None,
    verbose: int = 1,
    inventory_penalty_coef: float = 0.01,
    ent_coef: float = 0.0,
    target_qty_pct: float = 0.03,  # Percentage of daily volume (1-5% = 0.01-0.05)
):
    """
    Train a PPO agent on the LOB environment.
    
    CRITICAL: train_data_path and test_data_path must be completely separate.
    No overlap between training and test data to prevent data leakage.
    
    Args:
        train_data_path: Path to L3 market data CSV file or directory (TRAINING ONLY)
        test_data_path: Optional separate test data for evaluation (defaults to train_data_path if None)
        total_timesteps: Total training timesteps
        agent_type: Latency profile ("hft", "institutional", "retail", or "base_ms:sigma")
        volume_sensitivity: How much volume affects latency (η)
        target_qty: Target quantity to execute per episode
        execution_side: "BUY" or "SELL" - which side agent executes
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        net_arch: Network architecture (default [64, 64])
        save_dir: Directory for model checkpoints
        log_dir: Directory for tensorboard logs
        eval_freq: Evaluation frequency
        n_eval_episodes: Episodes per evaluation
        resume_path: Path to resume training from
        verbose: Verbosity level
        inventory_penalty_coef: Penalty coefficient for holding inventory (Almgren-Chriss style, default 0.01)
        ent_coef: Entropy coefficient for exploration (default 0.0, try 0.01-0.05 if stuck)
    """
    if net_arch is None:
        net_arch = [64, 64]
    
    # Find all CSV files for training (handles both file and directory)
    train_data_files = find_csv_files(train_data_path)
    
    # Find test data files (if provided)
    if test_data_path:
        test_data_files = find_csv_files(test_data_path)
        # CRITICAL: Verify no overlap between train and test
        train_set = set(Path(f).resolve() for f in train_data_files)
        test_set = set(Path(f).resolve() for f in test_data_files)
        overlap = train_set & test_set
        if overlap:
            raise ValueError(
                f"CRITICAL: Train and test data overlap detected!\n"
                f"Overlapping files: {[str(f) for f in overlap]}\n"
                f"This violates data leakage prevention. Use separate directories."
            )
    else:
        # If no test data provided, use train data for eval (not recommended for production)
        test_data_files = [train_data_files[0]]  # Use first file only
        if verbose > 0:
            print("⚠️  WARNING: No test_data_path provided. Using first training file for evaluation.")
            print("   This is acceptable for development but NOT for final experiments.")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_lob_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        # Check if running in sweep mode
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        if sweep_id:
            wandb.init()
        else:
            wandb.init(
                project="lob-rl",
                name=run_name,
                config={
                    "train_data_path": train_data_path,
                    "test_data_path": test_data_path,
                    "num_train_files": len(train_data_files),
                    "num_test_files": len(test_data_files) if test_data_path else 0,
                    "total_timesteps": total_timesteps,
                    "agent_type": agent_type,
                    "volume_sensitivity": volume_sensitivity,
                    "target_qty": target_qty,
                    "execution_side": execution_side,
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "net_arch": net_arch,
                    "inventory_penalty_coef": inventory_penalty_coef,
                    "ent_coef": ent_coef,
                },
                sync_tensorboard=True,
            )
    
    print("=" * 60)
    print("LOB RL Training (Implementation Shortfall Reward)")
    print("=" * 60)
    print(f"Training Data: {train_data_path}")
    if len(train_data_files) > 1:
        print(f"  Found {len(train_data_files)} CSV files")
        print(f"  Files: {', '.join([Path(f).name for f in train_data_files[:5]])}")
        if len(train_data_files) > 5:
            print(f"  ... and {len(train_data_files) - 5} more")
    else:
        print(f"  File: {Path(train_data_files[0]).name}")
    
    if test_data_path:
        print(f"\nTest Data: {test_data_path}")
        print(f"  Found {len(test_data_files)} CSV files (separate from training)")
        print("  ✅ Train/test separation verified: No overlap")
    else:
        print(f"\n⚠️  Test Data: Using first training file (NOT RECOMMENDED for production)")
    
    print(f"\nTimesteps: {total_timesteps:,}")
    print(f"Agent Type: {agent_type}")
    print(f"Volume Sensitivity: {volume_sensitivity}")
    print(f"Target Qty: {target_qty}")
    print(f"Execution Side: {execution_side}")
    print(f"Network: {net_arch}")
    print(f"Log Dir: {run_log_dir}")
    if WANDB_AVAILABLE and wandb.run is not None:
        print(f"WandB: {wandb.run.url}")
    print("=" * 60)
    
    # Create training environment (will randomly select from train_data_files each episode)
    train_env = DummyVecEnv([
            make_env(
            data_paths=train_data_files,  # Pass list of TRAINING files only
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            target_qty=target_qty,
            execution_side=execution_side,
            log_dir=run_log_dir,
            rank=0,
            allow_random_selection=True,  # Random selection for training diversity
            inventory_penalty_coef=inventory_penalty_coef,
            target_qty_pct=target_qty_pct,
        )
    ])
    
    # Create VecNormalize wrapper (CRITICAL: save this separately!)
    vec_normalize = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment (use TEST data, not training data)
    # Wrap with Monitor for proper metrics reporting (even without log_dir, Monitor tracks metrics)
    def _make_eval_env():
        env_factory = make_env(
            data_paths=test_data_files,  # Use TEST files only (prevents data leakage)
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            target_qty=target_qty,
            execution_side=execution_side,
            rank=0,
            allow_random_selection=False,  # Use first file for consistency in eval
            inventory_penalty_coef=inventory_penalty_coef,
            target_qty_pct=target_qty_pct,
        )
        env = env_factory()
        # Set max episode steps to prevent infinite episodes during eval
        env._max_episode_steps = max_eval_episode_steps
        # Wrap with Monitor for proper metrics reporting (filename=None means no file logging)
        return Monitor(env, filename=None)
    
    eval_env_unwrapped = DummyVecEnv([_make_eval_env])
    eval_env = VecNormalize(
        eval_env_unwrapped,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for eval
        clip_obs=10.0,
        training=False,
    )
    
    # Create or load model
    if resume_path and os.path.exists(resume_path + ".zip"):
        print(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=vec_normalize)
        # Load normalization stats (CRITICAL: use frozen train stats)
        vec_norm_path = resume_path + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            vec_normalize = VecNormalize.load(vec_norm_path, train_env)
            vec_normalize.training = False  # Freeze stats
            # Sync to eval env
            eval_env = VecNormalize.load(vec_norm_path, eval_env_unwrapped)
            eval_env.training = False
            eval_env.norm_reward = False
    else:
        model = PPO(
            "MlpPolicy",
            vec_normalize,  # Use VecNormalize wrapper
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,  # Entropy coefficient for exploration
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=run_log_dir,
            verbose=verbose,
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_dir,
        name_prefix=run_name,
        save_vecnormalize=True,
    )
    
    # Sync eval env normalization from training (CRITICAL!)
    # This ensures eval uses frozen train statistics
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=run_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        warn=False,  # Suppress warnings about Monitor wrapper (we already added it)
    )
    
    # Custom callback to sync normalization stats
    class SyncNormalizeCallback(BaseCallback):
        """Sync eval env normalization from training env."""
        def _on_step(self) -> bool:
            # Sync normalization stats from training to eval
            eval_env.obs_rms = vec_normalize.obs_rms
            eval_env.ret_rms = vec_normalize.ret_rms
            return True
    
    # Enhanced episode logging
    episode_logger = EpisodeLoggerCallback(verbose=verbose, log_freq=100)
    
    # Action distribution diagnostic
    action_dist_callback = ActionDistributionCallback(verbose=verbose, log_freq=1000)
    
    callbacks = CallbackList([
        checkpoint_callback, 
        eval_callback, 
        episode_logger,
        action_dist_callback,
        SyncNormalizeCallback(),
    ])
    
    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    # Save final model (CRITICAL: save VecNormalize separately!)
    final_path = os.path.join(save_dir, f"{run_name}_final")
    model.save(final_path)
    vec_normalize.save(final_path + "_vecnormalize.pkl")
    print(f"\nFinal model saved to: {final_path}")
    print(f"VecNormalize stats saved to: {final_path}_vecnormalize.pkl")
    
    # Also save as "latest" for easy resuming
    latest_path = os.path.join(save_dir, "ppo_lob_latest")
    model.save(latest_path)
    vec_normalize.save(latest_path + "_vecnormalize.pkl")
    
    vec_normalize.close()
    eval_env.close()
    
    return model


def evaluate(
    model_path: str,
    data_path: str,
    n_episodes: int = 10,
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    target_qty: Optional[int] = None,  # None = auto from daily volume
    execution_side: str = "SELL",
    render: bool = False,
    inventory_penalty_coef: float = 0.01,
    target_qty_pct: float = 0.03,  # Percentage of daily volume
):
    """Evaluate a trained model."""
    print("=" * 60)
    print("LOB RL Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Agent Type: {agent_type}")
    print(f"Target Qty: {target_qty}")
    print(f"Execution Side: {execution_side}")
    print("=" * 60)
    
    # Create environment
    env = LOBEnv(
        data_path=data_path,
        agent_type=agent_type,
        volume_sensitivity=volume_sensitivity,
        timestamp_unit_ns=1000,
        target_qty=target_qty,
        execution_side=execution_side,
        render_mode="human" if render else None,
        inventory_penalty_coef=inventory_penalty_coef,
        target_qty_pct=target_qty_pct,
    )
    
    # Load model
    model = PPO.load(model_path)
    
    # Check if model's observation space matches environment
    model_obs_shape = model.observation_space.shape
    env_obs_shape = env.observation_space.shape
    truncate_obs = False
    if model_obs_shape != env_obs_shape:
        print(f"⚠️  Warning: Model observation space shape {model_obs_shape} != environment shape {env_obs_shape}")
        print(f"   The model was trained with a different observation space.")
        if env_obs_shape[0] > model_obs_shape[0]:
            print(f"   Truncating observations from {env_obs_shape[0]} to {model_obs_shape[0]} dimensions.")
            truncate_obs = True
        else:
            print(f"   This will cause errors. Consider retraining the model with the updated environment.")
            print(f"   Attempting to evaluate anyway...")
    
    # Load normalization stats if available
    vec_norm_path = model_path + "_vecnormalize.pkl"
    normalize = False
    vec_env = None
    
    if os.path.exists(vec_norm_path):
        try:
            # We need to wrap in DummyVecEnv for VecNormalize
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            normalize = True
        except (AssertionError, ValueError) as e:
            # Handle shape mismatch (e.g., observation space changed)
            if "shape" in str(e).lower() or "space" in str(e).lower():
                print(f"⚠️  Warning: VecNormalize stats have incompatible observation space shape.")
                print(f"   This likely means the environment was updated (e.g., added time_remaining).")
                print(f"   Evaluating without normalization stats.")
                if vec_env is not None:
                    vec_env.close()
                vec_env = None
                normalize = False
            else:
                raise
    
    episode_rewards = []
    episode_lengths = []
    episode_pnls = []
    
    for ep in range(n_episodes):
        if normalize:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()
            # Truncate observation if needed to match model's expected input shape
            if truncate_obs:
                obs = obs[:model_obs_shape[0]]
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            if normalize:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                done = done[0]
                reward = reward[0]
                info = info[0]
            else:
                # Truncate observation if needed to match model's expected input shape
                obs_for_model = obs[:model_obs_shape[0]] if truncate_obs else obs
                action, _ = model.predict(obs_for_model, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if "pnl" in info:
            episode_pnls.append(info["pnl"])
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={total_reward:.2f}, Steps={steps}, "
              f"PnL={info.get('pnl', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    if episode_pnls:
        print(f"Mean PnL: {np.mean(episode_pnls):.2f} ± {np.std(episode_pnls):.2f}")
    print("=" * 60)
    
    if normalize:
        vec_env.close()
    else:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate PPO agent on LOB environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data (CRITICAL: Separate train and test to prevent data leakage)
    parser.add_argument(
        "--train-data", 
        default="data/csv",
        help="Path to L3 market data CSV file or directory for TRAINING (required)"
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help="Path to L3 market data CSV file or directory for TESTING (separate from training). "
             "If not provided, uses first training file (NOT recommended for production)."
    )
    
    # Training
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64], help="Network architecture")
    
    # Environment
    parser.add_argument("--agent-type", default="institutional", 
                        help="Latency profile: 'hft', 'institutional', 'retail', or 'base_ms:sigma'")
    parser.add_argument("--volume-sensitivity", type=float, default=0.1,
                        help="How much volume affects latency (η)")
    parser.add_argument("--target-qty", type=int, default=None, 
                        help="Target quantity to execute per episode (if None, calculated from target-qty-pct)")
    parser.add_argument("--target-qty-pct", type=float, default=0.03,
                        help="Target quantity as percentage of daily volume (default 0.03 = 3%%, range 0.01-0.05)")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="SELL", help="Execution side")
    parser.add_argument("--inventory-penalty-coef", type=float, default=0.01,
                        help="Penalty coefficient for holding inventory (Almgren-Chriss style, default 0.01)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient for exploration (default 0.0, try 0.01-0.05 if stuck)")
    
    # Saving/Loading
    parser.add_argument("--save-dir", default="models", help="Model save directory")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    
    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Evaluation mode only")
    parser.add_argument("--model", help="Model path for evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    # Misc
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Disable wandb if requested
    if args.no_wandb:
        global WANDB_AVAILABLE
        WANDB_AVAILABLE = False
    
    # Check training data exists (handles both files and directories)
    try:
        train_data_files = find_csv_files(args.train_data)
        if not train_data_files:
            print(f"Error: No CSV files found in: {args.train_data}")
            print("Please run: python data/fetch_l3.py --hours 1")
            sys.exit(1)
        print(f"Found {len(train_data_files)} CSV file(s) for training")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check test data if provided
    test_data_path = args.test_data
    if test_data_path:
        try:
            test_data_files = find_csv_files(test_data_path)
            if not test_data_files:
                print(f"Error: No CSV files found in test data: {test_data_path}")
                sys.exit(1)
            print(f"Found {len(test_data_files)} CSV file(s) for testing")
            
            # Verify no overlap
            train_set = set(Path(f).resolve() for f in train_data_files)
            test_set = set(Path(f).resolve() for f in test_data_files)
            overlap = train_set & test_set
            if overlap:
                print(f"\n❌ CRITICAL ERROR: Train and test data overlap!")
                print(f"Overlapping files: {[str(f) for f in overlap]}")
                print("This violates data leakage prevention.")
                sys.exit(1)
            print("✅ Train/test separation verified: No overlap")
        except ValueError as e:
            print(f"Error with test data: {e}")
            sys.exit(1)
    else:
        print("⚠️  WARNING: No --test-data provided. Using first training file for evaluation.")
        print("   This is acceptable for development but NOT for final experiments.")
    
    if args.eval_only:
        if not args.model:
            args.model = "models/ppo_lob_latest"
        # For evaluation, use test data if provided, otherwise first training file
        eval_data_files = find_csv_files(test_data_path) if test_data_path else train_data_files
        eval_data_path = eval_data_files[0] if eval_data_files else args.train_data
        if len(eval_data_files) > 1:
            print(f"Note: Using first file for evaluation: {Path(eval_data_path).name}")
        evaluate(
            model_path=args.model,
            data_path=eval_data_path,  # Use single file for eval
            n_episodes=args.n_eval_episodes,
            agent_type=args.agent_type,
            volume_sensitivity=args.volume_sensitivity,
            target_qty=args.target_qty,
            execution_side=args.side,
            render=args.render,
            inventory_penalty_coef=args.inventory_penalty_coef,
            target_qty_pct=args.target_qty_pct,
        )
    else:
        train(
            train_data_path=args.train_data,
            test_data_path=test_data_path,
            total_timesteps=args.timesteps,
            agent_type=args.agent_type,
            volume_sensitivity=args.volume_sensitivity,
            target_qty=args.target_qty,
            execution_side=args.side,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            net_arch=args.net_arch,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            resume_path=args.resume,
            verbose=0 if args.quiet else 1,
            inventory_penalty_coef=args.inventory_penalty_coef,
            ent_coef=args.ent_coef,
            target_qty_pct=args.target_qty_pct,
        )


if __name__ == "__main__":
    main()
