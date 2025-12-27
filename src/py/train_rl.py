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
from datetime import datetime
from pathlib import Path

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
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Please install stable-baselines3:")
    print("  pip install stable-baselines3[extra]")
    sys.exit(1)

from src.py.gym import LOBEnv


def make_env(
    data_path: str,
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    max_position: int = 100,
    step_duration_ns: int = 10_000_000,
    warmup_duration_ns: int = 60_000_000_000,
    target_qty: int = 100,
    execution_side: str = "SELL",
    latency_seed: int = None,
    rank: int = 0,
    log_dir: str = None,
):
    """Create a wrapped LOBEnv for training."""
    def _init():
        env = LOBEnv(
            data_path=data_path,
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            max_position=max_position,
            step_duration_ns=step_duration_ns,
            warmup_duration_ns=warmup_duration_ns,
            timestamp_unit_ns=1000,  # Tardis/Blockchain data uses microseconds
            target_qty=target_qty,
            execution_side=execution_side,
            latency_seed=latency_seed,
        )
        # Wrap with Monitor for logging
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init


def train(
    data_path: str,
    total_timesteps: int = 100_000,
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    target_qty: int = 100,
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
    resume_path: str = None,
    verbose: int = 1,
):
    """
    Train a PPO agent on the LOB environment.
    
    Args:
        data_path: Path to L3 market data CSV
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
    """
    if net_arch is None:
        net_arch = [64, 64]
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_lob_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    print("=" * 60)
    print("LOB RL Training (Implementation Shortfall Reward)")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Agent Type: {agent_type}")
    print(f"Volume Sensitivity: {volume_sensitivity}")
    print(f"Target Qty: {target_qty}")
    print(f"Execution Side: {execution_side}")
    print(f"Network: {net_arch}")
    print(f"Log Dir: {run_log_dir}")
    print("=" * 60)
    
    # Create training environment
    train_env = DummyVecEnv([
        make_env(
            data_path=data_path,
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            target_qty=target_qty,
            execution_side=execution_side,
            log_dir=run_log_dir,
            rank=0,
        )
    ])
    
    # Optionally normalize observations
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(
            data_path=data_path,
            agent_type=agent_type,
            volume_sensitivity=volume_sensitivity,
            target_qty=target_qty,
            execution_side=execution_side,
            rank=0,
        )
    ])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for eval
        clip_obs=10.0,
        training=False,
    )
    
    # Create or load model
    if resume_path and os.path.exists(resume_path + ".zip"):
        print(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=train_env)
        # Load normalization stats
        vec_norm_path = resume_path + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            train_env = VecNormalize.load(vec_norm_path, train_env)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
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
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=run_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
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
    
    # Save final model
    final_path = os.path.join(save_dir, f"{run_name}_final")
    model.save(final_path)
    train_env.save(final_path + "_vecnormalize.pkl")
    print(f"\nFinal model saved to: {final_path}")
    
    # Also save as "latest" for easy resuming
    latest_path = os.path.join(save_dir, "ppo_lob_latest")
    model.save(latest_path)
    train_env.save(latest_path + "_vecnormalize.pkl")
    
    train_env.close()
    eval_env.close()
    
    return model


def evaluate(
    model_path: str,
    data_path: str,
    n_episodes: int = 10,
    agent_type: str = "institutional",
    volume_sensitivity: float = 0.1,
    target_qty: int = 100,
    execution_side: str = "SELL",
    render: bool = False,
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
    )
    
    # Load model
    model = PPO.load(model_path)
    
    # Load normalization stats if available
    vec_norm_path = model_path + "_vecnormalize.pkl"
    normalize = os.path.exists(vec_norm_path)
    if normalize:
        # We need to wrap in DummyVecEnv for VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    episode_rewards = []
    episode_lengths = []
    episode_pnls = []
    
    for ep in range(n_episodes):
        if normalize:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()
        
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
                action, _ = model.predict(obs, deterministic=True)
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
    
    # Data
    parser.add_argument(
        "--data", 
        default="data/blockchain_l3_2023-03-01.csv",
        help="Path to L3 market data CSV"
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
    parser.add_argument("--target-qty", type=int, default=100, help="Target quantity to execute per episode")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="SELL", help="Execution side")
    
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
    
    args = parser.parse_args()
    
    # Check data exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print("Please run: python data/fetch_l3.py --hours 1")
        sys.exit(1)
    
    if args.eval_only:
        if not args.model:
            args.model = "models/ppo_lob_latest"
        evaluate(
            model_path=args.model,
            data_path=args.data,
            n_episodes=args.n_eval_episodes,
            agent_type=args.agent_type,
            volume_sensitivity=args.volume_sensitivity,
            target_qty=args.target_qty,
            execution_side=args.side,
            render=args.render,
        )
    else:
        train(
            data_path=args.data,
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
        )


if __name__ == "__main__":
    main()
