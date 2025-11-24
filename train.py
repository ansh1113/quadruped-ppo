"""
Training script for quadruped locomotion using PPO.

This script trains a quadruped robot to walk on various terrain types
using Proximal Policy Optimization (PPO) from Stable Baselines3.

Usage:
    python train.py --terrain flat --timesteps 1000000
    python train.py --terrain mixed --timesteps 2000000 --save-freq 50000
"""

import argparse
import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import custom environment (you'll implement this)
from quadruped_ppo.envs import QuadrupedEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train quadruped locomotion policy')
    
    # Environment arguments
    parser.add_argument('--terrain', type=str, default='flat',
                       choices=['flat', 'uneven', 'stairs', 'slopes', 'mixed'],
                       help='Terrain type for training')
    parser.add_argument('--difficulty', type=float, default=0.5,
                       help='Terrain difficulty (0.0 to 1.0)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate for optimizer')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per training iteration')
    
    # Saving arguments
    parser.add_argument('--save-path', type=str, default='models/quadruped',
                       help='Path to save trained model')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Save checkpoint every N timesteps')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load pretrained model')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs/',
                       help='Directory for tensorboard logs')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, or 2)')
    
    return parser.parse_args()


def make_env(terrain_type, difficulty, render=False):
    """
    Create and wrap environment.
    
    Args:
        terrain_type: Type of terrain
        difficulty: Terrain difficulty
        render: Whether to render
    
    Returns:
        Wrapped environment
    """
    def _init():
        env = QuadrupedEnv(
            terrain_type=terrain_type,
            terrain_difficulty=difficulty,
            render=render
        )
        env = Monitor(env)
        return env
    
    return _init


def create_callbacks(args, eval_env):
    """
    Create training callbacks.
    
    Args:
        args: Command line arguments
        eval_env: Evaluation environment
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.dirname(args.save_path),
        name_prefix=os.path.basename(args.save_path),
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - evaluate policy periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.save_path),
        log_path=args.log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    return callbacks


def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("=" * 50)
    print("Quadruped Locomotion Training")
    print("=" * 50)
    print(f"Terrain: {args.terrain}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)
    
    # Create training environment
    print("\nCreating training environment...")
    env = DummyVecEnv([make_env(args.terrain, args.difficulty, args.render)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(args.terrain, args.difficulty, False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Create or load model
    if args.load is not None:
        print(f"\nLoading pretrained model from {args.load}")
        model = PPO.load(args.load, env=env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=args.verbose,
            tensorboard_log=args.log_dir
        )
    
    # Create callbacks
    callbacks = create_callbacks(args, eval_env)
    
    # Train the model
    print("\nStarting training...")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Checkpoints will be saved to: {os.path.dirname(args.save_path)}")
    print("\nYou can monitor training with:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    print(f"\nSaving final model to {args.save_path}_final.zip")
    model.save(f"{args.save_path}_final")
    
    # Save normalization statistics
    env.save(f"{args.save_path}_vecnormalize.pkl")
    
    print("\nTraining complete!")
    print("=" * 50)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    obs = eval_env.reset()
    episode_rewards = []
    episode_lengths = []
    current_reward = 0
    current_length = 0
    
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        current_reward += reward[0]
        current_length += 1
        
        if done[0]:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            current_reward = 0
            current_length = 0
    
    print(f"\nFinal Evaluation Results:")
    print(f"  Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Number of episodes: {len(episode_rewards)}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
