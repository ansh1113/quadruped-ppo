"""
Metrics computation for policy evaluation.

This module provides functions to evaluate trained policies and compute
various performance metrics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3.common.base_class import BaseAlgorithm
import gym


def evaluate_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained policy over multiple episodes.
    
    Args:
        model: Trained RL model
        env: Environment to evaluate on
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_distances: List[float] = []
    falls: int = 0
    successful_episodes: int = 0
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        start_pos = obs[0]  # x position
        fell = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if info.get('fell', False):
                fell = True
                falls += 1
        
        end_pos = obs[0]
        distance = end_pos - start_pos
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_distances.append(distance)
        
        if not fell:
            successful_episodes += 1
        
        if verbose:
            print(f"Episode {ep + 1}/{n_episodes}: "
                  f"Reward={ep_reward:.2f}, "
                  f"Length={ep_length}, "
                  f"Distance={distance:.2f}m, "
                  f"Fell={fell}")
    
    # Compute statistics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'mean_distance': float(np.mean(episode_distances)),
        'std_distance': float(np.std(episode_distances)),
        'fall_rate': float(falls / n_episodes),
        'success_rate': float(successful_episodes / n_episodes),
        'n_episodes': n_episodes
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"  Mean Distance: {metrics['mean_distance']:.2f}m ± {metrics['std_distance']:.2f}m")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Fall Rate: {metrics['fall_rate']:.1%}")
        print("=" * 50)
    
    return metrics


def compute_gait_metrics(
    foot_contacts: np.ndarray,
    dt: float = 0.02
) -> Dict[str, Any]:
    """
    Compute gait metrics from foot contact data.
    
    Args:
        foot_contacts: Array of shape (timesteps, 4) with binary contact data
        dt: Time step in seconds
        
    Returns:
        Dictionary containing gait metrics
    """
    num_steps = foot_contacts.shape[0]
    num_feet = foot_contacts.shape[1]
    
    # Compute duty cycle (fraction of time each foot is on ground)
    duty_cycles = np.mean(foot_contacts, axis=0)
    
    # Compute stance/swing transitions
    transitions = np.diff(foot_contacts, axis=0)
    liftoffs = np.sum(transitions == -1, axis=0)  # Contact to no contact
    touchdowns = np.sum(transitions == 1, axis=0)  # No contact to contact
    
    # Compute stride frequency (average across feet)
    stride_frequencies = touchdowns / (num_steps * dt)
    
    # Compute phase relationships between legs
    # Cross-correlation between foot contacts
    phases = []
    for i in range(num_feet):
        for j in range(i + 1, num_feet):
            # Normalized cross-correlation
            corr = np.correlate(
                foot_contacts[:, i] - np.mean(foot_contacts[:, i]),
                foot_contacts[:, j] - np.mean(foot_contacts[:, j]),
                mode='valid'
            )[0]
            norm = np.sqrt(np.sum((foot_contacts[:, i] - np.mean(foot_contacts[:, i]))**2) *
                          np.sum((foot_contacts[:, j] - np.mean(foot_contacts[:, j]))**2))
            if norm > 0:
                phases.append(corr / norm)
            else:
                phases.append(0.0)
    
    # Check gait type based on duty cycle
    avg_duty_cycle = np.mean(duty_cycles)
    if avg_duty_cycle > 0.5:
        gait_type = "walk"
    elif avg_duty_cycle > 0.3:
        gait_type = "trot"
    else:
        gait_type = "gallop"
    
    metrics = {
        'duty_cycles': duty_cycles.tolist(),
        'mean_duty_cycle': float(np.mean(duty_cycles)),
        'stride_frequencies': stride_frequencies.tolist(),
        'mean_stride_frequency': float(np.mean(stride_frequencies)),
        'liftoffs': liftoffs.tolist(),
        'touchdowns': touchdowns.tolist(),
        'phase_correlations': phases,
        'mean_phase_correlation': float(np.mean(phases)),
        'estimated_gait_type': gait_type,
        'total_duration': float(num_steps * dt)
    }
    
    return metrics


def compute_energy_efficiency(
    actions: np.ndarray,
    velocities: np.ndarray,
    dt: float = 0.02
) -> Dict[str, float]:
    """
    Compute energy efficiency metrics.
    
    Args:
        actions: Array of actions taken (timesteps, action_dim)
        velocities: Array of forward velocities (timesteps,)
        dt: Time step in seconds
        
    Returns:
        Dictionary with energy efficiency metrics
    """
    # Approximate mechanical cost of transport
    # Cost of Transport (CoT) = Power / (Weight * Velocity)
    # Here we approximate power from action magnitude
    
    power = np.sum(np.abs(actions), axis=1)  # Approximation
    mean_power = float(np.mean(power))
    
    mean_velocity = float(np.mean(velocities))
    total_distance = float(np.sum(velocities) * dt)
    
    # Energy per unit distance
    if total_distance > 0:
        energy_per_meter = mean_power / mean_velocity if mean_velocity > 0 else np.inf
    else:
        energy_per_meter = np.inf
    
    metrics = {
        'mean_power': mean_power,
        'total_energy': float(mean_power * len(actions) * dt),
        'mean_velocity': mean_velocity,
        'total_distance': total_distance,
        'energy_per_meter': float(energy_per_meter),
        'action_smoothness': float(np.mean(np.abs(np.diff(actions, axis=0))))
    }
    
    return metrics


def compute_stability_metrics(
    orientations: np.ndarray,
    positions: np.ndarray
) -> Dict[str, float]:
    """
    Compute stability metrics from robot state.
    
    Args:
        orientations: Array of orientations as quaternions (timesteps, 4)
        positions: Array of positions (timesteps, 3)
        
    Returns:
        Dictionary with stability metrics
    """
    # Convert quaternions to euler angles
    rotations = Rotation.from_quat(orientations)
    euler_angles = rotations.as_euler('xyz')
    
    rolls = euler_angles[:, 0]
    pitches = euler_angles[:, 1]
    yaws = euler_angles[:, 2]
    
    # Height stability
    heights = positions[:, 2]
    mean_height = float(np.mean(heights))
    height_std = float(np.std(heights))
    
    # Orientation stability
    roll_std = float(np.std(rolls))
    pitch_std = float(np.std(pitches))
    
    # Maximum deviation
    max_roll = float(np.max(np.abs(rolls)))
    max_pitch = float(np.max(np.abs(pitches)))
    
    metrics = {
        'mean_height': mean_height,
        'height_std': height_std,
        'roll_std': roll_std,
        'pitch_std': pitch_std,
        'max_roll': max_roll,
        'max_pitch': max_pitch,
        'yaw_drift': float(yaws[-1] - yaws[0]) if len(yaws) > 0 else 0.0
    }
    
    return metrics
