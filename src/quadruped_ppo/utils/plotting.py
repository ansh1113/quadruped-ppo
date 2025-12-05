"""
Plotting utilities for visualization and analysis.

This module provides functions to create various plots for training
analysis and gait visualization.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def plot_training_curves(
    log_dir: str,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training curves from tensorboard logs.
    
    Args:
        log_dir: Directory containing tensorboard logs
        metrics: List of metrics to plot (default: ['reward', 'ep_len_mean'])
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if metrics is None:
        metrics = ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # Load tensorboard data
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Get available tags
        available_tags = ea.Tags()['scalars']
        
        # Plot each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in available_tags:
                data = ea.Scalars(metric)
                steps = [d.step for d in data]
                values = [d.value for d in data]
                
                axes[idx].plot(steps, values, linewidth=2)
                axes[idx].set_xlabel('Timesteps')
                axes[idx].set_ylabel(metric.split('/')[-1])
                axes[idx].set_title(f'{metric}')
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, f'Metric "{metric}" not found',
                             ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    except ImportError:
        print("Warning: tensorboard not installed. Cannot plot training curves.")
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def plot_gait_pattern(
    foot_contacts: np.ndarray,
    dt: float = 0.02,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 'Gait Pattern'
) -> None:
    """
    Visualize gait pattern from foot contact data.
    
    Args:
        foot_contacts: Array of shape (timesteps, 4) with binary contact data
        dt: Time step in seconds
        save_path: Path to save the plot
        show: Whether to display the plot
        title: Plot title
    """
    num_steps = foot_contacts.shape[0]
    time = np.arange(num_steps) * dt
    
    foot_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, name in enumerate(foot_names):
        contacts = foot_contacts[:, i]
        
        # Plot contact periods as filled regions
        y_pos = i * 1.5
        ax.fill_between(time, y_pos, y_pos + contacts, 
                        alpha=0.7, color=colors[i], label=name)
        
        # Add baseline
        ax.plot(time, np.ones_like(time) * y_pos, 
               color=colors[i], linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Foot', fontsize=12)
    ax.set_yticks([i * 1.5 + 0.5 for i in range(4)])
    ax.set_yticklabels(foot_names)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-0.5, 6.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gait pattern saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_foot_contacts(
    foot_contacts: np.ndarray,
    dt: float = 0.02,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot foot contact timing diagram.
    
    Args:
        foot_contacts: Array of shape (timesteps, 4) with binary contact data
        dt: Time step in seconds
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    num_steps = foot_contacts.shape[0]
    time = np.arange(num_steps) * dt
    
    foot_names = ['FL', 'FR', 'RL', 'RR']
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    for i, (ax, name) in enumerate(zip(axes, foot_names)):
        contacts = foot_contacts[:, i]
        ax.fill_between(time, 0, contacts, alpha=0.6, label=name)
        ax.set_ylabel(name, fontsize=12, rotation=0, labelpad=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Air', 'Ground'])
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_title('Foot Contact Timeline', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Foot contacts plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory(
    positions: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot robot trajectory in 3D.
    
    Args:
        positions: Array of shape (timesteps, 3) with x, y, z positions
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
           linewidth=2, alpha=0.7, label='Trajectory')
    
    # Mark start and end
    ax.scatter(*positions[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*positions[-1], color='red', s=100, marker='x', label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Robot Trajectory', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rewards(
    rewards: List[float],
    window_size: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot reward progression with moving average.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    episodes = np.arange(1, len(rewards) + 1)
    
    # Compute moving average
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    moving_avg_episodes = episodes[window_size-1:]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    ax.plot(moving_avg_episodes, moving_avg, linewidth=2, 
           color='red', label=f'{window_size}-Episode Moving Average')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rewards plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'mean_reward',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of different models/policies.
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        metric: Metric to compare
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    errors = [results[name].get(f'std_{metric.split("_")[1]}', 0) for name in names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_report(
    results: Dict[str, Any],
    plots_dir: str = 'plots',
    report_path: str = 'report.md'
) -> None:
    """
    Create a markdown report with plots and metrics.
    
    Args:
        results: Dictionary containing evaluation results
        plots_dir: Directory to save plots
        report_path: Path to save the markdown report
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Quadruped Locomotion Evaluation Report\n\n")
        
        f.write("## Performance Metrics\n\n")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n")
        
        f.write("\n## Plots\n\n")
        
        # List any plots that were generated
        if os.path.exists(plots_dir):
            plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
            for plot_file in plot_files:
                f.write(f"### {plot_file.replace('_', ' ').replace('.png', '').title()}\n\n")
                f.write(f"![{plot_file}]({os.path.join(plots_dir, plot_file)})\n\n")
    
    print(f"Report saved to {report_path}")
