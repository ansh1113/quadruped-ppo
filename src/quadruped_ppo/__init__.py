"""
Quadruped PPO - Quadruped Locomotion via Proximal Policy Optimization.

A reinforcement learning approach to train quadruped robots to walk on
various terrains using PPO in PyBullet simulation.
"""

__version__ = '0.1.0'
__author__ = 'Ansh Bhansali'
__email__ = 'anshbhansali5@gmail.com'

from .envs.quadruped_env import QuadrupedEnv
from .envs.terrain import TerrainGenerator

__all__ = ['QuadrupedEnv', 'TerrainGenerator']
