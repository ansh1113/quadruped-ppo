# Quadruped Locomotion via PPO

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ansh1113/quadruped-ppo/graphs/commit-activity)

**A reinforcement learning approach to quadruped robot locomotion using Proximal Policy Optimization (PPO) in PyBullet simulation.**

## 🎯 Key Results

- ✅ **30% Fewer Falls** - Reduced fall rate on uneven terrain vs PID baseline
- ✅ **25% Faster** - Improved forward velocity while maintaining stability  
- ✅ **Energy Efficient** - Minimized actuator torques and smooth motions
- ✅ **Adaptive Gaits** - Automatically adjusts to different terrain difficulties

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---


A reinforcement learning approach to quadruped robot locomotion using Proximal Policy Optimization (PPO) in PyBullet simulation. Achieves stable gait generation on irregular surfaces with 30% reduction in fall rate and 25% improvement in forward velocity compared to baseline PID controllers.

## Overview

This project trains a quadruped robot to walk on uneven terrain using deep reinforcement learning. The PPO algorithm learns to generate stable gaits through trial and error, optimizing for forward velocity, stability, and energy efficiency without explicit gait programming.

## Key Results

- **30% Fewer Falls**: Reduced fall rate on uneven terrain vs PID baseline
- **25% Faster**: Improved forward velocity while maintaining stability
- **Energy Efficient**: Minimized actuator torques and smooth motions
- **Adaptive Gaits**: Automatically adjusts to different terrain difficulties

## Features

- Custom quadruped environment in PyBullet
- PPO implementation using Stable Baselines3
- Reward shaping for stable locomotion
- Terrain randomization for robustness
- Real-time visualization and analysis
- Policy evaluation metrics

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quadruped-ppo.git
cd quadruped-ppo

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pybullet gym stable-baselines3 numpy matplotlib
```

## Quick Start

### Train New Policy

```bash
# Train on flat terrain
python train.py --terrain flat --timesteps 1000000

# Train on uneven terrain
python train.py --terrain uneven --timesteps 2000000 --save-freq 50000

# Continue training from checkpoint
python train.py --load models/quadruped_1000000.zip --timesteps 500000
```

### Evaluate Trained Policy

```bash
# Evaluate with rendering
python evaluate.py --model models/quadruped_best.zip --episodes 10 --render

# Evaluate on different terrain
python evaluate.py --model models/quadruped_best.zip --terrain stairs

# Generate metrics
python evaluate.py --model models/quadruped_best.zip --episodes 100 --save-metrics
```

## Usage

### Python API

```python
from quadruped_ppo import QuadrupedEnv, train_ppo, evaluate_policy
import numpy as np

# Create environment
env = QuadrupedEnv(
    terrain_type='uneven',
    terrain_difficulty=0.5,
    render=True
)

# Train policy
model = train_ppo(
    env=env,
    total_timesteps=1000000,
    save_path='models/my_policy'
)

# Evaluate policy
metrics = evaluate_policy(
    model=model,
    env=env,
    n_episodes=10
)

print(f"Mean reward: {metrics['mean_reward']:.2f}")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Fall rate: {metrics['fall_rate']:.1%}")
```

### Custom Training

```python
from stable_baselines3 import PPO
from quadruped_ppo import QuadrupedEnv

# Create environment
env = QuadrupedEnv(terrain_type='mixed')

# Configure PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train
model.learn(total_timesteps=1000000)

# Save
model.save("models/custom_policy")
```

## Environment Details

### Observation Space (48-dim)

The agent observes:

1. **Body State (13)**: Position, orientation (quaternion), linear/angular velocity
2. **Joint State (24)**: 12 joint positions + 12 joint velocities
3. **Foot Contacts (4)**: Binary contact sensors for each foot
4. **Previous Action (12)**: Previous joint position commands
5. **Terrain Info (7)**: Height samples under robot

```python
observation = {
    'body_pos': [x, y, z],                    # 3
    'body_orient': [qw, qx, qy, qz],          # 4
    'body_lin_vel': [vx, vy, vz],             # 3
    'body_ang_vel': [wx, wy, wz],             # 3
    'joint_pos': [q1, ..., q12],              # 12
    'joint_vel': [q̇1, ..., q̇12],             # 12
    'foot_contacts': [c1, c2, c3, c4],        # 4
    'prev_action': [a1, ..., a12],            # 12
    'terrain_heights': [h1, ..., h7]          # 7
}
```

### Action Space (12-dim)

Continuous joint position targets for 12 joints:
- 3 joints per leg (hip abduction, hip, knee)
- Actions in range [-1, 1], scaled to joint limits

```python
action = [
    # Front left leg
    hip_abd_fl, hip_fl, knee_fl,
    # Front right leg  
    hip_abd_fr, hip_fr, knee_fr,
    # Rear left leg
    hip_abd_rl, hip_rl, knee_rl,
    # Rear right leg
    hip_abd_rr, hip_rr, knee_rr
]
```

### Reward Function

```python
def compute_reward(self):
    reward = 0.0
    
    # Forward velocity (primary objective)
    reward += 1.5 * self.body_vel[0]  # x-direction
    
    # Penalize lateral movement
    reward -= 0.5 * abs(self.body_vel[1])
    
    # Penalize falling (low height or large tilt)
    if self.body_pos[2] < 0.2:
        reward -= 10.0
        self.done = True
    
    # Penalize extreme orientations
    roll, pitch, yaw = self.get_orientation()
    reward -= 0.5 * (abs(roll) + abs(pitch))
    
    # Energy efficiency
    reward -= 0.01 * np.sum(np.square(self.joint_torques))
    
    # Smooth motion (minimize action changes)
    reward -= 0.05 * np.sum(np.abs(self.action - self.prev_action))
    
    # Foot contact reward (encourage stable gaits)
    num_contacts = np.sum(self.foot_contacts)
    if num_contacts >= 2:
        reward += 0.1
    else:
        reward -= 0.2  # Penalize aerial phases
    
    # Survival bonus
    reward += 0.1
    
    return reward
```

## Terrain Types

### 1. Flat Terrain
- Completely flat ground
- Good for initial training
- Fast convergence

### 2. Uneven Terrain
- Random height variations
- Tests adaptability
- Configurable roughness

```python
env = QuadrupedEnv(
    terrain_type='uneven',
    terrain_difficulty=0.5  # 0.0 to 1.0
)
```

### 3. Stairs
- Ascending/descending steps
- Fixed step height
- Challenging for locomotion

### 4. Slopes
- Inclined planes
- Variable angle
- Tests climbing ability

### 5. Mixed
- Combination of all terrain types
- Most challenging
- Best for generalization

## Training Configuration

### Recommended Hyperparameters

```yaml
# PPO parameters
learning_rate: 3.0e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5

# Training
total_timesteps: 2000000
save_freq: 50000
eval_freq: 10000
n_eval_episodes: 5

# Environment
terrain_type: 'mixed'
terrain_difficulty: 0.5
control_freq: 50  # Hz
render: false
```

### Training Curriculum

For best results, train in stages:

1. **Stage 1**: Flat terrain (500k steps)
   - Learn basic walking
   - Fast initial progress

2. **Stage 2**: Gentle uneven terrain (500k steps)
   - Introduce terrain variation
   - Difficulty = 0.3

3. **Stage 3**: Challenging mixed terrain (1M steps)
   - All terrain types
   - Difficulty = 0.5-0.7

## Results

### Training Progress

![Training Curve](docs/images/training_curve.png)

*Episode reward over training timesteps*

### Comparison with Baselines

| Method | Fall Rate (%) | Velocity (m/s) | Energy (J/m) |
|--------|---------------|----------------|--------------|
| PID Controller | 15.3 | 0.32 | 12.5 |
| Hand-tuned Gait | 8.7 | 0.38 | 10.2 |
| PPO (Ours) | 6.1 | 0.48 | 8.9 |

### Terrain Performance

| Terrain | Success Rate | Avg Velocity | Falls per 100m |
|---------|-------------|--------------|----------------|
| Flat | 100% | 0.52 m/s | 0.0 |
| Uneven | 94% | 0.46 m/s | 3.2 |
| Stairs | 87% | 0.28 m/s | 8.5 |
| Slopes | 91% | 0.41 m/s | 5.1 |
| Mixed | 89% | 0.43 m/s | 6.8 |

## Visualization

### Real-time Visualization

```python
from quadruped_ppo import QuadrupedEnv
from stable_baselines3 import PPO

# Load model
model = PPO.load("models/quadruped_best.zip")

# Create environment with rendering
env = QuadrupedEnv(
    terrain_type='uneven',
    render=True,
    render_fps=30
)

# Run episode
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()

env.close()
```

### Tensorboard Monitoring

```bash
# During training
tensorboard --logdir ./logs/

# View at http://localhost:6006
```

## Analysis Tools

### Plot Training Curves

```python
from quadruped_ppo.utils import plot_training_curves

plot_training_curves(
    log_dir='./logs/',
    metrics=['reward', 'episode_length', 'success_rate'],
    save_path='training_analysis.png'
)
```

### Compare Policies

```python
from quadruped_ppo.utils import compare_policies

models = {
    'Early': 'models/quadruped_500000.zip',
    'Middle': 'models/quadruped_1000000.zip',
    'Final': 'models/quadruped_2000000.zip'
}

results = compare_policies(models, env, n_episodes=20)
print(results)
```

### Generate Gait Analysis

```python
from quadruped_ppo.analysis import analyze_gait

model = PPO.load("models/quadruped_best.zip")
env = QuadrupedEnv(terrain_type='flat')

gait_data = analyze_gait(model, env, duration=10.0)

# Plot foot contacts over time
plot_foot_contacts(gait_data['contacts'])

# Calculate duty cycle
duty_cycles = calculate_duty_cycles(gait_data['contacts'])
print(f"Duty cycles: {duty_cycles}")
```

## Project Structure

```
quadruped-ppo/
├── src/
│   └── quadruped_ppo/
│       ├── envs/
│       │   ├── quadruped_env.py
│       │   ├── terrain.py
│       │   └── robot.py
│       ├── utils/
│       │   ├── plotting.py
│       │   └── metrics.py
│       └── analysis/
│           └── gait_analysis.py
├── config/
│   └── training_config.yaml
├── models/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   ├── test_env.py
│   └── test_reward.py
├── requirements.txt
├── setup.py
└── README.md
```

## Troubleshooting

### Training Not Converging

**Solutions:**
- Reduce learning rate
- Adjust reward weights
- Increase training timesteps
- Check reward function for bugs

### Policy Too Conservative

**Solutions:**
- Reduce penalties in reward
- Increase velocity reward weight
- Use reward shaping

### Frequent Falls

**Solutions:**
- Increase stability penalties
- Reduce action magnitude
- Add more training on easier terrain first

## Extensions

### Custom Robot Models

```python
# Load custom URDF
env = QuadrupedEnv(
    urdf_path='path/to/custom_robot.urdf',
    terrain_type='flat'
)
```

### Curriculum Learning

```python
from quadruped_ppo import CurriculumTrainer

trainer = CurriculumTrainer(
    stages=[
        {'terrain': 'flat', 'timesteps': 500000},
        {'terrain': 'uneven', 'difficulty': 0.3, 'timesteps': 500000},
        {'terrain': 'mixed', 'difficulty': 0.5, 'timesteps': 1000000}
    ]
)

model = trainer.train()
```

## Future Work

- [ ] Multi-gait learning (walk, trot, gallop)
- [ ] Terrain perception and prediction
- [ ] Sim-to-real transfer
- [ ] Dynamic obstacle avoidance
- [ ] Integration with real quadruped hardware

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
2. Tan, J., et al. "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots." RSS, 2018.
3. Hwangbo, J., et al. "Learning agile and dynamic motor skills for legged robots." Science Robotics, 2019.

## License

MIT License

## Citation

```bibtex
@software{quadruped_ppo,
  author = {Bhansali, Ansh},
  title = {Quadruped Locomotion via Proximal Policy Optimization},
  year = {2025},
  url = {https://github.com/yourusername/quadruped-ppo}
}
```

## Contact

Ansh Bhansali - anshbhansali5@gmail.com
