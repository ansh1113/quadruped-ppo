# Development Roadmap

This document outlines a practical path forward to complete the quadruped-ppo project implementation.

---

## 🎯 Current Status

- ✅ Documentation (excellent)
- ✅ Project structure (good)
- ⚠️ Environment (skeleton only)
- ⚠️ Training pipeline (ready but untested)
- ❌ Tests (none)
- ❌ Results (none)

**Goal:** Close the gap between documentation and implementation.

---

## Phase 1: Foundation (Week 1)

### 1.1 Documentation Cleanup
**Effort:** 2-3 hours

```markdown
# Add to README.md after the title:

> **⚠️ Project Status:** This project is under active development. 
> Core environment implementation is in progress. See ROADMAP.md for details.

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Environment Structure | ✅ Complete | Basic Gym interface |
| PPO Training | ✅ Complete | Via Stable Baselines3 |
| Quadruped Model | 🚧 In Progress | Currently using placeholder |
| Flat Terrain | ✅ Complete | Basic implementation |
| Uneven Terrain | 📋 Planned | Q1 2025 |
| Complete Observations | 🚧 In Progress | 13/48 dims implemented |
| Contact Sensors | 📋 Planned | Q1 2025 |
| Analysis Tools | 📋 Planned | Q2 2025 |
```

**Tasks:**
- [ ] Add implementation status section to README
- [ ] Create ROADMAP.md (use this document)
- [ ] Add CHANGELOG.md
- [ ] Update badges to reflect current status
- [ ] Remove or clearly mark aspirational performance claims

---

### 1.2 Directory Structure
**Effort:** 30 minutes

```bash
# Create missing directories
mkdir -p tests config models logs docs/images src/quadruped_ppo/utils src/quadruped_ppo/analysis

# Create placeholder files
touch tests/__init__.py
touch tests/test_env.py
touch tests/test_reward.py
touch src/quadruped_ppo/utils/__init__.py
touch src/quadruped_ppo/utils/metrics.py
touch src/quadruped_ppo/utils/plotting.py
touch src/quadruped_ppo/analysis/__init__.py
touch src/quadruped_ppo/analysis/gait_analysis.py
touch config/training_config.yaml
touch CHANGELOG.md
touch ROADMAP.md
```

**Tasks:**
- [ ] Create directory structure
- [ ] Add .gitkeep files to empty dirs (models/, logs/)
- [ ] Create __init__.py files for packages
- [ ] Update .gitignore if needed

---

### 1.3 Cleanup
**Effort:** 1 hour

**Tasks:**
- [ ] Decide on train.py location (root vs scripts/)
- [ ] Remove duplicate or document difference
- [ ] Document or remove generate_code.py
- [ ] Run Black on entire codebase: `black .`
- [ ] Fix any flake8 errors: `flake8 src/ tests/`
- [ ] Remove README.md.backup

---

### 1.4 Basic Test Suite
**Effort:** 3-4 hours

Create `tests/test_env.py`:

```python
import pytest
import numpy as np
from quadruped_ppo import QuadrupedEnv


class TestQuadrupedEnv:
    """Test suite for QuadrupedEnv."""
    
    def test_env_creation(self):
        """Test environment can be created."""
        env = QuadrupedEnv(render=False)
        assert env is not None
        env.close()
    
    def test_observation_space(self):
        """Test observation space shape."""
        env = QuadrupedEnv(render=False)
        assert env.observation_space.shape == (48,)
        env.close()
    
    def test_action_space(self):
        """Test action space shape."""
        env = QuadrupedEnv(render=False)
        assert env.action_space.shape == (12,)
        env.close()
    
    def test_reset(self):
        """Test environment reset."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        assert obs.shape == (48,)
        assert np.isfinite(obs).all()
        env.close()
    
    def test_step(self):
        """Test environment step."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = np.zeros(12)
        obs, reward, done, info = env.step(action)
        
        assert obs.shape == (48,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_episode_termination(self):
        """Test episode terminates correctly."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Run for max steps
        for _ in range(1001):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                break
        
        assert done
        env.close()
    
    def test_reward_computation(self):
        """Test reward function returns valid values."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = np.zeros(12)
        _, reward, _, _ = env.step(action)
        
        assert np.isfinite(reward)
        assert isinstance(reward, (int, float))
        env.close()


@pytest.mark.parametrize("terrain_type", ["flat", "uneven", "stairs", "slopes", "mixed"])
def test_terrain_types(terrain_type):
    """Test different terrain types can be created."""
    env = QuadrupedEnv(terrain_type=terrain_type, render=False)
    obs = env.reset()
    assert obs.shape == (48,)
    env.close()
```

Create `tests/test_reward.py`:

```python
import pytest
import numpy as np
from quadruped_ppo.envs.quadruped_env import QuadrupedEnv


class TestRewardFunction:
    """Test reward function components."""
    
    def test_reward_forward_velocity(self):
        """Forward velocity should increase reward."""
        env = QuadrupedEnv(render=False)
        # Test would need mock of PyBullet state
        # Placeholder for now
        pass
    
    def test_reward_falling_penalty(self):
        """Falling should give large negative reward."""
        # Placeholder - implement when environment is complete
        pass
```

**Tasks:**
- [ ] Create test files as shown above
- [ ] Install pytest: Add to requirements-dev.txt
- [ ] Run tests: `pytest tests/ -v`
- [ ] Add test running to CI

---

## Phase 2: Core Implementation (Week 2-3)

### 2.1 Get/Create Quadruped URDF
**Effort:** 4-8 hours (depending on approach)

**Option A: Use Existing Model (Faster)**
- Find open-source quadruped URDF (ANYmal, Spot, MIT Cheetah)
- Adapt joint names and limits for your code
- Credit source in README

**Option B: Create Simple Model (Learning)**
- Create basic box-leg quadruped in PyBullet
- Define 12 joints (3 per leg)
- Set appropriate joint limits

Example simple URDF structure:
```xml
<!-- quadruped_simple.urdf -->
<robot name="quadruped">
  <link name="base">
    <visual>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.4 0.2 0.1"/></geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Repeat for each leg: hip_abd, hip, knee -->
  <!-- 4 legs × 3 joints = 12 joints -->
</robot>
```

**Tasks:**
- [ ] Obtain or create URDF file
- [ ] Place in `src/quadruped_ppo/assets/` directory
- [ ] Update environment to load proper URDF
- [ ] Test that robot loads correctly
- [ ] Document joint layout in README

---

### 2.2 Implement Joint Control
**Effort:** 4-6 hours

Update `quadruped_env.py`:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Joint configuration
    self.joint_ids = []  # Will store joint indices
    self.joint_limits = []  # Store (min, max) for each joint
    
def reset(self):
    # ... existing code ...
    
    # Get joint information
    self.joint_ids = []
    self.joint_limits = []
    for j in range(p.getNumJoints(self.robot_id)):
        joint_info = p.getJointInfo(self.robot_id, j)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            self.joint_ids.append(j)
            self.joint_limits.append((joint_info[8], joint_info[9]))
    
    assert len(self.joint_ids) == 12, f"Expected 12 joints, found {len(self.joint_ids)}"
    
    return self._get_observation()

def step(self, action):
    action = np.clip(action, -1.0, 1.0)
    
    # Convert normalized actions [-1, 1] to actual joint positions
    joint_positions = []
    for i, (action_val, (joint_min, joint_max)) in enumerate(zip(action, self.joint_limits)):
        # Map from [-1, 1] to [joint_min, joint_max]
        pos = joint_min + (action_val + 1.0) * 0.5 * (joint_max - joint_min)
        joint_positions.append(pos)
    
    # Apply joint control
    p.setJointMotorControlArray(
        bodyIndex=self.robot_id,
        jointIndices=self.joint_ids,
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_positions,
        forces=[50.0] * len(self.joint_ids)  # Max force per joint
    )
    
    # Simulate for control_steps
    for _ in range(self.control_steps):
        p.stepSimulation()
    
    # ... rest of step function ...
```

**Tasks:**
- [ ] Implement joint discovery on reset
- [ ] Implement joint position control
- [ ] Test that robot moves when actions applied
- [ ] Tune force/torque limits
- [ ] Add joint damping if needed

---

### 2.3 Complete Observation Space
**Effort:** 3-4 hours

```python
def _get_observation(self):
    """Get complete 48-dim observation."""
    obs = np.zeros(48, dtype=np.float32)
    
    if self.robot_id is None:
        return obs
    
    # Body state (13 dims)
    pos, orn = p.getBasePositionAndOrientation(self.robot_id)
    vel, ang_vel = p.getBaseVelocity(self.robot_id)
    
    idx = 0
    obs[idx:idx+3] = pos           # Body position (3)
    idx += 3
    obs[idx:idx+4] = orn           # Orientation quaternion (4)
    idx += 4
    obs[idx:idx+3] = vel           # Linear velocity (3)
    idx += 3
    obs[idx:idx+3] = ang_vel       # Angular velocity (3)
    idx += 3
    
    # Joint states (24 dims: 12 positions + 12 velocities)
    joint_states = p.getJointStates(self.robot_id, self.joint_ids)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    
    obs[idx:idx+12] = joint_positions   # Joint positions (12)
    idx += 12
    obs[idx:idx+12] = joint_velocities  # Joint velocities (12)
    idx += 12
    
    # Foot contacts (4 dims)
    foot_links = self._get_foot_link_ids()
    foot_contacts = []
    for foot_link in foot_links:
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            linkIndexA=foot_link
        )
        foot_contacts.append(1.0 if len(contact_points) > 0 else 0.0)
    
    obs[idx:idx+4] = foot_contacts  # Foot contacts (4)
    idx += 4
    
    # Previous action (12 dims)
    obs[idx:idx+12] = self.prev_action  # Previous action (12)
    idx += 12
    
    # Terrain info (7 dims)
    # Sample height at points around robot
    terrain_heights = self._sample_terrain_heights(pos[:2])
    obs[idx:idx+7] = terrain_heights  # Terrain heights (7)
    
    assert idx + 7 == 48, f"Observation size mismatch: {idx + 7} != 48"
    
    return obs

def _get_foot_link_ids(self):
    """Get link IDs for foot links."""
    # This depends on your URDF structure
    # For a typical quadruped, feet are at the end of each leg
    # You may need to adjust based on your model
    foot_names = ['foot_fl', 'foot_fr', 'foot_rl', 'foot_rr']
    foot_ids = []
    
    for j in range(p.getNumJoints(self.robot_id)):
        joint_info = p.getJointInfo(self.robot_id, j)
        link_name = joint_info[12].decode('utf-8')
        if any(foot in link_name.lower() for foot in ['foot', 'toe']):
            foot_ids.append(j)
    
    # Fallback: if can't find by name, use last joint of each leg
    if len(foot_ids) != 4:
        foot_ids = [2, 5, 8, 11]  # Assuming 3 joints per leg
    
    return foot_ids

def _sample_terrain_heights(self, robot_xy):
    """Sample terrain heights around robot."""
    # For now, return zeros (flat terrain)
    # TODO: Implement actual height sampling from terrain
    return np.zeros(7, dtype=np.float32)
```

**Tasks:**
- [ ] Implement complete observation vector
- [ ] Test observation shape is correct
- [ ] Test all values are finite
- [ ] Add observation normalization if needed
- [ ] Update tests to verify all 48 dims

---

### 2.4 Implement Contact Sensors
**Effort:** 2-3 hours

Already partially covered in observation space above. Additional tasks:

**Tasks:**
- [ ] Verify contact detection works
- [ ] Test with robot stepping
- [ ] Add contact visualization in GUI mode
- [ ] Consider adding contact forces to observations (optional)

---

## Phase 3: Terrain Generation (Week 4)

### 3.1 Heightfield Terrain
**Effort:** 6-8 hours

```python
# Add to quadruped_env.py or create terrain.py

def _create_terrain(self):
    """Create terrain based on type."""
    if self.terrain_type == 'flat':
        return p.loadURDF("plane.urdf")
    
    elif self.terrain_type == 'uneven':
        return self._create_heightfield_terrain()
    
    elif self.terrain_type == 'stairs':
        return self._create_stairs_terrain()
    
    # ... other types ...

def _create_heightfield_terrain(self):
    """Create random heightfield terrain."""
    terrain_size = 20  # meters
    mesh_scale = 0.1   # resolution
    grid_size = int(terrain_size / mesh_scale)
    
    # Generate random heights
    heights = np.random.uniform(
        -self.terrain_difficulty * 0.1,
        self.terrain_difficulty * 0.1,
        size=(grid_size, grid_size)
    )
    
    # Apply smoothing
    from scipy.ndimage import gaussian_filter
    heights = gaussian_filter(heights, sigma=2.0)
    
    # Create collision shape
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[mesh_scale, mesh_scale, 1],
        heightfieldData=heights.flatten(),
        numHeightfieldRows=grid_size,
        numHeightfieldColumns=grid_size
    )
    
    # Create terrain body
    terrain_id = p.createMultiBody(0, terrain_shape)
    p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
    
    return terrain_id
```

**Tasks:**
- [ ] Implement heightfield generation
- [ ] Test uneven terrain
- [ ] Adjust difficulty parameter
- [ ] Verify robot doesn't fall through terrain
- [ ] Add scipy to requirements if using filtering

---

### 3.2 Other Terrain Types
**Effort:** 4-6 hours

```python
def _create_stairs_terrain(self):
    """Create stairs terrain."""
    step_height = 0.05 + self.terrain_difficulty * 0.1
    step_length = 0.3
    num_steps = 10
    
    stairs_ids = []
    for i in range(num_steps):
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, step_height * i])
        body = p.createMultiBody(
            0, box,
            basePosition=[i * step_length, 0, step_height * i]
        )
        stairs_ids.append(body)
    
    return stairs_ids[0]  # Return first step as terrain ID

def _create_slopes_terrain(self):
    """Create sloped terrain."""
    angle = self.terrain_difficulty * 0.3  # Up to ~17 degrees
    # Implementation similar to stairs but with continuous slope
    pass
```

**Tasks:**
- [ ] Implement stairs terrain
- [ ] Implement slopes terrain
- [ ] Implement mixed terrain (random selection)
- [ ] Test all terrain types
- [ ] Update tests for terrain types

---

## Phase 4: Training & Validation (Week 5)

### 4.1 First Training Run
**Effort:** 1 day (mostly waiting for training)

```bash
# Start training on flat terrain
python train.py --terrain flat --timesteps 500000 --save-freq 50000

# Monitor with tensorboard
tensorboard --logdir logs/
```

**Tasks:**
- [ ] Run training for 500k steps on flat terrain
- [ ] Monitor tensorboard logs
- [ ] Verify model checkpoints are saved
- [ ] Check for any errors or crashes
- [ ] Document hyperparameters used

---

### 4.2 Evaluation & Metrics
**Effort:** 4-6 hours

Create `src/quadruped_ppo/utils/metrics.py`:

```python
import numpy as np
from typing import Dict, List

def evaluate_policy(model, env, n_episodes=10) -> Dict[str, float]:
    """Evaluate a trained policy."""
    episode_rewards = []
    episode_lengths = []
    falls = 0
    distances = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        start_pos = obs[0]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if info.get('fell', False):
                falls += 1
        
        end_pos = obs[0]
        distances.append(end_pos - start_pos)
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_distance': np.mean(distances),
        'fall_rate': falls / n_episodes,
        'success_rate': 1.0 - (falls / n_episodes)
    }
```

**Tasks:**
- [ ] Implement evaluation metrics
- [ ] Run evaluation on trained model
- [ ] Document actual results
- [ ] Compare with README claims
- [ ] Update README with real numbers

---

### 4.3 Visualization
**Effort:** 3-4 hours

```python
# src/quadruped_ppo/utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curve(log_dir, save_path=None):
    """Plot training curves from tensorboard logs."""
    from tensorboard.backend.event_processing import event_accumulator
    
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get episode rewards
    rewards = ea.Scalars('rollout/ep_rew_mean')
    steps = [r.step for r in rewards]
    values = [r.value for r in rewards]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_gait_pattern(foot_contacts, save_path=None):
    """Visualize gait pattern from foot contacts."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    foot_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    for i, name in enumerate(foot_names):
        contacts = foot_contacts[:, i]
        times = np.arange(len(contacts)) * 0.02  # Assuming 50Hz
        ax.plot(times, contacts + i*1.2, label=name)
        ax.fill_between(times, i*1.2, contacts + i*1.2, alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Foot')
    ax.set_yticks([0.6, 1.8, 3.0, 4.2])
    ax.set_yticklabels(foot_names)
    ax.legend()
    ax.set_title('Gait Pattern')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

**Tasks:**
- [ ] Implement plotting utilities
- [ ] Generate training curve plot
- [ ] Create gait visualization
- [ ] Save plots to docs/images/
- [ ] Update README with actual plots

---

## Phase 5: Polish & Documentation (Week 6)

### 5.1 Update Documentation
**Effort:** 4-6 hours

**Tasks:**
- [ ] Update README with actual results
- [ ] Add "Getting Started" video or GIF
- [ ] Document all configuration options
- [ ] Create examples/ directory with notebooks
- [ ] Write tutorial: "Training Your First Quadruped"
- [ ] Add FAQ section
- [ ] Update installation instructions
- [ ] Add troubleshooting for common issues

---

### 5.2 Code Quality
**Effort:** 3-4 hours

**Tasks:**
- [ ] Add type hints to all functions
- [ ] Improve docstrings (use Google/NumPy style)
- [ ] Run pylint and fix issues
- [ ] Add pre-commit hooks
- [ ] Set up code coverage reporting
- [ ] Aim for >80% test coverage
- [ ] Add docstring examples (doctest)

---

### 5.3 GitHub Repository
**Effort:** 2-3 hours

Create issue templates:

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: File a bug report
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear description of the bug
    validations:
      required: true
  # ... more fields ...
```

**Tasks:**
- [ ] Create issue templates (bug, feature, question)
- [ ] Create PR template
- [ ] Add CONTRIBUTING.md
- [ ] Add CODE_OF_CONDUCT.md
- [ ] Create GitHub project board
- [ ] Add useful labels
- [ ] Set up GitHub Discussions
- [ ] Add SECURITY.md

---

### 5.4 Release Preparation
**Effort:** 2-3 hours

**Tasks:**
- [ ] Tag version v0.1.0
- [ ] Create GitHub release
- [ ] Upload pretrained model as release asset
- [ ] Write release notes
- [ ] Update version in setup.py
- [ ] Create DOI with Zenodo (optional)
- [ ] Submit to relevant communities (Reddit, forums)

---

## Phase 6: Advanced Features (Month 2+)

### 6.1 Analysis Tools
- Implement gait analysis
- Add stability metrics
- Create energy efficiency calculator
- Add trajectory visualization

### 6.2 Advanced Training
- Implement curriculum learning
- Add domain randomization
- Support for distributed training
- Hyperparameter optimization with Optuna

### 6.3 Sim-to-Real
- Domain randomization enhancements
- Reality gap analysis
- Hardware interface (ROS2)
- Real robot testing framework

---

## Quick Wins Checklist

Things you can do RIGHT NOW (< 1 hour each):

- [ ] Add "Project Status" section to README
- [ ] Create missing directories (tests, config, etc.)
- [ ] Run Black on codebase: `black .`
- [ ] Create requirements-dev.txt
- [ ] Add basic .pytest.ini configuration
- [ ] Create CHANGELOG.md
- [ ] Remove README.md.backup
- [ ] Add shields.io badges for tests passing
- [ ] Create a simple issue template
- [ ] Star/Watch similar projects for inspiration

---

## Success Metrics

Track these metrics to measure progress:

| Metric | Current | Target (Phase 4) | Target (Phase 6) |
|--------|---------|------------------|------------------|
| Test Coverage | 0% | 70% | 85% |
| Observation Dims | 13/48 | 48/48 | 48/48 |
| Terrain Types | 1/5 | 3/5 | 5/5 |
| Training Success | N/A | Works on flat | Works on all |
| Documentation % | 95% | 100% | 100% |
| GitHub Stars | ? | ? | ? |
| Contributors | 1 | 1-2 | 3-5 |

---

## Resources & References

### Learning Materials
- Stable Baselines3 docs: https://stable-baselines3.readthedocs.io/
- PyBullet Quickstart: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/
- PPO Paper: https://arxiv.org/abs/1707.06347

### Similar Projects for Reference
- Legged Gym: https://github.com/leggedrobotics/legged_gym
- Isaac Gym: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
- PyBullet RL: https://github.com/bulletphysics/bullet3

### URDF Resources
- ANYmal URDF: https://github.com/ANYbotics/anymal_c_simple_description
- Spot URDF: https://github.com/chvmp/robots
- Create URDF: http://wiki.ros.org/urdf/Tutorials

---

## Getting Help

If you get stuck:

1. Check PyBullet forums: https://pybullet.org/wordpress/
2. Stable Baselines3 Discord: https://discord.com/invite/UM8Z5FzPVN
3. RL Discord: https://discord.gg/xhfNqQv
4. Stack Overflow tags: `pybullet`, `reinforcement-learning`, `stable-baselines3`

---

## Summary Timeline

**Conservative Estimate:**
- Week 1: Foundation & cleanup
- Week 2-3: Core implementation
- Week 4: Terrain & training
- Week 5: Validation & results
- Week 6: Polish & documentation
- **Total: 6 weeks to functional v0.1.0**

**Aggressive Estimate:**
- 2-3 weeks to functional v0.1.0 (if working full-time)

---

**Good luck! This is a great project with solid foundations. With focused effort, you can absolutely achieve the vision outlined in your README. 🚀**

---

_Last Updated: December 5, 2025_  
_For questions about this roadmap, create an issue on GitHub._
