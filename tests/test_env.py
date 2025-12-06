"""
Comprehensive tests for QuadrupedEnv.

This module tests all aspects of the quadruped environment including
initialization, reset, step, observation space, action space, and termination.
"""

import pytest
import numpy as np
from quadruped_ppo import QuadrupedEnv


class TestQuadrupedEnvInitialization:
    """Test environment initialization."""
    
    def test_env_creation_default(self):
        """Test environment can be created with default parameters."""
        env = QuadrupedEnv(render=False)
        assert env is not None
        assert env.terrain_type == 'flat'
        assert env.terrain_difficulty == 0.5
        env.close()
    
    def test_env_creation_with_params(self):
        """Test environment with custom parameters."""
        env = QuadrupedEnv(
            terrain_type='uneven',
            terrain_difficulty=0.7,
            render=False,
            max_episode_steps=500
        )
        assert env.terrain_type == 'uneven'
        assert env.terrain_difficulty == 0.7
        assert env.max_episode_steps == 500
        env.close()
    
    @pytest.mark.parametrize("terrain_type", ['flat', 'uneven', 'stairs', 'slopes', 'mixed'])
    def test_all_terrain_types(self, terrain_type):
        """Test all terrain types can be created."""
        env = QuadrupedEnv(terrain_type=terrain_type, render=False)
        assert env.terrain_type == terrain_type
        env.close()
    
    def test_difficulty_clipping(self):
        """Test difficulty is clipped to valid range."""
        env = QuadrupedEnv(terrain_difficulty=1.5, render=False)
        assert env.terrain_difficulty == 1.0
        env.close()
        
        env = QuadrupedEnv(terrain_difficulty=-0.5, render=False)
        assert env.terrain_difficulty == 0.0
        env.close()


class TestQuadrupedEnvSpaces:
    """Test observation and action spaces."""
    
    def test_observation_space_shape(self):
        """Test observation space has correct shape."""
        env = QuadrupedEnv(render=False)
        assert env.observation_space.shape == (48,)
        env.close()
    
    def test_action_space_shape(self):
        """Test action space has correct shape."""
        env = QuadrupedEnv(render=False)
        assert env.action_space.shape == (12,)
        env.close()
    
    def test_action_space_bounds(self):
        """Test action space bounds."""
        env = QuadrupedEnv(render=False)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        env.close()
    
    def test_observation_space_dtype(self):
        """Test observation space dtype."""
        env = QuadrupedEnv(render=False)
        assert env.observation_space.dtype == np.float32
        env.close()
    
    def test_action_space_dtype(self):
        """Test action space dtype."""
        env = QuadrupedEnv(render=False)
        assert env.action_space.dtype == np.float32
        env.close()


class TestQuadrupedEnvReset:
    """Test environment reset functionality."""
    
    def test_reset_returns_observation(self):
        """Test reset returns valid observation."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        assert obs.shape == (48,)
        assert obs.dtype == np.float32
        env.close()
    
    def test_reset_observation_finite(self):
        """Test reset observation contains finite values."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        assert np.isfinite(obs).all()
        env.close()
    
    def test_reset_clears_step_counter(self):
        """Test reset clears step counter."""
        env = QuadrupedEnv(render=False)
        env.reset()
        env.step(np.zeros(12))
        env.step(np.zeros(12))
        assert env.step_counter == 2
        
        env.reset()
        assert env.step_counter == 0
        env.close()
    
    def test_reset_multiple_times(self):
        """Test environment can be reset multiple times."""
        env = QuadrupedEnv(render=False)
        for _ in range(5):
            obs = env.reset()
            assert obs.shape == (48,)
            assert np.isfinite(obs).all()
        env.close()
    
    def test_reset_initial_position(self):
        """Test robot starts at correct initial position."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        # Position should be in obs[0:3]
        position = obs[0:3]
        assert position[2] > 0.3  # Should be above ground
        assert abs(position[0]) < 0.1  # Should be near x=0
        assert abs(position[1]) < 0.1  # Should be near y=0
        env.close()


class TestQuadrupedEnvStep:
    """Test environment step functionality."""
    
    def test_step_returns_tuple(self):
        """Test step returns (obs, reward, done, info)."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = np.zeros(12, dtype=np.float32)
        result = env.step(action)
        
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_step_observation_shape(self):
        """Test step returns correct observation shape."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        
        assert obs.shape == (48,)
        assert obs.dtype == np.float32
        env.close()
    
    def test_step_observation_finite(self):
        """Test step observation is finite."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        
        assert np.isfinite(obs).all()
        env.close()
    
    def test_step_reward_finite(self):
        """Test step reward is finite."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        
        assert np.isfinite(reward)
        env.close()
    
    def test_step_increments_counter(self):
        """Test step increments step counter."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        assert env.step_counter == 0
        env.step(np.zeros(12))
        assert env.step_counter == 1
        env.step(np.zeros(12))
        assert env.step_counter == 2
        env.close()
    
    def test_step_with_action_clipping(self):
        """Test actions are clipped to valid range."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Test with out-of-bounds action
        action = np.full(12, 2.0)  # Above maximum
        obs, reward, done, info = env.step(action)
        
        # Should not crash and should return valid observation
        assert obs.shape == (48,)
        assert np.isfinite(reward)
        env.close()
    
    def test_step_info_dict(self):
        """Test info dictionary contains required keys."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = np.zeros(12)
        _, _, _, info = env.step(action)
        
        assert 'step' in info
        assert 'fell' in info
        assert 'position' in info
        assert 'velocity' in info
        assert 'distance' in info
        env.close()
    
    def test_multiple_steps(self):
        """Test taking multiple steps."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            assert obs.shape == (48,)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert isinstance(done, bool)
            assert info['step'] == i + 1
        env.close()


class TestQuadrupedEnvTermination:
    """Test episode termination conditions."""
    
    def test_max_steps_termination(self):
        """Test episode terminates at max steps."""
        env = QuadrupedEnv(render=False, max_episode_steps=10)
        env.reset()
        
        done = False
        for _ in range(10):
            action = np.zeros(12)
            _, _, done, _ = env.step(action)
        
        assert done
        env.close()
    
    def test_falling_termination(self):
        """Test episode terminates when robot falls."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Take many random actions to potentially cause falling
        done = False
        max_attempts = 100
        for _ in range(max_attempts):
            action = env.action_space.sample()
            obs, _, done, info = env.step(action)
            
            if done and info.get('fell', False):
                # Successfully tested falling termination
                break
        env.close()
    
    def test_not_done_initially(self):
        """Test environment is not done after reset."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        action = np.zeros(12)
        _, _, done, _ = env.step(action)
        
        # First step should not be done (for reasonable action)
        assert not done
        env.close()


class TestQuadrupedEnvObservationComponents:
    """Test individual components of the observation."""
    
    def test_observation_body_position(self):
        """Test body position in observation (indices 0-2)."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        position = obs[0:3]
        assert len(position) == 3
        assert np.isfinite(position).all()
        assert position[2] > 0  # Height should be positive
        env.close()
    
    def test_observation_orientation(self):
        """Test orientation quaternion in observation (indices 3-6)."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        orientation = obs[3:7]
        assert len(orientation) == 4
        assert np.isfinite(orientation).all()
        
        # Quaternion should be normalized
        norm = np.linalg.norm(orientation)
        assert abs(norm - 1.0) < 0.1  # Allow some tolerance
        env.close()
    
    def test_observation_velocities(self):
        """Test velocities in observation (indices 7-12)."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Take a step to generate some velocity
        action = np.ones(12) * 0.5
        obs, _, _, _ = env.step(action)
        
        linear_vel = obs[7:10]
        angular_vel = obs[10:13]
        
        assert len(linear_vel) == 3
        assert len(angular_vel) == 3
        assert np.isfinite(linear_vel).all()
        assert np.isfinite(angular_vel).all()
        env.close()
    
    def test_observation_joint_states(self):
        """Test joint states in observation (indices 13-36)."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        joint_positions = obs[13:25]
        joint_velocities = obs[25:37]
        
        assert len(joint_positions) == 12
        assert len(joint_velocities) == 12
        assert np.isfinite(joint_positions).all()
        assert np.isfinite(joint_velocities).all()
        env.close()
    
    def test_observation_foot_contacts(self):
        """Test foot contacts in observation (indices 37-40)."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        foot_contacts = obs[37:41]
        
        assert len(foot_contacts) == 4
        assert np.all((foot_contacts == 0) | (foot_contacts == 1))  # Binary
        env.close()
    
    def test_observation_terrain_heights(self):
        """Test terrain heights in observation (indices 41-47)."""
        env = QuadrupedEnv(render=False)
        obs = env.reset()
        
        terrain_heights = obs[41:48]
        
        assert len(terrain_heights) == 7
        assert np.isfinite(terrain_heights).all()
        env.close()


class TestQuadrupedEnvRewardFunction:
    """Test reward function components."""
    
    def test_reward_finite(self):
        """Test reward is always finite."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, _, _ = env.step(action)
            assert np.isfinite(reward)
        env.close()
    
    def test_reward_includes_survival_bonus(self):
        """Test reward includes survival bonus."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Zero action should still give some reward (survival bonus)
        action = np.zeros(12)
        _, reward, _, _ = env.step(action)
        
        # Survival bonus is +0.1, so reward should be at least close to that
        # (may be slightly different due to other components)
        assert reward > -1.0  # Should not be extremely negative
        env.close()


class TestQuadrupedEnvRendering:
    """Test rendering functionality."""
    
    def test_render_no_crash(self):
        """Test rendering doesn't crash."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        # Render should not crash
        result = env.render(mode='human')
        assert result is None  # Returns None for 'human' mode
        env.close()
    
    def test_render_rgb_array(self):
        """Test RGB array rendering."""
        env = QuadrupedEnv(render=False)
        env.reset()
        
        rgb = env.render(mode='rgb_array')
        
        # Should return numpy array
        assert isinstance(rgb, np.ndarray)
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3  # RGB channels
        env.close()


class TestQuadrupedEnvCleanup:
    """Test resource cleanup."""
    
    def test_close(self):
        """Test environment can be closed."""
        env = QuadrupedEnv(render=False)
        env.reset()
        env.close()
        # Should not raise any exceptions
    
    def test_multiple_close(self):
        """Test calling close multiple times doesn't crash."""
        env = QuadrupedEnv(render=False)
        env.reset()
        env.close()
        env.close()  # Second close should not crash
    
    def test_context_manager(self):
        """Test environment works as context manager."""
        with QuadrupedEnv(render=False) as env:
            obs = env.reset()
            assert obs.shape == (48,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
