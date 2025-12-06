"""Tests for utility functions."""

import pytest
import numpy as np
from quadruped_ppo.utils.metrics import (
    compute_gait_metrics,
    compute_energy_efficiency,
    compute_stability_metrics
)


class TestGaitMetrics:
    """Test gait metrics computation."""
    
    def test_compute_gait_metrics_basic(self):
        """Test basic gait metrics computation."""
        # Create simple walking pattern
        num_steps = 100
        foot_contacts = np.zeros((num_steps, 4))
        
        # Simulate simple alternating pattern
        for i in range(num_steps):
            foot_contacts[i, 0] = 1 if (i // 10) % 2 == 0 else 0
            foot_contacts[i, 1] = 1 if (i // 10) % 2 == 1 else 0
            foot_contacts[i, 2] = 1 if (i // 10) % 2 == 1 else 0
            foot_contacts[i, 3] = 1 if (i // 10) % 2 == 0 else 0
        
        metrics = compute_gait_metrics(foot_contacts, dt=0.02)
        
        assert 'duty_cycles' in metrics
        assert 'mean_duty_cycle' in metrics
        assert 'stride_frequencies' in metrics
        assert 'estimated_gait_type' in metrics
        
        assert isinstance(metrics['duty_cycles'], list)
        assert len(metrics['duty_cycles']) == 4
    
    def test_compute_gait_metrics_all_grounded(self):
        """Test metrics when all feet are always on ground."""
        num_steps = 50
        foot_contacts = np.ones((num_steps, 4))
        
        metrics = compute_gait_metrics(foot_contacts, dt=0.02)
        
        # Duty cycle should be 1.0 for all feet
        assert np.allclose(metrics['duty_cycles'], [1.0, 1.0, 1.0, 1.0])
        assert metrics['mean_duty_cycle'] == 1.0


class TestEnergyEfficiency:
    """Test energy efficiency metrics."""
    
    def test_compute_energy_efficiency(self):
        """Test energy efficiency computation."""
        num_steps = 100
        actions = np.random.randn(num_steps, 12) * 0.5
        velocities = np.random.rand(num_steps) * 0.5 + 0.3
        
        metrics = compute_energy_efficiency(actions, velocities, dt=0.02)
        
        assert 'mean_power' in metrics
        assert 'total_energy' in metrics
        assert 'mean_velocity' in metrics
        assert 'total_distance' in metrics
        assert 'energy_per_meter' in metrics
        assert 'action_smoothness' in metrics
        
        assert metrics['mean_power'] > 0
        assert metrics['mean_velocity'] > 0
        assert metrics['total_distance'] > 0


class TestStabilityMetrics:
    """Test stability metrics."""
    
    def test_compute_stability_metrics(self):
        """Test stability metrics computation."""
        num_steps = 100
        
        # Create sample data
        # Quaternions (slightly varying)
        orientations = np.tile([1, 0, 0, 0], (num_steps, 1)).astype(float)
        orientations += np.random.randn(num_steps, 4) * 0.01
        
        # Normalize quaternions
        norms = np.linalg.norm(orientations, axis=1, keepdims=True)
        orientations = orientations / norms
        
        # Positions
        positions = np.zeros((num_steps, 3))
        positions[:, 0] = np.linspace(0, 5, num_steps)  # Forward movement
        positions[:, 2] = 0.4 + np.random.randn(num_steps) * 0.05  # Height variation
        
        metrics = compute_stability_metrics(orientations, positions)
        
        assert 'mean_height' in metrics
        assert 'height_std' in metrics
        assert 'roll_std' in metrics
        assert 'pitch_std' in metrics
        assert 'max_roll' in metrics
        assert 'max_pitch' in metrics
        
        assert 0.3 < metrics['mean_height'] < 0.5  # Reasonable height


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
