"""
Gait analysis for quadruped locomotion.

This module provides tools to analyze and classify gaits based on
foot contact patterns and other kinematic data.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr


class GaitAnalyzer:
    """
    Analyze quadruped gait patterns from simulation data.
    
    This class provides methods to extract gait characteristics such as
    duty cycle, stride frequency, phase relationships, and gait classification.
    """
    
    def __init__(self, dt: float = 0.02):
        """
        Initialize gait analyzer.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        self.foot_names = ['FL', 'FR', 'RL', 'RR']
    
    def analyze(
        self,
        foot_contacts: np.ndarray,
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform complete gait analysis.
        
        Args:
            foot_contacts: Array of shape (timesteps, 4) with binary contact data
            positions: Optional array of shape (timesteps, 3) with positions
            velocities: Optional array of shape (timesteps, 3) with velocities
            
        Returns:
            Dictionary containing all gait metrics
        """
        results = {}
        
        # Basic metrics
        results.update(self.compute_duty_cycles(foot_contacts))
        results.update(self.compute_stride_metrics(foot_contacts))
        results.update(self.compute_phase_relationships(foot_contacts))
        results.update(self.classify_gait(foot_contacts))
        
        # Optional metrics if position/velocity provided
        if positions is not None:
            results.update(self.compute_kinematic_metrics(positions))
        
        if velocities is not None:
            results.update(self.compute_velocity_metrics(velocities))
        
        return results
    
    def compute_duty_cycles(self, foot_contacts: np.ndarray) -> Dict[str, Any]:
        """
        Compute duty cycle for each foot.
        
        The duty cycle is the fraction of time a foot spends in contact
        with the ground during one complete stride cycle.
        
        Args:
            foot_contacts: Array of shape (timesteps, 4)
            
        Returns:
            Dictionary with duty cycle information
        """
        duty_cycles = np.mean(foot_contacts, axis=0)
        
        return {
            'duty_cycles': {name: float(dc) for name, dc in zip(self.foot_names, duty_cycles)},
            'mean_duty_cycle': float(np.mean(duty_cycles)),
            'duty_cycle_std': float(np.std(duty_cycles))
        }
    
    def compute_stride_metrics(self, foot_contacts: np.ndarray) -> Dict[str, Any]:
        """
        Compute stride-related metrics.
        
        Args:
            foot_contacts: Array of shape (timesteps, 4)
            
        Returns:
            Dictionary with stride metrics
        """
        num_feet = foot_contacts.shape[1]
        stride_durations = []
        stride_frequencies = []
        step_counts = []
        
        for i in range(num_feet):
            contacts = foot_contacts[:, i]
            
            # Find contact transitions (touchdowns)
            diff = np.diff(contacts)
            touchdowns = np.where(diff > 0)[0]
            
            if len(touchdowns) > 1:
                # Stride duration is time between consecutive touchdowns
                durations = np.diff(touchdowns) * self.dt
                stride_durations.extend(durations.tolist())
                stride_frequencies.append(1.0 / np.mean(durations) if len(durations) > 0 else 0.0)
            else:
                stride_frequencies.append(0.0)
            
            step_counts.append(len(touchdowns))
        
        return {
            'stride_frequencies': {name: float(freq) for name, freq in zip(self.foot_names, stride_frequencies)},
            'mean_stride_frequency': float(np.mean(stride_frequencies)) if stride_frequencies else 0.0,
            'mean_stride_duration': float(np.mean(stride_durations)) if stride_durations else 0.0,
            'step_counts': {name: int(count) for name, count in zip(self.foot_names, step_counts)},
            'total_steps': int(np.sum(step_counts))
        }
    
    def compute_phase_relationships(self, foot_contacts: np.ndarray) -> Dict[str, Any]:
        """
        Compute phase relationships between legs.
        
        Args:
            foot_contacts: Array of shape (timesteps, 4)
            
        Returns:
            Dictionary with phase information
        """
        # Compute normalized cross-correlation between all pairs
        num_feet = foot_contacts.shape[1]
        phase_lags = {}
        correlations = {}
        
        pairs = [
            ('FL', 'FR'), ('FL', 'RL'), ('FL', 'RR'),
            ('FR', 'RL'), ('FR', 'RR'), ('RL', 'RR')
        ]
        
        for i in range(num_feet):
            for j in range(i + 1, num_feet):
                name_i, name_j = self.foot_names[i], self.foot_names[j]
                pair_name = f"{name_i}-{name_j}"
                
                # Compute cross-correlation
                signal_i = foot_contacts[:, i] - np.mean(foot_contacts[:, i])
                signal_j = foot_contacts[:, j] - np.mean(foot_contacts[:, j])
                
                if np.std(signal_i) > 0 and np.std(signal_j) > 0:
                    corr, _ = pearsonr(signal_i, signal_j)
                    correlations[pair_name] = float(corr)
                    
                    # Compute phase lag
                    xcorr = np.correlate(signal_i, signal_j, mode='full')
                    lag = np.argmax(xcorr) - len(signal_i) + 1
                    phase_lag = (lag * self.dt) % (2 * np.pi)
                    phase_lags[pair_name] = float(phase_lag)
                else:
                    correlations[pair_name] = 0.0
                    phase_lags[pair_name] = 0.0
        
        return {
            'phase_correlations': correlations,
            'phase_lags': phase_lags,
            'mean_correlation': float(np.mean(list(correlations.values()))) if correlations else 0.0
        }
    
    def classify_gait(self, foot_contacts: np.ndarray) -> Dict[str, Any]:
        """
        Classify the gait type based on foot contact patterns.
        
        Common gaits:
        - Walk: duty cycle > 0.5, diagonal pairs in phase
        - Trot: duty cycle ~0.5, diagonal pairs in phase
        - Pace: duty cycle ~0.5, lateral pairs in phase  
        - Gallop: duty cycle < 0.5, asymmetric
        - Bound: duty cycle < 0.5, front/rear pairs in phase
        
        Args:
            foot_contacts: Array of shape (timesteps, 4)
            
        Returns:
            Dictionary with gait classification
        """
        # Compute duty cycle
        duty_cycles = np.mean(foot_contacts, axis=0)
        mean_duty = np.mean(duty_cycles)
        
        # Check synchronization patterns
        fl_fr_corr = np.corrcoef(foot_contacts[:, 0], foot_contacts[:, 1])[0, 1]
        rl_rr_corr = np.corrcoef(foot_contacts[:, 2], foot_contacts[:, 3])[0, 1]
        fl_rl_corr = np.corrcoef(foot_contacts[:, 0], foot_contacts[:, 2])[0, 1]
        fr_rr_corr = np.corrcoef(foot_contacts[:, 1], foot_contacts[:, 3])[0, 1]
        fl_rr_corr = np.corrcoef(foot_contacts[:, 0], foot_contacts[:, 3])[0, 1]
        fr_rl_corr = np.corrcoef(foot_contacts[:, 1], foot_contacts[:, 2])[0, 1]
        
        # Diagonal synchronization (FL-RR, FR-RL)
        diag_sync = (fl_rr_corr + fr_rl_corr) / 2
        
        # Lateral synchronization (FL-FR, RL-RR)
        lateral_sync = (fl_fr_corr + rl_rr_corr) / 2
        
        # Front-rear synchronization (FL-RL, FR-RR)
        frontrear_sync = (fl_rl_corr + fr_rr_corr) / 2
        
        # Classify
        gait_type = "unknown"
        confidence = 0.0
        
        if mean_duty > 0.6:
            gait_type = "walk"
            confidence = min(mean_duty, 0.9)
        elif 0.4 <= mean_duty <= 0.6:
            if diag_sync > 0.3:
                gait_type = "trot"
                confidence = abs(diag_sync)
            elif lateral_sync > 0.3:
                gait_type = "pace"
                confidence = abs(lateral_sync)
            else:
                gait_type = "transitional"
                confidence = 0.5
        elif mean_duty < 0.4:
            if frontrear_sync > 0.3:
                gait_type = "bound"
                confidence = abs(frontrear_sync)
            else:
                gait_type = "gallop"
                confidence = 1.0 - mean_duty
        
        return {
            'gait_type': gait_type,
            'gait_confidence': float(confidence),
            'diagonal_synchronization': float(diag_sync),
            'lateral_synchronization': float(lateral_sync),
            'frontrear_synchronization': float(frontrear_sync)
        }
    
    def compute_kinematic_metrics(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Compute kinematic metrics from position data.
        
        Args:
            positions: Array of shape (timesteps, 3)
            
        Returns:
            Dictionary with kinematic metrics
        """
        # Compute velocities
        velocities = np.diff(positions, axis=0) / self.dt
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0) / self.dt
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        return {
            'mean_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'mean_acceleration': float(np.mean(accel_magnitudes)),
            'max_acceleration': float(np.max(accel_magnitudes)),
            'total_distance': float(np.sum(speeds) * self.dt),
            'displacement': float(np.linalg.norm(positions[-1] - positions[0]))
        }
    
    def compute_velocity_metrics(self, velocities: np.ndarray) -> Dict[str, float]:
        """
        Compute velocity-specific metrics.
        
        Args:
            velocities: Array of shape (timesteps, 3)
            
        Returns:
            Dictionary with velocity metrics
        """
        forward_vel = velocities[:, 0]
        lateral_vel = velocities[:, 1]
        vertical_vel = velocities[:, 2]
        
        return {
            'mean_forward_velocity': float(np.mean(forward_vel)),
            'std_forward_velocity': float(np.std(forward_vel)),
            'mean_lateral_velocity': float(np.mean(np.abs(lateral_vel))),
            'mean_vertical_velocity': float(np.mean(np.abs(vertical_vel)))
        }
    
    def detect_gait_transitions(
        self,
        foot_contacts: np.ndarray,
        window_size: int = 50
    ) -> List[Tuple[int, str, str]]:
        """
        Detect transitions between different gaits.
        
        Args:
            foot_contacts: Array of shape (timesteps, 4)
            window_size: Size of sliding window for gait classification
            
        Returns:
            List of tuples (timestep, from_gait, to_gait)
        """
        transitions = []
        num_steps = foot_contacts.shape[0]
        
        prev_gait = None
        
        for i in range(0, num_steps - window_size, window_size // 2):
            window = foot_contacts[i:i + window_size]
            gait_info = self.classify_gait(window)
            current_gait = gait_info['gait_type']
            
            if prev_gait is not None and current_gait != prev_gait:
                transitions.append((i, prev_gait, current_gait))
            
            prev_gait = current_gait
        
        return transitions


def analyze_gait(
    model,
    env,
    duration: float = 10.0,
    dt: float = 0.02
) -> Dict[str, Any]:
    """
    Analyze gait of a trained model over a duration.
    
    Args:
        model: Trained RL model
        env: Environment
        duration: Duration to analyze in seconds
        dt: Time step
        
    Returns:
        Dictionary with complete gait analysis
    """
    analyzer = GaitAnalyzer(dt=dt)
    
    # Collect data
    obs = env.reset()
    done = False
    
    foot_contacts_list = []
    positions_list = []
    velocities_list = []
    
    steps = int(duration / dt)
    for _ in range(steps):
        if done:
            obs = env.reset()
            done = False
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        
        # Extract foot contacts from observation (indices 37-40)
        foot_contacts = obs[37:41]
        foot_contacts_list.append(foot_contacts)
        
        # Extract position (indices 0-2)
        position = obs[0:3]
        positions_list.append(position)
        
        # Extract velocity (indices 7-9)
        velocity = obs[7:10]
        velocities_list.append(velocity)
    
    # Convert to arrays
    foot_contacts_array = np.array(foot_contacts_list)
    positions_array = np.array(positions_list)
    velocities_array = np.array(velocities_list)
    
    # Perform analysis
    results = analyzer.analyze(
        foot_contacts_array,
        positions_array,
        velocities_array
    )
    
    # Add raw data
    results['foot_contacts'] = foot_contacts_array
    results['positions'] = positions_array
    results['velocities'] = velocities_array
    
    return results
