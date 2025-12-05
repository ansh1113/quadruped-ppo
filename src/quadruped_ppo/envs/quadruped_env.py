"""
Quadruped locomotion environment using PyBullet simulation.

This module provides a complete Gym environment for training quadruped
robots to walk on various terrains using reinforcement learning.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data

from .terrain import TerrainGenerator


class QuadrupedEnv(gym.Env):
    """
    Quadruped locomotion environment using PyBullet.
    
    The environment simulates a quadruped robot learning to walk on various
    terrains. The robot has 12 degrees of freedom (3 per leg) and must learn
    to generate stable gaits.
    
    Observation Space (48-dim):
        - Body state (13): position (3), orientation (4), linear velocity (3), angular velocity (3)
        - Joint state (24): joint positions (12) + joint velocities (12)
        - Foot contacts (4): binary contact sensors for each foot
        - Previous action (12): previous joint position commands
        - Terrain info (7): height samples around robot
    
    Action Space (12-dim):
        - Continuous joint position targets for 12 joints in range [-1, 1]
        - Mapped to actual joint limits during execution
    
    Reward Function:
        - Forward velocity: +1.5 * v_x
        - Lateral movement penalty: -0.5 * |v_y|
        - Falling penalty: -10.0
        - Orientation penalty: -0.5 * (|roll| + |pitch|)
        - Energy efficiency: -0.01 * Σ(action²)
        - Smooth motion: -0.05 * Σ(|action - prev_action|)
        - Foot contact reward: +0.1 if ≥2 feet on ground, -0.2 otherwise
        - Survival bonus: +0.1
    
    Args:
        terrain_type: Type of terrain ('flat', 'uneven', 'stairs', 'slopes', 'mixed')
        terrain_difficulty: Difficulty level from 0.0 to 1.0
        render: Whether to render the environment
        max_episode_steps: Maximum steps per episode
        control_freq: Control frequency in Hz
    
    Example:
        >>> env = QuadrupedEnv(terrain_type='flat', render=True)
        >>> obs = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, done, info = env.step(action)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        terrain_type: str = 'flat',
        terrain_difficulty: float = 0.5,
        render: bool = False,
        max_episode_steps: int = 1000,
        control_freq: int = 50
    ):
        super(QuadrupedEnv, self).__init__()
        
        # Environment parameters
        self.terrain_type = terrain_type
        self.terrain_difficulty = np.clip(terrain_difficulty, 0.0, 1.0)
        self.render_enabled = render
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq
        
        # Simulation parameters
        self.dt = 1.0 / 240.0  # PyBullet default timestep
        self.control_steps = int(1.0 / (self.control_freq * self.dt))
        
        # Robot parameters
        self.num_joints = 12
        self.num_feet = 4
        
        # Action space: joint position targets [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Observation space (48-dim)
        obs_dim = 48
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Connect to PyBullet
        if self.render_enabled:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.3]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # State variables
        self.robot_id: Optional[int] = None
        self.terrain_generator: Optional[TerrainGenerator] = None
        self.joint_ids: List[int] = []
        self.joint_limits: List[Tuple[float, float]] = []
        self.foot_link_ids: List[int] = []
        self.prev_action: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self.step_counter: int = 0
        self.initial_position: np.ndarray = np.array([0.0, 0.0, 0.45])
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset simulation
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        
        # Create terrain
        self.terrain_generator = TerrainGenerator(difficulty=self.terrain_difficulty)
        self.terrain_generator.generate(self.terrain_type)
        
        # Load quadruped robot
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'assets',
            'quadruped.urdf'
        )
        
        start_pos = self.initial_position
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot_id = p.loadURDF(
            urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=False,
            physicsClientId=self.client
        )
        
        # Discover and configure joints
        self._discover_joints()
        
        # Set initial joint positions (standing pose)
        initial_joint_positions = self._get_initial_pose()
        for joint_id, pos in zip(self.joint_ids, initial_joint_positions):
            p.resetJointState(
                self.robot_id,
                joint_id,
                pos,
                targetVelocity=0.0,
                physicsClientId=self.client
            )
        
        # Reset state variables
        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.step_counter = 0
        
        # Let the robot settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Joint position targets in range [-1, 1]
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information dictionary
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Convert normalized actions to actual joint positions
        joint_positions = self._action_to_joint_positions(action)
        
        # Apply joint control
        p.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[50.0] * len(self.joint_ids),
            physicsClientId=self.client
        )
        
        # Step simulation
        for _ in range(self.control_steps):
            p.stepSimulation(physicsClientId=self.client)
        
        self.step_counter += 1
        
        # Get observation and compute reward
        obs = self._get_observation()
        reward = self._compute_reward(action)
        done = self._is_done()
        
        # Additional info
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        
        info = {
            'step': self.step_counter,
            'fell': done and pos[2] < 0.15,
            'position': pos,
            'velocity': vel[0],  # Forward velocity
            'distance': pos[0] - self.initial_position[0]
        }
        
        self.prev_action = action.copy()
        
        return obs, reward, done, info
    
    def _discover_joints(self) -> None:
        """Discover and configure robot joints."""
        self.joint_ids = []
        self.joint_limits = []
        self.foot_link_ids = []
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            # Only consider revolute joints
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(joint_idx)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
            
            # Identify foot links (knee links are feet)
            if 'knee' in link_name.lower():
                self.foot_link_ids.append(joint_idx)
        
        # Verify we have correct number of joints
        if len(self.joint_ids) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, found {len(self.joint_ids)}"
            )
        if len(self.foot_link_ids) != self.num_feet:
            raise ValueError(
                f"Expected {self.num_feet} feet, found {len(self.foot_link_ids)}"
            )
    
    def _get_initial_pose(self) -> List[float]:
        """
        Get initial standing pose for the robot.
        
        Returns:
            List of initial joint positions
        """
        # Standing pose: slightly bent legs
        # Pattern: [hip_abd, hip, knee] for each leg
        return [
            0.0, 0.5, -1.0,  # Front left
            0.0, 0.5, -1.0,  # Front right
            0.0, 0.5, -1.0,  # Rear left
            0.0, 0.5, -1.0   # Rear right
        ]
    
    def _action_to_joint_positions(self, action: np.ndarray) -> List[float]:
        """
        Convert normalized actions to actual joint positions.
        
        Args:
            action: Normalized action in [-1, 1]
            
        Returns:
            List of target joint positions
        """
        joint_positions = []
        for act, (lower, upper) in zip(action, self.joint_limits):
            # Map from [-1, 1] to [lower, upper]
            pos = lower + (act + 1.0) * 0.5 * (upper - lower)
            joint_positions.append(pos)
        return joint_positions
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (48-dim).
        
        Returns:
            Observation vector
        """
        obs = np.zeros(48, dtype=np.float32)
        
        if self.robot_id is None:
            return obs
        
        idx = 0
        
        # Body state (13 dims)
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client
        )
        
        obs[idx:idx+3] = pos
        idx += 3
        obs[idx:idx+4] = orn
        idx += 4
        obs[idx:idx+3] = vel
        idx += 3
        obs[idx:idx+3] = ang_vel
        idx += 3
        
        # Joint state (24 dims: 12 positions + 12 velocities)
        joint_states = p.getJointStates(
            self.robot_id, self.joint_ids, physicsClientId=self.client
        )
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        obs[idx:idx+12] = joint_positions
        idx += 12
        obs[idx:idx+12] = joint_velocities
        idx += 12
        
        # Foot contacts (4 dims)
        foot_contacts = self._get_foot_contacts()
        obs[idx:idx+4] = foot_contacts
        idx += 4
        
        # Previous action (12 dims)
        obs[idx:idx+12] = self.prev_action
        idx += 12
        
        # Terrain info (7 dims) - height samples around robot
        terrain_heights = self._sample_terrain_heights(pos[:2])
        obs[idx:idx+7] = terrain_heights
        idx += 7
        
        # Verify observation size
        if idx != 48:
            raise RuntimeError(f"Observation size mismatch: {idx} != 48")
        
        return obs
    
    def _get_foot_contacts(self) -> np.ndarray:
        """
        Get binary contact sensors for each foot.
        
        Returns:
            Array of foot contact status (1.0 = contact, 0.0 = no contact)
        """
        contacts = np.zeros(self.num_feet, dtype=np.float32)
        
        for i, foot_link_id in enumerate(self.foot_link_ids):
            contact_points = p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=foot_link_id,
                physicsClientId=self.client
            )
            contacts[i] = 1.0 if len(contact_points) > 0 else 0.0
        
        return contacts
    
    def _sample_terrain_heights(self, robot_xy: np.ndarray) -> np.ndarray:
        """
        Sample terrain heights around robot position.
        
        Args:
            robot_xy: Robot's (x, y) position
            
        Returns:
            Array of 7 height samples
        """
        # Sample heights in a circle around the robot
        heights = np.zeros(7, dtype=np.float32)
        sample_radius = 0.3  # 30cm radius
        
        # Center
        heights[0] = self._get_height_at(robot_xy[0], robot_xy[1])
        
        # Six points around circle
        for i in range(6):
            angle = i * np.pi / 3
            x = robot_xy[0] + sample_radius * np.cos(angle)
            y = robot_xy[1] + sample_radius * np.sin(angle)
            heights[i + 1] = self._get_height_at(x, y)
        
        return heights
    
    def _get_height_at(self, x: float, y: float) -> float:
        """
        Get terrain height at given (x, y) position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Height at position
        """
        # Ray cast from above to get terrain height
        start_pos = [x, y, 10.0]
        end_pos = [x, y, -10.0]
        
        result = p.rayTest(start_pos, end_pos, physicsClientId=self.client)
        
        if result and result[0][0] != -1:  # Hit something
            hit_position = result[0][3]
            return hit_position[2]
        
        return 0.0  # Default ground level
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute reward for current state.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        if self.robot_id is None:
            return reward
        
        # Get robot state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client
        )
        
        # Forward velocity reward (primary objective)
        reward += 1.5 * vel[0]
        
        # Penalize lateral movement
        reward -= 0.5 * abs(vel[1])
        
        # Falling penalty
        if pos[2] < 0.2:
            reward -= 10.0
        
        # Orientation penalty
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        reward -= 0.5 * (abs(roll) + abs(pitch))
        
        # Energy efficiency - penalize large actions
        reward -= 0.01 * np.sum(np.square(action))
        
        # Smooth motion - penalize rapid action changes
        reward -= 0.05 * np.sum(np.abs(action - self.prev_action))
        
        # Foot contact reward - encourage stable gaits
        foot_contacts = self._get_foot_contacts()
        num_contacts = np.sum(foot_contacts)
        if num_contacts >= 2:
            reward += 0.1
        else:
            reward -= 0.2  # Penalize aerial phases
        
        # Survival bonus
        reward += 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if episode is done
        """
        # Maximum steps reached
        if self.step_counter >= self.max_episode_steps:
            return True
        
        # Robot fell
        if self.robot_id is not None:
            pos, orn = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.client
            )
            
            # Height check
            if pos[2] < 0.15:
                return True
            
            # Orientation check (flipped over)
            euler = p.getEulerFromQuaternion(orn)
            roll, pitch, _ = euler
            if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
                return True
        
        return False
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.3],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self.terrain_generator is not None:
            self.terrain_generator.cleanup()
        
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1
