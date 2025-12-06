"""
Terrain generation module for quadruped locomotion environment.

This module provides utilities for generating various terrain types
including flat, uneven, stairs, slopes, and mixed terrains.
"""

import numpy as np
import pybullet as p
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter, zoom


class TerrainGenerator:
    """Generate different types of terrains for quadruped locomotion."""
    
    def __init__(self, difficulty: float = 0.5):
        """
        Initialize terrain generator.
        
        Args:
            difficulty: Terrain difficulty level (0.0 to 1.0)
        """
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.terrain_id: Optional[int] = None
        self.terrain_objects: List[int] = []
    
    def generate(self, terrain_type: str) -> int:
        """
        Generate terrain of specified type.
        
        Args:
            terrain_type: One of 'flat', 'uneven', 'stairs', 'slopes', 'mixed'
            
        Returns:
            Body ID of the created terrain
            
        Raises:
            ValueError: If terrain type is not supported
        """
        if terrain_type == 'flat':
            return self._create_flat()
        elif terrain_type == 'uneven':
            return self._create_uneven()
        elif terrain_type == 'stairs':
            return self._create_stairs()
        elif terrain_type == 'slopes':
            return self._create_slopes()
        elif terrain_type == 'mixed':
            return self._create_mixed()
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")
    
    def _create_flat(self) -> int:
        """Create flat terrain."""
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)
        self.terrain_id = plane_id
        return plane_id
    
    def _create_uneven(self) -> int:
        """Create uneven heightfield terrain."""
        terrain_size = 20.0  # meters
        mesh_scale = 0.05    # resolution in meters
        grid_size = int(terrain_size / mesh_scale)
        
        # Generate random heights with Perlin-like noise
        heights = self._generate_perlin_noise(
            grid_size, 
            scale=self.difficulty * 0.1
        )
        
        # Apply smoothing to make it walkable
        heights = gaussian_filter(heights, sigma=3.0)
        
        # Create heightfield collision shape
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[mesh_scale, mesh_scale, 1.0],
            heightfieldData=heights.flatten().tolist(),
            numHeightfieldRows=grid_size,
            numHeightfieldColumns=grid_size,
            replaceHeightfieldIndex=-1
        )
        
        # Create visual mesh (optional, for better visualization)
        terrain_visual = p.createVisualShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[mesh_scale, mesh_scale, 1.0],
            heightfieldData=heights.flatten().tolist(),
            numHeightfieldRows=grid_size,
            numHeightfieldColumns=grid_size,
            rgbaColor=[0.5, 0.5, 0.5, 1.0]
        )
        
        # Create multibody
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            baseVisualShapeIndex=terrain_visual,
            basePosition=[0, 0, 0]
        )
        
        p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
        self.terrain_id = terrain_id
        self.terrain_objects = [terrain_id]
        
        return terrain_id
    
    def _create_stairs(self) -> int:
        """Create staircase terrain."""
        step_height = 0.02 + self.difficulty * 0.08  # 2-10cm per step
        step_length = 0.3
        step_width = 2.0
        num_steps = 15
        
        # Create ascending stairs
        for i in range(num_steps):
            height = step_height * (i + 1)
            position = [i * step_length, 0, height / 2]
            
            # Create collision shape
            box_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[step_length / 2, step_width / 2, height / 2]
            )
            
            # Create visual shape
            box_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[step_length / 2, step_width / 2, height / 2],
                rgbaColor=[0.6, 0.6, 0.6, 1.0]
            )
            
            # Create step
            step_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_shape,
                baseVisualShapeIndex=box_visual,
                basePosition=position
            )
            
            p.changeDynamics(step_id, -1, lateralFriction=1.0)
            self.terrain_objects.append(step_id)
        
        # Create descending stairs
        for i in range(num_steps):
            height = step_height * (num_steps - i)
            position = [(num_steps + i) * step_length, 0, height / 2]
            
            box_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[step_length / 2, step_width / 2, height / 2]
            )
            
            box_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[step_length / 2, step_width / 2, height / 2],
                rgbaColor=[0.6, 0.6, 0.6, 1.0]
            )
            
            step_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_shape,
                baseVisualShapeIndex=box_visual,
                basePosition=position
            )
            
            p.changeDynamics(step_id, -1, lateralFriction=1.0)
            self.terrain_objects.append(step_id)
        
        self.terrain_id = self.terrain_objects[0] if self.terrain_objects else None
        return self.terrain_id
    
    def _create_slopes(self) -> int:
        """Create sloped terrain."""
        slope_angle = self.difficulty * 0.3  # Up to ~17 degrees
        slope_length = 10.0
        slope_width = 2.0
        slope_thickness = 0.2
        
        # Create upward slope
        up_orientation = p.getQuaternionFromEuler([0, slope_angle, 0])
        up_position = [slope_length / 4, 0, np.sin(slope_angle) * slope_length / 4]
        
        slope_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[slope_length / 2, slope_width / 2, slope_thickness / 2]
        )
        
        slope_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[slope_length / 2, slope_width / 2, slope_thickness / 2],
            rgbaColor=[0.6, 0.6, 0.6, 1.0]
        )
        
        up_slope_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=slope_shape,
            baseVisualShapeIndex=slope_visual,
            basePosition=up_position,
            baseOrientation=up_orientation
        )
        
        p.changeDynamics(up_slope_id, -1, lateralFriction=1.0)
        self.terrain_objects.append(up_slope_id)
        
        # Create flat platform on top
        platform_height = np.sin(slope_angle) * slope_length / 2
        platform_position = [slope_length * 0.75, 0, platform_height]
        
        platform_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[slope_length / 2, slope_width / 2, slope_thickness / 2]
        )
        
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[slope_length / 2, slope_width / 2, slope_thickness / 2],
            rgbaColor=[0.6, 0.6, 0.6, 1.0]
        )
        
        platform_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_shape,
            baseVisualShapeIndex=platform_visual,
            basePosition=platform_position
        )
        
        p.changeDynamics(platform_id, -1, lateralFriction=1.0)
        self.terrain_objects.append(platform_id)
        
        # Create downward slope
        down_orientation = p.getQuaternionFromEuler([0, -slope_angle, 0])
        down_position = [slope_length * 1.5, 0, np.sin(slope_angle) * slope_length / 4]
        
        down_slope_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=slope_shape,
            baseVisualShapeIndex=slope_visual,
            basePosition=down_position,
            baseOrientation=down_orientation
        )
        
        p.changeDynamics(down_slope_id, -1, lateralFriction=1.0)
        self.terrain_objects.append(down_slope_id)
        
        self.terrain_id = self.terrain_objects[0] if self.terrain_objects else None
        return self.terrain_id
    
    def _create_mixed(self) -> int:
        """Create mixed terrain with random sections."""
        # Randomly select and create multiple terrain types
        section_types = ['flat', 'uneven', 'stairs', 'slopes']
        selected = np.random.choice(section_types, size=2, replace=False)
        
        # For simplicity, create uneven terrain with obstacles
        # This combines multiple terrain challenges
        terrain_id = self._create_uneven()
        
        # Add some random obstacles
        num_obstacles = int(5 * self.difficulty)
        for _ in range(num_obstacles):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            size = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.05, 0.15) * self.difficulty
            
            obstacle_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height]
            )
            
            obstacle_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height],
                rgbaColor=[0.7, 0.5, 0.3, 1.0]
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obstacle_shape,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=[x, y, height]
            )
            
            p.changeDynamics(obstacle_id, -1, lateralFriction=1.0)
            self.terrain_objects.append(obstacle_id)
        
        return terrain_id
    
    def _generate_perlin_noise(self, size: int, scale: float = 0.1) -> np.ndarray:
        """
        Generate Perlin-like noise for terrain.
        
        Args:
            size: Grid size
            scale: Height scale
            
        Returns:
            2D array of heights
        """
        # Simple noise generation using multiple frequencies
        heights = np.zeros((size, size))
        
        # Add multiple octaves
        for octave in range(4):
            freq = 2 ** octave
            amp = scale / (2 ** octave)
            
            # Generate random values at lower resolution
            low_res = np.random.randn(size // freq + 1, size // freq + 1) * amp
            
            # Upsample to full resolution
            upsampled = zoom(low_res, freq, order=1)
            
            # Add to heights (crop to exact size)
            heights += upsampled[:size, :size]
        
        return heights
    
    def sample_height(self, x: float, y: float) -> float:
        """
        Sample terrain height at given (x, y) position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Height at position
        """
        # This is a simplified version
        # For accurate height sampling, would need to query PyBullet
        return 0.0  # Default for flat terrain
    
    def cleanup(self):
        """Remove all terrain objects from simulation."""
        for obj_id in self.terrain_objects:
            try:
                p.removeBody(obj_id)
            except:
                pass
        
        if self.terrain_id is not None and self.terrain_id not in self.terrain_objects:
            try:
                p.removeBody(self.terrain_id)
            except:
                pass
        
        self.terrain_objects = []
        self.terrain_id = None
