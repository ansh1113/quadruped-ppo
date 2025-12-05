"""Tests for terrain generation."""

import pytest
import numpy as np
import pybullet as p
from quadruped_ppo.envs.terrain import TerrainGenerator


@pytest.fixture
def pybullet_env():
    """Setup and teardown PyBullet environment."""
    client = p.connect(p.DIRECT)
    yield client
    p.disconnect(client)


class TestTerrainGenerator:
    """Test terrain generation functionality."""
    
    def test_initialization(self):
        """Test terrain generator can be initialized."""
        gen = TerrainGenerator(difficulty=0.5)
        assert gen.difficulty == 0.5
    
    def test_difficulty_clipping(self):
        """Test difficulty is clipped to valid range."""
        gen = TerrainGenerator(difficulty=1.5)
        assert gen.difficulty == 1.0
        
        gen = TerrainGenerator(difficulty=-0.5)
        assert gen.difficulty == 0.0
    
    @pytest.mark.parametrize("terrain_type", ['flat', 'uneven', 'stairs', 'slopes', 'mixed'])
    def test_generate_all_types(self, pybullet_env, terrain_type):
        """Test all terrain types can be generated."""
        gen = TerrainGenerator(difficulty=0.5)
        terrain_id = gen.generate(terrain_type)
        
        assert terrain_id is not None
        assert isinstance(terrain_id, int)
    
    def test_invalid_terrain_type(self, pybullet_env):
        """Test invalid terrain type raises error."""
        gen = TerrainGenerator(difficulty=0.5)
        
        with pytest.raises(ValueError):
            gen.generate('invalid_type')
    
    def test_cleanup(self, pybullet_env):
        """Test cleanup removes terrain objects."""
        gen = TerrainGenerator(difficulty=0.5)
        gen.generate('flat')
        
        # Should not raise any exceptions
        gen.cleanup()
        
        assert gen.terrain_id is None
        assert len(gen.terrain_objects) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
