import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_free.movement import MovementManager

class TestColumnMovement(unittest.TestCase):
    def setUp(self):
        """Set up a mocked simulator for testing the MovementManager."""
        self.mock_simulator = MagicMock()
        self.mock_simulator.field_size = (10, 10)
        self.mock_simulator.side_length = 4
        self.mock_simulator.constraints = {
            'max_acceleration': 1e-6,  # m/s²
            'site_distance': 1e-6      # μm
        }
        self.mock_simulator.field = np.zeros((10, 10), dtype=int)
        self.mock_simulator.movement_history = []
        
        self.movement_manager = MovementManager(self.mock_simulator)
        self.movement_manager.initialize_target_region()
        
        # Target region should be (3, 3, 7, 7) for a 4x4 target in a 10x10 field
        self.expected_target = (3, 3, 7, 7)
        
    def test_empty_column(self):
        """Test centering with no atoms in the column."""
        # Column is empty
        moves = self.movement_manager.center_atoms_in_column(3, 3, 7)
        self.assertEqual(moves, 0)
        
    def test_perfectly_centered_column(self):
        """Test column where atoms are already in perfect positions."""
        # Setup atoms already in perfect center positions
        self.mock_simulator.field = np.zeros((10, 10), dtype=int)
        self.mock_simulator.field[4:6, 3] = 1  # 2 atoms in the center of column
        
        moves = self.movement_manager.center_atoms_in_column(3, 3, 7)
        self.assertEqual(moves, 0)  # No moves needed
        
    def test_simple_center_column(self):
        """Test a simple case where atoms in a column need to be centered."""
        # Setup atoms that need centering
        self.mock_simulator.field = np.zeros((10, 10), dtype=int)
        self.mock_simulator.field[1, 3] = 1  # Atom at top
        self.mock_simulator.field[8, 3] = 1  # Atom at bottom
        
        moves = self.movement_manager.center_atoms_in_column(3, 3, 7)
        self.assertEqual(moves, 2)  # Both atoms should move
        
        # Check that atoms are now in the center
        field_after = self.mock_simulator.field
        atom_positions = [i for i in range(10) if field_after[i, 3] == 1]
        self.assertEqual(len(atom_positions), 2)
        self.assertTrue(all(3 <= pos < 7 for pos in atom_positions))
        
    def test_blocked_paths_column(self):
        """Test when paths are blocked for atom movement in a column."""
        self.mock_simulator.field = np.zeros((10, 10), dtype=int)
        self.mock_simulator.field[1, 3] = 1  # Atom at top
        self.mock_simulator.field[3, 3] = 1  # Blocking atom - but also a target position
        self.mock_simulator.field[8, 3] = 1  # Atom at bottom
        
        moves = self.movement_manager.center_atoms_in_column(3, 3, 7)
        self.assertEqual(moves, 2)  # Both atoms at (1,3) and (8,3) should move
        
        # Verify the final positions
        field_after = self.mock_simulator.field
        expected_positions = [(4, 3), (5, 3), (3, 3)]  # (1,3) -> (4,3), (8,3) -> (5,3), (3,3) unchanged
        actual_positions = [(i, 3) for i in range(10) if field_after[i, 3] == 1]
        self.assertEqual(sorted(actual_positions), sorted(expected_positions))
        
    def test_column_wise_centering(self):
        """Test the complete column_wise_centering method."""
        # Setup a more complex field
        self.mock_simulator.field = np.zeros((10, 10), dtype=int)
        # Column 3: 2 atoms
        self.mock_simulator.field[1, 3] = 1
        self.mock_simulator.field[8, 3] = 1
        # Column 4: 3 atoms
        self.mock_simulator.field[0:3, 4] = 1
        # Column 5: No atoms
        # Column 6: 4 atoms, already centered
        self.mock_simulator.field[3:7, 6] = 1
        
        # Mock the visualizer so we don't actually try to visualize
        self.mock_simulator.visualizer = None
        
        result, retention_rate, _ = self.movement_manager.column_wise_centering(show_visualization=False)
        
        # Check that result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check for appropriate retention rate
        expected_retention = 7/16  # Similar to row-wise test
        self.assertAlmostEqual(retention_rate, expected_retention)

if __name__ == '__main__':
    unittest.main()
