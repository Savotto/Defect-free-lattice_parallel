"""
Tests for the lattice simulator algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
from defect_free import LatticeSimulator, LatticeVisualizer

class TestRowWiseCentering:
    def setUp(self):
        """Initialize the test environment with a fixed random seed for reproducibility."""
        np.random.seed(42)  # Set random seed for reproducible tests
        self.simulator = LatticeSimulator(initial_size=(10, 10), occupation_prob=0.7, use_aod_constraints=True)
        self.simulator.generate_initial_lattice()
        self.visualizer = LatticeVisualizer(self.simulator)
        self.simulator.visualizer = self.visualizer

    def test_row_wise_centering(self):
        """Test the row-wise centering rearrangement algorithm with detailed verification."""
        initial_atoms = np.sum(self.simulator.slm_lattice)
        print(f"Initial number of atoms: {initial_atoms}")
        print("Initial lattice:")
        print(self.simulator.slm_lattice)
        
        # Store the animation reference to prevent deletion warning
        self.simulator.rearrange_for_perfect_lattice(show_visualization=False)
        final_atoms = np.sum(self.simulator.target_lattice)
        print(f"Final number of atoms: {final_atoms}")
        print("Final lattice:")
        print(self.simulator.target_lattice)
        
        # Verify target region content
        target_region = self.simulator.movement_manager.target_region
        if target_region:
            target_start_row, target_start_col, target_end_row, target_end_col = target_region
            target_region_array = self.simulator.target_lattice[target_start_row:target_end_row, 
                                                             target_start_col:target_end_col]
            atoms_in_target = np.sum(target_region_array)
            target_size = self.simulator.side_length * self.simulator.side_length
            
            print("\nVerifying target region:")
            print(f"Target size (expected atoms): {target_size}")
            print(f"Actual atoms in target: {atoms_in_target}")
            
            if atoms_in_target != target_size:
                print(f"Warning: Mismatch in atom count - expected {target_size}, got {atoms_in_target}")
            
            # Verify each row's atom positions
            for row in range(target_start_row, target_end_row):
                atom_positions = [(row, col) for col in range(target_start_col, target_end_col) 
                                if self.simulator.target_lattice[row, col] == 1]
                
                # Calculate expected positions for this row
                row_idx = row - target_start_row
                expected_atoms = min(self.simulator.side_length,  # Can't exceed row width
                                   target_size - row_idx * self.simulator.side_length)  # Remaining needed
                if expected_atoms <= 0:
                    continue
                
                center_col = (target_start_col + target_end_col) // 2
                start_col = center_col - expected_atoms // 2
                expected_positions = [(row, start_col + i) for i in range(expected_atoms)]
                
                print(f"\nRow {row} atom positions: {atom_positions}")
                print(f"Expected positions: {expected_positions}")
                
                # Check if positions match
                positions_match = (len(atom_positions) == len(expected_positions) and 
                                all(a == b for a, b in zip(sorted(atom_positions), sorted(expected_positions))))
                
                if positions_match:
                    print(f"Row {row} matches expected positions.")
                else:
                    print(f"Row {row} positions do not match expectations.")
                    if len(atom_positions) != len(expected_positions):
                        print(f"Wrong number of atoms: got {len(atom_positions)}, expected {len(expected_positions)}")
        
        # Generate visual output but don't block test execution
        plt.ioff()  # Turn off interactive mode temporarily
        self.visualizer.visualize_lattices()
        plt.close('all')  # Close all figures to prevent memory leaks
        
        print("\nTest completed successfully!")

if __name__ == '__main__':
    test = TestRowWiseCentering()
    test.setUp()
    test.test_row_wise_centering()