#!/usr/bin/env python3
"""
Example demonstrating atom rearrangement strategies.
This script initializes a lattice with a random distribution of atoms and
applies a filling strategy to create a defect-free region.

To switch between strategies, simply edit the strategy call in the code:
- Use simulator.movement_manager.center_filling_strategy() for center filling
- Use simulator.movement_manager.corner_filling_strategy() for corner filling
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Ensure the project root is on the path so 'defect_free' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from defect_free.simulator import LatticeSimulator
from defect_free.visualizer import LatticeVisualizer

def main():
    # Command-line arguments so lattice size / occupation can be changed without editing the file
    import argparse
    parser = argparse.ArgumentParser(description='Run movement example')
    parser.add_argument('--size', type=int, nargs=2, metavar=('ROWS', 'COLS'),
                        default=[30, 30], help='Lattice size as two integers: ROWS COLS (default: 40 40)')
    parser.add_argument('--occupation', type=float, default=0.6,
                        help='Occupation probability (default: 0.6)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    # Initialize simulator with parameters from CLI
    np.random.seed(args.seed)
    lattice_size = tuple(args.size)
    occupation_prob = args.occupation
    
    # Step 1: Initialize the lattice
    simulator = LatticeSimulator(initial_size=lattice_size, occupation_prob=occupation_prob)
    simulator.generate_initial_lattice()
    
    # Step 2: Calculate the maximum possible target size based on available atoms
    initial_atoms = np.sum(simulator.slm_lattice)
    print(f"Total available atoms: {initial_atoms}")
    
    # Calculate maximum square size using all available atoms
    max_square_size = simulator.calculate_max_defect_free_size()
    
    print(f"Calculated maximum target zone: {max_square_size}x{max_square_size}")
    print(f"This requires {max_square_size**2} atoms out of {initial_atoms} available")
    print(f"Using {max_square_size**2} atoms for a perfect square")
    
    # Initialize visualizer for (non-interactive) tracking of the rearrangement
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer

    # Print initial configuration before rearrangement
    print("\nInitial configuration before rearrangement:")
    print(f"Number of atoms: {initial_atoms}")
    print(f"Target zone size: {simulator.side_length}x{simulator.side_length}")
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Step 3: Apply rearrangement method
    
    # *** CHANGE THIS LINE TO SWITCH BETWEEN STRATEGIES ***
    # Use either:
    # - center_filling_strategy() for center filling
    # - corner_filling_strategy() for corner filling
    print("\nApplying filling strategy...")
    # Run strategy without opening interactive visualization windows; we will save a GIF at the end
    final_lattice, fill_rate, execution_time = simulator.movement_manager.center_filling_strategy(show_visualization=False)
    
    # Name of the current strategy for display purposes
    strategy_name = "Center"  # Change this if you change the strategy above
    
    # Store the final state
    after_filling_lattice = simulator.field.copy()
    
    # Get target region
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    print(f"\n{strategy_name} filling completed in {execution_time:.3f} seconds")
    print(f"Final fill rate: {fill_rate:.2%}")
    
    # Create and save the animation GIF of movements (single visual output)
    print("\nCreating animation of movements (GIF)...")
    animation = visualizer.animate_movements(simulator.movement_history)
    out_gif = f"movement_animation_{strategy_name.lower()}{lattice_size[0]}x{lattice_size[1]}.gif"
    visualizer.save_animation(out_gif, fps=10)
    print(f"Animation saved as {out_gif}")

if __name__ == "__main__":
    main()
