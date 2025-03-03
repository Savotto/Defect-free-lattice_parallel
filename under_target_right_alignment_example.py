#!/usr/bin/env python
"""
Example script demonstrating moving atoms under the target grid to the right edge.
This aligns the rightmost atom in each row with the right edge of the target grid.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from src.core import LatticeSimulator
from src.visualization import LatticeVisualizer

def main():
    """Run a lattice simulation showing alignment of atoms below the target region."""
    print("Under-Target Right Alignment Demonstration")
    print("----------------------------------------")
    
    # Initialize the lattice simulator with a 20x20 grid and 40% occupation probability
    # Lower occupation probability to have more empty spaces
    initial_size = (20, 20)
    occupation_prob = 0.4
    
    print(f"Initializing lattice simulator with grid size {initial_size} and occupation probability {occupation_prob}")
    simulator = LatticeSimulator(initial_size, occupation_prob)
    
    # Generate the initial lattice
    simulator.generate_initial_lattice()
    print(f"Generated initial lattice with {simulator.total_atoms} atoms")
    print(f"Target lattice size will be {simulator.side_length}x{simulator.side_length}")
    
    # Create a visualizer and attach it to the simulator
    simulator.visualizer = LatticeVisualizer(simulator)
    
    # Reset movement history
    simulator.movement_history = []
    
    # First, do a left-up movement to set up the configuration
    print("\nPerforming initial left-up movement to create configuration...")
    simulator.movement_manager.move_atoms_with_constraints()
    
    # Visualize the initial state after left-up movement
    print("\nConfiguration after left-up movement:")
    simulator.visualizer.visualize_lattices()
    
    # Reset movement history to focus on our specific operation
    simulator.movement_history = []
    
    # Execute the under-target right alignment movement
    print("\nExecuting alignment of atoms below target grid...")
    start_time = time.time()
    
    # Call our new method
    moved = simulator.movement_manager.move_under_atoms_to_right_edge()
    
    execution_time = time.time() - start_time
    
    # If atoms were moved, animate the movement
    if moved:
        print("\nAnimating the movement...")
        simulator.visualizer.animate_rearrangement()
    
    # Print results
    print("\nMovement Results:")
    if moved:
        # Get target region coordinates to explain what happened
        start_row = (simulator.field_size[0] - simulator.initial_size[0]) // 2
        start_col = (simulator.field_size[1] - simulator.initial_size[1]) // 2
        end_row = start_row + simulator.side_length
        end_col = start_col + simulator.side_length
        
        print(f"Successfully aligned atoms below target region with right edge (column {end_col-1})")
        print(f"Algorithm execution time: {execution_time:.3f} seconds")
        print(f"Simulated physical movement time: {(simulator.total_transfer_time + simulator.movement_time)*1000:.3f} ms")
    else:
        print("No atoms were moved (either no atoms below target or already aligned)")
    
    # Visualize the final state
    print("\nFinal lattice configuration:")
    simulator.visualizer.visualize_lattices()
    
    print("\nAlignment Demonstration completed!")

if __name__ == "__main__":
    main()