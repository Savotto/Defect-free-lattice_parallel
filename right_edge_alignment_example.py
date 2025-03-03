#!/usr/bin/env python
"""
Example script demonstrating the updated atom rearrangement algorithm that aligns atoms 
below the target zone with the right edge of the target zone after the left-up phase.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from src.core import LatticeSimulator
from src.visualization import LatticeVisualizer

def main():
    """Run a lattice simulation demonstrating the new rearrangement algorithm."""
    print("Right-Edge Alignment Algorithm Demonstration")
    print("----------------------------------------")
    
    # Initialize with a larger initial size and lower occupation probability
    # to make the movement pattern more visible
    initial_size = (25, 25)
    occupation_prob = 0.3
    
    print(f"Initializing lattice simulator with grid size {initial_size} and occupation probability {occupation_prob}")
    simulator = LatticeSimulator(initial_size, occupation_prob)
    
    # Generate the initial lattice
    simulator.generate_initial_lattice()
    print(f"Generated initial lattice with {simulator.total_atoms} atoms")
    print(f"Target lattice size will be {simulator.side_length}x{simulator.side_length}")
    
    # Create a visualizer and attach it to the simulator
    simulator.visualizer = LatticeVisualizer(simulator)
    
    # Visualize the initial state
    print("\nInitial configuration:")
    simulator.visualizer.visualize_lattices()
    
    # Execute the new rearrangement algorithm
    print("\nExecuting rearrangement algorithm with right-edge alignment...")
    start_time = time.time()
    final_lattice, retention_rate, execution_time = simulator.rearrange_atoms(show_visualization=True)
    
    # Print results
    print("\nRearrangement Results:")
    print(f"Target lattice size: {simulator.side_length}x{simulator.side_length}")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"Algorithm execution time: {execution_time:.3f} seconds")
    print(f"Total movement steps: {len(simulator.movement_history)}")
    print(f"Total physical movement time: {(simulator.total_transfer_time + simulator.movement_time)*1000:.3f} ms")
    
    # Show final configuration
    print("\nFinal configuration:")
    simulator.visualizer.visualize_lattices()

if __name__ == "__main__":
    main()