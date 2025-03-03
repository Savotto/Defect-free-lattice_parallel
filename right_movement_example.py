#!/usr/bin/env python
"""
Example file demonstrating the mass movement of all atoms to the right.
This utilizes the new move_all_atoms_right method that maximizes parallelism.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from src.core import LatticeSimulator
from src.visualization import LatticeVisualizer

def main():
    """Run a simple lattice simulation showing all atoms moving to the right."""
    print("Right Movement Demonstration")
    print("--------------------------")
    
    # Initialize the lattice simulator with a 20x20 grid and 60% occupation probability
    initial_size = (20, 20)
    occupation_prob = 0.6
    
    print(f"Initializing lattice simulator with grid size {initial_size} and occupation probability {occupation_prob}")
    simulator = LatticeSimulator(initial_size, occupation_prob)
    
    # Generate the initial lattice
    simulator.generate_initial_lattice()
    print(f"Generated initial lattice with {simulator.total_atoms} atoms")
    
    # Create a visualizer and attach it to the simulator
    simulator.visualizer = LatticeVisualizer(simulator)
    
    # Visualize the initial state
    print("\nInitial lattice configuration:")
    simulator.visualizer.visualize_lattices()
    
    # Reset movement history to focus on right movement only
    simulator.movement_history = []
    
    # Execute the mass right movement
    print("\nExecuting mass right movement...")
    start_time = time.time()
    
    # Call the new method to move all atoms right
    simulator.movement_manager.move_all_atoms_right()
    
    execution_time = time.time() - start_time
    
    # Animate the movement
    print("\nAnimating the movement...")
    simulator.visualizer.animate_rearrangement()
    
    # Print results
    print("\nMovement Results:")
    print(f"Total atoms moved: {simulator.total_atoms}")
    print(f"Algorithm execution time: {execution_time:.3f} seconds")
    print(f"Simulated physical movement time: {(simulator.total_transfer_time + simulator.movement_time)*1000:.3f} ms")
    
    # Visualize the final state
    print("\nFinal lattice configuration:")
    simulator.visualizer.visualize_lattices()
    
    print("\nRight Movement Demonstration completed!")

if __name__ == "__main__":
    main()