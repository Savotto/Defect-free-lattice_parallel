#!/usr/bin/env python3
"""
Simplest possible script to run the defect-free lattice simulator and measure running time.
"""
import time
import numpy as np
from defect_free import LatticeSimulator, LatticeVisualizer

# Initialize simulator with a 30x30 lattice and 70% occupation probability
simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=0.4)

# Generate initial lattice
simulator.generate_initial_lattice()
print(f"Generated initial lattice with {np.sum(simulator.field)} atoms")

# Initialize visualizer
visualizer = LatticeVisualizer(simulator)
simulator.visualizer = visualizer

# Run the rearrangement and measure time
print("Starting rearrangement...")
start_time = time.time()
result, execution_time = simulator.rearrange_for_defect_free(
    strategy='center',
    show_visualization=False
)
final_lattice, fill_rate, _ = result 
total_time = time.time() - start_time

# Print results
print(f"\nResults:")
print(f"Created defect-free region of size {simulator.side_length}x{simulator.side_length}")
print(f"Fill rate: {fill_rate:.2%}") 
print(f"Total running time: {total_time:.3f} seconds")
print(f"Algorithm execution time: {execution_time:.3f} seconds")
print(f"Total movements: {len(simulator.movement_history)}")

# Calculate physical time from movement history
physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
print(f"Physical movement time: {physical_time:.6f} seconds")

if __name__ == "__main__":
    pass  # The code already runs at module level, but this is a good practice