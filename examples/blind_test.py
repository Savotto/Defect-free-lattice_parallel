#!/usr/bin/env python3
"""
Script to run the defect-free lattice simulator with iterative refinement
until achieving 100% fill rate.
"""
import time
import numpy as np
from defect_free import LatticeSimulator, LatticeVisualizer

# Initialize simulator with a 100x100 lattice and 70% occupation probability
simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=0.7)

# Generate initial lattice
simulator.generate_initial_lattice()
initial_atoms = np.sum(simulator.field)
print(f"Generated initial lattice with {initial_atoms} atoms")

# Initialize visualizer
visualizer = LatticeVisualizer(simulator)
simulator.visualizer = visualizer

# Parameters for iterative refinement
max_iterations = 5  # Maximum number of iterations to prevent infinite loops
target_fill_rate = 1.0  # 100% fill rate
current_fill_rate = 0.0
iteration = 0
cumulative_execution_time = 0.0
cumulative_physical_time = 0.0
total_movements = 0

# Run the rearrangement repeatedly until target fill rate is achieved
print("Starting iterative rearrangement process...")
overall_start_time = time.time()

while current_fill_rate < target_fill_rate and iteration < max_iterations:
    iteration += 1
    print(f"\nIteration {iteration}/{max_iterations}:")
    
    # Run the rearrangement and measure time
    start_time = time.time()
    result, execution_time = simulator.rearrange_for_defect_free(
        strategy='center',  # Change to 'corner' for corner filling
        show_visualization=True
    )
    final_lattice, current_fill_rate, _ = result 
    iteration_time = time.time() - start_time
    
    # Calculate physical time from movement history for this iteration
    iteration_physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
    
    # Update cumulative statistics
    cumulative_execution_time += execution_time
    cumulative_physical_time += iteration_physical_time
    total_movements += len(simulator.movement_history)
    
    # Print iteration results
    print(f"  Fill rate achieved: {current_fill_rate:.2%}")
    print(f"  Algorithm execution time: {execution_time:.3f} seconds")
    print(f"  Physical movement time: {iteration_physical_time:.6f} seconds")
    print(f"  Movements in this iteration: {len(simulator.movement_history)}")
    
    # Check if target fill rate achieved
    if current_fill_rate >= target_fill_rate:
        print(f"\nTarget fill rate of {target_fill_rate:.0%} achieved!")
        break
    
    # If another iteration is needed, preserve the current lattice state
    if iteration < max_iterations and current_fill_rate < target_fill_rate:
        print("  Continuing with next iteration using current lattice state...")
        # The simulator already preserves the current state for the next iteration

overall_total_time = time.time() - overall_start_time

# Print final results
print(f"\nFinal Results after {iteration} iterations:")
print(f"Created defect-free region of size {simulator.side_length}x{simulator.side_length}")
print(f"Final fill rate: {current_fill_rate:.2%}")
print(f"Total algorithm execution time: {cumulative_execution_time:.3f} seconds")
print(f"Total physical movement time: {cumulative_physical_time:.6f} seconds")
print(f"Total movements performed: {total_movements}")
print(f"Overall process time: {overall_total_time:.3f} seconds")

# Calculate atom retention rate
target_region = simulator.movement_manager.target_region
target_start_row, target_start_col, target_end_row, target_end_col = target_region
atoms_in_target = np.sum(simulator.field[target_start_row:target_end_row, 
                                         target_start_col:target_end_col])
retention_rate = atoms_in_target / initial_atoms
print(f"Atom retention rate: {retention_rate:.2%}")