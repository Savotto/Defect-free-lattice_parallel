#!/usr/bin/env python
"""
Example file demonstrating the usage of the refactored lattice simulator.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from src.core import LatticeSimulator
from src.visualization import LatticeVisualizer


def main():
    """Run a simple lattice simulation and visualize the results."""
    print("Lattice Simulator Example")
    print("-----------------------")
    
    # Initialize the lattice simulator with a larger initial size and lower occupation probability
    # to make the movement pattern more visible
    initial_size = (20, 20)
    occupation_prob = 0.7
    
    print(f"Initializing lattice simulator with grid size {initial_size} and occupation probability {occupation_prob}")
    simulator = LatticeSimulator(initial_size, occupation_prob)
    
    # Generate the initial lattice
    simulator.generate_initial_lattice()
    initial_atoms = simulator.total_atoms
    print(f"Generated initial lattice with {initial_atoms} atoms")
    print(f"Target lattice size will be {simulator.side_length}x{simulator.side_length}")
    
    # Create a visualizer and attach it to the simulator
    simulator.visualizer = LatticeVisualizer(simulator)
    
    # Visualize the initial state
    print("\nVisualizing initial lattice...")
    simulator.visualizer.visualize_lattices()
    
    # Print physical constraints
    print("\nPhysical Constraints:")
    print(f"- Site distance: {simulator.SITE_DISTANCE} µm")
    print(f"- Max acceleration: {simulator.MAX_ACCELERATION} m/s²")
    print(f"- Trap transfer time: {simulator.TRAP_TRANSFER_TIME*1e6:.1f} µs")
    print(f"- Trap transfer fidelity: {simulator.TRAP_TRANSFER_FIDELITY:.3%}")
    print(f"- Combined transfer fidelity per move: {simulator.TRAP_TRANSFER_FIDELITY**2:.3%}")
    
    # Run the rearrangement algorithm
    print("\nRearranging atoms...")
    start_time = time.time()
    final_lattice, retention_rate, execution_time = simulator.rearrange_atoms(show_visualization=True)
    actual_time = time.time() - start_time
    
    # Calculate simulated physical time in seconds
    physical_time_seconds = simulator.total_transfer_time + simulator.movement_time
    
    # Calculate total real-world time (computation + physical movement)
    total_real_time = actual_time + physical_time_seconds
    
    # Print results
    print("\nRearrangement Results:")
    print(f"Initial atoms: {initial_atoms}")
    print(f"Target lattice size: {simulator.side_length}x{simulator.side_length}")
    print(f"Target atoms needed: {simulator.side_length * simulator.side_length}")
    print(f"Atoms lost during transfers: {initial_atoms - np.sum(final_lattice)}")
    print(f"Final atoms in target: {np.sum(final_lattice)}")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"Algorithm execution time: {execution_time:.3f} seconds")
    print(f"Actual running time: {actual_time:.3f} seconds")
    print(f"Simulated physical movement time: {physical_time_seconds*1000:.3f} ms")
    print(f"Total trap transfer time: {simulator.total_transfer_time*1000:.3f} ms")
    print(f"Total movement time: {simulator.movement_time*1000:.3f} ms")
    print(f"\nTotal real-world rearrangement time: {total_real_time:.3f} seconds")
    print(f"  = {actual_time:.3f}s (computation) + {physical_time_seconds:.3f}s (physical movement)")
    
    # Visualize the final state
    print("\nVisualizing final lattice configuration...")
    simulator.visualizer.visualize_lattices()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()