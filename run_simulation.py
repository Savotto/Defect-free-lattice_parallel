import numpy as np
from lattice import LatticeSimulator
import time

def main():
    # Create a simulator with a 10x10 initial lattice and 70% occupation probability
    simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=0.7)

    # Generate the initial random lattice with SLM traps
    initial_lattice = simulator.generate_initial_lattice()
    
    # Count initial atoms
    total_atoms = np.sum(initial_lattice)
    side_length = int(np.floor(np.sqrt(total_atoms)))
    print(f"\nTotal atoms loaded: {total_atoms}")
    print(f"Target lattice size: {side_length}x{side_length}")
    print(f"Unused atoms: {total_atoms - side_length*side_length}")

    # Rearrange atoms using AOD movements without visualization
    target_lattice, retention_rate, execution_time = simulator.rearrange_atoms(show_visualization=False)

    # Print performance metrics
    print(f"\nPerformance metrics:")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"