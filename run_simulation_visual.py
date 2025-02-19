import numpy as np
from lattice import LatticeSimulator
import time

def main():
    # Create a simulator with a 10x10 initial lattice and 70% occupation probability
    simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=0.7)

    # Generate the initial random lattice with SLM traps
    initial_lattice = simulator.generate_initial_lattice()
    print("\nInitial configuration:")
    simulator.visualize_lattices()
    
    # Count initial atoms
    total_atoms = np.sum(initial_lattice)
    side_length = int(np.floor(np.sqrt(total_atoms)))
    print(f"\nTotal atoms loaded: {total_atoms}")
    print(f"Target lattice size: {side_length}x{side_length}")
    print(f"Unused atoms: {total_atoms - side_length*side_length}")
    
    time.sleep(2)  # Pause to show initial configuration

    # Rearrange atoms using AOD movements
    target_lattice, retention_rate, execution_time = simulator.rearrange_atoms()

    # Print performance metrics
    print(f"\nPerformance metrics:")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"Execution time: {execution_time:.3f} seconds")

    # Show final configuration
    print("\nFinal configuration:")
    simulator.visualize_lattices()

if __name__ == "__main__":
    main()