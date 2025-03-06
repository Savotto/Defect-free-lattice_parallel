"""
Basic example of using the lattice simulator to rearrange atoms.
"""
import numpy as np
from defect_free import LatticeSimulator, LatticeVisualizer
import matplotlib.pyplot as plt

def main():
    # Initialize simulator with a 10x10 lattice and 70% occupation probability
    simulator = LatticeSimulator(initial_size=(10, 10), occupation_prob=0.7)
    simulator.generate_initial_lattice()
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial configuration
    print("Initial lattice configuration:")
    print(simulator.slm_lattice)
    print(f"Number of atoms: {np.sum(simulator.slm_lattice)}")
    
    # Rearrange atoms into a perfect square lattice
    simulator.rearrange_for_perfect_lattice(show_visualization=True)
    
    # Print final configuration
    print("\nFinal lattice configuration:")
    print(simulator.target_lattice)
    print(f"Number of atoms in target: {np.sum(simulator.target_lattice)}")
    
    plt.show()

if __name__ == '__main__':
    main()