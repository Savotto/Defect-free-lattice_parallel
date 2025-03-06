"""
Example demonstrating the row-wise centering algorithm for atom rearrangement.
"""
import numpy as np
import matplotlib.pyplot as plt
from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Initialize simulator with a reasonable size and occupation probability
    # Using a fixed random seed for reproducible results
    np.random.seed(42)
    
    # Create simulator with initial 20x20 lattice and 70% occupation probability
    simulator = LatticeSimulator(initial_size=(20, 20), occupation_prob=0.7)
    simulator.generate_initial_lattice()
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial configuration
    print("Initial lattice configuration:")
    initial_atoms = np.sum(simulator.slm_lattice)
    print(f"Number of atoms: {initial_atoms}")
    
    # Visualize initial lattice
    fig1 = visualizer.plot_lattice(simulator.slm_lattice, title="Initial Lattice")
    plt.show(block=False)
    
    # Run row-wise centering algorithm
    print("\nRearranging atoms using row-wise centering method...")
    final_lattice, retention_rate, execution_time = simulator.rearrange_for_defect_free(
        show_visualization=True
    )
    
    # Show comprehensive analysis
    print(f"\nRearrangement completed in {execution_time:.3f} seconds")
    print(f"Retention rate: {retention_rate:.2%}")
    
    # Display final analysis with all metrics
    fig2 = visualizer.show_final_analysis()
    
    # Show density heatmap
    fig3 = visualizer.plot_density_heatmap()
    
    plt.show()

if __name__ == "__main__":
    main()
