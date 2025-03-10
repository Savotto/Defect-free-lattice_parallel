"""
Example demonstrating the row-wise centering algorithm for atom rearrangement.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
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
    
    # Run row-wise centering algorithm with parallel movement
    print("\nRearranging atoms using row-wise centering method (with parallel movements)...")
    final_lattice, retention_rate, execution_time = simulator.rearrange_for_defect_free(
        show_visualization=True,
        parallel=True  # Enable parallel movement of atoms within rows
    )
    
    # Show comprehensive analysis
    print(f"\nRearrangement completed in {execution_time:.3f} seconds")
    print(f"Retention rate: {retention_rate:.2%}")
    
    # Compare with sequential version
    print("\nComparing with sequential movement for reference...")
    # Reset simulator to have the same starting point
    np.random.seed(42)
    simulator_seq = LatticeSimulator(initial_size=(20, 20), occupation_prob=0.7)
    simulator_seq.generate_initial_lattice()
    
    # Run sequential version
    start_time = time.time()
    final_lattice_seq, retention_rate_seq, execution_time_seq = simulator_seq.rearrange_for_defect_free(
        show_visualization=False,
        parallel=False  # Disable parallel movement
    )
    
    # Print performance comparison
    speedup = execution_time_seq / execution_time if execution_time > 0 else 0
    print(f"Sequential execution time: {execution_time_seq:.3f} seconds")
    print(f"Parallel execution time: {execution_time:.3f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Display final analysis with all metrics
    fig2 = visualizer.show_final_analysis()
    
    # Show density heatmap
    fig3 = visualizer.plot_density_heatmap()
    
    # Analyze movement opportunities and unreachable defects
    fig4 = visualizer.visualize_movement_opportunities()
    print("Movement opportunity analysis: Green lines show reachable paths, red dotted lines show unreachable paths")
    
    plt.show()

if __name__ == "__main__":
    main()
