"""
Example demonstrating the column-wise centering algorithm for atom rearrangement.
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
    
    # Calculate the optimal target zone size based on available atoms
    initial_atoms = np.sum(simulator.slm_lattice)
    max_square_size = int(np.floor(np.sqrt(initial_atoms)))
    simulator.side_length = max_square_size
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial configuration
    print("Initial lattice configuration:")
    print(f"Number of atoms: {initial_atoms}")
    print(f"Optimal target zone size: {simulator.side_length}x{simulator.side_length} (using {simulator.side_length**2} atoms)")
    
    # Visualize initial lattice
    fig1 = visualizer.plot_lattice(simulator.slm_lattice, title="Initial Lattice")
    plt.show(block=False)
    
    # Run column-wise centering algorithm with parallel movement
    print("\nRearranging atoms using column-wise centering method (with parallel movements)...")
    final_lattice, retention_rate, execution_time = simulator.movement_manager.column_wise_centering(
        show_visualization=True,
        parallel=True  # Enable parallel movement of atoms within columns
    )
    
    # Show comprehensive analysis
    print(f"\nRearrangement completed in {execution_time:.3f} seconds")
    print(f"Retention rate: {retention_rate:.2%}")
    
    # Compare with row-wise version
    print("\nComparing with row-wise centering for reference...")
    # Reset simulator to have the same starting point
    np.random.seed(42)
    simulator_row = LatticeSimulator(initial_size=(20, 20), occupation_prob=0.7)
    simulator_row.generate_initial_lattice()
    # Use the same optimal target zone
    simulator_row.side_length = max_square_size
    
    # Run row-wise version
    start_time = time.time()
    final_lattice_row, retention_rate_row, execution_time_row = simulator_row.movement_manager.row_wise_centering(
        show_visualization=False,
        parallel=True
    )
    
    # Print performance comparison
    print(f"Row-wise execution time: {execution_time_row:.3f} seconds")
    print(f"Column-wise execution time: {execution_time:.3f} seconds")
    print(f"Row-wise retention rate: {retention_rate_row:.2%}")
    print(f"Column-wise retention rate: {retention_rate:.2%}")
    
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
