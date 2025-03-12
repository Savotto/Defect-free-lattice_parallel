"""
Example demonstrating the combined filling strategy for atom rearrangement.
This applies row-wise centering, column-wise centering, and defect repair in sequence.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Initialize simulator with a reasonable size and occupation probability
    # Using a fixed random seed for reproducible results
    np.random.seed(42)
    
    # Step 1: Initialize the lattice
    # Create simulator with initial 100x100 lattice and 50% occupation probability
    simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=1.0)
    simulator.generate_initial_lattice()
    
    # Step 2: Calculate the maximum possible target size based on available atoms
    initial_atoms = np.sum(simulator.slm_lattice)
    print(f"Total available atoms: {initial_atoms}")
    
    # Calculate maximum square size using ALL available atoms (no utilization factor)
    max_square_size = simulator.calculate_max_defect_free_size()
    
    print(f"Calculated maximum target zone: {max_square_size}x{max_square_size}")
    print(f"This requires {max_square_size**2} atoms out of {initial_atoms} available")
    print(f"Using {max_square_size**2} atoms for a perfect square")
    
    # Initialize visualizer for tracking the rearrangement
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial configuration before rearrangement
    print("\nInitial configuration before rearrangement:")
    print(f"Number of atoms: {initial_atoms}")
    print(f"Target zone size: {simulator.side_length}x{simulator.side_length}")
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Visualize initial lattice
    visualizer.plot_lattice(initial_lattice, title="Initial Lattice")
    plt.show(block=False)
    
    # Step 3: Apply rearrangement methods to make the target region defect-free
    # This will be executed in the combined_filling_strategy below
    print("\nApplying combined filling strategy...")
    final_lattice, fill_rate, execution_time = simulator.movement_manager.combined_filling_strategy(
        show_visualization=False
    )
    
    # Store the final state
    after_filling_lattice = simulator.field.copy()
    
    # Get target region
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    print(f"\nCombined filling completed in {execution_time:.3f} seconds")
    print(f"Final fill rate: {fill_rate:.2%}")
    
    # Create a figure comparing initial and final states
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot initial state
    visualizer.plot_lattice(
        initial_lattice, 
        title="Initial Lattice", 
        highlight_region=target_region,
        ax=axes[0]
    )
    
    # Plot final state
    visualizer.plot_lattice(
        after_filling_lattice, 
        title="After Combined Filling", 
        highlight_region=target_region,
        ax=axes[1]
    )
    
    plt.tight_layout()
    
    # Create a visualization of the defects before and after
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract target regions
    initial_target = initial_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    final_target = after_filling_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    
    # Calculate defect counts
    initial_defects = np.sum(initial_target == 0)
    final_defects = np.sum(final_target == 0)
    
    # Create heat maps of defects
    axes2[0].imshow(1-initial_target, cmap='Reds', vmin=0, vmax=1)
    axes2[0].set_title(f"Initial Defects: {initial_defects}")
    
    axes2[1].imshow(1-final_target, cmap='Reds', vmin=0, vmax=1)
    axes2[1].set_title(f"Remaining Defects: {final_defects}")
    
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Display final analysis with all metrics
    fig_analysis = visualizer.show_final_analysis()
    
    plt.show()

if __name__ == "__main__":
    main()
