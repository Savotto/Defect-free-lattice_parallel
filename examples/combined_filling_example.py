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
    
    # Create simulator with initial 20x20 lattice and 50% occupation probability
    simulator = LatticeSimulator(initial_size=(100, 100), occupation_prob=0.50)
    simulator.generate_initial_lattice()
    
    # Calculate the optimal target zone size
    initial_atoms = np.sum(simulator.slm_lattice)
    # Use a carefully calibrated target size to optimize fill rate
    # 90% of available atoms should be sufficient to create a nearly perfect square
    target_atoms = int(initial_atoms * 0.9)
    max_square_size = int(np.floor(np.sqrt(target_atoms)))
    simulator.side_length = max_square_size
    
    print(f"Using target zone size {simulator.side_length}x{simulator.side_length}")
    print(f"(Requiring {simulator.side_length**2} atoms out of {initial_atoms} available)")
    print(f"Target utilization: {simulator.side_length**2 / initial_atoms:.1%} of available atoms")
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial configuration
    print("Initial lattice configuration:")
    print(f"Number of atoms: {initial_atoms}")
    print(f"Target zone size: {simulator.side_length}x{simulator.side_length}")
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Visualize initial lattice
    visualizer.plot_lattice(initial_lattice, title="Initial Lattice")
    plt.show(block=False)
    
    # Apply the combined filling strategy
    print("\nApplying combined filling strategy...")
    final_lattice, fill_rate, execution_time = simulator.movement_manager.combined_filling_strategy(
        show_visualization=True
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
