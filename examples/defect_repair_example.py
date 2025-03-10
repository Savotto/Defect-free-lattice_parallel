"""
Example demonstrating the defect repair algorithm for filling holes in a target lattice region.
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
    # Using a lower occupation probability to ensure we have defects to repair
    simulator = LatticeSimulator(initial_size=(20, 20), occupation_prob=0.5)
    simulator.generate_initial_lattice()
    
    # Calculate the optimal target zone size based on available atoms
    initial_atoms = np.sum(simulator.slm_lattice)
    # Use a smaller target size to ensure we can fill most of it
    max_square_size = int(np.floor(np.sqrt(initial_atoms * 0.8)))
    simulator.side_length = max_square_size
    
    print(f"Using target zone size {simulator.side_length}x{simulator.side_length}")
    print(f"(Requiring {simulator.side_length**2} atoms out of {initial_atoms} available)")
    
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
    
    # Step 1: First create a centered arrangement with defects using row-wise centering
    print("\nStep 1: Performing row-wise centering to create initial arrangement...")
    row_lattice, row_retention, row_time = simulator.movement_manager.row_wise_centering(
        show_visualization=False  # Don't show animation yet
    )
    
    # Store the state after row-wise centering
    after_row_lattice = simulator.field.copy()
    
    # Get target region
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    # Count defects in target region
    defects_count = np.sum(after_row_lattice[target_start_row:target_end_row, 
                                            target_start_col:target_end_col] == 0)
    
    print(f"Row-wise centering completed in {row_time:.3f} seconds")
    print(f"Retention rate after centering: {row_retention:.2%}")
    print(f"Defects in target region: {defects_count}")
    
    # Step 2: Now repair the defects by moving atoms from outside the target region
    print("\nStep 2: Repairing defects by moving atoms from outside the target region...")
    repair_start_time = time.time()
    final_lattice, fill_rate, repair_time = simulator.movement_manager.repair_defects(
        show_visualization=True  # Show animation for the repair process
    )
    
    # Store the final state
    after_repair_lattice = simulator.field.copy()
    
    # Calculate overall metrics
    total_time = row_time + repair_time
    
    print(f"Defect repair completed in {repair_time:.3f} seconds")
    print(f"Final fill rate: {fill_rate:.2%}")
    print(f"Total execution time: {total_time:.3f} seconds")
    
    # Create a figure comparing all three states
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot initial state
    visualizer.plot_lattice(
        initial_lattice, 
        title="Initial Lattice", 
        highlight_region=target_region,
        ax=axes[0]
    )
    
    # Plot after row-wise centering
    visualizer.plot_lattice(
        after_row_lattice, 
        title="After Row-wise Centering\n(with defects)", 
        highlight_region=target_region,
        ax=axes[1]
    )
    
    # Plot after defect repair
    visualizer.plot_lattice(
        after_repair_lattice, 
        title="After Defect Repair", 
        highlight_region=target_region,
        ax=axes[2]
    )
    
    plt.tight_layout()
    
    # Create a visualization of the defects at each stage
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract target regions
    initial_target = initial_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    row_target = after_row_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    final_target = after_repair_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    
    # Calculate defect counts
    initial_defects = np.sum(initial_target == 0)
    row_defects = np.sum(row_target == 0)
    final_defects = np.sum(final_target == 0)
    
    # Create heat maps of defects
    axes2[0].imshow(1-initial_target, cmap='Reds', vmin=0, vmax=1)
    axes2[0].set_title(f"Initial Defects: {initial_defects}")
    
    axes2[1].imshow(1-row_target, cmap='Reds', vmin=0, vmax=1)
    axes2[1].set_title(f"After Centering: {row_defects} defects")
    
    axes2[2].imshow(1-final_target, cmap='Reds', vmin=0, vmax=1)
    axes2[2].set_title(f"After Repair: {final_defects} defects")
    
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Comprehensive analysis
    print("\nDefect repair effectiveness:")
    print(f"Initial defects in target region: {initial_defects}")
    print(f"Defects after centering: {row_defects}")
    print(f"Defects after repair: {final_defects}")
    print(f"Defects fixed: {row_defects - final_defects}")
    print(f"Repair effectiveness: {(row_defects - final_defects) / row_defects:.2%}")
    
    # Display final analysis with all metrics
    fig_analysis = visualizer.show_final_analysis()
    
    plt.show()

if __name__ == "__main__":
    main()
