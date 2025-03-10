"""
Example demonstrating a combined row-wise followed by column-wise centering approach.
This shows how using both strategies sequentially might improve the final result.
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
    print(f"Initial number of atoms: {initial_atoms}")
    print(f"Optimal target zone size: {simulator.side_length}x{simulator.side_length} (using {simulator.side_length**2} atoms)")
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Visualize initial lattice
    fig1 = visualizer.plot_lattice(initial_lattice, title="Initial Lattice")
    plt.show(block=False)
    
    # Step 1: Run row-wise centering algorithm first
    print("\nStep 1: Rearranging atoms using row-wise centering...")
    row_start_time = time.time()
    row_lattice, row_retention, row_time = simulator.movement_manager.row_wise_centering(
        show_visualization=False,  # Don't show animation yet
        parallel=True
    )
    
    # Store the state after row-wise centering
    after_row_lattice = simulator.field.copy()
    row_total_time = time.time() - row_start_time
    
    print(f"Row-wise centering completed in {row_time:.3f} seconds")
    print(f"Row-wise retention rate: {row_retention:.2%}")
    
    # Step 2: Run column-wise centering on the result
    print("\nStep 2: Applying column-wise centering to further improve the lattice...")
    col_start_time = time.time()
    final_lattice, final_retention, col_time = simulator.movement_manager.column_wise_centering(
        show_visualization=False,  # We'll create a custom visualization at the end
        parallel=True
    )
    
    # Store the final state
    after_col_lattice = simulator.field.copy()
    col_total_time = time.time() - col_start_time
    
    # Calculate overall metrics
    total_time = row_total_time + col_total_time
    print(f"Column-wise centering completed in {col_time:.3f} seconds")
    print(f"Final retention rate: {final_retention:.2%}")
    print(f"Total execution time: {total_time:.3f} seconds")
    
    # Compare results between steps
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    # Count atoms in target region at each stage
    initial_target_atoms = np.sum(initial_lattice[target_start_row:target_end_row, 
                                              target_start_col:target_end_col])
    row_target_atoms = np.sum(after_row_lattice[target_start_row:target_end_row, 
                                            target_start_col:target_end_col])
    final_target_atoms = np.sum(after_col_lattice[target_start_row:target_end_row, 
                                               target_start_col:target_end_col])
    target_size = (target_end_row - target_start_row) * (target_end_col - target_start_col)
    
    print("\nComparison of target region filling:")
    print(f"Initial target atoms: {initial_target_atoms}/{target_size} ({initial_target_atoms/target_size:.2%})")
    print(f"After row-wise: {row_target_atoms}/{target_size} ({row_target_atoms/target_size:.2%})")
    print(f"After column-wise: {final_target_atoms}/{target_size} ({final_target_atoms/target_size:.2%})")
    print(f"Improvement from row to combined: {(final_target_atoms - row_target_atoms)/target_size:.2%}")
    
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
        title="After Row-wise Centering", 
        highlight_region=target_region,
        ax=axes[1]
    )
    
    # Plot after column-wise centering
    visualizer.plot_lattice(
        after_col_lattice, 
        title="After Column-wise Centering", 
        highlight_region=target_region,
        ax=axes[2]
    )
    
    plt.tight_layout()
    
    # Show comprehensive analysis of the final state
    fig_analysis = visualizer.show_final_analysis()
    
    # Create a visualization of the defects at each stage
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract target regions
    initial_target = initial_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    row_target = after_row_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    final_target = after_col_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
    
    # Create heat maps of defects
    axes2[0].imshow(1-initial_target, cmap='Reds', vmin=0, vmax=1)
    axes2[0].set_title(f"Initial Defects: {target_size - initial_target_atoms}")
    
    axes2[1].imshow(1-row_target, cmap='Reds', vmin=0, vmax=1)
    axes2[1].set_title(f"After Row-wise: {target_size - row_target_atoms} defects")
    
    axes2[2].imshow(1-final_target, cmap='Reds', vmin=0, vmax=1)
    axes2[2].set_title(f"After Column-wise: {target_size - final_target_atoms} defects")
    
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Show all figures
    plt.show()

if __name__ == "__main__":
    main()
