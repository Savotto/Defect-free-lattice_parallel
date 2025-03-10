"""
Example demonstrating the atom spreading functionality that aligns atoms outside 
the target zone with the edges of the target zone.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Initialize simulator with a larger size for better visualization of spreading
    # Using a fixed random seed for reproducible results
    np.random.seed(42)
    
    # Create simulator with 30x30 lattice and 40% occupation probability
    simulator = LatticeSimulator(initial_size=(30, 30), occupation_prob=0.7)
    simulator.generate_initial_lattice()
    
    # Set a smaller target zone for better visibility of spreading effect
    initial_atoms = np.sum(simulator.slm_lattice)
    target_size = 20  # Small enough to leave plenty of atoms outside
    simulator.side_length = target_size
    
    print(f"Using target zone size {simulator.side_length}x{simulator.side_length}")
    print(f"(Requiring {simulator.side_length**2} atoms out of {initial_atoms} available)")
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Step 1: Perform row-wise centering first
    print("\n--- Step 1: Performing row-wise centering ---")
    simulator.movement_manager.row_wise_centering(show_visualization=False)
    after_row_centering = simulator.field.copy()
    
    # Step 2: Perform column-wise centering
    print("\n--- Step 2: Performing column-wise centering ---")
    simulator.movement_manager.column_wise_centering(show_visualization=False)
    after_col_centering = simulator.field.copy()
    
    # Get target region for visualization
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    # Step 3: Apply the first round of atom spreading
    print("\n--- Step 3: First round - Spreading atoms outside target zone ---")
    spread_lattice, spread_moves, spread_time = simulator.movement_manager.spread_outer_atoms(
        show_visualization=False
    )
    after_spreading = simulator.field.copy()
    
    # Step 4: Perform first round of column-wise centering after spreading
    print("\n--- Step 4: First round - Column-wise centering after spreading ---")
    simulator.movement_manager.column_wise_centering(show_visualization=False)
    after_first_squeeze = simulator.field.copy()
    
    # Step 5: Repair defects from atoms in aligned columns
    print("\n--- Step 5: Repairing defects using atoms from aligned columns ---")
    simulator.movement_manager.repair_defects_from_aligned_columns(show_visualization=False)
    after_aligned_repair = simulator.field.copy()
    
    # Step 6: Repair any remaining defects with general repair
    print("\n--- Step 6: Repairing remaining defects ---")
    simulator.movement_manager.repair_defects(show_visualization=False)
    after_repair = simulator.field.copy()
    
    print(f"\nComplete atom arrangement process finished")
    
    # Create a visualization showing all steps
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    # Plot initial state
    visualizer.plot_lattice(
        initial_lattice, 
        title="1. Initial Lattice", 
        highlight_region=target_region,
        ax=axes[0, 0]
    )
    
    # Plot after initial centering
    visualizer.plot_lattice(
        after_col_centering, 
        title="2. After Initial Centering", 
        highlight_region=target_region,
        ax=axes[0, 1]
    )
    
    # Plot after spreading and squeezing
    visualizer.plot_lattice(
        after_first_squeeze, 
        title="3. After Spread & Squeeze", 
        highlight_region=target_region,
        ax=axes[0, 2]
    )
    
    # Plot after aligned column repair
    visualizer.plot_lattice(
        after_aligned_repair, 
        title="4. After Aligned Column Repair", 
        highlight_region=target_region,
        ax=axes[0, 3]
    )
    
    # Plot after final repair
    visualizer.plot_lattice(
        after_repair, 
        title="5. After Final Repair", 
        highlight_region=target_region,
        ax=axes[1, 0]
    )
    
    # Hide the unused subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Advanced Atom Arrangement Process", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Update the explanation figure
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.axis('off')
    ax2.set_title("Advanced Filling Strategy", fontsize=14)
    
    explanation_text = """
    Advanced Atom Arrangement Process:
    
    1. Initial Setup: Row and column-wise centering creates basic structure
    
    2. Iterative Spread-Squeeze: Repeatedly spreads atoms outward and then 
       squeezes them back in until no further improvement
    
    3. Aligned Column Repair:
       - Identifies atoms already aligned with target columns (above/below target)
       - Uses path-finding to move these atoms directly into defects
       - Prioritizes atoms that require minimal movements
       
    4. Final Defect Repair:
       - Any remaining defects filled using general repair algorithm
       - Uses atoms from anywhere in the field to optimize fill rate
       
    This focused strategy maximizes target zone fill rate by systematically
    utilizing atoms from the most accessible locations first, with priority
    on atoms that are already column-aligned with defects.
    """
    
    ax2.text(0.05, 0.95, explanation_text, fontsize=12, va='top', ha='left', 
            wrap=True, transform=ax2.transAxes)
    
    # Display optimized paths graphic
    plt.tight_layout()
    
    # Animate the complete filling process
    print("\nShowing animation of the advanced process...")
    simulator.field = initial_lattice.copy()  # Reset to initial state
    final_lattice, fill_rate, execution_time = simulator.movement_manager.combined_filling_strategy(
        show_visualization=True
    )
    
    print(f"Advanced filling completed in {execution_time:.3f} seconds")
    print(f"Final fill rate: {fill_rate:.2%}")
    
    plt.show()

if __name__ == "__main__":
    main()
