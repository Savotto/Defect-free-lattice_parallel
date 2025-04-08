"""
Example demonstrating atom rearrangement strategies.
This script initializes a lattice with a random distribution of atoms and
applies a filling strategy to create a defect-free region.

To switch between strategies, simply edit the strategy call in the code:
- Use simulator.movement_manager.center_filling_strategy() for center filling
- Use simulator.movement_manager.corner_filling_strategy() for corner filling
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Initialize simulator with default parameters
    # Use a fixed random seed for reproducible results
    np.random.seed(42)
    
    # Configuration parameters - modify these as needed
    lattice_size = (100, 100)
    occupation_prob = 0.25
    
    # Step 1: Initialize the lattice
    simulator = LatticeSimulator(initial_size=lattice_size, occupation_prob=occupation_prob)
    simulator.generate_initial_lattice()
    
    # Step 2: Calculate the maximum possible target size based on available atoms
    initial_atoms = np.sum(simulator.slm_lattice)
    print(f"Total available atoms: {initial_atoms}")
    
    # Calculate maximum square size using ALL available atoms
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
    
    # Step 3: Apply rearrangement method
    
    # *** CHANGE THIS LINE TO SWITCH BETWEEN STRATEGIES ***
    # Use either:
    # - center_filling_strategy() for center filling
    # - corner_filling_strategy() for corner filling
    print("\nApplying filling strategy...")
    final_lattice, fill_rate, execution_time = simulator.movement_manager.corner_filling_strategy(show_visualization=True)
    
    # Name of the current strategy for display purposes
    strategy_name = "Corner"  # Change this if you change the strategy above
    
    # Store the final state
    after_filling_lattice = simulator.field.copy()
    
    # Get target region
    target_region = simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = target_region
    
    print(f"\n{strategy_name} filling completed in {execution_time:.3f} seconds")
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
        title=f"After {strategy_name} Filling", 
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

    # Create the animation
    print("\nCreating animation of movements...")
    #animation = visualizer.animate_movements(simulator.movement_history)

    # Save the animation as a GIF
    #visualizer.save_animation("movement_animation_center20x20.gif", fps=10)

if __name__ == "__main__":
    main()
