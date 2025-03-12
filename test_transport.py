
"""
Example demonstrating the transport efficiency implementation with both
perfect and realistic transport fidelities.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test transport efficiency in defect-free lattice')
    parser.add_argument('--no-losses', action='store_true', help='Disable transport losses')
    args = parser.parse_args()
    
    # Initialize physical constraints with transport efficiency
    physical_constraints = {
        'trap_transfer_fidelity': 1.0 if args.no_losses else 0.95  # Perfect fidelity if no losses
    }
    
    # Create simulator with reasonable parameters
    simulator = LatticeSimulator(
        initial_size=(100, 100),
        occupation_prob=0.7,
        physical_constraints=physical_constraints
    )
    
    simulator.generate_initial_lattice()
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Print initial information
    initial_atoms = np.sum(simulator.field)
    max_square_size = int(np.floor(np.sqrt(initial_atoms)))
    
    print(f"Initial setup:")
    print(f"- Transport efficiency: {simulator.constraints['trap_transfer_fidelity']:.2%}")
    print(f"- Total atoms available: {initial_atoms}")
    print(f"- Theoretical maximum square size: {max_square_size}x{max_square_size}")
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Create figure to compare approaches
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot initial lattice
    visualizer.plot_lattice(
        initial_lattice,
        title="Initial Lattice",
        ax=axes[0]
    )
    
    # Approach 1: With transport losses
    print("\n========== Testing with Transport Losses ==========")
    # Clone the simulator with transport losses enabled
    loss_constraints = {
        'trap_transfer_fidelity': 0.95  # 95% efficiency
    }
    
    loss_simulator = LatticeSimulator(
        initial_size=(100, 100),
        occupation_prob=0.7,
        physical_constraints=loss_constraints
    )
    loss_simulator.field = initial_lattice.copy()
    loss_vis = LatticeVisualizer(loss_simulator)
    loss_simulator.visualizer = loss_vis
    
    # Run with transport losses
    loss_lattice, loss_rate, loss_time = loss_simulator.rearrange_for_defect_free(
        show_visualization=False
    )
    
    # Get target region for the transport loss approach
    loss_target = loss_simulator.movement_manager.target_region
    
    # Plot transport loss result
    visualizer.plot_lattice(
        loss_lattice,
        title=f"With Transport Losses (95%)\nFill Rate: {loss_rate:.2%}",
        highlight_region=loss_target,
        ax=axes[1]
    )
    
    # Approach 2: No losses (perfect fidelity)
    print("\n========== Testing With Perfect Transport ==========")
    # Clone the simulator with perfect transport
    perfect_constraints = {
        'trap_transfer_fidelity': 1.0  # Perfect efficiency
    }
    
    perfect_simulator = LatticeSimulator(
        initial_size=(30, 30),
        occupation_prob=0.7,
        physical_constraints=perfect_constraints
    )
    perfect_simulator.field = initial_lattice.copy()
    perfect_vis = LatticeVisualizer(perfect_simulator)
    perfect_simulator.visualizer = perfect_vis
    
    # Run with perfect transport
    perfect_lattice, perfect_rate, perfect_time = perfect_simulator.rearrange_for_defect_free(
        show_visualization=False
    )
    
    # Get target region for the perfect approach
    perfect_target = perfect_simulator.movement_manager.target_region
    
    # Plot perfect transport result
    visualizer.plot_lattice(
        perfect_lattice,
        title=f"Perfect Transport (100%)\nFill Rate: {perfect_rate:.2%}",
        highlight_region=perfect_target,
        ax=axes[2]
    )
    
    # Compare the results
    print("\n========== Comparison of Approaches ==========")
    print(f"With Transport Losses (95%):")
    print(f"- Target size: {loss_simulator.side_length}x{loss_simulator.side_length}")
    print(f"- Fill rate: {loss_rate:.2%}")
    print(f"- Execution time: {loss_time:.3f} seconds")
    
    print(f"\nPerfect Transport (100%):")
    print(f"- Target size: {perfect_simulator.side_length}x{perfect_simulator.side_length}")
    print(f"- Fill rate: {perfect_rate:.2%}")
    print(f"- Execution time: {perfect_time:.3f} seconds")
    
    # Set overall title
    plt.suptitle("Impact of Transport Efficiency on Defect-Free Lattice", fontsize=16)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()