"""
Runner script for the defect-free lattice algorithm.
This script generates an initial atom lattice and rearranges it to create a defect-free region.
"""
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt
from defect_free.simulator import LatticeSimulator
from defect_free import LatticeVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run defect-free lattice rearrangement')
    parser.add_argument('--size', type=int, nargs=2, default=[100, 100],
                        help='Initial lattice size (rows, columns)')
    parser.add_argument('--occupation', type=float, default=0.5,
                        help='Occupation probability (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of the rearrangement process')
    parser.add_argument('--plot', action='store_true',
                        help='Show plots of initial and final states')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel algorithm for rearrangement')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output (default: quiet)')
    parser.add_argument('--algorithm', type=str, default='combined',
                        choices=['row_wise', 'column_wise', 'combined', 'repair'],
                        help='Algorithm to use for rearrangement')
    parser.add_argument('--save-plots', type=str, default=None,
                        help='Save plots to the specified directory')
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info(f"Initializing simulator with lattice size {args.size} and occupation {args.occupation}")
    
    # Create simulator instance
    simulator = LatticeSimulator(
        initial_size=tuple(args.size),
        occupation_prob=args.occupation
    )
    
    # Generate initial random lattice
    simulator.generate_initial_lattice(seed=args.seed)
    initial_atoms = np.sum(simulator.field)
    logger.info(f"Generated initial lattice with {initial_atoms} atoms")
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Store the initial lattice for comparison
    initial_lattice = simulator.field.copy()
    
    # Visualize initial lattice if plotting is enabled
    if args.plot:
        visualizer.plot_lattice(initial_lattice, title="Initial Lattice")
        plt.show(block=False)
    
    # Run the rearrangement algorithm through the simulator
    logger.info(f"Running defect-free rearrangement with {args.algorithm} algorithm...")
    start_time = time.time()
    
    # Choose the algorithm based on command-line argument
    if args.algorithm == 'row_wise':
        target_lattice, retention_rate, execution_time = simulator.movement_manager.row_wise_centering(
            show_visualization=args.visualize
        )
    elif args.algorithm == 'column_wise':
        target_lattice, retention_rate, execution_time = simulator.movement_manager.column_wise_centering(
            show_visualization=args.visualize
        )
    elif args.algorithm == 'repair':
        target_lattice, retention_rate, execution_time = simulator.movement_manager.repair_defects(
            show_visualization=args.visualize
        )
    else:  # Default to combined
        target_lattice, retention_rate, execution_time = simulator.movement_manager.combined_filling_strategy(
            show_visualization=args.visualize
        )
    
    total_time = time.time() - start_time
    
    # Store the final state
    final_lattice = simulator.field.copy()
    
    # Print results
    print("\nResults:")
    print(f"Algorithm: {args.algorithm}")
    print(f"Created defect-free region of size {simulator.side_length}x{simulator.side_length}")
    print(f"Initial atoms: {initial_atoms}")
    
    # Get target region for visualizing and calculating results
    target_region = simulator.movement_manager.target_region
    if target_region:
        target_start_row, target_start_col, target_end_row, target_end_col = target_region
        atoms_in_target = np.sum(simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col])
        print(f"Atoms in target region: {atoms_in_target}")
        print(f"Retention rate: {retention_rate:.2%} (atoms in target / initial atoms)")
    else:
        print(f"Retention rate: {retention_rate:.2%}")
        
    print(f"Algorithm execution time: {execution_time:.3f} seconds")
    print(f"Total time (including visualization): {total_time:.3f} seconds")
    print(f"Total atom movements: {len(simulator.movement_history)}")
    
    if args.verbose:
        # Additional detailed statistics in verbose mode
        total_distance = sum(move.get('time', 0) for move in simulator.movement_history)
        avg_distance = total_distance / len(simulator.movement_history) if simulator.movement_history else 0
        print(f"\nDetailed Statistics:")
        print(f"Average movement time: {avg_distance:.6f} seconds")
        print(f"Total physical movement time: {total_distance:.6f} seconds")
        
        # Movement types breakdown
        move_types = {}
        for move in simulator.movement_history:
            move_type = move.get('type', 'unknown')
            if move_type not in move_types:
                move_types[move_type] = 0
            move_types[move_type] += 1
            
        print("\nMovement type breakdown:")
        for move_type, count in move_types.items():
            print(f"  {move_type}: {count} moves")
    
    # Create comparison plots if plotting is enabled
    if args.plot and target_region:
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
            final_lattice, 
            title=f"After {args.algorithm.replace('_', ' ').title()} Algorithm", 
            highlight_region=target_region,
            ax=axes[1]
        )
        
        plt.tight_layout()
        
        # Save the figure if requested
        if args.save_plots:
            fig.savefig(f"{args.save_plots}/{args.algorithm}_comparison.png", dpi=300, bbox_inches='tight')
        
        # Create a visualization of the defects before and after
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract target regions
        initial_target = initial_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
        final_target = final_lattice[target_start_row:target_end_row, target_start_col:target_end_col]
        
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
        
        # Save the defect figure if requested
        if args.save_plots:
            fig2.savefig(f"{args.save_plots}/{args.algorithm}_defects.png", dpi=300, bbox_inches='tight')
        
        # Display final analysis with all metrics if verbose
        if args.verbose:
            fig_analysis = visualizer.show_final_analysis()
            
            # Save the analysis figure if requested
            if args.save_plots:
                fig_analysis.savefig(f"{args.save_plots}/{args.algorithm}_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    main()
