"""
Runner script for the defect-free lattice algorithm.
This script generates an initial atom lattice and rearranges it to create a defect-free region.
"""
import numpy as np
import argparse
import time
from defect_free.simulator import LatticeSimulator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run defect-free lattice rearrangement')
    parser.add_argument('--size', type=int, nargs=2, default=[50, 50],
                        help='Initial lattice size (rows, columns)')
    parser.add_argument('--occupation', type=float, default=0.5,
                        help='Occupation probability (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of the rearrangement')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel algorithm for rearrangement')
    args = parser.parse_args()
    
    print(f"Initializing simulator with lattice size {args.size} and occupation {args.occupation}")
    
    # Create simulator instance
    simulator = LatticeSimulator(
        initial_size=tuple(args.size),
        occupation_prob=args.occupation
    )
    
    # Generate initial random lattice
    simulator.generate_initial_lattice(seed=args.seed)
    initial_atoms = np.sum(simulator.field)
    print(f"Generated initial lattice with {initial_atoms} atoms")
    
    # Run the rearrangement algorithm
    print("Running defect-free rearrangement...")
    start_time = time.time()
    target_lattice, retention_rate, execution_time = simulator.rearrange_for_defect_free(
        show_visualization=args.visualize,
        parallel=args.parallel
    )
    total_time = time.time() - start_time
    
    # Print results
    print("\nResults:")
    print(f"Created defect-free region of size {simulator.side_length}x{simulator.side_length}")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"Algorithm execution time: {execution_time:.3f} seconds")
    print(f"Total time (including visualization): {total_time:.3f} seconds")
    print(f"Total atom movements: {len(simulator.movement_history)}")
    
    # You can add code here to save the result or further analyze it

if __name__ == "__main__":
    main()
