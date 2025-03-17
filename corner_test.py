"""
Test file to compare the corner-based filling strategy against the existing center-based approach.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from defect_free import LatticeSimulator, LatticeVisualizer
from defect_free.movement import MovementManager

# No need to import or monkey-patch the corner filling methods,
# as they are already defined in the MovementManager class

def test_filling_strategies(initial_size=(50, 50), occupation_prob=0.7, random_seed=42):
    """
    Test both filling strategies with the same initial conditions and compare performance.
    
    Args:
        initial_size: Initial lattice dimensions
        occupation_prob: Probability of atom occupation
        random_seed: Random seed for reproducibility
    """
    print(f"Testing with lattice size: {initial_size}, occupation: {occupation_prob}")
    
    # Fix random seed for reproducible results
    np.random.seed(random_seed)
    
    # ---------------------- Center-based strategy ----------------------
    # Initialize simulator for center-based approach
    center_simulator = LatticeSimulator(
        initial_size=initial_size, 
        occupation_prob=occupation_prob
    )
    center_simulator.generate_initial_lattice()
    
    # Initialize visualizer
    center_visualizer = LatticeVisualizer(center_simulator)
    center_simulator.visualizer = center_visualizer
    
    # Store initial lattice for both approaches
    initial_lattice = center_simulator.field.copy()
    
    # Apply center-based filling strategy
    print("\n========== Testing Center-Based Strategy ==========")
    start_time = time.time()
    center_lattice, center_fill_rate, center_time = center_simulator.movement_manager.combined_filling_strategy(
        show_visualization=False
    )
    center_total_time = time.time() - start_time
    
    # Get metrics for center-based approach
    center_target_region = center_simulator.movement_manager.target_region
    target_start_row, target_start_col, target_end_row, target_end_col = center_target_region
    center_defects = np.sum(center_lattice[target_start_row:target_end_row, 
                                         target_start_col:target_end_col] == 0)
    center_movements = len(center_simulator.movement_history)
    center_physical_time = sum(move['time'] for move in center_simulator.movement_history)
    
    print(f"Center-based strategy completed in {center_total_time:.3f} seconds")
    print(f"Fill rate: {center_fill_rate:.2%}")
    print(f"Remaining defects: {center_defects}")
    print(f"Total movements: {center_movements}")
    print(f"Physical movement time: {center_physical_time:.6f} seconds")
    
    # ---------------------- Corner-based strategy ----------------------
    # Re-initialize simulator with the same conditions for corner-based approach
    np.random.seed(random_seed)  # Reset seed to ensure same initial lattice
    corner_simulator = LatticeSimulator(
        initial_size=initial_size, 
        occupation_prob=occupation_prob
    )
    corner_simulator.generate_initial_lattice()
    
    # Verify that initial lattice is the same
    assert np.array_equal(initial_lattice, corner_simulator.field)
    
    # Initialize visualizer
    corner_visualizer = LatticeVisualizer(corner_simulator)
    corner_simulator.visualizer = corner_visualizer
    
    # Apply corner-based filling strategy
    print("\n========== Testing Corner-Based Strategy ==========")
    start_time = time.time()
    corner_lattice, corner_fill_rate, corner_time = corner_simulator.movement_manager.corner_filling_strategy(
        show_visualization=False
    )
    corner_total_time = time.time() - start_time
    
    # Get metrics for corner-based approach
    corner_target_region = corner_simulator.movement_manager.corner_target_region
    target_start_row, target_start_col, target_end_row, target_end_col = corner_target_region
    corner_defects = np.sum(corner_lattice[target_start_row:target_end_row, 
                                         target_start_col:target_end_col] == 0)
    corner_movements = len(corner_simulator.movement_history)
    corner_physical_time = sum(move['time'] for move in corner_simulator.movement_history)
    
    print(f"Corner-based strategy completed in {corner_total_time:.3f} seconds")
    print(f"Fill rate: {corner_fill_rate:.2%}")
    print(f"Remaining defects: {corner_defects}")
    print(f"Total movements: {corner_movements}")
    print(f"Physical movement time: {corner_physical_time:.6f} seconds")
    
    # ---------------------- Comparison and Visualization ----------------------
    # Compare the results
    print("\n========== Strategy Comparison ==========")
    print(f"Fill rate - Center: {center_fill_rate:.2%}, Corner: {corner_fill_rate:.2%}")
    print(f"Execution time - Center: {center_total_time:.3f}s, Corner: {corner_total_time:.3f}s")
    print(f"Physical time - Center: {center_physical_time:.6f}s, Corner: {corner_physical_time:.6f}s")
    print(f"Movements - Center: {center_movements}, Corner: {corner_movements}")
    
    # Determine which strategy was better
    fill_diff = corner_fill_rate - center_fill_rate
    time_ratio = corner_total_time / center_total_time if center_total_time > 0 else float('inf')
    physical_time_ratio = corner_physical_time / center_physical_time if center_physical_time > 0 else float('inf')
    
    print("\nPerformance Summary:")
    if fill_diff > 0.01:  # 1% better fill rate
        print(f"Corner-based strategy achieved {fill_diff:.2%} better fill rate")
    elif fill_diff < -0.01:  # 1% worse fill rate
        print(f"Center-based strategy achieved {-fill_diff:.2%} better fill rate")
    else:
        print(f"Both strategies achieved similar fill rates (difference: {fill_diff:.2%})")
    
    if time_ratio < 0.9:  # 10% faster
        print(f"Corner-based strategy was {(1-time_ratio)*100:.1f}% faster in execution time")
    elif time_ratio > 1.1:  # 10% slower
        print(f"Corner-based strategy was {(time_ratio-1)*100:.1f}% slower in execution time")
    else:
        print(f"Both strategies had similar execution times (ratio: {time_ratio:.2f})")
    
    if physical_time_ratio < 0.9:  # 10% faster
        print(f"Corner-based strategy was {(1-physical_time_ratio)*100:.1f}% faster in physical movement time")
    elif physical_time_ratio > 1.1:  # 10% slower
        print(f"Corner-based strategy was {(physical_time_ratio-1)*100:.1f}% slower in physical movement time")
    else:
        print(f"Both strategies had similar physical movement times (ratio: {physical_time_ratio:.2f})")
    
    # Create a figure comparing both approaches
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot initial state
    center_visualizer.plot_lattice(
        initial_lattice, 
        title="Initial Lattice", 
        ax=axes[0]
    )
    
    # Plot center-based result
    center_visualizer.plot_lattice(
        center_lattice, 
        title=f"Center-Based Strategy\nFill Rate: {center_fill_rate:.2%}", 
        highlight_region=center_target_region,
        ax=axes[1]
    )
    
    # Plot corner-based result
    corner_visualizer.plot_lattice(
        corner_lattice, 
        title=f"Corner-Based Strategy\nFill Rate: {corner_fill_rate:.2%}", 
        highlight_region=corner_target_region,
        ax=axes[2]
    )
    
    plt.tight_layout()
    
    # Create a visualization of the defects
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract target regions
    center_target = center_lattice[center_target_region[0]:center_target_region[2], 
                                 center_target_region[1]:center_target_region[3]]
    
    corner_target = corner_lattice[corner_target_region[0]:corner_target_region[2], 
                                 corner_target_region[1]:corner_target_region[3]]
    
    # Create heat maps of defects
    axes2[0].imshow(1-center_target, cmap='Reds', vmin=0, vmax=1)
    axes2[0].set_title(f"Center: {center_defects} defects")
    
    axes2[1].imshow(1-corner_target, cmap='Reds', vmin=0, vmax=1)
    axes2[1].set_title(f"Corner: {corner_defects} defects")
    
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'center': {
            'fill_rate': center_fill_rate,
            'execution_time': center_total_time,
            'physical_time': center_physical_time,
            'movements': center_movements,
            'defects': center_defects
        },
        'corner': {
            'fill_rate': corner_fill_rate,
            'execution_time': corner_total_time,
            'physical_time': corner_physical_time,
            'movements': corner_movements,
            'defects': corner_defects
        }
    }

def run_multi_test(sizes=[(20, 20), (50, 50), (80, 80)], occupations=[0.5, 0.7, 0.9]):
    """
    Run tests across multiple lattice sizes and occupation probabilities.
    
    Args:
        sizes: List of (rows, cols) tuples for initial lattice sizes
        occupations: List of occupation probabilities
    """
    results = []
    
    for size in sizes:
        for occ in occupations:
            print(f"\n=== Testing lattice size {size} with occupation {occ} ===")
            test_result = test_filling_strategies(initial_size=size, occupation_prob=occ)
            
            # Add parameters to result
            test_result['parameters'] = {
                'size': size,
                'occupation': occ
            }
            
            results.append(test_result)
    
    # Print summary table
    print("\n=== Summary of All Tests ===")
    print("Size\tOcc\tCenter Fill\tCorner Fill\tCenter Time\tCorner Time\tCenter Phys\tCorner Phys")
    for r in results:
        params = r['parameters']
        center = r['center']
        corner = r['corner']
        
        print(f"{params['size']}\t{params['occupation']:.1f}\t" +
              f"{center['fill_rate']:.2%}\t{corner['fill_rate']:.2%}\t" +
              f"{center['execution_time']:.2f}s\t{corner['execution_time']:.2f}s\t" +
              f"{center['physical_time']:.6f}s\t{corner['physical_time']:.6f}s")
    
    return results

def main():
    """Main function to run the test."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test corner-based vs center-based filling strategies')
    parser.add_argument('--size', type=int, default=50, help='Initial lattice size (default: 50)')
    parser.add_argument('--occupation', type=float, default=0.7, help='Atom occupation probability (default: 0.7)')
    parser.add_argument('--multi', action='store_true', help='Run multiple tests with different parameters')
    args = parser.parse_args()
    
    if args.multi:
        run_multi_test()
    else:
        test_filling_strategies(
            initial_size=(args.size, args.size),
            occupation_prob=args.occupation
        )

if __name__ == "__main__":
    main()