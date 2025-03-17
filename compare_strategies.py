#!/usr/bin/env python3
"""
Script to compare the center-based and corner-based filling strategies.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pandas as pd
from defect_free import LatticeSimulator, LatticeVisualizer

def run_comparison(size=(30, 30), occupation_prob=0.7, random_seed=42, 
                  show_visualization=True, save_visualization=False):
    """
    Run both filling strategies with the same initial conditions and compare performance.
    
    Args:
        size: Initial lattice dimensions (rows, columns)
        occupation_prob: Probability of atom occupation
        random_seed: Random seed for reproducibility
        show_visualization: Whether to show animated visualization
        save_visualization: Whether to save visualization to file
    
    Returns:
        Dictionary with results from both strategies
    """
    print(f"Testing with lattice size: {size}, occupation: {occupation_prob}")
    
    # Fix random seed for reproducible results
    np.random.seed(random_seed)
    
    # Initialize simulator with specified parameters
    simulator = LatticeSimulator(initial_size=size, occupation_prob=occupation_prob)
    simulator.generate_initial_lattice()
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Store initial field
    initial_field = simulator.field.copy()
    
    # Run center-based strategy
    print("\n============= Testing Center-Based Strategy =============")
    center_start_time = time.time()
    center_lattice, center_fill_rate, center_execution_time = simulator.movement_manager.rearrange_for_defect_free(
        strategy='center', 
        show_visualization=show_visualization
    )
    center_total_time = time.time() - center_start_time
    
    # Store center strategy results
    # Make sure target_region is initialized
    if simulator.movement_manager.center_manager.target_region is None:
        simulator.movement_manager.center_manager.initialize_target_region()
    center_target_region = simulator.movement_manager.center_manager.target_region
    
    # Continue only if we have a valid target region
    if center_target_region is None:
        print("ERROR: Center target region is None. Cannot continue with comparison.")
        return None
        
    center_movements = len(simulator.movement_history)
    center_physical_time = sum(move['time'] for move in simulator.movement_history)
    center_history = simulator.movement_history.copy()
    
    # Count defects in center target region
    center_target_start_row, center_target_start_col, center_target_end_row, center_target_end_col = center_target_region
    center_defects = np.sum(center_lattice[center_target_start_row:center_target_end_row, 
                                         center_target_start_col:center_target_end_col] == 0)
    
    print(f"Center-based strategy results:")
    print(f"  Target region: {center_target_region}")
    print(f"  Fill rate: {center_fill_rate:.2%}")
    print(f"  Execution time: {center_execution_time:.3f} seconds")
    print(f"  Physical movement time: {center_physical_time:.6f} seconds")
    print(f"  Total movements: {center_movements}")
    print(f"  Remaining defects: {center_defects}")
    
    # Save center visualization if requested
    if save_visualization:
        if hasattr(visualizer, 'ani') and visualizer.ani is not None:
            visualizer.save_animation('center_strategy_animation.mp4', fps=10)
            print("Center strategy animation saved to 'center_strategy_animation.mp4'")
    
    # Reset simulator for corner-based strategy
    simulator = LatticeSimulator(initial_size=size, occupation_prob=occupation_prob)
    # Make sure we have the exact same initial field
    simulator.field = initial_field.copy()
    simulator.total_atoms = np.sum(initial_field)
    
    simulator.slm_lattice = initial_field.copy()
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Run corner-based strategy
    print("\n============= Testing Corner-Based Strategy =============")
    corner_start_time = time.time()
    corner_lattice, corner_fill_rate, corner_execution_time = simulator.movement_manager.rearrange_for_defect_free(
        strategy='corner', 
        show_visualization=show_visualization
    )
    corner_total_time = time.time() - corner_start_time
    
    # Store corner strategy results
    # Make sure target_region is initialized
    if simulator.movement_manager.corner_manager.target_region is None:
        simulator.movement_manager.corner_manager.initialize_target_region()
    corner_target_region = simulator.movement_manager.corner_manager.target_region
    
    # Continue only if we have a valid target region
    if corner_target_region is None:
        print("ERROR: Corner target region is None. Cannot continue with comparison.")
        return None
    
    corner_movements = len(simulator.movement_history)
    corner_physical_time = sum(move['time'] for move in simulator.movement_history)
    corner_history = simulator.movement_history.copy()
    
    # Count defects in corner target region
    corner_target_start_row, corner_target_start_col, corner_target_end_row, corner_target_end_col = corner_target_region
    corner_defects = np.sum(corner_lattice[corner_target_start_row:corner_target_end_row, 
                                         corner_target_start_col:corner_target_end_col] == 0)
    
    print(f"Corner-based strategy results:")
    print(f"  Target region: {corner_target_region}")
    print(f"  Fill rate: {corner_fill_rate:.2%}")
    print(f"  Execution time: {corner_execution_time:.3f} seconds")
    print(f"  Physical movement time: {corner_physical_time:.6f} seconds")
    print(f"  Total movements: {corner_movements}")
    print(f"  Remaining defects: {corner_defects}")
    
    # Save corner visualization if requested
    if save_visualization:
        if hasattr(visualizer, 'ani') and visualizer.ani is not None:
            visualizer.save_animation('corner_strategy_animation.mp4', fps=10)
            print("Corner strategy animation saved to 'corner_strategy_animation.mp4'")
    
    # Create comparison visualization
    create_comparison_visualization(
        initial_field=initial_field,
        center_field=center_lattice,
        corner_field=corner_lattice,
        center_target=center_target_region,
        corner_target=corner_target_region,
        center_results={
            'fill_rate': center_fill_rate,
            'execution_time': center_execution_time,
            'physical_time': center_physical_time,
            'movements': center_movements,
            'defects': center_defects
        },
        corner_results={
            'fill_rate': corner_fill_rate,
            'execution_time': corner_execution_time,
            'physical_time': corner_physical_time,
            'movements': corner_movements,
            'defects': corner_defects
        },
        save_fig=save_visualization
    )
    
    # Compute detailed comparison metrics
    print("\n============= Strategy Comparison =============")
    print(f"Fill rate - Center: {center_fill_rate:.2%}, Corner: {corner_fill_rate:.2%}")
    print(f"Execution time - Center: {center_execution_time:.3f}s, Corner: {corner_execution_time:.3f}s")
    print(f"Physical time - Center: {center_physical_time:.6f}s, Corner: {corner_physical_time:.6f}s")
    print(f"Movements - Center: {center_movements}, Corner: {corner_movements}")
    print(f"Defects - Center: {center_defects}, Corner: {corner_defects}")
    
    # Determine which strategy was better
    fill_diff = corner_fill_rate - center_fill_rate
    time_ratio = corner_execution_time / center_execution_time if center_execution_time > 0 else float('inf')
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
    
    # Return results for potential further analysis
    return {
        'parameters': {
            'size': size,
            'occupation_prob': occupation_prob,
            'random_seed': random_seed
        },
        'center': {
            'field': center_lattice,
            'target_region': center_target_region,
            'fill_rate': center_fill_rate,
            'execution_time': center_execution_time,
            'physical_time': center_physical_time,
            'movements': center_movements,
            'defects': center_defects,
            'history': center_history
        },
        'corner': {
            'field': corner_lattice,
            'target_region': corner_target_region,
            'fill_rate': corner_fill_rate,
            'execution_time': corner_execution_time,
            'physical_time': corner_physical_time,
            'movements': corner_movements,
            'defects': corner_defects,
            'history': corner_history
        }
    }

def create_comparison_visualization(initial_field, center_field, corner_field,
                                   center_target, corner_target, center_results,
                                   corner_results, save_fig=False):
    """
    Create visualizations comparing both strategies.
    
    Args:
        initial_field: The initial lattice field
        center_field: The final field after center-based strategy
        corner_field: The final field after corner-based strategy
        center_target: Target region for center strategy
        corner_target: Target region for corner strategy
        center_results: Dictionary with center strategy metrics
        corner_results: Dictionary with corner strategy metrics
        save_fig: Whether to save figures to files
    """
    # Create a visualizer (just for its plotting methods)
    tmp_simulator = LatticeSimulator()
    tmp_simulator.field = initial_field
    visualizer = LatticeVisualizer(tmp_simulator)
    
    # Create figure for comparing fields
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot initial field
    visualizer.plot_lattice(
        initial_field, 
        title="Initial Lattice", 
        ax=axes[0]
    )
    
    # Plot center strategy result
    visualizer.plot_lattice(
        center_field, 
        title=f"Center Strategy (Fill: {center_results['fill_rate']:.2%})", 
        highlight_region=center_target,
        ax=axes[1]
    )
    
    # Plot corner strategy result
    visualizer.plot_lattice(
        corner_field, 
        title=f"Corner Strategy (Fill: {corner_results['fill_rate']:.2%})", 
        highlight_region=corner_target,
        ax=axes[2]
    )
    
    plt.tight_layout()
    fig.suptitle(f"Comparison of Center vs Corner Filling Strategies", fontsize=16, y=1.05)
    
    if save_fig:
        plt.savefig("strategy_comparison_fields.png", bbox_inches='tight', dpi=300)
        print("Field comparison saved to 'strategy_comparison_fields.png'")
    
    # Create figure for comparing target regions
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract target regions
    center_target_field = center_field[center_target[0]:center_target[2], 
                                     center_target[1]:center_target[3]]
    corner_target_field = corner_field[corner_target[0]:corner_target[2], 
                                     corner_target[1]:corner_target[3]]
    
    # Create heat maps of defects (1 = defect, 0 = atom)
    im1 = axes2[0].imshow(1-center_target_field, cmap='Reds', vmin=0, vmax=1)
    axes2[0].set_title(f"Center Strategy: {center_results['defects']} defects")
    
    im2 = axes2[1].imshow(1-corner_target_field, cmap='Reds', vmin=0, vmax=1)
    axes2[1].set_title(f"Corner Strategy: {corner_results['defects']} defects")
    
    # Add colorbars
    plt.colorbar(im1, ax=axes2[0], label='Defect (1 = defect, 0 = atom)')
    plt.colorbar(im2, ax=axes2[1], label='Defect (1 = defect, 0 = atom)')
    
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    fig2.suptitle(f"Defect Comparison in Target Regions", fontsize=16, y=1.05)
    
    if save_fig:
        plt.savefig("strategy_comparison_defects.png", bbox_inches='tight', dpi=300)
        print("Defect comparison saved to 'strategy_comparison_defects.png'")
    
    # Create performance metrics comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Extract metrics
    metrics = ['Fill Rate (%)', 'Execution Time (s)', 'Physical Time (s)', 'Movements', 'Defects']
    center_values = [
        center_results['fill_rate'] * 100,
        center_results['execution_time'],
        center_results['physical_time'],
        center_results['movements'],
        center_results['defects']
    ]
    corner_values = [
        corner_results['fill_rate'] * 100,
        corner_results['execution_time'],
        corner_results['physical_time'],
        corner_results['movements'],
        corner_results['defects']
    ]
    
    # Convert to log scale for better visualization of diverse metrics
    center_values_log = np.log10([max(0.001, v) for v in center_values])
    corner_values_log = np.log10([max(0.001, v) for v in corner_values])
    
    # Create bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, center_values_log, width, label='Center Strategy')
    ax3.bar(x + width/2, corner_values_log, width, label='Corner Strategy')
    
    # Add labels and ticks
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Log10 Scale')
    ax3.set_title('Performance Metrics Comparison (Log Scale)')
    ax3.legend()
    
    # Add actual values as text on bars
    for i, v in enumerate(center_values):
        if v >= 100:
            text = f"{v:.0f}"
        elif v >= 10:
            text = f"{v:.1f}"
        elif v >= 1:
            text = f"{v:.2f}"
        elif v >= 0.001:
            text = f"{v:.4f}"
        else:
            text = f"{v:.0e}"
        ax3.text(i - width/2, center_values_log[i] + 0.1, text, 
                ha='center', va='bottom', rotation=90, fontsize=8)
    
    for i, v in enumerate(corner_values):
        if v >= 100:
            text = f"{v:.0f}"
        elif v >= 10:
            text = f"{v:.1f}"
        elif v >= 1:
            text = f"{v:.2f}"
        elif v >= 0.001:
            text = f"{v:.4f}"
        else:
            text = f"{v:.0e}"
        ax3.text(i + width/2, corner_values_log[i] + 0.1, text, 
                ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig("strategy_comparison_metrics.png", bbox_inches='tight', dpi=300)
        print("Metrics comparison saved to 'strategy_comparison_metrics.png'")
    
    # Show all plots
    plt.show()

def run_multi_test(sizes=[(20, 20), (40, 40), (60, 60)], 
                  occupations=[0.5, 0.7, 0.9],
                  save_results=True,
                  show_visualization=False):
    """
    Run tests across multiple lattice sizes and occupation probabilities.
    
    Args:
        sizes: List of (rows, cols) tuples for initial lattice sizes
        occupations: List of occupation probabilities
        save_results: Whether to save results to CSV
        show_visualization: Whether to show visualizations
    """
    results = []
    
    for size in sizes:
        for occupation in occupations:
            print(f"\n{'='*60}")
            print(f"Testing with lattice size: {size}, occupation: {occupation}")
            print(f"{'='*60}")
            
            # Run the comparison
            result = run_comparison(
                size=size,
                occupation_prob=occupation,
                show_visualization=show_visualization,
                save_visualization=save_results
            )
            
            # Add to results
            results.append({
                'size': f"{size[0]}x{size[1]}",
                'occupation': occupation,
                'center_fill_rate': result['center']['fill_rate'],
                'corner_fill_rate': result['corner']['fill_rate'],
                'center_execution_time': result['center']['execution_time'],
                'corner_execution_time': result['corner']['execution_time'],
                'center_physical_time': result['center']['physical_time'],
                'corner_physical_time': result['corner']['physical_time'],
                'center_movements': result['center']['movements'],
                'corner_movements': result['corner']['movements'],
                'center_defects': result['center']['defects'],
                'corner_defects': result['corner']['defects'],
                'fill_rate_diff': result['corner']['fill_rate'] - result['center']['fill_rate'],
                'exec_time_ratio': result['corner']['execution_time'] / max(0.001, result['center']['execution_time']),
                'phys_time_ratio': result['corner']['physical_time'] / max(0.001, result['center']['physical_time']),
                'movement_ratio': result['corner']['movements'] / max(1, result['center']['movements'])
            })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n============= Summary of All Tests =============")
    print(df[['size', 'occupation', 'center_fill_rate', 'corner_fill_rate', 'fill_rate_diff',
              'center_execution_time', 'corner_execution_time', 'exec_time_ratio']].to_string(index=False))
    
    if save_results:
        # Save to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"strategy_comparison_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
        # Create summary plots
        create_summary_plots(df, save_fig=True)
    
    return df

def create_summary_plots(df, save_fig=False):
    """
    Create summary plots from multi-test results.
    
    Args:
        df: DataFrame with test results
        save_fig: Whether to save figures to files
    """
    # Create figure for fill rate comparison
    plt.figure(figsize=(12, 8))
    
    # Extract unique sizes and occupations
    sizes = df['size'].unique()
    occupations = df['occupation'].unique()
    
    # Plot fill rate by size for each occupation
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        plt.plot(subset['size'], subset['center_fill_rate'], 'o-', 
                label=f'Center (occ={occupation})')
        plt.plot(subset['size'], subset['corner_fill_rate'], 's--', 
                label=f'Corner (occ={occupation})')
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Fill Rate')
    plt.title('Fill Rate Comparison by Size and Occupation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_fig:
        plt.savefig("fill_rate_comparison.png", bbox_inches='tight', dpi=300)
    
    # Create figure for execution time comparison
    plt.figure(figsize=(12, 8))
    
    # Plot execution time by size for each occupation
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        plt.plot(subset['size'], subset['center_execution_time'], 'o-', 
                label=f'Center (occ={occupation})')
        plt.plot(subset['size'], subset['corner_execution_time'], 's--', 
                label=f'Corner (occ={occupation})')
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison by Size and Occupation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_fig:
        plt.savefig("execution_time_comparison.png", bbox_inches='tight', dpi=300)
    
    # Create figure for physical time comparison
    plt.figure(figsize=(12, 8))
    
    # Plot physical time by size for each occupation
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        plt.plot(subset['size'], subset['center_physical_time'], 'o-', 
                label=f'Center (occ={occupation})')
        plt.plot(subset['size'], subset['corner_physical_time'], 's--', 
                label=f'Corner (occ={occupation})')
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Physical Movement Time (s)')
    plt.title('Physical Movement Time Comparison by Size and Occupation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_fig:
        plt.savefig("physical_time_comparison.png", bbox_inches='tight', dpi=300)
    
    # Create figure for ratio comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot fill rate difference (corner - center)
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        axes[0, 0].plot(subset['size'], subset['fill_rate_diff'], 'o-', 
                       label=f'Occupation={occupation}')
    
    axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 0].set_xlabel('Lattice Size')
    axes[0, 0].set_ylabel('Fill Rate Difference (Corner - Center)')
    axes[0, 0].set_title('Fill Rate Difference by Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot execution time ratio (corner / center)
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        axes[0, 1].plot(subset['size'], subset['exec_time_ratio'], 'o-', 
                       label=f'Occupation={occupation}')
    
    axes[0, 1].axhline(y=1, color='r', linestyle='-', alpha=0.3)
    axes[0, 1].set_xlabel('Lattice Size')
    axes[0, 1].set_ylabel('Execution Time Ratio (Corner / Center)')
    axes[0, 1].set_title('Execution Time Ratio by Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot physical time ratio (corner / center)
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        axes[1, 0].plot(subset['size'], subset['phys_time_ratio'], 'o-', 
                       label=f'Occupation={occupation}')
    
    axes[1, 0].axhline(y=1, color='r', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Lattice Size')
    axes[1, 0].set_ylabel('Physical Time Ratio (Corner / Center)')
    axes[1, 0].set_title('Physical Movement Time Ratio by Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot movement ratio (corner / center)
    for occupation in occupations:
        subset = df[df['occupation'] == occupation]
        axes[1, 1].plot(subset['size'], subset['movement_ratio'], 'o-', 
                       label=f'Occupation={occupation}')
    
    axes[1, 1].axhline(y=1, color='r', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Lattice Size')
    axes[1, 1].set_ylabel('Movement Count Ratio (Corner / Center)')
    axes[1, 1].set_title('Movement Count Ratio by Size')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig("ratio_comparisons.png", bbox_inches='tight', dpi=300)
    
    plt.show()

def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description='Compare center and corner filling strategies')
    parser.add_argument('--size', type=int, default=30, help='Initial lattice size (default: 30)')
    parser.add_argument('--occupation', type=float, default=0.7, help='Atom occupation probability (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--multi', action='store_true', help='Run multiple tests with different parameters')
    parser.add_argument('--save', action='store_true', help='Save visualizations and results')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualizations')
    args = parser.parse_args()
    
    if args.multi:
        run_multi_test(
            sizes=[(20, 20), (40, 40), (60, 60)],
            occupations=[0.5, 0.7, 0.9],
            save_results=args.save,
            show_visualization=not args.no_viz
        )
    else:
        run_comparison(
            size=(args.size, args.size),
            occupation_prob=args.occupation,
            random_seed=args.seed,
            show_visualization=not args.no_viz,
            save_visualization=args.save
        )

if __name__ == "__main__":
    main()