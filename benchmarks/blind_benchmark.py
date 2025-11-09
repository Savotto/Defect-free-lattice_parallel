#!/usr/bin/env python3
"""
Enhanced benchmark for the blind_center_filling_strategy.

This script measures:
- Number of moves per iteration and in total
- Atom retention rate
- Iterations needed to achieve 100% fill rate (or max iterations)
- Computational and physical time for each iteration
- Total time to achieve perfect fill rate

Results are presented as a summary table and can be saved to CSV.
Includes visualization of Total Time vs Lattice Size for different loss probabilities.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re 
from defect_free import LatticeSimulator, LatticeVisualizer

def run_benchmark(lattice_size, occupation_prob, atom_loss_prob, max_iterations=6, 
                 target_fill_rate=1.0, seed=None, save_results=True, output_dir='benchmark_results',
                 visualize=False):
    """
    Run a comprehensive benchmark on the blind_center_filling_strategy.
    
    Args:
        lattice_size: Tuple of (height, width) for the lattice
        occupation_prob: Probability of atom occupation (0.0 to 1.0)
        atom_loss_prob: Probability of atom loss during transport
        max_iterations: Maximum number of iterations to run
        target_fill_rate: Target fill rate to achieve (1.0 = perfect)
        seed: Random seed for reproducibility
        save_results: Whether to save results to CSV
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        
    Returns:
        DataFrame with benchmark results and summary statistics
    """
    # Create output directory if needed
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize simulator with specified parameters
    simulator = LatticeSimulator(
        initial_size=lattice_size, 
        occupation_prob=occupation_prob,
        physical_constraints={'atom_loss_probability': atom_loss_prob}
    )
    
    # Generate initial lattice
    simulator.generate_initial_lattice(seed=seed)
    initial_atoms = np.sum(simulator.field)
    
    # Initialize data structures to track metrics
    metrics = {
        'iteration': [],
        'computational_time': [],
        'physical_time': [],
        'total_time': [],
        'moves': [],
        'fill_rate': [],
        'defects': [],
        'atoms_in_target': [],
        'retention_rate': [],
        'target_size': []
    }
    
    # Run the rearrangement process iteratively
    print(f"Starting benchmark with lattice size {lattice_size}, occupation {occupation_prob}, atom loss {atom_loss_prob}")
    print(f"Initial atoms: {initial_atoms}")
    
    current_fill_rate = 0.0
    iteration = 0
    cumulative_computational_time = 0.0
    cumulative_physical_time = 0.0
    total_moves = 0
    perfect_fill_achieved = False
    overall_start_time = time.time()
    
    while current_fill_rate < target_fill_rate and iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}/{max_iterations}:")
        
        # Run the rearrangement and measure computational time
        start_time = time.time()
        result, execution_time = simulator.rearrange_for_defect_free(
            strategy='center',
            show_visualization=visualize
        )
        final_lattice, current_fill_rate, _ = result
        iteration_computational_time = execution_time
        
        # Calculate physical time from movement history for this iteration
        iteration_physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
        
        # Count movements in this iteration
        iteration_moves = len(simulator.movement_history)
        
        # Update cumulative metrics
        cumulative_computational_time += iteration_computational_time
        cumulative_physical_time += iteration_physical_time
        total_moves += iteration_moves
        
        # Get target region details
        target_region = simulator.movement_manager.target_region
        target_start_row, target_start_col, target_end_row, target_end_col = target_region
        target_zone = simulator.field[target_start_row:target_end_row, target_start_col:target_end_col]
        target_size = (target_end_row - target_start_row) * (target_end_col - target_start_col)  # Calculate actual target size
        
        # Calculate atoms in target and defects
        atoms_in_target = np.sum(target_zone)
        defects = target_size - atoms_in_target
        
        # Calculate retention rate
        retention_rate = atoms_in_target / initial_atoms if initial_atoms > 0 else 0
        
        # Record metrics for this iteration
        metrics['iteration'].append(iteration)
        metrics['computational_time'].append(iteration_computational_time)
        metrics['physical_time'].append(iteration_physical_time)
        metrics['total_time'].append(iteration_computational_time + iteration_physical_time)
        metrics['moves'].append(iteration_moves)
        metrics['fill_rate'].append(current_fill_rate)
        metrics['defects'].append(defects)
        metrics['atoms_in_target'].append(atoms_in_target)
        metrics['retention_rate'].append(retention_rate)
        metrics['target_size'].append(target_size)
        
        # Print iteration metrics
        print(f"  Fill rate: {current_fill_rate:.2%}")
        print(f"  Defects remaining: {defects}")
        print(f"  Computational time: {iteration_computational_time:.3f} seconds")
        print(f"  Physical time: {iteration_physical_time:.6f} seconds")
        print(f"  Total time: {(iteration_computational_time + iteration_physical_time):.3f} seconds")
        print(f"  Moves: {iteration_moves}")
        print(f"  Retention rate: {retention_rate:.2%}")
        
        # Check if target fill rate achieved
        if current_fill_rate >= target_fill_rate:
            print(f"\nTarget fill rate of {target_fill_rate:.0%} achieved in {iteration} iterations!")
            perfect_fill_achieved = True
            break
    
    # Calculate overall process time
    overall_total_time = time.time() - overall_start_time
    
    # Convert metrics to DataFrame
    results_df = pd.DataFrame(metrics)
    
    # Calculate summary statistics
    summary = {
        'lattice_size': f"{lattice_size[0]}x{lattice_size[1]}",
        'occupation_probability': occupation_prob,
        'atom_loss_probability': atom_loss_prob,
        'initial_atoms': initial_atoms,
        'target_size': target_size,
        'iterations_to_perfect_fill': iteration if perfect_fill_achieved else None,
        'final_fill_rate': current_fill_rate,
        'total_moves': total_moves,
        'final_retention_rate': metrics['retention_rate'][-1],
        'total_computational_time': cumulative_computational_time,
        'total_physical_time': cumulative_physical_time,
        'total_combined_time': cumulative_computational_time + cumulative_physical_time,
        'overall_process_time': overall_total_time
    }
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Lattice size: {lattice_size[0]}x{lattice_size[1]}")
    print(f"Occupation probability: {occupation_prob}")
    print(f"Atom loss probability: {atom_loss_prob}")
    print(f"Initial atoms: {initial_atoms}")
    print(f"Target zone size: {simulator.side_length}x{simulator.side_length} ({target_size} positions)")
    print(f"Iterations to perfect fill: {summary['iterations_to_perfect_fill'] or 'Not achieved'}")
    print(f"Final fill rate: {current_fill_rate:.2%}")
    print(f"Total moves: {total_moves}")
    print(f"Final retention rate: {metrics['retention_rate'][-1]:.2%}")
    print(f"Total computational time: {cumulative_computational_time:.3f} seconds")
    print(f"Total physical time: {cumulative_physical_time:.6f} seconds")
    print(f"Total combined time: {(cumulative_computational_time + cumulative_physical_time):.3f} seconds")
    print(f"Overall process time: {overall_total_time:.3f} seconds")
    
    # Save results if requested
    if save_results:
        # Save detailed results
        results_path = os.path.join(output_dir, f"blind_center_{lattice_size[0]}x{lattice_size[1]}_occ{occupation_prob}_loss{atom_loss_prob}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, f"summary_{lattice_size[0]}x{lattice_size[1]}_occ{occupation_prob}_loss{atom_loss_prob}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nResults saved to {output_dir}")      
    
    return results_df, summary

def create_summary_tables(summary_df, output_dir):
    """
    Save summary tables per lattice size as CSV.
    Includes mean and standard deviation statistics from Monte Carlo runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    for size, group in summary_df.groupby('lattice_size'):
        out = group[['occupation_probability', 'atom_loss_probability',
                     'mean_iterations_to_perfect_fill', 'std_iterations_to_perfect_fill',
                     'mean_final_fill_rate', 'std_final_fill_rate',
                     'mean_total_moves', 'std_total_moves',
                     'mean_final_retention_rate', 'std_final_retention_rate',
                     'mean_total_computational_time', 'std_total_computational_time',
                     'mean_total_physical_time', 'std_total_physical_time']]
        out.to_csv(os.path.join(output_dir, f'summary_table_{size}.csv'), index=False)

def create_time_scaling_for_params(summary_df, results_dir, output_dir, occ=0.7, loss=0.05):
    """
    Plot total combined time vs lattice size for given occ & loss.
    Also plots the first iteration time for comparison.
    Includes error bars from Monte Carlo simulations.
    
    Args:
        summary_df: DataFrame with Monte Carlo summary statistics
        results_dir: Directory containing Monte Carlo iteration stats
        output_dir: Directory to save visualizations
        occ: Occupation probability to filter for
        loss: Loss probability to filter for
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total time from summary data (with mean and std)
    df = summary_df[(summary_df['occupation_probability']==occ) &
                    (summary_df['atom_loss_probability']==loss)].copy()
    df['size'] = df['lattice_size'].apply(lambda s: int(s.split('x')[0]))
    df = df.sort_values('size')
    
    # Collect first iteration time data from iteration statistics
    first_iter_times = []
    
    # Get the Monte Carlo iteration stats files
    files = [f for f in os.listdir(results_dir) 
             if f.startswith('mc_iteration_stats_') and f.endswith('.csv') 
             and f'_occ{occ}_loss{loss}' in f]
    
    for f in files:
        m = re.search(r"_(\d+)x\d+_occ[\d.]+_loss[\d.]+\.csv", f)
        if m:
            size = int(m.group(1))
            
            # Read data and get first iteration time mean and std
            iter_stats = pd.read_csv(os.path.join(results_dir, f))
            if 1 in iter_stats['iteration'].values:
                first_iter_row = iter_stats[iter_stats['iteration'] == 1].iloc[0]
                first_iter_times.append({
                    'size': size, 
                    'time_mean': first_iter_row['total_time_mean'],
                    'time_std': first_iter_row['total_time_std']
                })
    
    iter1_df = pd.DataFrame(first_iter_times)
    iter1_df = iter1_df.sort_values('size')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot total time across all iterations with error bars
    plt.errorbar(df['size'], df['mean_total_combined_time'], yerr=df['std_total_combined_time'],
                 marker='o', label='All Iterations', color='blue', capsize=5)
    
    # Plot first iteration time only with error bars if data exists
    if not iter1_df.empty:
        plt.errorbar(iter1_df['size'], iter1_df['time_mean'], yerr=iter1_df['time_std'], 
                     marker='s', label='First Iteration Only', color='green', capsize=5)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Time Scaling (Occ={occ}, Loss={loss}, n=100 runs)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'time_vs_size_occ{occ}_loss{loss}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Time scaling plot saved to {plot_path}")

def create_moves_scaling_plots(summary_df, results_dir, output_dir):
    """
    Plot number of moves vs lattice size with Monte Carlo statistics:
      - Only iteration 1 moves (mean and std)
      - Total moves across all iterations (mean and std)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total moves data from summary_df (already contains mean/std)
    moves_total = summary_df.copy()
    moves_total['size'] = moves_total['lattice_size'].apply(lambda s: int(s.split('x')[0]))
    
    # Collect iteration 1 moves from Monte Carlo stats
    iter1_moves_data = []
    files = [f for f in os.listdir(results_dir) if f.startswith('mc_iteration_stats_') and f.endswith('.csv')]
    
    for f in files:
        m = re.search(r"_(\d+)x\d+_occ([\d.]+)_loss([\d.]+)\.csv", f)
        if m:
            size = int(m.group(1))
            occ = float(m.group(2))
            loss = float(m.group(3))
            
            # Read the iteration stats file
            iter_stats = pd.read_csv(os.path.join(results_dir, f))
            if 1 in iter_stats['iteration'].values:
                iter1_row = iter_stats[iter_stats['iteration'] == 1].iloc[0]
                iter1_moves_data.append({
                    'size': size,
                    'occ': occ,
                    'loss': loss,
                    'moves_mean': iter1_row['moves_mean'],
                    'moves_std': iter1_row['moves_std']
                })
    
    iter1_df = pd.DataFrame(iter1_moves_data)
    
    # Plot for occ=0.7, loss=0.05 with error bars
    sub_total = moves_total[(moves_total['occupation_probability']==0.7) & 
                          (moves_total['atom_loss_probability']==0.05)].sort_values('size')
    
    sub_iter1 = iter1_df[(iter1_df['occ']==0.7) & 
                        (iter1_df['loss']==0.05)].sort_values('size')
    
    plt.figure(figsize=(8, 5))
    
    # Plot total moves with error bars
    plt.errorbar(sub_total['size'], sub_total['mean_total_moves'], yerr=sub_total['std_total_moves'],
                marker='s', label='All Iterations', color='blue', capsize=5)
    
    # Plot iteration 1 moves with error bars if data exists
    if not sub_iter1.empty:
        plt.errorbar(sub_iter1['size'], sub_iter1['moves_mean'], yerr=sub_iter1['moves_std'],
                    marker='o', label='Iteration 1', color='green', capsize=5)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Moves')
    plt.title('Moves Scaling (Occ=0.7, Loss=0.05, n=100 runs)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir,'mc_moves_vs_size_occ0.7_loss0.05.png'), dpi=300)
    plt.close()
    print("Monte Carlo moves scaling plot saved")

def create_retention_vs_size_plot(summary_df, output_dir, fixed_occ=0.7):
    """
    Plot final retention rate vs lattice size for various loss probs.
    Includes error bars from Monte Carlo simulations.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = summary_df[summary_df['occupation_probability']==fixed_occ].copy()
    df['size'] = df['lattice_size'].apply(lambda s: int(s.split('x')[0]))
    probs = sorted(df['atom_loss_probability'].unique())
    
    plt.figure(figsize=(8, 5))
    for loss in probs:
        sub = df[df['atom_loss_probability']==loss].sort_values('size')
        plt.errorbar(sub['size'], sub['mean_final_retention_rate'], 
                    yerr=sub['std_final_retention_rate'],
                    marker='o', label=f'Loss {loss}', capsize=5)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Retention Rate')
    plt.title(f'Retention vs Size (Occ={fixed_occ}, n=100 runs)')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, f'mc_retention_vs_size_occ{fixed_occ}.png'), dpi=300)
    plt.close()
    print(f"Monte Carlo retention rate plot saved")

def create_fill_rate_iteration_plot_fixed_occ(results_dir, output_dir, fixed_occ=0.7):
    """
    Plot fill rate vs iteration number for different lattice sizes.
    Uses Monte Carlo statistics with error bands.
    Creates a separate figure for each lattice size.
    
    Args:
        results_dir: Directory containing Monte Carlo iteration stats
        output_dir: Directory to save visualizations
        fixed_occ: Fixed occupation probability to use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Monte Carlo iteration stats files with the specified occupation probability
    files = [f for f in os.listdir(results_dir) 
             if f.startswith('mc_iteration_stats_') and f.endswith('.csv') 
             and f'_occ{fixed_occ}_' in f]
    
    # Group files by lattice size
    size_groups = {}
    for f in files:
        m = re.search(r"_(\d+)x\d+_occ[\d.]+_loss([\d.]+)\.csv", f)
        if m:
            size = int(m.group(1))
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(f)
    
    # Create a separate figure for each lattice size
    for size, size_files in size_groups.items():
        plt.figure(figsize=(10, 6))
        
        for f in size_files:
            # Extract loss probability from filename
            m = re.search(r"_\d+x\d+_occ[\d.]+_loss([\d.]+)\.csv", f)
            if m:
                loss = float(m.group(1))
                
                # Read Monte Carlo iteration stats
                df = pd.read_csv(os.path.join(results_dir, f))
                
                # Plot mean with error bands
                x = df['iteration']
                y = df['fill_rate_mean']
                error = df['fill_rate_std']
                
                plt.plot(x, y, marker='o', label=f'Loss {loss}')
                plt.fill_between(x, y-error, y+error, alpha=0.2)
        
        plt.title(f'Fill Rate vs Iteration (Size {size}x{size}, Occ={fixed_occ}, n=100 runs)')
        plt.xlabel('Iteration')
        plt.ylabel('Fill Rate')
        plt.ylim(0.8, 1.01)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'mc_fill_rate_vs_iteration_size{size}_occ{fixed_occ}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Monte Carlo fill rate comparison for size {size}x{size} saved")

def aggregate_monte_carlo_data(results_list):
    """
    Calculate mean and standard deviation for each metric from multiple Monte Carlo runs.
    
    Args:
        results_list: List of summary dictionaries from multiple runs
        
    Returns:
        Dictionary with mean and std for each metric
    """
    df = pd.DataFrame(results_list)
    
    # Calculate statistics for each column
    agg_results = {}
    
    # Add identifier columns directly
    for col in ['lattice_size', 'occupation_probability', 'atom_loss_probability']:
        if col in df.columns:
            agg_results[col] = df[col].iloc[0]  # These should be the same for all runs
    
    # Add mean and std for numeric columns
    for col in df.columns:
        if col not in ['lattice_size', 'occupation_probability', 'atom_loss_probability']:
            try:
                # Skip None values for standard deviation calculation
                values = df[col].dropna()
                if not values.empty:
                    agg_results[f'mean_{col}'] = values.mean()
                    agg_results[f'std_{col}'] = values.std() if len(values) > 1 else 0
            except:
                # Skip columns that can't be averaged
                pass
    
    return agg_results

def save_monte_carlo_iteration_stats(iteration_dfs, output_dir, size, occ, loss):
    """
    Aggregate and save per-iteration statistics across Monte Carlo runs.
    
    Args:
        iteration_dfs: List of DataFrames containing per-iteration metrics from multiple runs
        output_dir: Directory to save results
        size, occ, loss: Parameters to include in filename
    """
    # Add run identifier to each DataFrame
    for i, df in enumerate(iteration_dfs):
        df['run'] = i
    
    # Combine all iteration data
    combined = pd.concat(iteration_dfs, ignore_index=True)
    
    # Group by iteration number and calculate statistics
    stats = combined.groupby('iteration').agg({
        'computational_time': ['mean', 'std'],
        'physical_time': ['mean', 'std'],
        'total_time': ['mean', 'std'],
        'moves': ['mean', 'std'],
        'fill_rate': ['mean', 'std'],
        'defects': ['mean', 'std'],
        'atoms_in_target': ['mean', 'std'],
        'retention_rate': ['mean', 'std']
    })
    
    # Flatten column names
    stats.columns = ['_'.join(col) for col in stats.columns]
    stats = stats.reset_index()
    
    # Save to file
    filename = f"mc_iteration_stats_{size[0]}x{size[1]}_occ{occ}_loss{loss}.csv"
    stats.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Monte Carlo iteration stats saved to {filename}")
    
    return stats

# -- Main entrypoint updated to run Monte Carlo simulations --
def main():
    """Main function to run benchmarks with Monte Carlo simulations."""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks for blind center filling strategy")
    
    # Add arguments
    parser.add_argument('--single', action='store_true', 
                        help='Run a single benchmark instead of multiple configurations')
    parser.add_argument('--size', type=int, default=20,
                        help='Lattice size (will be size x size)')
    parser.add_argument('--occupation', type=float, default=0.7,
                        help='Initial occupation probability')
    parser.add_argument('--loss', type=float, default=0.0,
                        help='Atom loss probability')
    parser.add_argument('--output', default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of Monte Carlo iterations per configuration')
    
    args = parser.parse_args()
    summary_df = None
    
    # Set number of Monte Carlo iterations to 100
    monte_carlo_iterations = args.iterations
    
    if args.single:
        # Run a single benchmark with Monte Carlo iterations
        lattice_size = (args.size, args.size)
        
        monte_carlo_results = []
        monte_carlo_iteration_dfs = []
        
        for i in range(monte_carlo_iterations):
            print(f"\n=== Monte Carlo Run {i+1}/{monte_carlo_iterations} ===")
            
            # Use different seed for each iteration if seed is provided
            iter_seed = args.seed + i if args.seed is not None else None
            
            results_df, summary = run_benchmark(
                lattice_size=lattice_size,
                occupation_prob=args.occupation,
                atom_loss_prob=args.loss,
                save_results=(i == 0),  # Only save detailed results for first run
                output_dir=args.output,
                visualize=args.visualize if i == 0 else False,  # Only visualize first run if requested
                seed=iter_seed
            )
            
            monte_carlo_results.append(summary)
            monte_carlo_iteration_dfs.append(results_df)
        
        # Calculate aggregate Monte Carlo statistics
        mc_stats = aggregate_monte_carlo_data(monte_carlo_results)
        summary_df = pd.DataFrame([mc_stats])
        
        # Save Monte Carlo summary
        summary_df.to_csv(os.path.join(args.output, f"mc_summary_{lattice_size[0]}x{lattice_size[1]}_occ{args.occupation}_loss{args.loss}.csv"), index=False)
        
        # Save raw results for reference
        raw_df = pd.DataFrame(monte_carlo_results)
        raw_df.to_csv(os.path.join(args.output, f"mc_raw_results_{lattice_size[0]}x{lattice_size[1]}_occ{args.occupation}_loss{args.loss}.csv"), index=False)
        
        # Save per-iteration statistics
        save_monte_carlo_iteration_stats(monte_carlo_iteration_dfs, args.output, lattice_size, args.occupation, args.loss)
        
    else:
        # Run multiple benchmark configurations with Monte Carlo iterations
        mc_aggregate_results = []
        
        # Define parameter ranges for benchmarks
        lattice_sizes = [(10, 10), (20, 20), (50, 50), (75, 75), (100, 100)]
        occupation_probs = [0.5, 0.7, 0.9]
        loss_probs = [0.0, 0.01, 0.05]
        
        # Run benchmarks for all combinations
        for size in lattice_sizes:
            for occ in occupation_probs:
                for loss in loss_probs:
                    print(f"\n\n=== Running configuration: Size {size}, Occupation {occ}, Loss {loss} ===")
                    
                    # Run Monte Carlo iterations for this configuration
                    mc_results = []
                    mc_iteration_dfs = []
                    
                    for i in range(monte_carlo_iterations):
                        print(f"\n--- Monte Carlo Run {i+1}/{monte_carlo_iterations} ---")
                        
                        # Use different seed for each iteration if seed is provided
                        iter_seed = args.seed + i if args.seed is not None else None
                        
                        results_df, summary = run_benchmark(
                            lattice_size=size,
                            occupation_prob=occ,
                            atom_loss_prob=loss,
                            save_results=(i == 0),  # Only save detailed results for first run
                            output_dir=args.output,
                            visualize=args.visualize if i == 0 else False,  # Only visualize first run if requested
                            seed=iter_seed
                        )
                        
                        mc_results.append(summary)
                        mc_iteration_dfs.append(results_df)
                    
                    # Calculate aggregate Monte Carlo statistics for this configuration
                    mc_stats = aggregate_monte_carlo_data(mc_results)
                    mc_aggregate_results.append(mc_stats)
                    
                    # Save per-iteration statistics
                    save_monte_carlo_iteration_stats(mc_iteration_dfs, args.output, size, occ, loss)
        
        # Create summary DataFrame from all aggregated results
        summary_df = pd.DataFrame(mc_aggregate_results)
        
        # Save combined Monte Carlo summary
        summary_df.to_csv(os.path.join(args.output, 'mc_combined_summary.csv'), index=False)
    
    # Generate visualizations if we have summary data
    if summary_df is not None:
        # Create visualizations using the Monte Carlo statistics
        create_fill_rate_iteration_plot_fixed_occ(args.output, args.output)
        create_summary_tables(summary_df, args.output)
        create_time_scaling_for_params(summary_df, args.output, args.output)
        create_moves_scaling_plots(summary_df, args.output, args.output)
        create_retention_vs_size_plot(summary_df, args.output)

if __name__ == '__main__':
    main()