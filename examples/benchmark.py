"""
Benchmarking script for defect-free lattice rearrangement center strategy.
Focuses on detailed performance analysis of the center strategy for thesis work.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from defect_free import LatticeSimulator

def benchmark_center_strategy(lattice_size, occupation_prob, loss_prob, 
                             iterations=10, detailed_timing=False, seed=None):
    """
    Benchmark the center strategy with the given parameters.
    
    Args:
        lattice_size: Tuple of (height, width) for the lattice
        occupation_prob: Probability of atom occupation (0.0 to 1.0)
        loss_prob: Probability of atom loss during movement
        iterations: Number of iterations to run
        detailed_timing: Whether to collect detailed timing information
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of averaged metrics and detailed timings
    """
    metrics = {
        'execution_time': [],
        'physical_time': [],
        'fill_rate': [],
        'perfect_fill': [],
        'retention_rate': [],
        'target_size': [],
        'total_moves': [],
        'total_time': [],
        'atoms_moved': []
    }
    
    for i in range(iterations):
        # Set seed for reproducibility if provided
        if seed is not None:
            np_seed = seed + i
        else:
            np_seed = None
            
        # Initialize simulator
        simulator = LatticeSimulator(
            initial_size=lattice_size, 
            occupation_prob=occupation_prob,
            physical_constraints={'atom_loss_probability': loss_prob}
        )
        
        # Generate initial lattice
        simulator.generate_initial_lattice(seed=np_seed)
        initial_atoms = np.sum(simulator.field)
        
        # Run the rearrangement with center strategy
        result, execution_time = simulator.rearrange_for_defect_free(
            strategy='center',
            show_visualization=False
        )
        final_lattice, fill_rate, _ = result
        
        # Calculate standard metrics
        physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
        total_time = execution_time + physical_time
        total_moves = len(simulator.movement_history)
        
        # Target region metrics
        target_region = simulator.movement_manager.target_region
        start_row, start_col, end_row, end_col = target_region
        target_zone = simulator.field[start_row:end_row, start_col:end_col]
        atoms_in_target = np.sum(target_zone)
        target_size = simulator.side_length ** 2
        atoms_moved = len(simulator.movement_history)
        
        perfect_fill = 1.0 if fill_rate == 1.0 else 0.0
        retention_rate = atoms_in_target / initial_atoms if initial_atoms > 0 else 0
        
        # Store only the most meaningful metrics
        metrics['execution_time'].append(execution_time)
        metrics['physical_time'].append(physical_time)
        metrics['fill_rate'].append(fill_rate)
        metrics['perfect_fill'].append(perfect_fill)
        metrics['retention_rate'].append(retention_rate)
        metrics['target_size'].append(target_size)
        metrics['total_moves'].append(total_moves)
        metrics['total_time'].append(total_time)
        metrics['atoms_moved'].append(atoms_moved)

    # Average the metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {f"{k}_std": np.std(v) for k, v in metrics.items()}
    size_str = f"{lattice_size[0]}x{lattice_size[1]}"
    combined_metrics = {
        **avg_metrics,
        **std_metrics,
        'lattice_size': size_str,
        'lattice_width': lattice_size[0],
        'lattice_height': lattice_size[1],
        'occupation_prob': occupation_prob,
        'loss_prob': loss_prob
    }
    
    return combined_metrics

def visualize_center_benchmarks(df, output_dir):
    """
    Create visualizations of center strategy benchmark results.
    
    Args:
        df: DataFrame containing benchmark results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define color for center strategy
    center_color = '#3366CC'  # Blue for center strategy
    
    # 1. Execution Time by Lattice Size
    plt.figure(figsize=(12, 7))
    sizes = sorted(df['lattice_width'].unique())
    times = [df[df['lattice_width'] == size]['execution_time'].mean() for size in sizes]
    errors = [df[df['lattice_width'] == size]['execution_time_std'].mean() for size in sizes]
    
    plt.errorbar(sizes, times, yerr=errors, marker='o', capsize=5,
                label="Center Strategy", color=center_color, linewidth=2)
    
    # Fit polynomial regression for scaling analysis
    if len(sizes) >= 3:
        # Try quadratic fit for computational complexity analysis
        z = np.polyfit(sizes, times, 2)
        p = np.poly1d(z)
        x_line = np.linspace(min(sizes), max(sizes), 100)
        plt.plot(x_line, p(x_line), '--', color='black', 
                label=f'Fit: {z[0]:.2e}n² + {z[1]:.2e}n + {z[2]:.2e}')
    
    plt.xlabel('Lattice Size (width)')
    plt.ylabel('Execution Time (s)')
    plt.title('Center Strategy: Algorithm execution Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_execution_time.png'), dpi=300)
    plt.close()
    
    # 2. Physical Time by Lattice Size
    plt.figure(figsize=(12, 7))
    sizes = sorted(df['lattice_width'].unique())
    times = [df[df['lattice_width'] == size]['physical_time'].mean() for size in sizes]
    errors = [df[df['lattice_width'] == size]['physical_time_std'].mean() for size in sizes]
    
    plt.errorbar(sizes, times, yerr=errors, marker='o', capsize=5,
                label="Center Strategy", color=center_color, linewidth=2)
    
    plt.xlabel('Lattice Size (width)')
    plt.ylabel('Physical Time (s)')
    plt.title('Center Strategy: Physical Atom Movement Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_physical_time.png'), dpi=300)
    plt.close()
    
    # 3. Total Number of Moves by Lattice Size
    plt.figure(figsize=(12, 7))
    sizes = sorted(df['lattice_width'].unique())
    moves = [df[df['lattice_width'] == size]['total_moves'].mean() for size in sizes]
    errors = [df[df['lattice_width'] == size]['total_moves_std'].mean() for size in sizes]
    
    plt.errorbar(sizes, moves, yerr=errors, marker='o', capsize=5,
                label="Center Strategy", color=center_color, linewidth=2)
    
    # Fit linear regression for scaling analysis
    if len(sizes) >= 3:
        z = np.polyfit(sizes, moves, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(sizes), max(sizes), 100)
        plt.plot(x_line, p(x_line), '--', color='black', 
                label=f'Linear fit: {z[0]:.2f}n + {z[1]:.2f}')
    
    plt.xlabel('Lattice Size (width)')
    plt.ylabel('Number of Atom Moves')
    plt.title('Center Strategy: Total Number of Atom Movements')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_total_moves.png'), dpi=300)
    plt.close()
    
    # 4. Moves per Atom Ratio
    plt.figure(figsize=(12, 7))
    sizes = sorted(df['lattice_width'].unique())
    
    # Calculate moves per target atom
    efficiency = []
    errors = []
    for size in sizes:
        size_data = df[df['lattice_width'] == size]
        moves_per_atom = size_data['total_moves'] / size_data['target_size']
        efficiency.append(moves_per_atom.mean())
        errors.append(moves_per_atom.std())
    
    plt.errorbar(sizes, efficiency, yerr=errors, marker='o', capsize=5,
                label="Center Strategy", color=center_color, linewidth=2)
    
    plt.xlabel('Lattice Size (width)')
    plt.ylabel('Moves per Target Atom')
    plt.title('Center Strategy: Movement Efficiency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_movement_efficiency.png'), dpi=300)
    plt.close()
    
    # 5. Fill Rate by Lattice Size
    plt.figure(figsize=(12, 7))
    sizes = sorted(df['lattice_width'].unique())
    fill_rates = [df[df['lattice_width'] == size]['fill_rate'].mean() for size in sizes]
    errors = [df[df['lattice_width'] == size]['fill_rate_std'].mean() for size in sizes]
    
    plt.errorbar(sizes, fill_rates, yerr=errors, marker='o', capsize=5,
                label="Center Strategy", color=center_color, linewidth=2)
    
    plt.xlabel('Lattice Size (width)')
    plt.ylabel('Fill Rate')
    plt.title('Center Strategy: Fill Rate vs Lattice Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_fill_rate.png'), dpi=300)
    plt.close()
    
    # 6. Analysis of Occupation Probability Effect on Performance
    # Group results by occupation probability
    plt.figure(figsize=(12, 8))
    occupation_probs = sorted(df['occupation_prob'].unique())
    
    # For each loss probability, plot execution time vs occupation probability
    loss_probs = sorted(df['loss_prob'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, loss_prob in enumerate(loss_probs):
        loss_data = df[df['loss_prob'] == loss_prob]
        
        if not loss_data.empty:
            exec_times = [loss_data[loss_data['occupation_prob'] == o]['execution_time'].mean() 
                         for o in occupation_probs]
            errors = [loss_data[loss_data['occupation_prob'] == o]['execution_time_std'].mean()
                     for o in occupation_probs]
            
            plt.errorbar(occupation_probs, exec_times, yerr=errors, marker='o', capsize=5,
                        label=f'Loss Prob: {loss_prob}', color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Occupation Probability')
    plt.ylabel('Execution Time (s)')
    plt.title('Center Strategy: Effect of Occupation Probability on Execution Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_occupation_effect.png'), dpi=300)
    plt.close()
    
    # 7. Total Time (Computation + Physical) by Lattice Size for Different Loss Probabilities
    plt.figure(figsize=(14, 8))
    sizes = sorted(df['lattice_width'].unique())
    loss_probs = sorted(df['loss_prob'].unique())
    
    # Use these specific occupation probabilities
    target_occ_probs = [0.5, 0.7]
    line_styles = ['-', '--']
    markers = ['o', 's']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Filter for target occupation probabilities
    for occ_idx, occ_prob in enumerate(target_occ_probs):
        occ_data = df[df['occupation_prob'] == occ_prob]
        
        if not occ_data.empty:
            for i, loss_prob in enumerate(loss_probs):
                loss_data = occ_data[occ_data['loss_prob'] == loss_prob]
                
                if not loss_data.empty:
                    total_times = [loss_data[loss_data['lattice_width'] == size]['total_time'].mean() for size in sizes]
                    errors = [loss_data[loss_data['lattice_width'] == size]['total_time_std'].mean() for size in sizes]
                    
                    plt.errorbar(sizes, total_times, yerr=errors, marker=markers[occ_idx], capsize=5,
                                linestyle=line_styles[occ_idx], color=colors[i % len(colors)],
                                label=f'Occ: {occ_prob}, Loss: {loss_prob}', linewidth=2)
    
    plt.xlabel('Lattice Size (width)', fontsize=12)
    plt.ylabel('Total Time (computation + physical) in seconds', fontsize=12)
    plt.title('Center Strategy: Total Time vs Lattice Size for Different Loss Probabilities', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_total_time_comparison.png'), dpi=300)
    plt.close()
    
    # 8. Analysis of how atom loss probability affects fill rate
    plt.figure(figsize=(12, 7))
    loss_probs = sorted(df['loss_prob'].unique())
    
    for size in sizes:
        size_data = df[df['lattice_width'] == size]
        fill_rates = [size_data[size_data['loss_prob'] == l]['fill_rate'].mean() for l in loss_probs]
        errors = [size_data[size_data['loss_prob'] == l]['fill_rate_std'].mean() for l in loss_probs]
        
        plt.errorbar(loss_probs, fill_rates, yerr=errors, marker='o', capsize=5,
                    label=f'Size: {size}x{size}', linewidth=2)
    
    plt.xlabel('Atom Loss Probability')
    plt.ylabel('Fill Rate')
    plt.title('Center Strategy: Effect of Atom Loss on Fill Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_loss_effect.png'), dpi=300)
    plt.close()

def run_center_benchmark_suite(sizes, occupation_probs, loss_probs, iterations, 
                              detailed_timing=False, seed=None, output_dir="center_benchmark_results"):
    """
    Run a complete benchmark suite for the center strategy across all parameter combinations.
    
    Args:
        sizes: List of lattice sizes (square dimensions)
        occupation_probs: List of occupation probabilities
        loss_probs: List of atom loss probabilities
        iterations: Number of iterations per parameter combination
        detailed_timing: Whether to collect detailed timing information
        seed: Random seed for reproducibility
        output_dir: Directory to save results and visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert sizes to lattice dimensions
    lattice_sizes = [(x, x) for x in sizes]
    
    # Collect all parameter combinations
    parameter_combinations = []
    for size in lattice_sizes:
        for occ_prob in occupation_probs:
            for loss_prob in loss_probs:
                parameter_combinations.append({
                    'lattice_size': size,
                    'occupation_prob': occ_prob,
                    'loss_prob': loss_prob
                })
    
    # Print benchmark configuration
    print("Center Strategy Benchmark Configuration:")
    print(f"- Lattice Sizes: {sizes} x {sizes}")
    print(f"- Occupation Probabilities: {occupation_probs}")
    print(f"- Atom Loss Probabilities: {loss_probs}")
    print(f"- Iterations per combination: {iterations}")
    print(f"- Total parameter combinations: {len(parameter_combinations)}")
    print(f"- Detailed timing: {'Enabled' if detailed_timing else 'Disabled'}")
    
    # Run benchmarks for all parameter combinations
    results = []
    
    try:
        for params in tqdm(parameter_combinations):
            print(f"\nBenchmarking: Size={params['lattice_size']}, "
                  f"Occupation={params['occupation_prob']}, "
                  f"Loss={params['loss_prob']}")
            
            metrics = benchmark_center_strategy(
                params['lattice_size'], 
                params['occupation_prob'], 
                params['loss_prob'],
                iterations=iterations,
                detailed_timing=detailed_timing,
                seed=seed
            )
            
            # Print key metrics for immediate feedback
            print(f"  - Execution time: {metrics['execution_time']:.4f} s")
            print(f"  - Physical time: {metrics['physical_time']:.4f} s")
            print(f"  - Total moves: {metrics['total_moves']:.1f}")
            print(f"  - Fill rate: {metrics['fill_rate']:.2%}")
            
            results.append(metrics)
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Saving partial results...")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'detailed_timings'} for r in results])
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'center_benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate visualizations
    visualize_center_benchmarks(df, output_dir)
    
    # Generate benchmark summary
    generate_center_summary(df, output_dir)
    
    return df

def generate_center_summary(df, output_dir):
    """Generate a summary of center strategy benchmark results."""
    # Function to convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Process the data structure recursively to convert numpy types
    def process_data(data):
        if isinstance(data, dict):
            return {k: process_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [process_data(item) for item in data]
        else:
            return convert_to_serializable(data)
    
    # Create summary structure
    summary = {
        'lattice_sizes': sorted(df['lattice_width'].unique()),
        'metrics': {}
    }
    
    # Calculate overall averages
    summary['metrics']['overall'] = {
        'avg_execution_time': df['execution_time'].mean(),
        'avg_physical_time': df['physical_time'].mean(),
        'avg_total_moves': df['total_moves'].mean(),
        'avg_fill_rate': df['fill_rate'].mean(),
        'avg_retention_rate': df['retention_rate'].mean()
    }
    
    # By lattice size
    summary['metrics']['by_size'] = {}
    for size in summary['lattice_sizes']:
        size_data = df[df['lattice_width'] == size]
        if not size_data.empty:
            summary['metrics']['by_size'][int(size)] = {
                'execution_time': size_data['execution_time'].mean(),
                'physical_time': size_data['physical_time'].mean(),
                'total_moves': size_data['total_moves'].mean(),
                'fill_rate': size_data['fill_rate'].mean()
            }
    
    # By occupation probability
    occ_probs = sorted(df['occupation_prob'].unique())
    summary['metrics']['by_occupation'] = {}
    for occ in occ_probs:
        occ_data = df[df['occupation_prob'] == occ]
        summary['metrics']['by_occupation'][float(occ)] = {
            'execution_time': occ_data['execution_time'].mean(),
            'physical_time': occ_data['physical_time'].mean(),
            'total_moves': occ_data['total_moves'].mean(),
            'fill_rate': occ_data['fill_rate'].mean()
        }
    
    # By loss probability
    loss_probs = sorted(df['loss_prob'].unique())
    summary['metrics']['by_loss'] = {}
    for loss in loss_probs:
        loss_data = df[df['loss_prob'] == loss]
        summary['metrics']['by_loss'][float(loss)] = {
            'execution_time': loss_data['execution_time'].mean(),
            'physical_time': loss_data['physical_time'].mean(),
            'total_moves': loss_data['total_moves'].mean(),
            'fill_rate': loss_data['fill_rate'].mean()
        }
    
    # Process the entire summary to convert numpy types to Python native types
    processed_summary = process_data(summary)
    
    # Output summary as JSON
    with open(os.path.join(output_dir, 'center_benchmark_summary.json'), 'w') as f:
        json.dump(processed_summary, f, indent=2)
    
    # Generate a text summary
    with open(os.path.join(output_dir, 'center_benchmark_summary.txt'), 'w') as f:
        f.write("CENTER STRATEGY BENCHMARK SUMMARY\n")
        f.write("================================\n\n")
        
        metrics = summary['metrics']['overall']
        f.write(f"Average Execution Time: {metrics['avg_execution_time']:.4f} s\n")
        f.write(f"Average Physical Time: {metrics['avg_physical_time']:.4f} s\n")
        f.write(f"Average Total Moves: {metrics['avg_total_moves']:.1f}\n")
        f.write(f"Average Fill Rate: {metrics['avg_fill_rate']:.2%}\n")
        f.write(f"Average Retention Rate: {metrics['avg_retention_rate']:.2%}\n\n")
        
        f.write("By Lattice Size:\n")
        for size, size_metrics in summary['metrics']['by_size'].items():
            f.write(f"  {size}x{size}:\n")
            f.write(f"    Execution Time: {size_metrics['execution_time']:.4f} s\n")
            f.write(f"    Physical Time: {size_metrics['physical_time']:.4f} s\n")
            f.write(f"    Total Moves: {size_metrics['total_moves']:.1f}\n")
            f.write(f"    Fill Rate: {size_metrics['fill_rate']:.2%}\n\n")
        
        f.write("By Occupation Probability:\n")
        for occ, occ_metrics in summary['metrics']['by_occupation'].items():
            f.write(f"  {occ:.1f}:\n")
            f.write(f"    Execution Time: {occ_metrics['execution_time']:.4f} s\n")
            f.write(f"    Fill Rate: {occ_metrics['fill_rate']:.2%}\n\n")
        
        f.write("By Loss Probability:\n")
        for loss, loss_metrics in summary['metrics']['by_loss'].items():
            f.write(f"  {loss:.2f}:\n")
            f.write(f"    Execution Time: {loss_metrics['execution_time']:.4f} s\n")
            f.write(f"    Fill Rate: {loss_metrics['fill_rate']:.2%}\n\n")
        
        # Scaling analysis
        f.write("SCALING ANALYSIS\n")
        f.write("===============\n\n")
        
        # Fit execution time to polynomial
        try:
            sizes = sorted(df['lattice_width'].unique())
            times = [df[df['lattice_width'] == size]['execution_time'].mean() for size in sizes]
            
            if len(sizes) >= 3:
                # Try quadratic fit for computational complexity
                z = np.polyfit(sizes, times, 2)
                f.write(f"Execution Time Scaling: ~{z[0]:.2e}n² + {z[1]:.2e}n + {z[2]:.2e}\n")
                
                # Try linear fit for moves
                moves = [df[df['lattice_width'] == size]['total_moves'].mean() for size in sizes]
                z_moves = np.polyfit(sizes, moves, 1)
                f.write(f"Moves Scaling: ~{z_moves[0]:.2f}n + {z_moves[1]:.2f}\n")
        except:
            f.write("Not enough data points for scaling analysis\n")

def main():
    parser = argparse.ArgumentParser(description='Center strategy benchmarking for defect-free lattice simulator')
    parser.add_argument('--sizes', type=str, default='10,20,30,50,70,100',
                       help='Comma-separated list of lattice sizes (square dimensions)')
    parser.add_argument('--occupation', type=str, default='0.5,0.7,0.9',
                       help='Comma-separated list of occupation probabilities')
    parser.add_argument('--loss', type=str, default='0.0,0.01,0.05',
                       help='Comma-separated list of atom loss probabilities')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for each parameter combination')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='center_benchmark_results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--detailed-timing', action='store_true',
                       help='Collect detailed timing information')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Path to existing benchmark CSV to visualize without running new benchmarks')
    
    args = parser.parse_args()
    
    # If just visualizing existing results
    if args.visualize:
        if os.path.exists(args.visualize):
            print(f"Visualizing existing benchmark results from {args.visualize}")
            df = pd.read_csv(args.visualize)
            visualize_center_benchmarks(df, args.output)
            generate_center_summary(df, args.output)
            return
        else:
            print(f"Error: File {args.visualize} not found.")
            return
    
    # Parse parameters
    sizes = [int(x) for x in args.sizes.split(',')]
    occupation_probs = [float(x) for x in args.occupation.split(',')]
    loss_probs = [float(x) for x in args.loss.split(',')]
    
    # Run the benchmark suite
    run_center_benchmark_suite(
        sizes=sizes,
        occupation_probs=occupation_probs,
        loss_probs=loss_probs,
        iterations=args.iterations,
        detailed_timing=args.detailed_timing,
        seed=args.seed,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
