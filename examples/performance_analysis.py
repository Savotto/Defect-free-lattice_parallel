"""
Performance analysis script for the defect-free lattice rearrangement algorithm.
Tests different lattice sizes, filling probabilities, and atom loss rates.
Compares center and corner strategies on multiple metrics.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm  # Optional: for progress bar
import argparse
from defect_free import LatticeSimulator, LatticeVisualizer

def run_simulation(lattice_size, occupation_prob, loss_prob, strategy, iterations=3, seed=None):
    """
    Run a simulation with the given parameters multiple times and return average metrics.
    
    Args:
        lattice_size: Tuple of (height, width) for the lattice
        occupation_prob: Probability of atom occupation (0.0 to 1.0)
        loss_prob: Probability of atom loss during movement
        strategy: 'center' or 'corner'
        iterations: Number of iterations to run for statistical significance
        seed: Random seed for reproducibility (if None, uses different seeds)
        
    Returns:
        Dictionary of averaged metrics
    """
    metrics = {
        'execution_time': [],
        'physical_time': [],
        'fill_rate': [],
        'perfect_fill': [],
        'retention_rate': [],
        'target_size': [],
        'total_moves': [],
        'total_time': []
    }
    
    for i in range(iterations):
        # Set seed for reproducibility if provided
        if seed is not None:
            np_seed = seed + i  # Different seed for each iteration but reproducible
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
        
        # Run the rearrangement
        start_time = time.time()
        result, execution_time = simulator.rearrange_for_defect_free(
            strategy=strategy,
            show_visualization=False
        )
        final_lattice, fill_rate, _ = result
        
        # Calculate metrics
        physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
        total_time = execution_time + physical_time
        total_moves = len(simulator.movement_history)
        
        # Target region metrics
        target_region = simulator.movement_manager.target_region
        start_row, start_col, end_row, end_col = target_region
        target_zone = simulator.field[start_row:end_row, start_col:end_col]
        atoms_in_target = np.sum(target_zone)
        target_size = simulator.side_length ** 2
        
        # Additional metrics
        perfect_fill = 1.0 if fill_rate == 1.0 else 0.0
        retention_rate = atoms_in_target / initial_atoms if initial_atoms > 0 else 0
        
        # Store metrics
        metrics['execution_time'].append(execution_time)
        metrics['physical_time'].append(physical_time)
        metrics['fill_rate'].append(fill_rate)
        metrics['perfect_fill'].append(perfect_fill)
        metrics['retention_rate'].append(retention_rate)
        metrics['target_size'].append(target_size)
        metrics['total_moves'].append(total_moves)
        metrics['total_time'].append(total_time)
    
    # Average the metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {f"{k}_std": np.std(v) for k, v in metrics.items()}
    
    # Format lattice size for readability
    size_str = f"{lattice_size[0]}x{lattice_size[1]}"
    
    # Combine averages and standard deviations
    combined_metrics = {
        **avg_metrics,
        **std_metrics,
        'lattice_size': size_str,
        'lattice_width': lattice_size[0],
        'lattice_height': lattice_size[1],
        'occupation_prob': occupation_prob,
        'loss_prob': loss_prob,
        'strategy': strategy
    }
    
    return combined_metrics

def visualize_results(df, output_dir):
    """
    Create visualizations to compare performance metrics using matplotlib.
    
    Args:
        df: DataFrame containing simulation results
        output_dir: Directory to save visualizations
    """
    # Define common plot parameters
    plt.rcParams.update({'font.size': 12})
    center_color = '#3366CC'  # Blue for center strategy
    corner_color = '#CC3366'  # Pink/red for corner strategy
    
    # Helper function to create grouped bar plots
    def create_grouped_bar_plot(data, x_field, y_field, title, xlabel, ylabel, filename):
        # Get unique values for x_field and strategies
        x_values = sorted(data[x_field].unique())
        strategies = sorted(data['strategy'].unique())
        
        # Create figure
        plt.figure(figsize=(12, 7))
        
        # Set bar width and positions
        bar_width = 0.35
        x_positions = np.arange(len(x_values))
        
        # Plot bars for each strategy
        for i, strategy in enumerate(strategies):
            strategy_data = data[data['strategy'] == strategy]
            # Calculate means and errors for each x value
            means = []
            errors = []
            for x in x_values:
                values = strategy_data[strategy_data[x_field] == x][y_field]
                means.append(values.mean())
                errors.append(values.std() / np.sqrt(len(values)))  # Standard error
            
            # Plot the bars with error bars
            plt.bar(x_positions + (i - 0.5) * bar_width, means, bar_width,
                   label=strategy.capitalize(),
                   color=center_color if strategy == 'center' else corner_color,
                   yerr=errors, capsize=5)
        
        # Configure the plot
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(x_positions, x_values)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1. Compare execution time between strategies
    create_grouped_bar_plot(
        df, 'lattice_size', 'execution_time',
        'Execution Time Comparison by Lattice Size and Strategy',
        'Lattice Size', 'Execution Time (s)',
        'execution_time_comparison.png'
    )
    
    # 2. Compare fill rates between strategies
    create_grouped_bar_plot(
        df, 'lattice_size', 'fill_rate',
        'Fill Rate Comparison by Lattice Size and Strategy',
        'Lattice Size', 'Fill Rate',
        'fill_rate_comparison.png'
    )
    
    # 3. Compare performance across different atom loss probabilities
    # Create separate plots for each strategy and loss probability combination
    strategies = sorted(df['strategy'].unique())
    loss_probs = sorted(df['loss_prob'].unique())
    
    # Create a grid of plots
    fig, axes = plt.subplots(len(strategies), len(loss_probs), figsize=(15, 10), squeeze=False)
    
    for i, strategy in enumerate(strategies):
        for j, loss_prob in enumerate(loss_probs):
            # Filter data
            plot_data = df[(df['strategy'] == strategy) & (df['loss_prob'] == loss_prob)]
            
            # Skip if no data
            if plot_data.empty:
                continue
                
            # Group by occupation probability
            occ_probs = sorted(plot_data['occupation_prob'].unique())
            fill_rates = [plot_data[plot_data['occupation_prob'] == occ]['fill_rate'].mean() for occ in occ_probs]
            errors = [plot_data[plot_data['occupation_prob'] == occ]['fill_rate'].std() for occ in occ_probs]
            
            # Plot on the corresponding axis
            ax = axes[i, j]
            ax.errorbar(occ_probs, fill_rates, yerr=errors, marker='o', capsize=5,
                       color=center_color if strategy == 'center' else corner_color)
            
            ax.set_xlabel('Occupation Probability')
            ax.set_ylabel('Fill Rate')
            ax.set_title(f"{strategy.capitalize()} - Loss: {loss_prob}")
            ax.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fill_rate_by_loss_prob.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Compare physical movement time
    create_grouped_bar_plot(
        df, 'lattice_size', 'physical_time',
        'Physical Movement Time Comparison',
        'Lattice Size', 'Physical Time (s)',
        'physical_time_comparison.png'
    )
    
    # 5. Success rate (perfect fill) comparison
    create_grouped_bar_plot(
        df, 'lattice_size', 'perfect_fill',
        'Perfect Fill Success Rate',
        'Lattice Size', 'Success Rate',
        'perfect_fill_comparison.png'
    )
    
    # 6. Target size vs lattice size
    create_grouped_bar_plot(
        df, 'lattice_size', 'target_size',
        'Target Size by Lattice Size',
        'Lattice Size', 'Target Size (atoms)',
        'target_size_comparison.png'
    )
    
    # 7. Retention rate comparison
    create_grouped_bar_plot(
        df, 'lattice_size', 'retention_rate',
        'Retention Rate Comparison',
        'Lattice Size', 'Retention Rate',
        'retention_rate_comparison.png'
    )
    
    # 8. Retention rate vs. atom loss probability
    create_grouped_bar_plot(
        df, 'loss_prob', 'retention_rate',
        'Retention Rate vs. Atom Loss Probability',
        'Atom Loss Probability', 'Retention Rate',
        'retention_vs_loss.png'
    )
    
    # 9. Total moves by lattice size
    create_grouped_bar_plot(
        df, 'lattice_size', 'total_moves',
        'Total Moves by Lattice Size',
        'Lattice Size', 'Total Moves',
        'total_moves_comparison.png'
    )
    
    # 10. NEW: Occupation probability vs execution time with different atom loss probabilities
    # This is specifically for the corner method as requested
    plt.figure(figsize=(12, 8))
    
    # Define a colormap for atom loss probabilities
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # For each strategy, create a plot
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        loss_probs = sorted(strategy_data['loss_prob'].unique())
        lattice_sizes = sorted(strategy_data['lattice_width'].unique())
        
        for size_idx, lattice_size in enumerate(lattice_sizes):
            plt.figure(figsize=(12, 8))
            
            for loss_idx, loss_prob in enumerate(loss_probs):
                # Filter data for this loss probability and lattice size
                filtered_data = strategy_data[(strategy_data['loss_prob'] == loss_prob) & 
                                            (strategy_data['lattice_width'] == lattice_size)]
                
                # Group by occupation probability
                occ_probs = sorted(filtered_data['occupation_prob'].unique())
                exec_times = [filtered_data[filtered_data['occupation_prob'] == occ]['execution_time'].mean() for occ in occ_probs]
                errors = [filtered_data[filtered_data['occupation_prob'] == occ]['execution_time'].std() / 
                        np.sqrt(len(filtered_data[filtered_data['occupation_prob'] == occ])) for occ in occ_probs]
                
                # Choose color based on loss probability
                color_idx = loss_idx % len(colors)
                
                # Plot with error bars
                plt.errorbar(occ_probs, exec_times, yerr=errors, marker='o', capsize=5,
                          label=f'Loss Prob: {loss_prob}', color=colors[color_idx], linewidth=2)
            
            plt.xlabel('Occupation Probability')
            plt.ylabel('Execution Time (s)')
            plt.title(f'{strategy.capitalize()} Strategy: Occupation Probability vs Execution Time\nLattice Size: {lattice_size}x{lattice_size}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'occ_vs_exec_time_{strategy}_{lattice_size}x{lattice_size}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 11. NEW: All lattice sizes in one plot (specifically for corner method)
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        loss_probs = sorted(strategy_data['loss_prob'].unique())
        
        for loss_idx, loss_prob in enumerate(loss_probs):
            plt.figure(figsize=(12, 8))
            
            # Filter data for this loss probability
            filtered_data = strategy_data[strategy_data['loss_prob'] == loss_prob]
            lattice_sizes = sorted(filtered_data['lattice_width'].unique())
            
            for size_idx, lattice_size in enumerate(lattice_sizes):
                size_data = filtered_data[filtered_data['lattice_width'] == lattice_size]
                
                # Group by occupation probability
                occ_probs = sorted(size_data['occupation_prob'].unique())
                exec_times = [size_data[size_data['occupation_prob'] == occ]['execution_time'].mean() for occ in occ_probs]
                errors = [size_data[size_data['occupation_prob'] == occ]['execution_time'].std() / 
                        np.sqrt(len(size_data[size_data['occupation_prob'] == occ])) for occ in occ_probs]
                
                # Choose color based on lattice size
                color_idx = size_idx % len(colors)
                
                # Plot with error bars
                plt.errorbar(occ_probs, exec_times, yerr=errors, marker='o', capsize=5,
                          label=f'Size: {lattice_size}x{lattice_size}', color=colors[color_idx], linewidth=2)
            
            plt.xlabel('Occupation Probability')
            plt.ylabel('Execution Time (s)')
            plt.title(f'{strategy.capitalize()} Strategy: Occupation Probability vs Execution Time\nAtom Loss Probability: {loss_prob}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'occ_vs_exec_time_{strategy}_loss_{loss_prob}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Performance analysis for defect-free lattice simulator')
    parser.add_argument('--strategies', type=str, default='corner', choices=['center', 'corner', 'both'],
                       help='Which strategy to analyze (center, corner, or both)')
    parser.add_argument('--sizes', type=str, default='10,20,30,50,70,100',
                       help='Comma-separated list of lattice sizes (square dimensions)')
    parser.add_argument('--occupation', type=str, default='0.3,0.5,0.7,0.9,1.0',
                       help='Comma-separated list of occupation probabilities')
    parser.add_argument('--loss', type=str, default='0.0,0.01,0.05',
                       help='Comma-separated list of atom loss probabilities')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations for each parameter combination')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='corner_analysis_results',
                       help='Output directory for results and visualizations')
    
    args = parser.parse_args()
    
    # Parse parameters
    if args.strategies == 'both':
        strategies = ['center', 'corner']
    else:
        strategies = [args.strategies]
        
    # Parse sizes to create square lattices
    sizes = [int(x) for x in args.sizes.split(',')]
    lattice_sizes = [(x, x) for x in sizes]
    
    # Parse other parameters
    occupation_probs = [float(x) for x in args.occupation.split(',')]
    loss_probs = [float(x) for x in args.loss.split(',')]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Collect all results
    results = []
    
    # Configuration to show which parameters to iterate over
    parameter_combinations = []
    for size in lattice_sizes:
        for occ_prob in occupation_probs:
            for loss_prob in loss_probs:
                for strategy in strategies:
                    parameter_combinations.append({
                        'lattice_size': size,
                        'occupation_prob': occ_prob,
                        'loss_prob': loss_prob,
                        'strategy': strategy
                    })
    
    # Print configuration
    print("Performance Analysis Configuration:")
    print(f"- Strategies: {strategies}")
    print(f"- Lattice Sizes: {sizes} x {sizes}")
    print(f"- Occupation Probabilities: {occupation_probs}")
    print(f"- Atom Loss Probabilities: {loss_probs}")
    print(f"- Iterations per combination: {args.iterations}")
    print(f"- Total parameter combinations: {len(parameter_combinations)}")
    
    # Run simulations for all parameter combinations
    print(f"Running simulations for {len(parameter_combinations)} parameter combinations...")
    
    # Try using tqdm for a progress bar if available, otherwise use regular loop
    try:
        for params in tqdm(parameter_combinations):
            metrics = run_simulation(
                params['lattice_size'], 
                params['occupation_prob'], 
                params['loss_prob'], 
                params['strategy'],
                args.iterations,
                args.seed
            )
            results.append(metrics)
    except NameError:
        # If tqdm is not available
        total_combinations = len(parameter_combinations)
        for i, params in enumerate(parameter_combinations):
            print(f"Processing combination {i+1}/{total_combinations}...")
            metrics = run_simulation(
                params['lattice_size'], 
                params['occupation_prob'], 
                params['loss_prob'], 
                params['strategy'],
                args.iterations,
                args.seed
            )
            results.append(metrics)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Save results to CSV for later analysis
    csv_path = os.path.join(args.output, 'lattice_performance_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Visualize key comparisons
    visualize_results(df, args.output)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]
        print(f"\n{strategy.capitalize()} Strategy:")
        print(f"- Average Execution Time: {strategy_df['execution_time'].mean():.3f} s")
        print(f"- Average Physical Time: {strategy_df['physical_time'].mean():.6f} s")
        print(f"- Average Fill Rate: {strategy_df['fill_rate'].mean():.2%}")
        print(f"- Perfect Fill Success Rate: {strategy_df['perfect_fill'].mean():.2%}")

if __name__ == "__main__":
    main()