#!/usr/bin/env python3
"""
Benchmarking script for iterative blind center filling strategy.
Tracks metrics per iteration to analyze convergence and performance characteristics.
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

def benchmark_blind_center(lattice_size, occupation_prob, loss_prob, 
                                     max_iterations=5, min_improvement=0.01,
                                     high_fill_threshold=0.9, samples=5, seed=None):
    """
    Benchmark the iterative blind center filling strategy with detailed per-iteration metrics.
    
    Args:
        lattice_size: Tuple of (height, width) for the lattice
        occupation_prob: Probability of atom occupation (0.0 to 1.0)
        loss_prob: Probability of atom loss during movement
        max_iterations: Maximum iterations for the iterative strategy
        min_improvement: Minimum improvement threshold to continue iterations
        high_fill_threshold: Fill rate threshold for optimization strategies
        samples: Number of samples to average over
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing averaged metrics and per-iteration data
    """
    # Overall metrics to be averaged across samples
    overall_metrics = {
        'total_execution_time': [],
        'total_physical_time': [],
        'final_fill_rate': [],
        'iterations_used': [],
        'perfect_fill': [],
        'retention_rate': []
    }
    
    # Per-iteration metrics for each sample
    per_iteration_metrics = {
        'fill_rates': [[] for _ in range(max_iterations)],
        'execution_times': [[] for _ in range(max_iterations)],
        'physical_times': [[] for _ in range(max_iterations)],
        'moves_counts': [[] for _ in range(max_iterations)]
    }
    
    for sample in range(samples):
        print(f"Running sample {sample+1}/{samples}")
        
        # Set seed for reproducibility if provided
        if seed is not None:
            sample_seed = seed + sample
        else:
            sample_seed = None
            
        # Initialize simulator
        simulator = LatticeSimulator(
            initial_size=lattice_size, 
            occupation_prob=occupation_prob,
            physical_constraints={'atom_loss_probability': loss_prob}
        )
        
        # Generate initial lattice
        simulator.generate_initial_lattice(seed=sample_seed)
        initial_atoms = np.sum(simulator.field)
        
        # Prepare to track per-iteration metrics
        sample_iter_metrics = {
            'fill_rates': [],
            'execution_times': [],
            'physical_times': [],
            'moves_counts': []
        }
        
        # Monkey-patch the iterative_blind_center_filling method to capture per-iteration metrics
        original_method = simulator.movement_manager.center_manager.iterative_blind_center_filling
        
        def patched_method(*args, **kwargs):
            """Monkey-patched method that tracks per-iteration metrics"""
            # Override show_visualization to ensure it's off during benchmarking
            kwargs['show_visualization'] = False
            
            start_time = time.time()
            
            # Store initial state of the simulator
            original_field = simulator.field.copy()
            original_total_atoms = np.sum(original_field)
            
            # Initialize target region
            simulator.movement_manager.center_manager.initialize_target_region()
            target_region = simulator.movement_manager.center_manager.target_region
            target_start_row, target_start_col, target_end_row, target_end_col = target_region
            target_size = (target_end_row - target_start_row) * (target_end_col - target_start_col)
            
            # Track fill rates and times per iteration
            iter_fill_rates = []
            iter_execution_times = []
            iter_physical_times = []
            iter_moves_counts = []
            all_movement_history = []
            iterations_used = 0
            
            # Run for a maximum number of iterations
            for iteration in range(1, max_iterations + 1):
                iterations_used = iteration
                
                # Reset movement history for this iteration
                simulator.movement_history = []
                
                # Measure time for this iteration
                iter_start_time = time.time()
                
                # For first iteration, always use full algorithm
                if iteration == 1:
                    # Call the blind_center_filling_strategy directly on the center_manager
                    simulator.movement_manager.center_manager.blind_center_filling_strategy(show_visualization=False)
                else:
                    # For high fill rates, use simplified strategies
                    current_fill_rate = iter_fill_rates[-1] if iter_fill_rates else 0.0
                    
                    if current_fill_rate >= high_fill_threshold:
                        # Use the optimized column-wise squeezing
                        simulator.movement_manager.center_manager._perform_column_wise_squeezing(target_region)
                    else:
                        # For lower fill rates, continue with full algorithm
                        simulator.movement_manager.center_manager.blind_center_filling_strategy(show_visualization=False)
                
                # Calculate metrics for this iteration
                iter_execution_time = time.time() - iter_start_time
                
                # Calculate physical time for this iteration
                iter_physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
                
                # Count total moves for this iteration
                iter_moves_count = len(simulator.movement_history)
                
                # Save the movement history from this iteration
                all_movement_history.extend(simulator.movement_history)
                
                # Calculate fill rate after this iteration
                target_zone = simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
                defects = np.sum(target_zone == 0)
                actual_fill_rate = 1.0 - (defects / target_size)
                
                # Record metrics for this iteration
                iter_fill_rates.append(actual_fill_rate)
                iter_execution_times.append(iter_execution_time)
                iter_physical_times.append(iter_physical_time)
                iter_moves_counts.append(iter_moves_count)
                
                # Check if we've achieved perfect fill
                if defects == 0:
                    break
                
                # Check improvement if not the first iteration
                if iteration > 1:
                    improvement = actual_fill_rate - iter_fill_rates[-2]
                    
                    # Stop if improvement is below threshold
                    if improvement < min_improvement:
                        break
            
            # Save per-iteration metrics
            for i in range(len(iter_fill_rates)):
                sample_iter_metrics['fill_rates'].append(iter_fill_rates[i])
                sample_iter_metrics['execution_times'].append(iter_execution_times[i])
                sample_iter_metrics['physical_times'].append(iter_physical_times[i])
                sample_iter_metrics['moves_counts'].append(iter_moves_counts[i])
            
            # Calculate final metrics
            final_fill_rate = iter_fill_rates[-1] if iter_fill_rates else 0.0
            total_execution_time = sum(iter_execution_times)
            total_physical_time = sum(iter_physical_times)
            perfect_fill = 1.0 if defects == 0 else 0.0
            
            # Calculate retention rate
            atoms_in_target = np.sum(target_zone == 1)
            retention_rate = atoms_in_target / original_total_atoms if original_total_atoms > 0 else 0
            
            # Restore original movement history for compatibility
            simulator.movement_history = all_movement_history
            
            execution_time = time.time() - start_time
            
            # Return a tuple matching the original method's return format
            return (simulator.field.copy(), final_fill_rate, execution_time, iterations_used)
        
        # Replace method with our instrumented version
        simulator.movement_manager.center_manager.iterative_blind_center_filling = patched_method
        
        # Run the benchmark by calling the iterative_blind_center_filling method directly
        overall_start_time = time.time()
        final_lattice, final_fill_rate, execution_time, iterations_used = simulator.movement_manager.center_manager.iterative_blind_center_filling(
            max_iterations=max_iterations,
            min_improvement=min_improvement,
            high_fill_threshold=high_fill_threshold,
            show_visualization=False
        )
        overall_time = time.time() - overall_start_time
        
        # Calculate metrics
        total_physical_time = sum(sample_iter_metrics['physical_times'])
        
        # Target region metrics
        target_region = simulator.movement_manager.target_region
        start_row, start_col, end_row, end_col = target_region
        target_zone = simulator.field[start_row:end_row, start_col:end_col]
        atoms_in_target = np.sum(target_zone)
        perfect_fill = 1.0 if final_fill_rate == 1.0 else 0.0
        retention_rate = atoms_in_target / initial_atoms if initial_atoms > 0 else 0
        
        # Record overall metrics for this sample
        overall_metrics['total_execution_time'].append(execution_time)
        overall_metrics['total_physical_time'].append(total_physical_time)
        overall_metrics['final_fill_rate'].append(final_fill_rate)
        overall_metrics['iterations_used'].append(iterations_used)
        overall_metrics['perfect_fill'].append(perfect_fill)
        overall_metrics['retention_rate'].append(retention_rate)
        
        # Record per-iteration metrics for this sample
        for i in range(min(iterations_used, max_iterations)):
            if i < len(sample_iter_metrics['fill_rates']):
                per_iteration_metrics['fill_rates'][i].append(sample_iter_metrics['fill_rates'][i])
                per_iteration_metrics['execution_times'][i].append(sample_iter_metrics['execution_times'][i])
                per_iteration_metrics['physical_times'][i].append(sample_iter_metrics['physical_times'][i])
                per_iteration_metrics['moves_counts'][i].append(sample_iter_metrics['moves_counts'][i])
        
        # Restore original method
        simulator.movement_manager.center_manager.iterative_blind_center_filling = original_method
    
    # Average the overall metrics
    avg_overall_metrics = {k: np.mean(v) for k, v in overall_metrics.items()}
    std_overall_metrics = {f"{k}_std": np.std(v) for k, v in overall_metrics.items()}
    
    # Average the per-iteration metrics
    avg_per_iteration = {
        'fill_rates': [np.mean(rates) if rates else None for rates in per_iteration_metrics['fill_rates']],
        'execution_times': [np.mean(times) if times else None for times in per_iteration_metrics['execution_times']],
        'physical_times': [np.mean(times) if times else None for times in per_iteration_metrics['physical_times']],
        'moves_counts': [np.mean(counts) if counts else None for counts in per_iteration_metrics['moves_counts']]
    }
    
    std_per_iteration = {
        'fill_rates_std': [np.std(rates) if rates else None for rates in per_iteration_metrics['fill_rates']],
        'execution_times_std': [np.std(times) if times else None for times in per_iteration_metrics['execution_times']],
        'physical_times_std': [np.std(times) if times else None for times in per_iteration_metrics['physical_times']],
        'moves_counts_std': [np.std(counts) if counts else None for counts in per_iteration_metrics['moves_counts']]
    }
    
    # Combine all metrics
    size_str = f"{lattice_size[0]}x{lattice_size[1]}"
    combined_metrics = {
        **avg_overall_metrics,
        **std_overall_metrics,
        'per_iteration': {
            **avg_per_iteration,
            **std_per_iteration
        },
        'lattice_size': size_str,
        'lattice_width': lattice_size[0],
        'lattice_height': lattice_size[1],
        'occupation_prob': occupation_prob,
        'loss_prob': loss_prob,
        'max_iterations': max_iterations,
        'min_improvement': min_improvement,
        'high_fill_threshold': high_fill_threshold
    }
    
    return combined_metrics


# The rest of the benchmark script remains the same...
# (visualize_iterative_benchmarks and run_benchmark_suite functions are unchanged)

def visualize_iterative_benchmarks(results, output_dir):
    """
    Create visualizations of iterative blind center strategy benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'per_iteration'} for r in results])
    
    # Extract unique parameter values
    lattice_sizes = sorted(df['lattice_size'].unique())
    occupation_probs = sorted(df['occupation_prob'].unique())
    loss_probs = sorted(df['loss_prob'].unique())
    
    # Define colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(lattice_sizes)))
    
    # 1. Fill Rate Progression by Iteration for different lattice sizes
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(lattice_sizes):
        # Get the result dictionary for this size with average occupation and loss
        mid_occ = occupation_probs[len(occupation_probs)//2]
        mid_loss = loss_probs[len(loss_probs)//2]
        
        size_results = [r for r in results 
                       if r['lattice_size'] == size 
                       and r['occupation_prob'] == mid_occ
                       and r['loss_prob'] == mid_loss]
        
        if size_results:
            result = size_results[0]
            fill_rates = result['per_iteration']['fill_rates']
            fill_rates_std = result['per_iteration']['fill_rates_std']
            
            # Remove None values
            iterations = list(range(1, len(fill_rates) + 1))
            valid_indices = [i for i, x in enumerate(fill_rates) if x is not None]
            valid_fill_rates = [fill_rates[i] for i in valid_indices]
            valid_iterations = [iterations[i] for i in valid_indices]
            valid_errors = [fill_rates_std[i] for i in valid_indices]
            
            plt.errorbar(valid_iterations, valid_fill_rates, yerr=valid_errors, 
                       marker='o', capsize=5, color=colors[i], label=size, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Fill Rate')
    plt.title(f'Fill Rate Progression by Iteration (Occ={mid_occ}, Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lattice Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fill_rate_progression.png'), dpi=300)
    plt.close()
    
    # 2. Execution Time by Iteration for different lattice sizes
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(lattice_sizes):
        # Get the result dictionary for this size with average occupation and loss
        mid_occ = occupation_probs[len(occupation_probs)//2]
        mid_loss = loss_probs[len(loss_probs)//2]
        
        size_results = [r for r in results 
                       if r['lattice_size'] == size 
                       and r['occupation_prob'] == mid_occ
                       and r['loss_prob'] == mid_loss]
        
        if size_results:
            result = size_results[0]
            exec_times = result['per_iteration']['execution_times']
            exec_times_std = result['per_iteration']['execution_times_std']
            
            # Remove None values
            iterations = list(range(1, len(exec_times) + 1))
            valid_indices = [i for i, x in enumerate(exec_times) if x is not None]
            valid_times = [exec_times[i] for i in valid_indices]
            valid_iterations = [iterations[i] for i in valid_indices]
            valid_errors = [exec_times_std[i] for i in valid_indices]
            
            plt.errorbar(valid_iterations, valid_times, yerr=valid_errors, 
                       marker='o', capsize=5, color=colors[i], label=size, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time by Iteration (Occ={mid_occ}, Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lattice Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_by_iteration.png'), dpi=300)
    plt.close()
    
    # 3. Improvement in Fill Rate by Iteration
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(lattice_sizes):
        # Get the result dictionary for this size with average occupation and loss
        mid_occ = occupation_probs[len(occupation_probs)//2]
        mid_loss = loss_probs[len(loss_probs)//2]
        
        size_results = [r for r in results 
                       if r['lattice_size'] == size 
                       and r['occupation_prob'] == mid_occ
                       and r['loss_prob'] == mid_loss]
        
        if size_results:
            result = size_results[0]
            fill_rates = result['per_iteration']['fill_rates']
            
            # Calculate improvement
            improvements = []
            for j in range(1, len(fill_rates)):
                if fill_rates[j] is not None and fill_rates[j-1] is not None:
                    improvements.append(fill_rates[j] - fill_rates[j-1])
            
            if improvements:
                plt.bar([f"{size}-{j+1}" for j in range(len(improvements))], 
                       improvements, color=colors[i], alpha=0.7)
    
    plt.xlabel('Lattice Size - Iteration')
    plt.ylabel('Improvement in Fill Rate')
    plt.title(f'Improvement in Fill Rate by Iteration (Occ={mid_occ}, Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fill_rate_improvement.png'), dpi=300)
    plt.close()
    
    # 4. Total Execution Time vs Lattice Size for Different Occupation Probabilities
    plt.figure(figsize=(12, 8))
    
    for i, occ in enumerate(occupation_probs):
        sizes = []
        times = []
        errors = []
        
        # Use middle loss probability
        mid_loss = loss_probs[len(loss_probs)//2]
        
        for size in lattice_sizes:
            size_occ_data = df[(df['lattice_size'] == size) & 
                             (df['occupation_prob'] == occ) & 
                             (df['loss_prob'] == mid_loss)]
            
            if not size_occ_data.empty:
                sizes.append(size)
                times.append(size_occ_data['total_execution_time'].mean())
                errors.append(size_occ_data['total_execution_time_std'].mean() 
                            if 'total_execution_time_std' in size_occ_data.columns else 0)
        
        if sizes:
            plt.errorbar(sizes, times, yerr=errors, marker='o', capsize=5,
                       label=f'Occ={occ}', linewidth=2)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Total Execution Time (s)')
    plt.title(f'Total Execution Time vs Lattice Size (Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_execution_time.png'), dpi=300)
    plt.close()
    
    # 5. Number of Iterations Used vs Lattice Size
    plt.figure(figsize=(12, 8))
    
    for i, occ in enumerate(occupation_probs):
        sizes = []
        iterations = []
        errors = []
        
        # Use middle loss probability
        mid_loss = loss_probs[len(loss_probs)//2]
        
        for size in lattice_sizes:
            size_occ_data = df[(df['lattice_size'] == size) & 
                             (df['occupation_prob'] == occ) & 
                             (df['loss_prob'] == mid_loss)]
            
            if not size_occ_data.empty:
                sizes.append(size)
                iterations.append(size_occ_data['iterations_used'].mean())
                errors.append(size_occ_data['iterations_used_std'].mean() 
                            if 'iterations_used_std' in size_occ_data.columns else 0)
        
        if sizes:
            plt.errorbar(sizes, iterations, yerr=errors, marker='o', capsize=5,
                       label=f'Occ={occ}', linewidth=2)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Average Iterations Used')
    plt.title(f'Iterations Used vs Lattice Size (Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iterations_used.png'), dpi=300)
    plt.close()
    
    # 6. Physical Time per Iteration
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(lattice_sizes):
        # Get the result dictionary for this size with average occupation and loss
        mid_occ = occupation_probs[len(occupation_probs)//2]
        mid_loss = loss_probs[len(loss_probs)//2]
        
        size_results = [r for r in results 
                       if r['lattice_size'] == size 
                       and r['occupation_prob'] == mid_occ
                       and r['loss_prob'] == mid_loss]
        
        if size_results:
            result = size_results[0]
            phy_times = result['per_iteration']['physical_times']
            phy_times_std = result['per_iteration']['physical_times_std']
            
            # Remove None values
            iterations = list(range(1, len(phy_times) + 1))
            valid_indices = [i for i, x in enumerate(phy_times) if x is not None]
            valid_times = [phy_times[i] for i in valid_indices]
            valid_iterations = [iterations[i] for i in valid_indices]
            valid_errors = [phy_times_std[i] for i in valid_indices]
            
            plt.errorbar(valid_iterations, valid_times, yerr=valid_errors, 
                       marker='o', capsize=5, color=colors[i], label=size, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Physical Time (s)')
    plt.title(f'Physical Movement Time by Iteration (Occ={mid_occ}, Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lattice Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'physical_time_by_iteration.png'), dpi=300)
    plt.close()
    
    # 7. Final Fill Rate vs Lattice Size
    plt.figure(figsize=(12, 8))
    
    # Group by atom loss probability
    for i, loss in enumerate(loss_probs):
        sizes = []
        rates = []
        errors = []
        
        # Use middle occupation probability
        mid_occ = occupation_probs[len(occupation_probs)//2]
        
        for size in lattice_sizes:
            size_loss_data = df[(df['lattice_size'] == size) & 
                              (df['occupation_prob'] == mid_occ) & 
                              (df['loss_prob'] == loss)]
            
            if not size_loss_data.empty:
                sizes.append(size)
                rates.append(size_loss_data['final_fill_rate'].mean())
                errors.append(size_loss_data['final_fill_rate_std'].mean() 
                            if 'final_fill_rate_std' in size_loss_data.columns else 0)
        
        if sizes:
            plt.errorbar(sizes, rates, yerr=errors, marker='o', capsize=5,
                       label=f'Loss={loss}', linewidth=2)
    
    plt.xlabel('Lattice Size')
    plt.ylabel('Final Fill Rate')
    plt.title(f'Final Fill Rate vs Lattice Size (Occ={mid_occ})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_fill_rate.png'), dpi=300)
    plt.close()
    
    # 8. Execution/Physical Time Ratio by Iteration
    plt.figure(figsize=(12, 8))
    
    for i, size in enumerate(lattice_sizes):
        # Get the result dictionary for this size with average occupation and loss
        mid_occ = occupation_probs[len(occupation_probs)//2]
        mid_loss = loss_probs[len(loss_probs)//2]
        
        size_results = [r for r in results 
                       if r['lattice_size'] == size 
                       and r['occupation_prob'] == mid_occ
                       and r['loss_prob'] == mid_loss]
        
        if size_results:
            result = size_results[0]
            exec_times = result['per_iteration']['execution_times']
            phy_times = result['per_iteration']['physical_times']
            
            # Calculate ratio for each valid iteration
            ratios = []
            iterations = []
            for j in range(len(exec_times)):
                if exec_times[j] is not None and phy_times[j] is not None and phy_times[j] > 0:
                    ratios.append(exec_times[j] / phy_times[j])
                    iterations.append(j + 1)
            
            if ratios:
                plt.plot(iterations, ratios, marker='o', label=size, color=colors[i], linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time / Physical Time Ratio')
    plt.title(f'Algorithm Efficiency by Iteration (Occ={mid_occ}, Loss={mid_loss})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lattice Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_ratio.png'), dpi=300)
    plt.close()
    
    # Save raw results to JSON for later analysis
    with open(os.path.join(output_dir, 'iterative_benchmark_results.json'), 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        json.dump(json_results, f, indent=2)

def run_benchmark_suite(sizes, occupation_probs, loss_probs, samples=5, 
                       max_iterations=5, min_improvement=0.01, high_fill_threshold=0.9,
                       seed=None, output_dir="iterative_benchmark_results"):
    """
    Run a complete benchmark suite for the iterative blind center strategy.
    
    Args:
        sizes: List of lattice sizes (square dimensions)
        occupation_probs: List of occupation probabilities
        loss_probs: List of atom loss probabilities
        samples: Number of samples to average over
        max_iterations: Maximum iterations for the iterative strategy
        min_improvement: Minimum improvement threshold
        high_fill_threshold: Fill rate threshold for optimization
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
    print("Iterative Blind Center Strategy Benchmark Configuration:")
    print(f"- Lattice Sizes: {sizes} x {sizes}")
    print(f"- Occupation Probabilities: {occupation_probs}")
    print(f"- Atom Loss Probabilities: {loss_probs}")
    print(f"- Samples per combination: {samples}")
    print(f"- Maximum iterations: {max_iterations}")
    print(f"- Minimum improvement threshold: {min_improvement}")
    print(f"- High fill rate threshold: {high_fill_threshold}")
    print(f"- Total parameter combinations: {len(parameter_combinations)}")
    
    # Run benchmarks for all parameter combinations
    all_results = []
    
    try:
        for i, params in enumerate(parameter_combinations):
            print(f"\nBenchmarking configuration {i+1}/{len(parameter_combinations)}:")
            print(f"  - Size: {params['lattice_size']}")
            print(f"  - Occupation: {params['occupation_prob']}")
            print(f"  - Loss: {params['loss_prob']}")
            
            # Run the benchmark for this configuration
            result = benchmark_iterative_blind_center(
                lattice_size=params['lattice_size'],
                occupation_prob=params['occupation_prob'],
                loss_prob=params['loss_prob'],
                max_iterations=max_iterations,
                min_improvement=min_improvement,
                high_fill_threshold=high_fill_threshold,
                samples=samples,
                seed=seed
            )
            
            # Print some key results
            print("\nResults:")
            print(f"  - Final Fill Rate: {result['final_fill_rate']:.2%}")
            print(f"  - Iterations Used: {result['iterations_used']:.1f}")
            print(f"  - Total Execution Time: {result['total_execution_time']:.3f} s")
            
            all_results.append(result)
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Saving partial results...")
    
    # Generate visualizations
    visualize_iterative_benchmarks(all_results, output_dir)
    
    print(f"\nBenchmark complete! Results saved to {output_dir}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Benchmark the iterative blind center filling strategy')
    parser.add_argument('--sizes', type=str, default='20,30,50,70,100',
                      help='Comma-separated list of lattice sizes (square dimensions)')
    parser.add_argument('--occupation', type=str, default='0.5,0.7,0.9',
                      help='Comma-separated list of occupation probabilities')
    parser.add_argument('--loss', type=str, default='0.0,0.01,0.05',
                      help='Comma-separated list of atom loss probabilities')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of samples to average over')
    parser.add_argument('--iterations', type=int, default=7,
                      help='Maximum iterations for the iterative strategy')
    parser.add_argument('--improvement', type=float, default=0.001,
                      help='Minimum improvement threshold')
    parser.add_argument('--threshold', type=float, default=0.9,
                      help='High fill rate threshold for optimization')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='iterative_benchmark_results',
                      help='Output directory for results and visualizations')
    parser.add_argument('--quick', action='store_true',
                      help='Run a quicker benchmark with limited parameters')
    
    args = parser.parse_args()
    
    # Parse parameters
    if args.quick:
        # Quick benchmark with minimal parameters
        sizes = [20, 30]
        occupation_probs = [0.7]
        loss_probs = [0.01]
        samples = 2
    else:
        # Full benchmark
        sizes = [int(x) for x in args.sizes.split(',')]
        occupation_probs = [float(x) for x in args.occupation.split(',')]
        loss_probs = [float(x) for x in args.loss.split(',')]
        samples = args.samples
    
    # Run the benchmark suite
    run_benchmark_suite(
        sizes=sizes,
        occupation_probs=occupation_probs,
        loss_probs=loss_probs,
        samples=samples,
        max_iterations=args.iterations,
        min_improvement=args.improvement,
        high_fill_threshold=args.threshold,
        seed=args.seed,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()