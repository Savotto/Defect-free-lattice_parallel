#!/usr/bin/env python3
"""
Complete Algorithm Move Batches Benchmark 

This script benchmarks the number of move batches (operation complexity) 
for the complete atom rearrangement algorithm across different lattice sizes.
Focuses on comparison with literature values:
- Modified LSAP algorithm: scales as N^1.16(1)
- PSCA algorithm: scales as N^0.48(2)

This analyzes the full algorithm's performance until reaching 100% fill rate
or maximum allowed cycles, letting the algorithm determine its own target region size.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import os
import sys
from pathlib import Path
import argparse
from scipy.optimize import curve_fit
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from defect_free import LatticeSimulator, LatticeVisualizer

def analyze_complete_algorithm(initial_sizes, occupation_prob=0.75, loss_prob=0.0, 
                             trials=10, seed=None, strategy='center',
                             max_cycles=100, target_fill_rate=1.0):
    """
    Analyze move batch scaling for the complete algorithm until target fill rate.
    
    Args:
        initial_sizes: List of initial lattice side lengths L (L×L lattices)
        occupation_prob: Probability of atom occupation
        loss_prob: Probability of atom loss during movement
        trials: Number of different random initial configurations to try
        seed: Random seed for reproducibility
        strategy: 'center' or 'corner'
        max_cycles: Maximum number of repair cycles to attempt
        target_fill_rate: Target fill rate to stop algorithm (default 1.0 = 100%)
        
    Returns:
        Dictionary with scaling analysis results
    """
    scaling_data = {
        'initial_size': [],        # Initial lattice side length L
        'achieved_N': [],          # Size of target region achieved
        'achieved_N_std': [],      # Standard deviation of achieved N
        'move_batches_total': [],  # Total number of move batches across all cycles
        'move_batches_total_std': [], # Std dev of total batches
        'initial_batches': [],     # Initial planned batches before parallel merge
        'initial_batches_std': [],
        'reduced_batches': [],     # Reduced planned batches after parallel merge
        'reduced_batches_std': [],
        'repair_cycles': [],         # Number of cycles needed
        'repair_cycles_std': [],     # Std dev of cycles
        'calculation_time': [],      # Total computational time (ms)
        'calculation_time_std': [],  # Std dev of calculation time
        'final_fill_rate': [],       # Final achieved fill rate
        'final_fill_rate_std': []    # Std dev of fill rate
    }
    
    # For each initial size L, run multiple trials
    for L in initial_sizes:
        print(f"\nAnalyzing initial lattice size L = {L} with complete algorithm")

        # Metrics for this initial size across trials
        size_achieved_N = []
        size_total_batches = []
        size_initial_batches = []
        size_reduced_batches = []
        size_cycles = []
        size_calc_time = []
        size_fill_rate = []

        # Run multiple trials
        for trial in range(trials):
            if seed is not None:
                trial_seed = seed + trial
            else:
                trial_seed = None
                
            # Initialize simulator with initial lattice size
            simulator = LatticeSimulator(
                initial_size=(L, L),
                occupation_prob=occupation_prob,
                physical_constraints={'atom_loss_probability': loss_prob}
            )
            
            # Generate initial lattice
            simulator.generate_initial_lattice(seed=trial_seed)
            
            # Run complete algorithm until target fill rate or max cycles reached
            calculation_start_time = time.time()
            
            cycles = 0
            current_fill_rate = 0
            total_batches = 0
            target_N = 0
            
            # Let the algorithm calculate optimal target size based on available atoms
            optimal_size = simulator.calculate_max_defect_free_size(strategy=strategy)
            simulator.side_length = optimal_size
            
            # Let the movement manager initialize the target region
            if strategy == 'center':
                simulator.movement_manager.center_manager.initialize_target_region()
                target_region = simulator.movement_manager.center_manager.target_region
            else:
                simulator.movement_manager.corner_manager.initialize_target_region()
                target_region = simulator.movement_manager.corner_manager.target_region
                
            print(f"    Initialized target region with side length {optimal_size}")
            
            # Modified to use target_fill_rate instead of hardcoded 1.0
            while current_fill_rate < target_fill_rate and cycles < max_cycles:
                # Reset move history for this cycle
                simulator.movement_history = []
                
                # Use the correct method from movement_manager that handles both center and corner strategies
                result = simulator.movement_manager.rearrange_for_defect_free(
                    strategy=strategy,
                    show_visualization=False
                )
                
                # The target region should now be properly defined in the movement manager
                if strategy == 'center':
                    target_region = simulator.movement_manager.center_manager.target_region
                else:
                    target_region = simulator.movement_manager.corner_manager.target_region
                
                # Calculate fill rate and region size if target region exists
                if target_region:
                    start_row, start_col, end_row, end_col = target_region
                    target_zone = simulator.field[start_row:end_row, start_col:end_col]
                    target_N = (end_row - start_row) * (end_col - start_col)
                    current_fill_rate = np.sum(target_zone) / target_N
                    
                    # Add detailed debug info about the target region on first cycle
                    if cycles == 0:
                        atoms_in_target = np.sum(target_zone)
                        print(f"    Target region: {start_row},{start_col} to {end_row},{end_col}")
                        print(f"    Target size: {target_N} sites ({end_row-start_row}×{end_col-start_col})")
                        print(f"    Atoms in target region: {atoms_in_target} ({atoms_in_target/target_N:.1%} filled)")
                else:
                    print("    Warning: Target region not defined!")
                    target_N = 0
                    current_fill_rate = 0
                
                # Add move batches from this cycle
                cycle_batches = len(simulator.movement_history)
                total_batches += cycle_batches
                
                cycles += 1
                
                print(f"    Trial {trial+1}/{trials}, Cycle {cycles}: Fill rate = {current_fill_rate:.1%}, " +
                      f"Target N = {target_N}, Batches = {cycle_batches}")
                
                # Add diagnostic prints for why algorithm is stopping
                if cycles >= max_cycles - 1:
                    print(f"    Reached maximum cycles ({max_cycles}) with {current_fill_rate:.1%} fill rate")
                    if current_fill_rate < target_fill_rate and current_fill_rate <= occupation_prob + 0.01:
                        print(f"    Note: Achieved fill rate {current_fill_rate:.1%} is close to initial occupation {occupation_prob:.1%}")
                        print(f"    This is expected when target region and initial lattice are similar in size")
                
                if current_fill_rate >= target_fill_rate:
                    print(f"    Reached target fill rate ({target_fill_rate:.1%}) after {cycles} cycles with {total_batches} total batches")
                    break
            
            calculation_time = time.time() - calculation_start_time
            
            # Record metrics for this trial
            size_achieved_N.append(target_N)
            size_total_batches.append(total_batches)
            # Read parallel planning batch counts if available
            init_b = getattr(simulator, 'last_planned_initial_batches', None)
            red_b = getattr(simulator, 'last_planned_reduced_batches', None)
            if init_b is None:
                # If not available, set to the observed total_batches as fallback
                init_b = total_batches
            if red_b is None:
                red_b = total_batches
            size_initial_batches.append(init_b)
            size_reduced_batches.append(red_b)
            size_cycles.append(cycles)
            size_calc_time.append(calculation_time * 1000)  # Convert to ms
            size_fill_rate.append(current_fill_rate)

        # End of trials loop for this L — now record metrics for this initial size
        scaling_data['initial_size'].append(L)
        scaling_data['achieved_N'].append(np.mean(size_achieved_N))
        scaling_data['achieved_N_std'].append(np.std(size_achieved_N))
        scaling_data['move_batches_total'].append(np.mean(size_total_batches))
        scaling_data['move_batches_total_std'].append(np.std(size_total_batches))
        scaling_data['initial_batches'].append(np.mean(size_initial_batches))
        scaling_data['initial_batches_std'].append(np.std(size_initial_batches))
        scaling_data['reduced_batches'].append(np.mean(size_reduced_batches))
        scaling_data['reduced_batches_std'].append(np.std(size_reduced_batches))
        scaling_data['repair_cycles'].append(np.mean(size_cycles))
        scaling_data['repair_cycles_std'].append(np.std(size_cycles))
        scaling_data['calculation_time'].append(np.mean(size_calc_time))
        scaling_data['calculation_time_std'].append(np.std(size_calc_time))
        scaling_data['final_fill_rate'].append(np.mean(size_fill_rate))
        scaling_data['final_fill_rate_std'].append(np.std(size_fill_rate))

        # Print summary for this size
        print(f"  Complete {trials} trials for initial size L = {L}")
        print(f"  Avg achieved target N: {np.mean(size_achieved_N):.1f} ± {np.std(size_achieved_N):.1f}")
        print(f"  Avg total move batches: {np.mean(size_total_batches):.1f} ± {np.std(size_total_batches):.1f}")
        print(f"  Avg repair cycles: {np.mean(size_cycles):.1f} ± {np.std(size_cycles):.1f}")
        print(f"  Avg calculation time: {np.mean(size_calc_time):.2f} ± {np.std(size_calc_time):.2f} ms")
        print(f"  Avg final fill rate: {np.mean(size_fill_rate)*100:.2f}% ± {np.std(size_fill_rate)*100:.2f}%")

    # Fit data to power law model for complete algorithm scaling (done after all sizes processed)
    N_values = np.array(scaling_data['achieved_N'])  # Use achieved N instead of target size
    total_batches = np.array(scaling_data['move_batches_total'])

    # Fit log-log power law; be robust to degenerate cases (too few points / singular matrix)
    if len(N_values) < 2 or len(total_batches) < 2:
        total_batches_exponent = float('nan')
        total_batches_exponent_err = float('nan')
        total_batches_prefactor = float('nan')
    else:
        log_N = np.log(N_values)
        log_total_batches = np.log(total_batches)
        try:
            total_batches_params, total_batches_cov = np.polyfit(log_N, log_total_batches, 1, cov=True)
            total_batches_exponent = total_batches_params[0]
            total_batches_exponent_err = np.sqrt(total_batches_cov[0, 0])
            total_batches_prefactor = np.exp(total_batches_params[1])
        except np.linalg.LinAlgError:
            total_batches_exponent = float('nan')
            total_batches_exponent_err = float('nan')
            total_batches_prefactor = float('nan')

    # Create complete result dictionary
    result = {
        'data': scaling_data,
        'scaling': {
            'batches_exponent': total_batches_exponent,
            'batches_exponent_err': total_batches_exponent_err,
            'batches_prefactor': total_batches_prefactor,
            'lsap_exponent': 1.16,  # Literature value
            'psca_exponent': 0.48,  # Literature value
            'lsap_exponent_err': 0.01, # Literature value uncertainty
            'psca_exponent_err': 0.02  # Literature value uncertainty
        },
        'config': {
            'occupation_prob': occupation_prob,
            'loss_prob': loss_prob,
            'trials': trials,
            'strategy': strategy,
            'max_cycles': max_cycles
        }
    }

    return result

def visualize_results(results, output_dir):
    """
    Create visualizations of the complete algorithm scaling analysis.
    
    Args:
        results: Dictionary with scaling analysis results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    data = results['data']
    scaling = results['scaling']
    config = results['config']
    
    # Extract arrays from data - use achieved_N instead of initial_size for x-axis
    initial_sizes = np.array(data['initial_size'])  # Initial sizes for some plots
    N_values = np.array(data['achieved_N'])  # Actual achieved target region sizes
    N_std = np.array(data['achieved_N_std'])  # Standard deviation of achieved sizes
    total_batches = np.array(data['move_batches_total'])
    total_batches_std = np.array(data['move_batches_total_std'])
    
    # Generate smooth curves for power law fits
    N_smooth = np.logspace(np.log10(min(N_values)), np.log10(max(N_values)), 100)
    batches_fit = scaling['batches_prefactor'] * np.power(N_smooth, scaling['batches_exponent'])
    
    # Generate literature comparison curves
    # Scale to match at approximately N=100 for better visual comparison
    ref_N = 100
    ref_idx = np.argmin(np.abs(N_values - ref_N))
    ref_batch_value = total_batches[ref_idx]
    
    ref_N_idx = np.argmin(np.abs(N_smooth - ref_N))
    lsap_prefactor = ref_batch_value / (ref_N ** scaling['lsap_exponent'])
    psca_prefactor = ref_batch_value / (ref_N ** scaling['psca_exponent'])
    
    lsap_curve = lsap_prefactor * np.power(N_smooth, scaling['lsap_exponent'])
    psca_curve = psca_prefactor * np.power(N_smooth, scaling['psca_exponent'])
    
    # 1. Move Batches Scaling (log-log plot)
    plt.figure(figsize=(12, 8))
    
    # Plot data points with error bars
    plt.errorbar(N_values, total_batches, yerr=total_batches_std, fmt='o', markersize=8, capsize=5,
               label='Our Algorithm (Complete)')
    
    # Plot power law fit
    plt.loglog(N_smooth, batches_fit, '-', linewidth=2, 
             label=f'Fit: {scaling["batches_prefactor"]:.2f} × N^{scaling["batches_exponent"]:.3f}±{scaling["batches_exponent_err"]:.3f}')
    
    # Plot literature comparison curves
    plt.loglog(N_smooth, lsap_curve, '--', linewidth=2, 
             label=f'Modified LSAP: ∝ N^{scaling["lsap_exponent"]:.2f}')
    plt.loglog(N_smooth, psca_curve, '-.', linewidth=2, 
             label=f'PSCA: ∝ N^{scaling["psca_exponent"]:.2f}')
    
    plt.xlabel('Achieved Target Region Size (N)', fontsize=14)
    plt.ylabel('Total Number of Move Batches', fontsize=14)
    plt.title(f'Complete Algorithm Move Batch Scaling: {config["strategy"].capitalize()} Strategy\n' +
             f'Occupation: {config["occupation_prob"]:.0%}, {config["trials"]} Trials', 
             fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # Add annotation comparing scaling exponents
    comparison_text = f'Our Algorithm: N^{scaling["batches_exponent"]:.3f}±{scaling["batches_exponent_err"]:.3f}\n'
    
    if abs(scaling["batches_exponent"] - scaling["psca_exponent"]) < 0.1:
        comparison_text += f'≈ PSCA: N^{scaling["psca_exponent"]:.2f}\n'
    elif scaling["batches_exponent"] < scaling["psca_exponent"]:
        comparison_text += f'Better than PSCA: N^{scaling["psca_exponent"]:.2f}\n'
    else:
        comparison_text += f'vs. PSCA: N^{scaling["psca_exponent"]:.2f}\n'
        
    if abs(scaling["batches_exponent"] - scaling["lsap_exponent"]) < 0.1:
        comparison_text += f'≈ Modified LSAP: N^{scaling["lsap_exponent"]:.2f}'
    elif scaling["batches_exponent"] < scaling["lsap_exponent"]:
        comparison_text += f'Better than Modified LSAP: N^{scaling["lsap_exponent"]:.2f}'
    else:
        comparison_text += f'vs. Modified LSAP: N^{scaling["lsap_exponent"]:.2f}'
    
    plt.annotate(
        comparison_text,
        xy=(0.02, 0.02), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_algorithm_move_batches.png'), dpi=300)
    plt.close()
    
    # 2. Repair Cycles vs Target Size
    plt.figure(figsize=(12, 6))
    plt.errorbar(N_values, np.array(data['repair_cycles']), 
                yerr=np.array(data['repair_cycles_std']),
                fmt='o-', markersize=8, linewidth=2, capsize=5, color='blue')
    
    plt.xlabel('Achieved Target Region Size (N)', fontsize=14)
    plt.ylabel('Number of Repair Cycles', fontsize=14)
    plt.title(f'Complete Algorithm Repair Cycles: {config["strategy"].capitalize()} Strategy\n' +
             f'Occupation: {config["occupation_prob"]:.0%}, {config["trials"]} Trials', 
             fontsize=16)
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_algorithm_repair_cycles.png'), dpi=300)
    plt.close()
    
    # 3. Final Fill Rate vs Target Size
    plt.figure(figsize=(12, 6))
    plt.errorbar(N_values, np.array(data['final_fill_rate']) * 100, 
                yerr=np.array(data['final_fill_rate_std']) * 100,
                fmt='o-', markersize=8, linewidth=2, capsize=5, color='green')
    
    plt.xlabel('Achieved Target Region Size (N)', fontsize=14)
    plt.ylabel('Final Fill Rate (%)', fontsize=14)
    plt.title(f'Complete Algorithm Final Fill Rate: {config["strategy"].capitalize()} Strategy\n' +
             f'Occupation: {config["occupation_prob"]:.0%}, {config["trials"]} Trials', 
             fontsize=16)
    
    # Add reference line showing initial occupation probability
    plt.axhline(y=config['occupation_prob']*100, color='gray', linestyle='--', 
               label=f'Initial Occupation: {config["occupation_prob"]*100:.0f}%')
    
    # Add explanatory note
    if np.mean(data['final_fill_rate']) < 0.99:
        plt.annotate(
            "Note: Final fill rate limited by initial atom availability.\n" +
            f"With {config['occupation_prob']*100:.0f}% loading, 100% fill is only\n" +
            "possible when target region is smaller than initial lattice.",
            xy=(0.02, 0.02), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10
        )
    
    plt.grid(True, alpha=0.2)
    plt.ylim(min(min(data['final_fill_rate'])*100 - 5, 90), 101)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_algorithm_fill_rate.png'), dpi=300)
    plt.close()
    
    # 4. Calculation Time vs Target Size
    plt.figure(figsize=(12, 6))
    plt.errorbar(N_values, data['calculation_time'], 
                yerr=data['calculation_time_std'],
                fmt='o-', markersize=8, linewidth=2, capsize=5, color='red')
    
    plt.xlabel('Achieved Target Region Size (N)', fontsize=14)
    plt.ylabel('Calculation Time (ms)', fontsize=14)
    plt.title(f'Complete Algorithm Calculation Time: {config["strategy"].capitalize()} Strategy\n' +
             f'Occupation: {config["occupation_prob"]:.0%}, {config["trials"]} Trials', 
             fontsize=16)
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_algorithm_calculation_time.png'), dpi=300)
    plt.close()
    
    # 5. Combined Plot (all metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top left: Total move batches scaling (log-log)
    axes[0,0].errorbar(N_values, total_batches, yerr=total_batches_std, fmt='o', markersize=8, capsize=5)
    axes[0,0].loglog(N_smooth, batches_fit, '-', linewidth=2, 
                   label=f'Fit: N^{scaling["batches_exponent"]:.3f}')
    axes[0,0].loglog(N_smooth, lsap_curve, '--', linewidth=2, 
                   label=f'LSAP: N^{scaling["lsap_exponent"]:.2f}')
    axes[0,0].loglog(N_smooth, psca_curve, '-.', linewidth=2, 
                   label=f'PSCA: N^{scaling["psca_exponent"]:.2f}')
    
    axes[0,0].set_xlabel('Achieved Target Region Size (N)', fontsize=12)
    axes[0,0].set_ylabel('Total Move Batches', fontsize=12)
    axes[0,0].set_title('Total Move Batch Scaling (log-log)', fontsize=14)
    axes[0,0].grid(True, which="both", ls="-", alpha=0.2)
    axes[0,0].legend(fontsize=10)
    
    # Top right: Repair cycles vs N
    axes[0,1].errorbar(N_values, data['repair_cycles'], yerr=data['repair_cycles_std'], 
                      fmt='o-', markersize=8, linewidth=2, capsize=5, color='blue')
    
    axes[0,1].set_xlabel('Achieved Target Region Size (N)', fontsize=12)
    axes[0,1].set_ylabel('Repair Cycles', fontsize=12)
    axes[0,1].set_title('Repair Cycles vs Array Size', fontsize=14)
    axes[0,1].grid(True, alpha=0.2)
    
    # Bottom left: Final fill rate
    axes[1,0].errorbar(N_values, np.array(data['final_fill_rate']) * 100, 
                      yerr=np.array(data['final_fill_rate_std']) * 100,
                      fmt='o-', markersize=8, linewidth=2, capsize=5, color='green')
    
    axes[1,0].set_xlabel('Achieved Target Region Size (N)', fontsize=12)
    axes[1,0].set_ylabel('Final Fill Rate (%)', fontsize=12)
    axes[1,0].set_title('Final Fill Rate', fontsize=14)
    axes[1,0].grid(True, alpha=0.2)
    axes[1,0].set_ylim(min(min(data['final_fill_rate'])*100 - 5, 90), 101)
    
    # Bottom right: Calculation time
    axes[1,1].errorbar(N_values, data['calculation_time'], yerr=data['calculation_time_std'],
                      fmt='o-', markersize=8, linewidth=2, capsize=5, color='red')
    
    axes[1,1].set_xlabel('Achieved Target Region Size (N)', fontsize=12)
    axes[1,1].set_ylabel('Calculation Time (ms)', fontsize=12)
    axes[1,1].set_title('Calculation Time', fontsize=14)
    axes[1,1].grid(True, alpha=0.2)
    
    plt.suptitle(f'Complete Algorithm Analysis: {config["strategy"].capitalize()} Strategy\n' +
                f'Occupation Probability: {config["occupation_prob"]:.0%}, Max Cycles: {config["max_cycles"]}', 
                fontsize=18)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, 'complete_algorithm_combined.png'), dpi=300)
    plt.close()
    
    # Update the scaling comparison table with more info
    scaling_table = [
        "Move Batch Scaling Comparison - Complete Algorithm Analysis",
        "=====================================================",
        "",
        f"Target Array Size Range: {min(N_values)} to {max(N_values)}",
        f"Occupation Probability: {config['occupation_prob']:.0%}",
        f"Number of Trials per Size: {config['trials']}",
        f"Strategy: {config['strategy'].capitalize()}",
        f"Max Repair Cycles: {config['max_cycles']}",
        f"Final fill rate: {np.mean(data['final_fill_rate'])*100:.1f}% (with {config['occupation_prob']*100:.0f}% initial occupation)",
        "",
        "Algorithm            | Scaling Exponent | Interpretation",
        "-------------------- | ---------------- | -----------------------------",
        f"Our Algorithm        | N^{scaling['batches_exponent']:.3f}±{scaling['batches_exponent_err']:.3f} | {'Better than Modified LSAP' if scaling['batches_exponent'] < scaling['lsap_exponent'] else 'Worse than Modified LSAP'}",
        f"Modified LSAP (Ref.) | N^{scaling['lsap_exponent']:.2f}±{scaling['lsap_exponent_err']:.2f} | Literature reference",
        f"PSCA (Ref.)          | N^{scaling['psca_exponent']:.2f}±{scaling['psca_exponent_err']:.2f} | Literature reference",
        "",
        "Performance Metrics:",
        f"- Average repair cycles: {np.mean(data['repair_cycles']):.1f}",
        f"- Final fill rate for largest array: {data['final_fill_rate'][-1]*100:.1f}%",
        f"- Calculation time for largest array: {data['calculation_time'][-1]:.1f} ms",
        f"- Initial planned batches (avg): {np.mean(data.get('initial_batches', [])):.1f}",
        f"- Reduced planned batches (avg): {np.mean(data.get('reduced_batches', [])):.1f}",
        "",
        "Conclusion:",
    ]
    if scaling['batches_exponent'] < scaling['psca_exponent']:
        scaling_table.append("Our complete algorithm outperforms both Modified LSAP and PSCA in move batch scaling.")
    elif scaling['batches_exponent'] < scaling['lsap_exponent']:
        scaling_table.append("Our complete algorithm outperforms Modified LSAP but not PSCA in move batch scaling.")
    else:
        scaling_table.append("Our complete algorithm has worse move batch scaling than both Modified LSAP and PSCA.")
    
    with open(os.path.join(output_dir, 'complete_algorithm_scaling_comparison.txt'), 'w') as f:
        f.write('\n'.join(scaling_table))
    
    # Save raw data for future analysis - include both initial and achieved sizes
    np.savez(
        os.path.join(output_dir, 'complete_algorithm_data.npz'),
        initial_sizes=initial_sizes,
        N_values=N_values,
        N_std=N_std,
        total_batches=total_batches,
        total_batches_std=total_batches_std,
    initial_batches=np.array(data.get('initial_batches')),
    initial_batches_std=np.array(data.get('initial_batches_std')),
    reduced_batches=np.array(data.get('reduced_batches')),
    reduced_batches_std=np.array(data.get('reduced_batches_std')),
        repair_cycles=np.array(data['repair_cycles']),
        repair_cycles_std=np.array(data['repair_cycles_std']),
        fill_rate=np.array(data['final_fill_rate']),
        fill_rate_std=np.array(data['final_fill_rate_std']),
        calculation_time=np.array(data['calculation_time']),
        calculation_time_std=np.array(data['calculation_time_std']),
        batches_exponent=scaling['batches_exponent'],
        batches_exponent_err=scaling['batches_exponent_err']
    )
    
    # Create a structured results file with all parameters
    with open(os.path.join(output_dir, 'complete_algorithm_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray, list)):
                return [clean_for_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            else:
                return obj
        
        json.dump(clean_for_json(results), f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Analyze complete algorithm move batch scaling')
    
    # Default initial sizes from 10 to 100 with step 10
    default_sizes = range(10, 100, 10)
    default_sizes_str = ','.join(str(x) for x in default_sizes)
    
    parser.add_argument('--initial-sizes', type=str, default=default_sizes_str,
                       help='Comma-separated list of initial lattice side lengths L')
    parser.add_argument('--occupation', type=float, default=0.75,
                       help='Atom occupation probability (default: 0.75 as in paper)')
    parser.add_argument('--loss', type=float, default=0.0,
                       help='Atom loss probability (default: 0.0)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per configuration')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='150_190_N_scaling_075_results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--strategy', type=str, default='center', choices=['center', 'corner'],
                       help='Which movement strategy to analyze')
    parser.add_argument('--max-cycles', type=int, default=6,
                       help='Maximum number of repair cycles to attempt')
    parser.add_argument('--target-fill', type=float, default=1.0,
                       help='Target fill rate to achieve (default: 1.0)')
    
    args = parser.parse_args()
    
    # Parse initial sizes
    initial_sizes = [int(x) for x in args.initial_sizes.split(',')]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nAnalyzing complete algorithm move batch scaling with occupation probability: {args.occupation:.0%}")
    print(f"Initial lattice sizes: {min(initial_sizes)} to {max(initial_sizes)}")
    print(f"Strategy: {args.strategy}")
    print(f"Trials per configuration: {args.trials}")
    print(f"Max repair cycles: {args.max_cycles}")
    print(f"Target fill rate: {args.target_fill:.0%}")
    
    # Calculate theoretical max fill
    max_fill_possible = args.occupation
    if max_fill_possible < args.target_fill:
        print(f"\nWarning: Target fill rate ({args.target_fill:.0%}) exceeds maximum possible fill ({max_fill_possible:.0%})")
        print(f"with current occupation probability. Algorithm will run to max cycles.")
    
    # Run analysis with initial sizes
    results = analyze_complete_algorithm(
        initial_sizes=initial_sizes,
        occupation_prob=args.occupation,
        loss_prob=args.loss,
        trials=args.trials,
        seed=args.seed,
        strategy=args.strategy,
        max_cycles=args.max_cycles,
        target_fill_rate=args.target_fill
    )
    
    # Visualize the results
    visualize_results(results, args.output)
    
    # Print key results
    scaling = results['scaling']
    print("\nScaling Analysis Results:")
    print(f"Total move batches scaling: N^{scaling['batches_exponent']:.3f}±{scaling['batches_exponent_err']:.3f}")
    print("\nComparison with Literature:")
    print(f"Our Algorithm: N^{scaling['batches_exponent']:.3f}")
    print(f"Modified LSAP: N^{scaling['lsap_exponent']:.2f}")
    print(f"PSCA: N^{scaling['psca_exponent']:.2f}")
    
    print(f"\nAnalysis complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()