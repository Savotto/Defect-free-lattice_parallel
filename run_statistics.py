import numpy as np
from lattice import LatticeSimulator
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import json
from datetime import datetime

def run_lattice_statistics(
    lattice_sizes: List[Tuple[int, int]] = [(6,6), (8,8), (10,10), (12,12)],
    filling_probs: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    runs_per_config: int = 10
) -> Dict:
    """
    Run multiple simulations with different lattice sizes and filling probabilities.
    
    Args:
        lattice_sizes: List of (rows, cols) tuples for different lattice sizes
        filling_probs: List of filling probabilities to test
        runs_per_config: Number of runs for each configuration
    
    Returns:
        Dictionary containing all statistics
    """
    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'lattice_sizes': lattice_sizes,
            'filling_probs': filling_probs,
            'runs_per_config': runs_per_config
        },
        'data': {}
    }
    
    total_runs = len(lattice_sizes) * len(filling_probs) * runs_per_config
    run_count = 0
    
    for size in lattice_sizes:
        size_key = f"{size[0]}x{size[1]}"
        results['data'][size_key] = {}
        
        for prob in filling_probs:
            prob_key = f"{prob:.1f}"
            current_results = {
                'retention_rates': [],
                'execution_times': [],
                'initial_atoms': [],
                'final_atoms': [],
                'target_sizes': []
            }
            
            print(f"\nTesting {size[0]}x{size[1]} lattice with {prob:.0%} filling probability")
            
            for run in range(runs_per_config):
                run_count += 1
                print(f"Progress: {run_count}/{total_runs} "
                      f"(Run {run + 1}/{runs_per_config} for current config)")
                
                # Create simulator and run
                simulator = LatticeSimulator(initial_size=size, occupation_prob=prob)
                initial_lattice = simulator.generate_initial_lattice()
                initial_atoms = np.sum(initial_lattice)
                
                # Run simulation without visualization
                target_lattice, retention_rate, execution_time = simulator.rearrange_atoms(
                    show_visualization=False
                )
                
                # Store results
                target_size = int(np.sqrt(np.sum(initial_lattice)))
                final_atoms = np.sum(target_lattice)
                
                current_results['retention_rates'].append(float(retention_rate))
                current_results['execution_times'].append(float(execution_time))
                current_results['initial_atoms'].append(int(initial_atoms))
                current_results['final_atoms'].append(int(final_atoms))
                current_results['target_sizes'].append(int(target_size))
            
            # Calculate statistics for this configuration
            results['data'][size_key][prob_key] = {
                'mean_retention_rate': float(np.mean(current_results['retention_rates'])),
                'std_retention_rate': float(np.std(current_results['retention_rates'])),
                'mean_execution_time': float(np.mean(current_results['execution_times'])),
                'std_execution_time': float(np.std(current_results['execution_times'])),
                'mean_initial_atoms': float(np.mean(current_results['initial_atoms'])),
                'mean_final_atoms': float(np.mean(current_results['final_atoms'])),
                'mean_target_size': float(np.mean(current_results['target_sizes'])),
                'raw_data': current_results
            }
    
    return results

def plot_statistics(results: Dict) -> None:
    """Plot the statistics from the simulation runs."""
    lattice_sizes = list(results['data'].keys())
    filling_probs = list(results['data'][lattice_sizes[0]].keys())
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Plot retention rates
    for size in lattice_sizes:
        retention_rates = [results['data'][size][prob]['mean_retention_rate'] 
                         for prob in filling_probs]
        retention_errors = [results['data'][size][prob]['std_retention_rate'] 
                          for prob in filling_probs]
        ax1.errorbar([float(p) for p in filling_probs], retention_rates, 
                    yerr=retention_errors, label=size, marker='o')
    
    ax1.set_xlabel('Filling Probability')
    ax1.set_ylabel('Retention Rate')
    ax1.set_title('Retention Rate vs Filling Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot execution times
    for size in lattice_sizes:
        exec_times = [results['data'][size][prob]['mean_execution_time'] 
                     for prob in filling_probs]
        exec_errors = [results['data'][size][prob]['std_execution_time'] 
                      for prob in filling_probs]
        ax2.errorbar([float(p) for p in filling_probs], exec_times, 
                    yerr=exec_errors, label=size, marker='o')
    
    ax2.set_xlabel('Filling Probability')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Execution Time vs Filling Probability')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot initial vs target size
    for size in lattice_sizes:
        initial_atoms = [results['data'][size][prob]['mean_initial_atoms'] 
                        for prob in filling_probs]
        target_sizes = [results['data'][size][prob]['mean_target_size'] ** 2
                       for prob in filling_probs]
        ax3.plot([float(p) for p in filling_probs], initial_atoms, 
                 label=f'{size} Initial', marker='o')
        ax3.plot([float(p) for p in filling_probs], target_sizes, 
                 label=f'{size} Target', marker='s', linestyle='--')
    
    ax3.set_xlabel('Filling Probability')
    ax3.set_ylabel('Number of Atoms')
    ax3.set_title('Initial vs Target Atom Count')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot defect-free size achieved
    for size in lattice_sizes:
        target_sizes = [results['data'][size][prob]['mean_target_size'] 
                       for prob in filling_probs]
        ax4.plot([float(p) for p in filling_probs], target_sizes, 
                 label=size, marker='o')
    
    ax4.set_xlabel('Filling Probability')
    ax4.set_ylabel('Side Length of Defect-Free Square')
    ax4.set_title('Achieved Defect-Free Lattice Size')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'lattice_statistics_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Run simulations
    results = run_lattice_statistics()
    
    # Plot and save results
    plot_statistics(results)