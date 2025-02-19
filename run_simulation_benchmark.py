import numpy as np
from lattice import LatticeSimulator
import time
from typing import List, Tuple
import matplotlib.pyplot as plt

def run_benchmark_simulation(num_runs: int = 10) -> List[Tuple[float, float]]:
    """
    Run multiple simulations without visualization and collect performance metrics.
    
    Args:
        num_runs: Number of simulation runs to perform
        
    Returns:
        List of (retention_rate, execution_time) tuples for each run
    """
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Create a simulator with a 10x10 initial lattice and 70% occupation probability
        simulator = LatticeSimulator(initial_size=(10, 10), occupation_prob=0.7)
        
        # Generate initial lattice
        initial_lattice = simulator.generate_initial_lattice()
        total_atoms = np.sum(initial_lattice)
        
        # Rearrange atoms without visualization
        target_lattice, retention_rate, execution_time = simulator.rearrange_atoms(show_visualization=False)
        
        results.append((retention_rate, execution_time))
        print(f"Run {run + 1} complete - Retention: {retention_rate:.2%}, Time: {execution_time:.3f}s")
    
    return results

def plot_benchmark_results(results: List[Tuple[float, float]]) -> None:
    """Plot the benchmark results."""
    retention_rates, execution_times = zip(*results)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot retention rate distribution
    ax1.hist(retention_rates, bins=10, edgecolor='black')
    ax1.set_title('Retention Rate Distribution')
    ax1.set_xlabel('Retention Rate')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Plot execution time distribution
    ax2.hist(execution_times, bins=10, edgecolor='black')
    ax2.set_title('Execution Time Distribution')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"Average retention rate: {np.mean(retention_rates):.2%} ± {np.std(retention_rates):.2%}")
    print(f"Average execution time: {np.mean(execution_times):.3f}s ± {np.std(execution_times):.3f}s")

if __name__ == "__main__":
    # Run benchmark simulations
    results = run_benchmark_simulation(num_runs=10)
    
    # Plot and analyze results
    plot_benchmark_results(results)