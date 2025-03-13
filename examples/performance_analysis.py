"""
Performance analysis script for the defect-free lattice rearrangement algorithm.
Tests different lattice sizes and filling probabilities with and without atom loss.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from defect_free.simulator import LatticeSimulator

def run_performance_test(size, occupation_prob, atom_loss_prob):
    """Run a single performance test with given parameters."""
    # Configure simulator with specific parameters
    simulator = LatticeSimulator(
        initial_size=(size, size),
        occupation_prob=occupation_prob,
        physical_constraints={'atom_loss_probability': atom_loss_prob}
    )
    
    # Generate initial lattice
    simulator.generate_initial_lattice()
    
    # Time the rearrangement
    start_time = time.time()
    result_lattice, retention_rate, _ = simulator.rearrange_for_defect_free(show_visualization=False)
    execution_time = time.time() - start_time
    
    # Calculate metrics
    initial_atoms = np.sum(simulator.slm_lattice)
    final_atoms = np.sum(result_lattice)
    defect_free_size = simulator.side_length
    
    return {
        'size': size,
        'occupation_prob': occupation_prob,
        'atom_loss_prob': atom_loss_prob,
        'initial_atoms': initial_atoms,
        'final_atoms': final_atoms,
        'defect_free_size': defect_free_size,
        'retention_rate': retention_rate,
        'execution_time': execution_time
    }

def main():
    # Test parameters
    sizes = np.arange(10, 101, 10)  # 10x10 to 100x100 with step 10
    occupation_probs = np.arange(0.7, 1.01, 0.1)  # 0.7 to 1.0 with step 0.1
    atom_loss_probs = [0.0, 0.05]  # Test with and without atom loss
    
    results = []
    total_tests = len(sizes) * len(occupation_probs) * len(atom_loss_probs)
    test_count = 0
    
    print(f"Running {total_tests} test configurations...")
    
    for size in sizes:
        for prob in occupation_probs:
            for loss_prob in atom_loss_probs:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}")
                print(f"Size: {size}x{size}, Occupation: {prob:.1f}, Loss Prob: {loss_prob}")
                
                try:
                    result = run_performance_test(size, prob, loss_prob)
                    results.append(result)
                    print(f"Completed - Defect-free size: {result['defect_free_size']}x{result['defect_free_size']}")
                except Exception as e:
                    print(f"Error in test: {e}")
                    continue
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f'performance_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Defect-free size vs Initial size for different occupation probabilities
    plt.subplot(2, 2, 1)
    for prob in occupation_probs:
        for loss_prob in atom_loss_probs:
            data = df[(df['occupation_prob'] == prob) & (df['atom_loss_prob'] == loss_prob)]
            label = f'p={prob:.1f}, loss={loss_prob}'
            plt.plot(data['size'], data['defect_free_size'], 'o-', label=label)
    
    plt.xlabel('Initial Lattice Size')
    plt.ylabel('Defect-free Region Size')
    plt.title('Defect-free Size vs Initial Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Execution time vs Initial size
    plt.subplot(2, 2, 2)
    for prob in occupation_probs:
        for loss_prob in atom_loss_probs:
            data = df[(df['occupation_prob'] == prob) & (df['atom_loss_prob'] == loss_prob)]
            label = f'p={prob:.1f}, loss={loss_prob}'
            plt.plot(data['size'], data['execution_time'], 'o-', label=label)
    
    plt.xlabel('Initial Lattice Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Initial Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Retention rate vs Initial size
    plt.subplot(2, 2, 3)
    for prob in occupation_probs:
        for loss_prob in atom_loss_probs:
            data = df[(df['occupation_prob'] == prob) & (df['atom_loss_prob'] == loss_prob)]
            label = f'p={prob:.1f}, loss={loss_prob}'
            plt.plot(data['size'], data['retention_rate'], 'o-', label=label)
    
    plt.xlabel('Initial Lattice Size')
    plt.ylabel('Retention Rate')
    plt.title('Retention Rate vs Initial Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_filename = f'performance_plots_{timestamp}.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plots saved to {plot_filename}")

    # Create a second figure for direct comparisons
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    # Plot execution time vs number of initial atoms
    plt.scatter(df['initial_atoms'], df['execution_time'], c=df['atom_loss_prob'], cmap='viridis')
    plt.colorbar(label='Atom Loss Probability')
    plt.xlabel('Initial Atom Count')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Initial Atom Count')

    plt.subplot(1, 2, 2)
    # Compare execution with/without atom loss
    no_loss = df[df['atom_loss_prob'] == 0.0]
    with_loss = df[df['atom_loss_prob'] > 0.0]
    plt.scatter(no_loss['size'], no_loss['execution_time'], label='No Loss', color='blue')
    plt.scatter(with_loss['size'], with_loss['execution_time'], label='With Loss', color='red')
    plt.xlabel('Initial Lattice Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Impact of Atom Loss on Performance')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'performance_comparison_{timestamp}.png', bbox_inches='tight')

if __name__ == "__main__":
    main()