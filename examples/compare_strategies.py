#!/usr/bin/env python3
"""
Compare corner and center movement strategies for defect-free atom rearrangement.
This script allows testing both strategies with the same initial conditions
and comparing their performance metrics.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from defect_free import LatticeSimulator, LatticeVisualizer

def run_comparison(initial_size=(100, 100), 
                  occupation_prob=0.7, 
                  seed=42, 
                  atom_loss_probability=0.05,
                  show_visualization=True,
                  save_figures=False,
                  output_dir="comparison_results"):
    """
    Run a comparison between corner and center movement strategies.
    
    Args:
        initial_size: Tuple of (rows, cols) for initial lattice size
        occupation_prob: Probability of atom occupation (0.0 to 1.0)
        seed: Random seed for reproducible results
        atom_loss_probability: Probability of losing atoms during movement
        show_visualization: Whether to display visualizations
        save_figures: Whether to save figures to files
        output_dir: Directory to save output files (if save_figures is True)
    
    Returns:
        Dictionary with comparison results
    """
    # Set random seed for reproducible results
    np.random.seed(seed)
    
    # Dictionary to store comparison results
    results = {
        "center": {},
        "corner": {},
        "parameters": {
            "initial_size": initial_size,
            "occupation_prob": occupation_prob,
            "seed": seed,
            "atom_loss_probability": atom_loss_probability
        }
    }
    
    # Create output directory if saving figures
    if save_figures:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Create initial lattice that will be copied for both strategies
    print("\n" + "="*50)
    print(f"Initializing lattice with size {initial_size} and {occupation_prob:.1%} occupation")
    print(f"Atom loss probability: {atom_loss_probability:.1%}")
    
    temp_simulator = LatticeSimulator(
        initial_size=initial_size, 
        occupation_prob=occupation_prob,
        physical_constraints={"atom_loss_probability": atom_loss_probability}
    )
    initial_lattice = temp_simulator.generate_initial_lattice()
    total_atoms = np.sum(initial_lattice)
    print(f"Generated initial lattice with {total_atoms} atoms")
    
    # Estimate max defect-free size
    max_square_size = temp_simulator.calculate_max_defect_free_size()
    print(f"Estimated maximum defect-free size: {max_square_size}x{max_square_size}")
    
    # Import the time module separately in the function scope
    import time as time_module
    
    # Function to run a single strategy
    def run_strategy(strategy):
        print("\n" + "-"*50)
        print(f"Testing {strategy.upper()} strategy")
        
        # Initialize simulator with the same initial lattice
        simulator = LatticeSimulator(
            initial_size=initial_size, 
            occupation_prob=occupation_prob,
            physical_constraints={"atom_loss_probability": atom_loss_probability}
        )
        
        # Use the same initial lattice for fair comparison
        simulator.field = initial_lattice.copy()
        simulator.slm_lattice = initial_lattice.copy()
        simulator.total_atoms = total_atoms
        simulator.side_length = max_square_size
        
        # Initialize visualizer
        visualizer = LatticeVisualizer(simulator)
        simulator.visualizer = visualizer
        
        # Run the rearrangement
        start_time = time_module.time()
        if strategy == "center":
            final_lattice, fill_rate, execution_time = simulator.movement_manager.center_filling_strategy(
                show_visualization=False  # We'll show visualizations at the end
            )
        else:  # corner
            final_lattice, fill_rate, execution_time = simulator.movement_manager.corner_filling_strategy(
                show_visualization=False  # We'll show visualizations at the end
            )
        total_time = time_module.time() - start_time
        
        # Calculate physical time from movement history
        physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
        
        # Get target region
        target_region = simulator.movement_manager.target_region
        target_start_row, target_start_col, target_end_row, target_end_col = target_region
        
        # Count defects
        target_zone = simulator.field[target_start_row:target_end_row, 
                                      target_start_col:target_end_col]
        defects = np.sum(target_zone == 0)
        
        # Count movements
        total_movements = len(simulator.movement_history)
        total_atom_moves = sum(len(move.get('moves', [])) for move in simulator.movement_history)
        
        # Store results
        results[strategy] = {
            "fill_rate": fill_rate,
            "execution_time": execution_time,
            "total_time": total_time,
            "physical_time": physical_time,
            "defects": defects,
            "total_movements": total_movements,
            "total_atom_moves": total_atom_moves,
            "simulator": simulator,
            "final_lattice": final_lattice
        }
        
        # Print results
        print(f"\nResults for {strategy.upper()} strategy:")
        print(f"Fill rate: {fill_rate:.2%}")
        print(f"Remaining defects: {defects}")
        print(f"Algorithm execution time: {execution_time:.3f} seconds")
        print(f"Physical movement time: {physical_time:.6f} seconds")
        print(f"Total movements: {total_movements}")
        print(f"Total atom moves: {total_atom_moves}")
        
        # Create visualization if requested
        if show_visualization or save_figures:
            # Show final analysis figure
            fig_analysis = visualizer.show_final_analysis()
            if save_figures:
                fig_analysis.savefig(f"{output_dir}/{strategy}_final_analysis.png", dpi=300, bbox_inches='tight')
            
            # Show movement opportunities (helps diagnose why some defects couldn't be filled)
            if defects > 0:
                fig_opportunities = visualizer.visualize_movement_opportunities()
                if save_figures:
                    fig_opportunities.savefig(f"{output_dir}/{strategy}_movement_opportunities.png", dpi=300, bbox_inches='tight')
            
            # Don't display figures yet if we're comparing both strategies
            if not show_visualization:
                plt.close('all')
                
        return simulator
    
    # Run both strategies
    center_simulator = run_strategy("center")
    corner_simulator = run_strategy("corner")
    
    # Compare and print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("-"*50)
    
    # Compare fill rates
    center_fill = results["center"]["fill_rate"]
    corner_fill = results["corner"]["fill_rate"]
    print(f"Fill rates: Center = {center_fill:.2%}, Corner = {corner_fill:.2%}")
    if center_fill > corner_fill:
        fill_diff = center_fill - corner_fill
        print(f"Center strategy achieved {fill_diff:.2%} higher fill rate")
    elif corner_fill > center_fill:
        fill_diff = corner_fill - center_fill
        print(f"Corner strategy achieved {fill_diff:.2%} higher fill rate")
    else:
        print("Both strategies achieved the same fill rate")
    
    # Compare execution timesf
    center_time = results["center"]["execution_time"]
    corner_time = results["corner"]["execution_time"]
    print(f"Execution times: Center = {center_time:.3f}s, Corner = {corner_time:.3f}s")
    if center_time < corner_time:
        time_diff = corner_time - center_time
        time_pct = (time_diff / corner_time) * 100
        print(f"Center strategy was {time_diff:.3f}s ({time_pct:.1f}%) faster")
    elif corner_time < center_time:
        time_diff = center_time - corner_time
        time_pct = (time_diff / center_time) * 100
        print(f"Corner strategy was {time_diff:.3f}s ({time_pct:.1f}%) faster")
    else:
        print("Both strategies took the same execution time")
    
    # Compare physical times
    center_physical = results["center"]["physical_time"]
    corner_physical = results["corner"]["physical_time"]
    print(f"Physical times: Center = {center_physical:.6f}s, Corner = {corner_physical:.6f}s")
    if center_physical < corner_physical:
        physical_diff = corner_physical - center_physical
        physical_pct = (physical_diff / corner_physical) * 100
        print(f"Center strategy was {physical_diff:.6f}s ({physical_pct:.1f}%) more efficient in physical time")
    elif corner_physical < center_physical:
        physical_diff = center_physical - corner_physical
        physical_pct = (physical_diff / center_physical) * 100
        print(f"Corner strategy was {physical_diff:.6f}s ({physical_pct:.1f}%) more efficient in physical time")
    else:
        print("Both strategies had same physical movement time")
    
    # Compare movement complexity
    center_moves = results["center"]["total_atom_moves"]
    corner_moves = results["corner"]["total_atom_moves"]
    print(f"Atom moves: Center = {center_moves}, Corner = {corner_moves}")
    if center_moves < corner_moves:
        moves_diff = corner_moves - center_moves
        moves_pct = (moves_diff / corner_moves) * 100
        print(f"Center strategy required {moves_diff} ({moves_pct:.1f}%) fewer atom movements")
    elif corner_moves < center_moves:
        moves_diff = center_moves - corner_moves
        moves_pct = (moves_diff / center_moves) * 100
        print(f"Corner strategy required {moves_diff} ({moves_pct:.1f}%) fewer atom movements")
    else:
        print("Both strategies required the same number of atom movements")
    
    # Create direct comparison visualization if requested
    if show_visualization or save_figures:
        # Create a figure comparing both strategies
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Get target regions
        center_target = center_simulator.movement_manager.target_region
        corner_target = corner_simulator.movement_manager.target_region
        
        # Plot initial lattice and target regions
        center_visualizer = center_simulator.visualizer
        corner_visualizer = corner_simulator.visualizer
        
        center_visualizer.plot_lattice(
            initial_lattice,
            title="Initial Lattice with Center Target",
            highlight_region=center_target,
            ax=axes[0, 0]
        )
        
        corner_visualizer.plot_lattice(
            initial_lattice,
            title="Initial Lattice with Corner Target",
            highlight_region=corner_target,
            ax=axes[0, 1]
        )
        
        # Plot final lattices
        center_final = results["center"]["final_lattice"]
        corner_final = results["corner"]["final_lattice"]
        
        center_visualizer.plot_lattice(
            center_final,
            title=f"Center Strategy Result (Fill: {center_fill:.1%})",
            highlight_region=center_target,
            ax=axes[1, 0]
        )
        
        corner_visualizer.plot_lattice(
            corner_final,
            title=f"Corner Strategy Result (Fill: {corner_fill:.1%})",
            highlight_region=corner_target,
            ax=axes[1, 1]
        )
        
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
        
        # Display all figures if requested
        if show_visualization:
            plt.show()
    
    # Run the detailed movement efficiency analysis
    efficiency_analysis = analyze_movement_efficiency(results)
    results["efficiency_analysis"] = efficiency_analysis
    
    # Add a visualization of phase-by-phase movement progress
    if show_visualization or save_figures:
        # Create phase-by-phase movement analysis plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract movement phase data
        center_phases = []
        center_times = []
        center_atoms = []
        running_time = 0
        running_atoms = 0
        
        for move in center_simulator.movement_history:
            move_type = move.get('type', 'unknown')
            atoms_moved = len(move.get('moves', []))
            time_taken = move.get('time', 0)
            
            running_time += time_taken
            running_atoms += atoms_moved
            
            center_phases.append(move_type)
            center_times.append(running_time * 1000)  # Convert to ms
            center_atoms.append(running_atoms)
        
        corner_phases = []
        corner_times = []
        corner_atoms = []
        running_time = 0
        running_atoms = 0
        
        for move in corner_simulator.movement_history:
            move_type = move.get('type', 'unknown')
            atoms_moved = len(move.get('moves', []))
            time_taken = move.get('time', 0)
            
            running_time += time_taken
            running_atoms += atoms_moved
            
            corner_phases.append(move_type)
            corner_times.append(running_time * 1000)  # Convert to ms
            corner_atoms.append(running_atoms)
        
        # Plot cumulative movement time
        axes[0].plot(center_atoms, center_times, 'o-', label='Center Strategy')
        axes[0].plot(corner_atoms, corner_times, 's-', label='Corner Strategy')
        axes[0].set_xlabel('Cumulative Atoms Moved')
        axes[0].set_ylabel('Cumulative Movement Time (ms)')
        axes[0].set_title('Movement Efficiency Comparison')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot phase durations
        if center_simulator.movement_history and corner_simulator.movement_history:
            # Get unique movement types
            all_types = set()
            for move in center_simulator.movement_history + corner_simulator.movement_history:
                all_types.add(move.get('type', 'unknown'))
            
            # Calculate time spent in each phase
            center_phase_times = {phase: 0 for phase in all_types}
            corner_phase_times = {phase: 0 for phase in all_types}
            
            for move in center_simulator.movement_history:
                move_type = move.get('type', 'unknown')
                center_phase_times[move_type] += move.get('time', 0) * 1000  # Convert to ms
            
            for move in corner_simulator.movement_history:
                move_type = move.get('type', 'unknown')
                corner_phase_times[move_type] += move.get('time', 0) * 1000  # Convert to ms
            
            # Plot stacked bar chart
            center_phases = []
            center_times = []
            corner_phases = []
            corner_times = []
            
            for phase, time in center_phase_times.items():
                if time > 0:
                    center_phases.append(phase)
                    center_times.append(time)
            
            for phase, time in corner_phase_times.items():
                if time > 0:
                    corner_phases.append(phase)
                    corner_times.append(time)
            
            # Create indexes for bars
            x = np.arange(2)
            width = 0.35
            
            # Plot total times as reference
            axes[1].bar(x, [sum(center_times), sum(corner_times)], width, 
                     alpha=0.3, label='Total Time')
            
            # Plot phase breakdown for center strategy
            bottom = 0
            for i, (phase, time) in enumerate(zip(center_phases, center_times)):
                if i < 5:  # Limit to top 5 phases for readability
                    axes[1].bar(x[0], time, width, bottom=bottom, alpha=0.7, 
                             label=f'Center: {phase}' if i == 0 else None)
                    bottom += time
            
            # Plot phase breakdown for corner strategy
            bottom = 0
            for i, (phase, time) in enumerate(zip(corner_phases, corner_times)):
                if i < 5:  # Limit to top 5 phases for readability
                    axes[1].bar(x[1], time, width, bottom=bottom, alpha=0.7,
                             label=f'Corner: {phase}' if i == 0 else None)
                    bottom += time
            
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(['Center', 'Corner'])
            axes[1].set_ylabel('Movement Time (ms)')
            axes[1].set_title('Movement Phase Times')
            axes[1].grid(True, axis='y')
        
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"{output_dir}/movement_efficiency_analysis.png", dpi=300, bbox_inches='tight')
    
    return results

def analyze_movement_efficiency(results):
    """
    Analyze the movement efficiency of both strategies in detail.
    
    Args:
        results: Dictionary with results from both strategies
        
    Returns:
        Dictionary with detailed analysis metrics
    """
    print("\n" + "="*50)
    print("DETAILED MOVEMENT EFFICIENCY ANALYSIS")
    print("="*50)
    
    # Extract key data
    center_data = results["center"]
    corner_data = results["corner"]
    center_moves = center_data["simulator"].movement_history
    corner_moves = corner_data["simulator"].movement_history
    
    # Initialize analysis dict
    analysis = {
        "center": {},
        "corner": {}
    }
    
    # Analyze movement phases
    for strategy, moves in [("center", center_moves), ("corner", corner_moves)]:
        # Count moves by type
        move_types = {}
        for move in moves:
            move_type = move.get('type', 'unknown')
            if move_type not in move_types:
                move_types[move_type] = {
                    'count': 0,
                    'atoms_moved': 0,
                    'total_time': 0,
                    'successful': 0,
                    'failed': 0
                }
            
            move_types[move_type]['count'] += 1
            move_types[move_type]['atoms_moved'] += len(move.get('moves', []))
            move_types[move_type]['total_time'] += move.get('time', 0)
            move_types[move_type]['successful'] += move.get('successful', 0)
            move_types[move_type]['failed'] += move.get('failed', 0)
        
        # Analyze movement distances
        total_distance = 0
        max_distance = 0
        move_counts = []
        for move in moves:
            for atom_move in move.get('moves', []):
                if isinstance(atom_move, dict) and 'from' in atom_move and 'to' in atom_move:
                    from_pos = atom_move['from']
                    to_pos = atom_move['to']
                    # Manhattan distance
                    distance = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
                    total_distance += distance
                    max_distance = max(max_distance, distance)
                    move_counts.append(distance)
        
        # Calculate average distance
        avg_distance = total_distance / sum(len(move.get('moves', [])) for move in moves) if moves else 0
        
        # Store analysis
        analysis[strategy] = {
            'move_types': move_types,
            'total_distance': total_distance,
            'max_distance': max_distance,
            'avg_distance': avg_distance,
            'move_counts': move_counts
        }
    
    # Print detailed analysis
    print("\nMovement Type Analysis:")
    print("\nCenter Strategy:")
    for move_type, stats in sorted(analysis["center"]['move_types'].items()):
        print(f"  {move_type}: {stats['count']} operations, {stats['atoms_moved']} atoms moved, {stats['total_time']*1000:.2f} ms")
    
    print("\nCorner Strategy:")
    for move_type, stats in sorted(analysis["corner"]['move_types'].items()):
        print(f"  {move_type}: {stats['count']} operations, {stats['atoms_moved']} atoms moved, {stats['total_time']*1000:.2f} ms")
    
    print("\nMovement Efficiency Metrics:")
    print(f"Center - Average move distance: {analysis['center']['avg_distance']:.2f} units")
    print(f"Corner - Average move distance: {analysis['corner']['avg_distance']:.2f} units")
    print(f"Center - Maximum move distance: {analysis['center']['max_distance']} units")
    print(f"Corner - Maximum move distance: {analysis['corner']['max_distance']} units")
    print(f"Center - Total movement distance: {analysis['center']['total_distance']} units")
    print(f"Corner - Total movement distance: {analysis['corner']['total_distance']} units")
    
    # Additional analysis on target placement
    center_simulator = results["center"]["simulator"]
    corner_simulator = results["corner"]["simulator"]
    
    center_target = center_simulator.movement_manager.target_region
    corner_target = corner_simulator.movement_manager.target_region
    
    center_start_row, center_start_col, center_end_row, center_end_col = center_target
    corner_start_row, corner_start_col, corner_end_row, corner_end_col = corner_target
    
    # Analyze initial atom distribution within target region
    initial_lattice = center_simulator.slm_lattice  # Both use the same initial lattice
    
    center_target_initial = initial_lattice[center_start_row:center_end_row, center_start_col:center_end_col]
    corner_target_initial = initial_lattice[corner_start_row:corner_end_row, corner_start_col:corner_end_col]
    
    center_initial_fill = np.sum(center_target_initial) / center_target_initial.size
    corner_initial_fill = np.sum(corner_target_initial) / corner_target_initial.size
    
    print("\nTarget Placement Analysis:")
    print(f"Center target location: ({center_start_row}, {center_start_col}) to ({center_end_row}, {center_end_col})")
    print(f"Corner target location: ({corner_start_row}, {corner_start_col}) to ({corner_end_row}, {corner_end_col})")
    print(f"Center initial target fill rate: {center_initial_fill:.2%}")
    print(f"Corner initial target fill rate: {corner_initial_fill:.2%}")
    
    # Compare initial distribution advantages
    center_advantage = center_initial_fill
    corner_advantage = corner_initial_fill
    
    if center_advantage > corner_advantage:
        advantage_diff = center_advantage - corner_advantage
        print(f"\nCenter target had {advantage_diff:.2%} better initial filling - this gives it a head start.")
    elif corner_advantage > center_advantage:
        advantage_diff = corner_advantage - center_advantage
        print(f"\nCorner target had {advantage_diff:.2%} better initial filling - this gives it a head start.")
    
    # Plot movement distance distribution
    if len(analysis['center']['move_counts']) > 0 and len(analysis['corner']['move_counts']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.arange(0, max(max(analysis['center']['move_counts']), 
                              max(analysis['corner']['move_counts'])) + 2)
        
        ax.hist(analysis['center']['move_counts'], bins=bins, alpha=0.5, label='Center Strategy', 
                density=True)
        ax.hist(analysis['corner']['move_counts'], bins=bins, alpha=0.5, label='Corner Strategy', 
                density=True)
        
        ax.set_xlabel('Movement Distance (Manhattan)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Atom Movement Distances')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        print("\nCreated movement distance distribution plot.")
    
    return analysis


def main():
    """Main function to run the comparison with default parameters."""
    run_comparison(
        initial_size=(100, 100),
        occupation_prob=0.7,
        seed=42,
        atom_loss_probability=0.1,
        show_visualization=True,
        save_figures=False,
        output_dir="comparison_results"
    )


if __name__ == "__main__":
    main()