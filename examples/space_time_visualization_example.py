#!/usr/bin/env python3
"""
Example script demonstrating the space-time visualization feature.

This script:
1. Initializes a lattice with random atom positions
2. Performs atom rearrangement using either center or corner strategy
3. Creates a 3D space-time visualization of the atom movements
4. Opens the visualization in a web browser (if available)
"""
import time
import numpy as np
import os
import webbrowser
import sys
import shutil

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defect_free import LatticeSimulator, LatticeVisualizer

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Space-time visualization example')
    parser.add_argument('--size', type=int, default=50, help='Size of the lattice (default: 50)')
    parser.add_argument('--occupation', type=float, default=0.5, help='Occupation probability (default: 0.5)')
    parser.add_argument('--loss', type=float, default=0.05, help='Atom loss probability (default: 0.05)')
    parser.add_argument('--strategy', type=str, default='center', choices=['center', 'corner'],
                        help='Rearrangement strategy (default: center)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='space_time_output', 
                        help='Output directory for visualization files')
    parser.add_argument('--open-browser', action='store_true', 
                        help='Automatically open the visualization in a web browser')
    args = parser.parse_args()
    # python examples/space_time_visualization_example.py --open-browser
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print(f"Initializing lattice simulator with:")
    print(f"- Lattice size: {args.size}x{args.size}")
    print(f"- Occupation probability: {args.occupation}")
    print(f"- Atom loss probability: {args.loss}")
    print(f"- Strategy: {args.strategy}")
    
    # Initialize simulator with the specified parameters
    simulator = LatticeSimulator(
        initial_size=(args.size, args.size),
        occupation_prob=args.occupation,
        physical_constraints={'atom_loss_probability': args.loss}
    )
    
    # Generate initial lattice
    simulator.generate_initial_lattice()
    print(f"Generated initial lattice with {np.sum(simulator.field)} atoms")
    
    # Initialize visualizer
    visualizer = LatticeVisualizer(simulator)
    simulator.visualizer = visualizer
    
    # Run the rearrangement
    print(f"\nStarting rearrangement using {args.strategy} strategy...")
    start_time = time.time()
    result, execution_time = simulator.rearrange_for_defect_free(
        strategy=args.strategy,
        show_visualization=False  # Don't show animation during execution
    )
    final_lattice, fill_rate, _ = result
    total_time = time.time() - start_time
    
    # Print results summary
    print(f"\nRearrangement completed in {execution_time:.3f} seconds")
    print(f"Created defect-free region of size {simulator.side_length}x{simulator.side_length}")
    print(f"Fill rate: {fill_rate:.2%}")
    print(f"Total movements: {len(simulator.movement_history)}")
    
    # Calculate physical time from movement history
    physical_time = sum(move.get('time', 0) for move in simulator.movement_history)
    print(f"Physical movement time: {physical_time:.6f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the space-time visualization
    print(f"\nGenerating 3D space-time visualization...")
    visualization_html = visualizer.create_space_time_visualization(output_dir=args.output_dir)
    
    if visualization_html and os.path.exists(visualization_html):
        print(f"\nVisualization successfully created at:")
        print(f"  {os.path.abspath(visualization_html)}")
        
        # Open in browser if requested
        if args.open_browser:
            visualization_url = 'file://' + os.path.abspath(visualization_html)
            print(f"\nOpening visualization in web browser...")
            webbrowser.open(visualization_url)
    else:
        print("Failed to create visualization.")
    
    # Additional analysis and visualization
    print("\nCreating standard visualizations...")
    
    # Show the final analysis plot
    analysis_fig = visualizer.show_final_analysis()
    
    # Save the analysis plot
    analysis_path = os.path.join(args.output_dir, 'final_analysis.png')
    analysis_fig.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"Final analysis saved to: {analysis_path}")
    
    # Copy the space_time_extractor.py file to the project directory if it's not already there
    extractor_path = os.path.join(os.path.dirname(__file__), '..', 'defect_free', 'space_time_extractor.py')
    if not os.path.exists(extractor_path):
        source_path = os.path.join(os.path.dirname(__file__), '..', 'space_time_extractor.py')
        if os.path.exists(source_path):
            # Copy to the defect_free directory
            target_dir = os.path.join(os.path.dirname(__file__), '..', 'defect_free')
            shutil.copy(source_path, os.path.join(target_dir, 'space_time_extractor.py'))
            print(f"Copied space_time_extractor.py to the defect_free package directory")
    
    print("\nAll visualizations completed successfully!")
    print(f"Results are available in: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()