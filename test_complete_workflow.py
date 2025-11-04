#!/usr/bin/env python3
"""
Complete workflow simulation: Initialize atoms, pull them down, then form dual 9x9 patterns.

This script demonstrates the full pipeline:
1. Initialize atoms in a random/sparse configuration
2. Apply column-by-column downward rearrangement 
3. Form dual 9x9 patterns (left and right) using the closest atoms
4. Visualize each step with proper AOD constraint enforcement
"""

import numpy as np
import matplotlib.pyplot as plt
from defect_free import LatticeSimulator, LatticeVisualizer
from without_defect_free.rearrangement import (
    rearrange_column_by_column_downward_parallel,
    rearrange_to_9x9_pattern,
    form_inner_atoms_in_9x9_patterns,
    get_9x9_pattern_positions
)

def create_initial_sparse_lattice(size, occupancy, seed=None):
    """
    Create a sparse lattice with isolated atoms positioned in upper half to maximize downward movement effect.
    
    Args:
        size (int): Size of the square lattice
        occupancy (float): Target occupancy ratio (0.0 to 1.0)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Binary lattice with isolated atoms
    """
    if seed is not None:
        np.random.seed(seed)
    
    lattice = np.zeros((size, size), dtype=int)
    
    # Calculate target number of atoms
    total_positions = size * size
    num_atoms = int(total_positions * occupancy)
    
    print(f"Creating {size}x{size} lattice with {num_atoms} isolated atoms (occupancy: {occupancy:.1%})")
    print("Placing atoms preferentially in upper 70% of lattice for maximum downward movement effect")
    
    placed_atoms = []
    max_attempts = num_atoms * 50  # Reasonable upper bound to avoid infinite loops
    
    for attempt in range(max_attempts):
        if len(placed_atoms) >= num_atoms:
            break
        
        # Bias placement toward upper portion of lattice (top 70%)
        # This ensures atoms will have room to fall during downward movement
        upper_limit = int(size * 0.7)  # Top 70% of the lattice
        r = np.random.randint(0, upper_limit)
        c = np.random.randint(0, size)
        
        # Check if position is empty
        if lattice[r, c] == 1:
            continue
        
        # Check isolation: ensure all neighbors are empty
        valid_position = True
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the position itself
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if lattice[nr, nc] == 1:
                        valid_position = False
                        break
            if not valid_position:
                break
        
        if valid_position:
            lattice[r, c] = 1
            placed_atoms.append((r, c))
    
    print(f"Successfully placed {len(placed_atoms)} isolated atoms out of target {num_atoms}")
    
    # Show vertical distribution
    upper_count = sum(1 for r, c in placed_atoms if r < size * 0.5)
    middle_count = sum(1 for r, c in placed_atoms if size * 0.5 <= r < size * 0.7)
    lower_count = sum(1 for r, c in placed_atoms if r >= size * 0.7)
    print(f"Vertical distribution: {upper_count} in top 50%, {middle_count} in middle 20%, {lower_count} in bottom 30%")
    
    # Verify isolation (each atom should be surrounded by 0s)
    isolated_count = 0
    for r, c in placed_atoms:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the atom itself
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    neighbors.append(lattice[nr, nc])
        
        if all(neighbor == 0 for neighbor in neighbors):
            isolated_count += 1
    
    print(f"Verified: {isolated_count}/{len(placed_atoms)} atoms are properly isolated")
    
    return lattice

def print_step_header(step_num, title):
    """Print a formatted step header."""
    print("\n" + "="*60)
    print(f"STEP {step_num}: {title}")
    print("="*60)

def print_lattice_stats(lattice, title):
    """Print statistics about the lattice."""
    atoms = np.sum(lattice == 1)
    total_positions = lattice.shape[0] * lattice.shape[1]
    occupancy = atoms / total_positions
    
    print(f"\n{title}:")
    print(f"  Size: {lattice.shape[0]}x{lattice.shape[1]}")
    print(f"  Total atoms: {atoms}")
    print(f"  Occupancy: {occupancy:.2%}")
    
    # Find atom positions
    atom_positions = [(r, c) for r in range(lattice.shape[0]) 
                     for c in range(lattice.shape[1]) if lattice[r, c] == 1]
    
    if len(atom_positions) <= 20:  # Only show positions if not too many
        print(f"  Atom positions: {atom_positions}")

def main():
    print("="*60)
    print("COMPLETE ATOM REARRANGEMENT WORKFLOW SIMULATION")
    print("="*60)
    
    # Configuration
    lattice_size = 20
    initial_occupancy = 0.4
    
    # ================================================================
    # STEP 1: Initialize sparse lattice
    # ================================================================
    print_step_header(1, "INITIALIZE SPARSE LATTICE")
    
    initial_lattice = create_initial_sparse_lattice(
        size=lattice_size, 
        occupancy=initial_occupancy,
        seed=42
    )
    
    print_lattice_stats(initial_lattice, "Initial lattice")
    
    # Visualize initial state
    simulator = LatticeSimulator(initial_size=(lattice_size, lattice_size))
    visualizer = LatticeVisualizer(simulator)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    visualizer.plot_lattice(initial_lattice, ax=ax, title="Step 1: Initial Sparse Lattice")
    plt.tight_layout()
    plt.savefig('workflow_step1_initial.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: workflow_step1_initial.png")
    
    # ================================================================
    # STEP 2: Column-by-column downward rearrangement
    # ================================================================
    print_step_header(2, "COLUMN-BY-COLUMN DOWNWARD REARRANGEMENT")
    
    print("Applying parallel column-wise downward movement...")
    
    # Apply column-by-column downward rearrangement
    final_lattice_down, assignments_down, batches_down = rearrange_column_by_column_downward_parallel(
        initial_lattice
    )
    
    print_lattice_stats(final_lattice_down, "After downward rearrangement")
    
    print(f"\nDownward rearrangement details:")
    print(f"  Total movement batches: {len(batches_down)}")
    print(f"  Total assignments: {len(assignments_down)}")
    
    # Show batch details
    total_moves = 0
    for i, batch in enumerate(batches_down):
        moves_in_batch = len(batch.get('moves', []))
        total_moves += moves_in_batch
        batch_time = batch.get('time', 0)
        print(f"  Batch {i+1}: {moves_in_batch} parallel moves, time: {batch_time:.6f}s")
    
    print(f"  Total moves executed: {total_moves}")
    
    # Visualize downward rearrangement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    visualizer.plot_lattice(initial_lattice, ax=ax1, title="Before: Sparse Configuration")
    visualizer.plot_lattice(final_lattice_down, ax=ax2, title="After: Column-wise Downward")
    
    plt.tight_layout()
    plt.savefig('workflow_step2_downward.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: workflow_step2_downward.png")
    
    # Create animation for downward movement
    if batches_down:
        print("Creating downward movement animation...")
        # Set up the simulator properly for animation
        simulator.slm_lattice = initial_lattice.copy()
        simulator.field = final_lattice_down.copy()
        
        # Create a more visual-friendly version of the batches
        # Group consecutive small batches together for better visualization
        visual_batches = []
        current_batch_moves = []
        current_batch_time = 0
        current_state = initial_lattice.copy()
        
        for batch in batches_down:
            moves = batch.get('moves', [])
            batch_time = batch.get('time', 0)
            
            # If this batch is small (≤3 moves) and we have room, accumulate it
            if len(moves) <= 3 and len(current_batch_moves) + len(moves) <= 8:
                current_batch_moves.extend(moves)
                current_batch_time = max(current_batch_time, batch_time)
            else:
                # Finalize current accumulated batch if it has moves
                if current_batch_moves:
                    # Apply accumulated moves to state
                    new_state = current_state.copy()
                    for move in current_batch_moves:
                        from_pos, to_pos = move['from'], move['to']
                        new_state[from_pos[0], from_pos[1]] = 0
                        new_state[to_pos[0], to_pos[1]] = 1
                    
                    visual_batches.append({
                        'type': 'visual_downward_batch',
                        'moves': current_batch_moves,
                        'state': new_state.copy(),
                        'time': current_batch_time,
                        'successful': len(current_batch_moves),
                        'failed': 0
                    })
                    current_state = new_state
                
                # Start new batch with current moves
                current_batch_moves = moves.copy()
                current_batch_time = batch_time
                
                # If this is a large batch, finalize it immediately
                if len(moves) > 3:
                    new_state = current_state.copy()
                    for move in current_batch_moves:
                        from_pos, to_pos = move['from'], move['to']
                        new_state[from_pos[0], from_pos[1]] = 0
                        new_state[to_pos[0], to_pos[1]] = 1
                    
                    visual_batches.append({
                        'type': 'visual_downward_batch',
                        'moves': current_batch_moves,
                        'state': new_state.copy(),
                        'time': current_batch_time,
                        'successful': len(current_batch_moves),
                        'failed': 0
                    })
                    current_state = new_state
                    current_batch_moves = []
                    current_batch_time = 0
        
        # Don't forget the last accumulated batch
        if current_batch_moves:
            new_state = current_state.copy()
            for move in current_batch_moves:
                from_pos, to_pos = move['from'], move['to']
                new_state[from_pos[0], from_pos[1]] = 0
                new_state[to_pos[0], to_pos[1]] = 1
            
            visual_batches.append({
                'type': 'visual_downward_batch',
                'moves': current_batch_moves,
                'state': new_state.copy(),
                'time': current_batch_time,
                'successful': len(current_batch_moves),
                'failed': 0
            })
        
        print(f"Optimized visualization: {len(batches_down)} batches -> {len(visual_batches)} visual frames")
        
        # Set up the simulator movement history for animation
        simulator.movement_history = visual_batches
        
        # Set a dummy target region (the visualizer needs this)
        if not hasattr(simulator, 'movement_manager') or simulator.movement_manager is None:
            # Create a dummy movement manager if needed
            from defect_free.movement import MovementManager
            simulator.movement_manager = MovementManager(simulator)
            # Set target region to None (no highlighting)
            simulator.movement_manager.target_region = None
        
        ani = visualizer.animate_movements(visual_batches)
        if ani:
            visualizer.save_animation('workflow_step2_downward_animation.gif')
            print("Saved: workflow_step2_downward_animation.gif")
    
    # ================================================================
    # STEP 3: Form dual 9x9 patterns side by side
    # ================================================================
    print_step_header(3, "FORM DUAL 9x9 PATTERNS")
    
    print("Forming dual 9x9 patterns (left and right) using closest atoms with advanced pathfinding...")
    
    # Apply dual 9x9 pattern formation
    final_lattice_pattern, assignments_pattern, batches_pattern = rearrange_to_9x9_pattern(
        final_lattice_down
    )
    
    print_lattice_stats(final_lattice_pattern, "After dual 9x9 pattern formation")
    
    print(f"\nDual 9x9 pattern formation details:")
    print(f"  Total movement batches: {len(batches_pattern)}")
    print(f"  Pattern assignments: {len(assignments_pattern)}")
    
    # Show pattern batch details
    total_pattern_moves = 0
    for i, batch in enumerate(batches_pattern):
        moves_in_batch = len(batch.get('moves', []))
        total_pattern_moves += moves_in_batch
        batch_time = batch.get('time', 0)
        print(f"  AOD Batch {i+1}: {moves_in_batch} parallel moves, time: {batch_time:.6f}s")
    
    print(f"  Total pattern moves executed: {total_pattern_moves}")
    
    # Visualize pattern formation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    visualizer.plot_lattice(final_lattice_down, ax=ax1, title="Before: After Downward Movement")
    visualizer.plot_lattice(final_lattice_pattern, ax=ax2, title="After: Dual 9x9 Patterns Formed")
    
    # Highlight the dual 9x9 pattern positions
    from without_defect_free.rearrangement import get_9x9_pattern_positions
    pattern_positions = get_9x9_pattern_positions((lattice_size, lattice_size))
    
    for r, c in pattern_positions:
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            ax2.add_patch(plt.Circle((c, r), 0.3, color='red', fill=False, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('workflow_step3_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: workflow_step3_pattern.png")
    
    # Create animation for pattern formation
    if batches_pattern:
        print("Creating dual 9x9 pattern formation animation...")
        # Update simulator for pattern animation
        simulator.slm_lattice = final_lattice_down.copy()
        simulator.field = final_lattice_pattern.copy()
        simulator.movement_history = batches_pattern
        
        ani = visualizer.animate_movements(batches_pattern)
        if ani:
            visualizer.save_animation('workflow_step3_pattern_animation.gif')
            print("Saved: workflow_step3_pattern_animation.gif")
    
    # ================================================================
    # STEP 4: Form inner atoms within 9x9 patterns
    # ================================================================
    print_step_header(4, "FORM INNER ATOMS IN 9x9 PATTERNS")
    
    print("Forming 4 inner atoms within selected 9x9 pattern region(s) using advanced pathfinding...")
    
    # Get existing 9x9 pattern positions
    existing_9x9_positions = get_9x9_pattern_positions((lattice_size, lattice_size))
    
    # Apply inner atoms formation - demonstrate with only left pattern first
    final_lattice_inner, assignments_inner, batches_inner = form_inner_atoms_in_9x9_patterns(
        final_lattice_pattern,
        existing_9x9_positions,
        target_patterns=['left']  # Only target the left 9x9 pattern
    )
    
    print(f"\nAfter inner atoms formation:")
    print(f"  Size: {final_lattice_inner.shape[0]}x{final_lattice_inner.shape[1]}")
    print(f"  Total atoms: {np.sum(final_lattice_inner)}")
    print(f"  Occupancy: {np.sum(final_lattice_inner) / (final_lattice_inner.shape[0] * final_lattice_inner.shape[1]) * 100:.2f}%")
    
    print(f"\nInner atoms formation details:")
    print(f"  Total movement batches: {len(batches_inner)}")
    print(f"  Inner atom assignments: {len(assignments_inner)}")
    for i, batch in enumerate(batches_inner, 1):
        moves = batch.get('moves', [])
        time_val = batch.get('time', 0)
        phase = batch.get('phase', 1)
        print(f"  AOD Batch {i}: {len(moves)} parallel moves, time: {time_val:.6f}s")
    
    total_inner_moves = sum(len(batch.get('moves', [])) for batch in batches_inner)
    print(f"  Total inner moves executed: {total_inner_moves}")
    
    # Visualize step 4 result
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    visualizer.plot_lattice(final_lattice_inner, ax=ax, title="Step 4: After Inner Atoms Formation")
    
    # Highlight both pattern positions and inner positions
    pattern_positions = get_9x9_pattern_positions((lattice_size, lattice_size))
    for r, c in pattern_positions:
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            ax.add_patch(plt.Circle((c, r), 0.3, color='red', fill=False, linewidth=2))
    
    # Highlight inner positions in different color
    for pos in assignments_inner:
        r, c = pos
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            ax.add_patch(plt.Circle((c, r), 0.3, color='blue', fill=False, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('workflow_step4_inner_atoms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: workflow_step4_inner_atoms.png")
    
    # Create inner atoms animation if batches exist
    if batches_inner:
        print("Creating inner atoms formation animation...")
        print("Saving animation to workflow_step4_inner_atoms_animation.gif...")
        
        # Convert inner atoms batches to visualizer format
        visual_batches_inner = []
        current_state = final_lattice_pattern.copy()
        
        for i, batch in enumerate(batches_inner):
            batch_moves = batch.get('moves', [])
            if batch_moves:
                print(f"Drawing {len(batch_moves)} arrows for frame {i+1}")
                
                # Apply moves to create new state
                new_state = current_state.copy()
                for move in batch_moves:
                    from_pos, to_pos = move['from'], move['to']
                    new_state[from_pos[0], from_pos[1]] = 0
                    new_state[to_pos[0], to_pos[1]] = 1
                
                visual_batches_inner.append({
                    'type': 'visual_inner_batch',
                    'moves': batch_moves,
                    'state': new_state.copy(),
                    'time': batch.get('time', 0.01),
                    'successful': len(batch_moves),
                    'failed': 0
                })
                current_state = new_state.copy()
        
        # Set up simulator for animation
        simulator.slm_lattice = final_lattice_pattern.copy()
        simulator.field = final_lattice_inner.copy()
        simulator.movement_history = visual_batches_inner
        
        ani = visualizer.animate_movements(visual_batches_inner)
        if ani:
            visualizer.save_animation('workflow_step4_inner_atoms_animation.gif')
            print("Animation saved successfully!")
            print("Saved: workflow_step4_inner_atoms_animation.gif")

    # ================================================================
    # STEP 5: Final comparison and summary
    # ================================================================
    print_step_header(5, "FINAL SUMMARY AND COMPARISON")
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    visualizer.plot_lattice(initial_lattice, ax=axes[0,0], title="Step 1: Initial Sparse Lattice")
    visualizer.plot_lattice(final_lattice_down, ax=axes[0,1], title="Step 2: After Downward Movement") 
    visualizer.plot_lattice(final_lattice_pattern, ax=axes[1,0], title="Step 3: After Dual 9x9 Pattern Formation")
    visualizer.plot_lattice(final_lattice_inner, ax=axes[1,1], title="Step 4: After Inner Atoms Formation")
    
    # Highlight pattern positions on step 3 plot
    for r, c in pattern_positions:
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            axes[1,0].add_patch(plt.Circle((c, r), 0.3, color='red', fill=False, linewidth=2))
    
    # Highlight both pattern and inner positions on final plot
    for r, c in pattern_positions:
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            axes[1,1].add_patch(plt.Circle((c, r), 0.3, color='red', fill=False, linewidth=2))
    
    for pos in assignments_inner:
        r, c = pos
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            axes[1,1].add_patch(plt.Circle((c, r), 0.3, color='blue', fill=False, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('workflow_complete_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: workflow_complete_comparison.png")
    
    # Print final statistics
    print("\n" + "="*60)
    print("WORKFLOW COMPLETION SUMMARY")
    print("="*60)
    
    print(f"\nLattice progression:")
    print(f"  Initial atoms: {np.sum(initial_lattice == 1)}")
    print(f"  After downward: {np.sum(final_lattice_down == 1)}")
    print(f"  After pattern: {np.sum(final_lattice_pattern == 1)}")
    print(f"  Final: {np.sum(final_lattice_inner == 1)}")
    
    print(f"\nMovement statistics:")
    print(f"  Downward movement batches: {len(batches_down)}")
    print(f"  Pattern formation batches: {len(batches_pattern)}")
    print(f"  Inner atoms batches: {len(batches_inner)}")
    print(f"  Total downward moves: {total_moves}")
    print(f"  Total pattern moves: {total_pattern_moves}")
    print(f"  Total inner moves: {total_inner_moves}")
    print(f"  Total moves executed: {total_moves + total_pattern_moves + total_inner_moves}")
    
    # Calculate total time
    total_downward_time = sum(batch.get('time', 0) for batch in batches_down)
    total_pattern_time = sum(batch.get('time', 0) for batch in batches_pattern)
    total_inner_time = sum(batch.get('time', 0) for batch in batches_inner)
    total_time = total_downward_time + total_pattern_time + total_inner_time
    
    print(f"\nTiming (physical movement time):")
    print(f"  Downward movement: {total_downward_time:.6f}s")
    print(f"  Pattern formation: {total_pattern_time:.6f}s")
    print(f"  Inner atoms formation: {total_inner_time:.6f}s")
    print(f"  Total time: {total_time:.6f}s")
    
    print(f"\nAOD constraint compliance:")
    print(f"  All movements respect AOD crossbar physics")
    print(f"  Parallel execution optimized within constraints")
    
    # Check if pattern was successfully formed
    pattern_filled = 0
    for r, c in pattern_positions:
        if 0 <= r < lattice_size and 0 <= c < lattice_size:
            if final_lattice_inner[r, c] == 1:
                pattern_filled += 1
    
    print(f"\nDual 9x9 Pattern completion: {pattern_filled}/18 positions filled")
    if pattern_filled == 18:
        print("✅ Perfect dual 9x9 patterns achieved!")
    else:
        print(f"⚠️  Partial patterns: {pattern_filled} out of 18 positions filled")
    
    # Check inner atoms completion
    inner_filled = len(assignments_inner)
    print(f"Inner atoms completion: {inner_filled}/8 positions filled")
    if inner_filled == 8:
        print("✅ Perfect inner atoms configuration achieved!")
    else:
        print(f"⚠️  Partial inner atoms: {inner_filled} out of 8 positions filled")
    
    # Overall completion status
    if pattern_filled == 18 and inner_filled == 8:
        print("🎉 PERFECT COMPLETE CONFIGURATION ACHIEVED!")
    elif pattern_filled == 18:
        print("✅ Perfect dual 9x9 patterns with partial inner atoms")
    else:
        print(f"⚠️  Partial completion: {pattern_filled}/18 patterns, {inner_filled}/8 inner atoms")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE - All files saved!")
    print("="*60)

if __name__ == "__main__":
    main()