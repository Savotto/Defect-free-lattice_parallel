"""
This module implements column-by-column downward rearrangement of atoms.

The main function rearranges atoms by pulling them down in each column simultaneously,
using parallel movement validation from defect_free.parallel_utils.
"""

import numpy as np
from typing import Dict, Tuple, List, Set
import heapq
import time

from defect_free.parallel_utils import can_parallelize_moves, merge_parallel_batches


def find_atoms(lattice: np.ndarray) -> List[Tuple[int, int]]:
    """Find all positions where atoms are present (value == 1)."""
    atoms = []
    rows, cols = lattice.shape
    for r in range(rows):
        for c in range(cols):
            if lattice[r, c] == 1:
                atoms.append((r, c))
    return atoms


def rearrange_column_by_column_downward_parallel(lattice: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[Dict]]:
    """
    Rearrange atoms by moving them downward in each column one at a time.
    
    Process each column sequentially to ensure proper gravity behavior:
    1. Find atoms in each column
    2. Sort them by row (preserving relative order)
    3. Move them down to form a compact stack at the bottom
    4. Create movement batches that respect AOD constraints
    
    Args:
        lattice: Binary numpy array (modified copy returned).
        
    Returns:
        Tuple of:
        - final_lattice: The lattice after rearrangement
        - assignments: Dict mapping final positions to 1  
        - movement_log: List of movement batches with proper timing
    """
    rows, cols = lattice.shape
    final_lattice = lattice.copy()
    
    # Find all current atoms
    current_atoms = find_atoms(lattice)
    
    # Group atoms by column
    atoms_by_col: Dict[int, List[Tuple[int, int]]] = {}
    for atom_pos in current_atoms:
        row, col = atom_pos
        if col not in atoms_by_col:
            atoms_by_col[col] = []
        atoms_by_col[col].append(atom_pos)
    
    all_moves = []
    assignments = {}
    
    print(f"Processing {len(atoms_by_col)} columns with atoms")
    
    # Process each column independently
    for col in sorted(atoms_by_col.keys()):
        atoms_in_col = atoms_by_col[col]
        
        if len(atoms_in_col) <= 1:
            # Single atom or empty column - check if it needs to move down
            if len(atoms_in_col) == 1:
                atom_row, atom_col = atoms_in_col[0]
                target_row = rows - 1  # Move to bottom row
                target_pos = (target_row, atom_col)
                
                if atom_row != target_row:
                    # Need to move this atom down
                    all_moves.append({
                        'from': (atom_row, atom_col),
                        'to': target_pos,
                        'distance': target_row - atom_row,
                        'column': col
                    })
                    # Update final lattice
                    final_lattice[atom_row, atom_col] = 0
                    final_lattice[target_row, atom_col] = 1
                
                assignments[target_pos] = 1
            continue
        
        # Multiple atoms in column - preserve relative order
        # Sort by row position (top to bottom)
        atoms_in_col.sort(key=lambda pos: pos[0])
        
        print(f"  Column {col}: {len(atoms_in_col)} atoms at rows {[pos[0] for pos in atoms_in_col]}")
        
        # Calculate target positions (stack from bottom, preserving order)
        # Bottom-most target position for this stack
        bottom_target_row = rows - len(atoms_in_col)
        
        # Clear original positions in final lattice
        for atom_row, atom_col in atoms_in_col:
            final_lattice[atom_row, atom_col] = 0
        
        # Assign new positions and create moves
        for i, (atom_row, atom_col) in enumerate(atoms_in_col):
            # Target position: i=0 (topmost atom) goes to bottom_target_row
            # i=1 goes to bottom_target_row + 1, etc.
            target_row = bottom_target_row + i
            target_pos = (target_row, atom_col)
            
            # Place atom in final lattice
            final_lattice[target_row, atom_col] = 1
            assignments[target_pos] = 1
            
            # Create move if position changed
            if atom_row != target_row:
                all_moves.append({
                    'from': (atom_row, atom_col),
                    'to': target_pos,
                    'distance': abs(target_row - atom_row),
                    'column': col
                })
                
        print(f"    -> Moved to rows {bottom_target_row}-{bottom_target_row + len(atoms_in_col) - 1}")
    
    # Create movement batches
    movement_batches = []
    
    if all_moves:
        print(f"Created {len(all_moves)} total moves")
        
        # Group moves by similar distance for better parallelization
        moves_by_distance = {}
        for move in all_moves:
            dist = move['distance']
            if dist not in moves_by_distance:
                moves_by_distance[dist] = []
            moves_by_distance[dist].append(move)
        
        # Create batches for each distance group
        for distance in sorted(moves_by_distance.keys()):
            moves_group = moves_by_distance[distance]
            
            # For moves of the same distance, we can potentially parallelize
            # But need to ensure no column conflicts
            remaining_moves = moves_group.copy()
            
            while remaining_moves:
                # Create a batch of non-conflicting moves
                current_batch_moves = []
                used_columns = set()
                
                moves_to_remove = []
                for move in remaining_moves:
                    col = move['column']
                    if col not in used_columns:
                        current_batch_moves.append({
                            'from': move['from'],
                            'to': move['to']
                        })
                        used_columns.add(col)
                        moves_to_remove.append(move)
                
                # Remove moves that were added to this batch
                for move in moves_to_remove:
                    remaining_moves.remove(move)
                
                # Create movement batch
                if current_batch_moves:
                    # Create intermediate state for this batch
                    batch_state = lattice.copy()
                    for move in current_batch_moves:
                        from_pos, to_pos = move['from'], move['to']
                        batch_state[from_pos[0], from_pos[1]] = 0
                        batch_state[to_pos[0], to_pos[1]] = 1
                    
                    movement_batch = {
                        'type': 'column_gravity_batch',
                        'moves': current_batch_moves,
                        'state': batch_state,
                        'time': float(distance),  # Movement time proportional to distance
                        'successful': len(current_batch_moves),
                        'failed': 0,
                        'distance_group': distance
                    }
                    movement_batches.append(movement_batch)
        
        # Update the final batch to have the correct final state
        if movement_batches:
            movement_batches[-1]['state'] = final_lattice.copy()
            
        print(f"Organized into {len(movement_batches)} movement batches")
        
        # Show batch summary
        for i, batch in enumerate(movement_batches):
            moves_count = len(batch['moves'])
            batch_time = batch['time']
            distance_group = batch.get('distance_group', 'unknown')
            print(f"  Batch {i+1}: {moves_count} moves, distance {distance_group}, time {batch_time}s")
    
    return final_lattice, assignments, movement_batches


def get_9x9_pattern_positions(lattice_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get the target positions for two 9x9 patterns side by side - one on the left and one on the right.
    
    Pattern for each 9x9 block:
    [100010001]
    [000000000]
    [000000000] 
    [000000000]
    [100010001]
    [000000000]
    [000000000]
    [000000000]
    [100010001]
    
    Args:
        lattice_shape: (rows, cols) shape of the lattice
        
    Returns:
        List of (row, col) positions where atoms should be placed (18 positions total)
    """
    rows, cols = lattice_shape
    
    pattern_positions = []
    
    # Calculate vertical center for both patterns
    center_row = rows // 2
    pattern_start_row = center_row - 4  # 9//2 = 4
    
    # Calculate horizontal positions for left and right patterns
    # Leave some space between the patterns
    pattern_spacing = 12  # Space between the two patterns
    
    # Left pattern: positioned at 1/4 of the width
    left_center_col = cols // 4
    left_start_col = left_center_col - 4
    
    # Right pattern: positioned at 3/4 of the width  
    right_center_col = (3 * cols) // 4
    right_start_col = right_center_col - 4
    
    # Define the pattern rows and columns that contain atoms (0, 4, 8)
    atom_rows = [0, 4, 8]
    atom_cols = [0, 4, 8]
    
    # Generate positions for left pattern
    for pattern_row in atom_rows:
        for pattern_col in atom_cols:
            actual_row = pattern_start_row + pattern_row
            actual_col = left_start_col + pattern_col
            
            # Make sure the position is within the lattice bounds
            if 0 <= actual_row < rows and 0 <= actual_col < cols:
                pattern_positions.append((actual_row, actual_col))
    
    # Generate positions for right pattern
    for pattern_row in atom_rows:
        for pattern_col in atom_cols:
            actual_row = pattern_start_row + pattern_row
            actual_col = right_start_col + pattern_col
            
            # Make sure the position is within the lattice bounds
            if 0 <= actual_row < rows and 0 <= actual_col < cols:
                pattern_positions.append((actual_row, actual_col))
    
    return pattern_positions


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_direct_path(field: np.ndarray, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Find a direct path between two points (horizontal or vertical)."""
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    
    # Only handle direct paths (same row or column)
    if start_row != end_row and start_col != end_col:
        return None
        
    if start_row == end_row:  # Same row
        # Check if horizontal path is clear
        path_clear = True
        start_c = min(start_col, end_col) + 1
        end_c = max(start_col, end_col)
        for col in range(start_c, end_c):
            if field[start_row, col] == 1:
                path_clear = False
                break
        
        if path_clear:
            return [start_pos, end_pos]
            
    elif start_col == end_col:  # Same column
        # Check if vertical path is clear
        path_clear = True
        start_r = min(start_row, end_row) + 1
        end_r = max(start_row, end_row)
        for row in range(start_r, end_r):
            if field[row, start_col] == 1:
                path_clear = False
                break
        
        if path_clear:
            return [start_pos, end_pos]
    
    return None


def find_l_shaped_path(field: np.ndarray, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Find L-shaped path with one turn (horizontal then vertical or vertical then horizontal)."""
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    
    # Try horizontal then vertical
    intermediate_pos1 = (start_row, end_col)
    if (intermediate_pos1 != start_pos and intermediate_pos1 != end_pos and 
        field[intermediate_pos1] == 0):
        # Check horizontal path segment
        h_path_clear = True
        start_c = min(start_col, end_col) + 1
        end_c = max(start_col, end_col)
        for col in range(start_c, end_c):
            if field[start_row, col] == 1:
                h_path_clear = False
                break
        
        # Check vertical path segment
        v_path_clear = True
        start_r = min(start_row, end_row) + 1
        end_r = max(start_row, end_row)
        for row in range(start_r, end_r):
            if field[row, end_col] == 1:
                v_path_clear = False
                break
        
        if h_path_clear and v_path_clear:
            return [start_pos, intermediate_pos1, end_pos]
    
    # Try vertical then horizontal 
    intermediate_pos2 = (end_row, start_col)
    if (intermediate_pos2 != start_pos and intermediate_pos2 != end_pos and
        field[intermediate_pos2] == 0):
        # Check vertical path segment
        v_path_clear = True
        start_r = min(start_row, end_row) + 1
        end_r = max(start_row, end_row)
        for row in range(start_r, end_r):
            if field[row, start_col] == 1:
                v_path_clear = False
                break
        
        # Check horizontal path segment
        h_path_clear = True
        start_c = min(start_col, end_col) + 1
        end_c = max(start_col, end_col)
        for col in range(start_c, end_c):
            if field[end_row, col] == 1:
                h_path_clear = False
                break
        
        if v_path_clear and h_path_clear:
            return [start_pos, intermediate_pos2, end_pos]
    
    return None


def find_a_star_path(field: np.ndarray, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                     reserved_positions: Set[Tuple[int, int]] = None, max_iterations: int = 1000) -> List[Tuple[int, int]]:
    """Implements A* search algorithm to find the shortest path."""
    if reserved_positions is None:
        reserved_positions = set()
    
    end_row, end_col = end_pos
    
    # Define heuristic (Manhattan distance)
    def heuristic(pos):
        return abs(pos[0] - end_row) + abs(pos[1] - end_col)
    
    # Initialize open and closed sets
    open_set = []
    closed_set = set()
    
    # Map to track the best path to each position
    came_from = {}
    
    # Initialize g_score (cost from start to current) and f_score (g_score + heuristic)
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos)}
    
    # Priority queue entry: (f_score, position)
    heapq.heappush(open_set, (f_score[start_pos], start_pos))
    
    # Define possible moves (up, right, down, left)
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    iterations = 0
    while open_set and iterations < max_iterations:
        iterations += 1
        
        # Get position with lowest f_score
        _, current_pos = heapq.heappop(open_set)
        
        # Check if we've reached the target
        if current_pos == end_pos:
            # Reconstruct path
            path = [current_pos]
            while current_pos in came_from:
                current_pos = came_from[current_pos]
                path.append(current_pos)
            path.reverse()
            return path
        
        # Mark as explored
        closed_set.add(current_pos)
        
        # Generate neighboring positions
        row, col = current_pos
        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            next_pos = (next_row, next_col)
            
            # Check if valid move
            if (0 <= next_row < field.shape[0] and 
                0 <= next_col < field.shape[1] and
                (field[next_row, next_col] == 0 or next_pos == end_pos) and  # Must be empty or the goal
                next_pos not in closed_set and
                next_pos not in reserved_positions):  # Avoid reserved positions
                
                # Calculate tentative g_score (path length so far)
                tentative_g_score = g_score.get(current_pos, float('inf')) + 1
                
                # If this path to next_pos is better than any previous one
                if tentative_g_score < g_score.get(next_pos, float('inf')):
                    # Update path and scores
                    came_from[next_pos] = current_pos
                    g_score[next_pos] = tentative_g_score
                    f_score[next_pos] = tentative_g_score + heuristic(next_pos)
                    
                    # Add to open set if not already there
                    for i, (_, pos) in enumerate(open_set):
                        if pos == next_pos:
                            # Remove old entry
                            open_set[i] = open_set[-1]
                            open_set.pop()
                            heapq.heapify(open_set)
                            break
                    heapq.heappush(open_set, (f_score[next_pos], next_pos))
    
    # No path found or exceeded max iterations
    return None


def compress_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Compresses consecutive horizontal or vertical movements in a path into single steps."""
    if not path or len(path) <= 2:
        return path  # Nothing to compress for trivial paths
        
    compressed = [path[0]]  # Always include the first position
    i = 1
    
    while i < len(path):
        # Track the current direction by checking adjacent points
        row_dir = path[i][0] - path[i-1][0]  # -1: up, 0: same row, 1: down
        col_dir = path[i][1] - path[i-1][1]  # -1: left, 0: same col, 1: right
        
        # Find the end of this direction segment
        curr_i = i
        while curr_i + 1 < len(path):
            next_row_dir = path[curr_i+1][0] - path[curr_i][0]
            next_col_dir = path[curr_i+1][1] - path[curr_i][1]
            
            # If direction changes, stop here
            if next_row_dir != row_dir or next_col_dir != col_dir:
                break

            curr_i += 1
        
        # Add the end point of this segment
        compressed.append(path[curr_i])
        i = curr_i + 1
        
    return compressed


def find_optimal_path(field: np.ndarray, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                     reserved_positions: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """Finds the optimal path using a tiered approach: direct, L-shaped, then A* search.
    
    Args:
        field: The lattice field
        start_pos: Starting position
        end_pos: Target position  
        reserved_positions: Set of positions that are reserved by other atoms' paths
    """
    if reserved_positions is None:
        reserved_positions = set()
    
    # Try direct path first (same row or column)
    direct_path = find_direct_path(field, start_pos, end_pos)
    if direct_path and not any(pos in reserved_positions for pos in direct_path[1:-1]):  # Exclude start/end
        return direct_path
    
    # Try L-shaped path (1 turn)
    l_shaped_path = find_l_shaped_path(field, start_pos, end_pos)
    if l_shaped_path and not any(pos in reserved_positions for pos in l_shaped_path[1:-1]):  # Exclude start/end
        return l_shaped_path
    
    # If simpler paths fail, use A* search for complex paths
    a_star_path = find_a_star_path(field, start_pos, end_pos, reserved_positions)
    if a_star_path:
        compressed_path = compress_path(a_star_path)
        if not any(pos in reserved_positions for pos in compressed_path[1:-1]):  # Exclude start/end
            return compressed_path
    
    return None


def calculate_realistic_movement_time(distance: float) -> float:
    """Calculate movement time with a trapezoidal velocity profile."""
    # Physical constants (simplified version)
    max_acceleration = 50.0  # m/s²
    max_velocity = 5.0  # m/s 
    site_distance = 0.5  # μm
    trap_transfer_time = 0.001  # 1ms
    
    # Convert distance from lattice units to meters
    distance_m = distance * site_distance * 1e-6
    
    # Acceleration distance
    accel_distance = (max_velocity**2) / (2 * max_acceleration)
    
    # Calculate the time based on kinematics
    if 2 * accel_distance <= distance_m:
        # Trapezoidal profile (reach max velocity)
        accel_time = max_velocity / max_acceleration
        constant_velocity_time = (distance_m - 2 * accel_distance) / max_velocity
        kinematic_time = 2 * accel_time + constant_velocity_time
    else:
        # Triangular profile (never reach max velocity)
        kinematic_time = 2 * np.sqrt(distance_m / max_acceleration)
    
    # Add trap transfer times (before and after movement)
    total_time = 2 * trap_transfer_time + kinematic_time
    return total_time


def rearrange_to_9x9_pattern(lattice: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[Dict]]:
    """
    Move atoms to fill two 9x9 pattern positions side by side using sophisticated pathfinding.
    
    Uses greedy assignment with optimal pathfinding for each move:
    1. Find the closest available atom for each pattern position
    2. Use tiered pathfinding (direct -> L-shaped -> A*) with path compression
    3. Calculate realistic movement times and create proper movement batches
    
    Args:
        lattice: Binary numpy array with atoms to rearrange.
        
    Returns:
        Tuple of:
        - final_lattice: The lattice with atoms in dual 9x9 patterns
        - assignments: Dict mapping pattern positions to 1
        - movement_log: List of movement batches with proper timing
    """
    print("=== ADVANCED DUAL 9x9 PATTERN FORMATION ===")
    start_time = time.time()
    
    # Get target positions for dual 9x9 patterns
    target_positions = get_9x9_pattern_positions(lattice.shape)
    
    if len(target_positions) != 18:
        raise ValueError(f"Expected 18 target positions for dual patterns, got {len(target_positions)}")
    
    # Find current atoms 
    current_atoms = find_atoms(lattice)
    
    if len(current_atoms) < 18:
        raise ValueError(f"Not enough atoms for dual 9x9 patterns. Have {len(current_atoms)}, need 18")
    
    print(f"Target positions: {len(target_positions)} positions for dual 9x9 patterns")
    print(f"Available atoms: {len(current_atoms)}")
    
    # Greedy assignment: for each target position, find the closest available atom
    # with the best path
    assignments = {}
    available_atoms = current_atoms.copy()
    working_field = lattice.copy()
    
    # Sort target positions by distance from center (fill center first for stability)
    rows, cols = lattice.shape
    center_row, center_col = rows // 2, cols // 2
    target_positions.sort(key=lambda pos: manhattan_distance(pos, (center_row, center_col)))
    
    print("\n=== GREEDY ASSIGNMENT WITH PATHFINDING ===")
    
    # Track reserved positions from all assigned paths
    reserved_positions = set()
    
    for i, target_pos in enumerate(target_positions):
        print(f"\nAssigning atom to target {i+1}/18: {target_pos}")
        
        best_atom = None
        best_path = None
        best_cost = float('inf')
        
        # Consider all available atoms
        for atom_pos in available_atoms:
            # Skip if atom position is already taken by a previous assignment
            if working_field[atom_pos[0], atom_pos[1]] == 0:
                continue
                
            # Find optimal path using sophisticated pathfinding with reserved positions
            path = find_optimal_path(working_field, atom_pos, target_pos, reserved_positions)
            
            if path:
                # Calculate path cost (length + complexity)
                path_length = len(path) - 1  # Number of moves
                total_distance = sum(manhattan_distance(path[j], path[j+1]) 
                                   for j in range(len(path)-1))
                
                # Cost function: prefer shorter paths with fewer turns
                cost = path_length * 1.0 + total_distance * 0.1
                
                if cost < best_cost:
                    best_atom = atom_pos
                    best_path = path
                    best_cost = cost
                    
                    print(f"  Candidate: {atom_pos} -> path length {path_length}, cost {cost:.2f}")
        
        if best_atom and best_path:
            assignments[best_atom] = (target_pos, best_path)
            available_atoms.remove(best_atom)
            
            # Reserve all intermediate positions from this path (excluding start/end)
            for pos in best_path[1:-1]:  # Exclude start and end positions
                reserved_positions.add(pos)
            
            # Temporarily mark the atom as moved in working field for next assignments
            working_field[best_atom[0], best_atom[1]] = 0
            working_field[target_pos[0], target_pos[1]] = 1
            
            print(f"  ✅ Selected: {best_atom} -> {target_pos}")
            print(f"     Path: {' -> '.join(map(str, best_path))}")
        else:
            print(f"  ❌ No valid path found for target {target_pos}")
    
    print(f"\n=== CREATING MOVEMENT BATCHES ===")
    print(f"Assigned {len(assignments)} atoms to pattern positions")
    
    # Create movement batches ensuring correct sequential execution order
    individual_batches = []
    working_lattice = lattice.copy()
    
    # Find maximum path length to determine number of sequential phases
    max_path_length = 0
    for source_pos, (target_pos, path) in assignments.items():
        if len(path) > max_path_length:
            max_path_length = len(path)
    
    print(f"Maximum path length: {max_path_length - 1} segments")
    
    # Create batches phase by phase to ensure correct sequential order
    for phase in range(max_path_length - 1):
        phase_moves = []
        max_time = 0
        
        print(f"\nPhase {phase + 1}: Processing path segment {phase + 1}")
        
        # Collect all moves for this phase (same segment index from all paths)
        for source_pos, (target_pos, path) in assignments.items():
            if phase < len(path) - 1:  # Check if this path has a segment for this phase
                from_pos = path[phase]
                to_pos = path[phase + 1]
                
                phase_moves.append({
                    'from': from_pos, 
                    'to': to_pos,
                    'source_atom': source_pos,
                    'target_pattern': target_pos
                })
                
                # Calculate movement time for this segment
                segment_distance = manhattan_distance(from_pos, to_pos)
                movement_time = calculate_realistic_movement_time(segment_distance)
                max_time = max(max_time, movement_time)
                
                print(f"  Atom from {source_pos}: {from_pos} → {to_pos} (distance: {segment_distance})")
        
        # Split phase moves into AOD-compliant sub-batches if needed
        if phase_moves:
            print(f"  Checking AOD constraints for {len(phase_moves)} phase moves...")
            
            # For each phase, we need to process moves sequentially to avoid conflicts
            # caused by intermediate states from earlier moves in the same phase
            remaining_moves = phase_moves.copy()
            phase_working_lattice = working_lattice.copy()
            
            while remaining_moves:
                print(f"    Processing {len(remaining_moves)} remaining moves...")
                
                # Check which moves can be made without blocking other atoms' paths
                valid_moves = []
                
                # First, calculate paths for all remaining moves using current field state
                move_paths = {}
                for move in remaining_moves:
                    from_pos = move['from']
                    to_pos = move['to']
                    
                    # Check if the atom is still at the source position
                    if phase_working_lattice[from_pos[0], from_pos[1]] != 1:
                        print(f"      Skipping move {from_pos} -> {to_pos}: atom not at source")
                        continue
                    
                    # Calculate path to final target using current field state
                    final_target = move['target_pattern']
                    full_path = find_optimal_path(phase_working_lattice, from_pos, final_target)
                    if full_path:
                        move_paths[move['source_atom']] = {
                            'current_move': move,
                            'full_path': full_path,
                            'next_pos': full_path[1] if len(full_path) > 1 else None
                        }
                
                # Now check each potential move to see if it would block others
                for source_atom, path_info in move_paths.items():
                    move = path_info['current_move']
                    next_pos = path_info['next_pos']
                    from_pos = move['from']
                    
                    if next_pos is None or phase_working_lattice[next_pos[0], next_pos[1]] != 0:
                        print(f"      Cannot move {from_pos} -> {next_pos}: position not available")
                        continue
                    
                    # Check if this move would block other atoms' paths
                    would_block_others = False
                    
                    # Simulate the move temporarily
                    temp_lattice = phase_working_lattice.copy()
                    temp_lattice[from_pos[0], from_pos[1]] = 0  # Remove from source
                    temp_lattice[next_pos[0], next_pos[1]] = 1   # Add to destination
                    
                    # Check if other atoms can still reach their targets AND make their next moves
                    for other_source, other_path_info in move_paths.items():
                        if other_source == source_atom:
                            continue  # Skip self
                        
                        other_move = other_path_info['current_move']
                        other_from = other_move['from']
                        other_target = other_move['target_pattern']
                        other_next_pos = other_path_info['next_pos']
                        
                        # Check if the other atom still exists and can make its next move
                        if temp_lattice[other_from[0], other_from[1]] == 1:  # Other atom still exists
                            # First check: Can the other atom make its immediate next move?
                            if other_next_pos and temp_lattice[other_next_pos[0], other_next_pos[1]] != 0:
                                would_block_others = True
                                print(f"      Move {from_pos} -> {next_pos} would block atom at {other_from} from making immediate move to {other_next_pos}")
                                break
                            
                            # Second check: Can the other atom still find a path to its final target?
                            other_new_path = find_optimal_path(temp_lattice, other_from, other_target)
                            if not other_new_path:
                                would_block_others = True
                                print(f"      Move {from_pos} -> {next_pos} would block atom at {other_from} from reaching final target {other_target}")
                                break
                    
                    if not would_block_others:
                        valid_moves.append({
                            'from': from_pos,
                            'to': next_pos,
                            'source_atom': move['source_atom'],
                            'target_pattern': move['target_pattern']
                        })
                        print(f"      Valid move: {from_pos} -> {next_pos} (doesn't block others)")
                    else:
                        print(f"      Deferred move: {from_pos} -> {next_pos} (would block other atoms)")
                
                if not valid_moves:
                    print("    No valid moves found in this iteration")
                    # If we can't make any moves without blocking others, we're done with this phase
                    print("    All remaining moves would block other atoms' paths")
                    break
                
                # Create individual batches for valid moves using current state
                phase_individual_batches = []
                for move in valid_moves:
                    segment_distance = manhattan_distance(move['from'], move['to'])
                    movement_time = calculate_realistic_movement_time(segment_distance)
                    
                    phase_individual_batches.append({
                        'type': 'advanced_9x9_phase_move',
                        'moves': [move],
                        'state': phase_working_lattice.copy(),
                        'time': movement_time,
                        'successful': 1,
                        'failed': 0,
                        'phase': phase + 1,
                        'total_phases': max_path_length - 1
                    })
                
                # Use parallel_utils to merge compatible moves using current field state
                from defect_free.parallel_utils import merge_parallel_batches
                aod_batches = merge_parallel_batches(phase_individual_batches, phase_working_lattice)
                
                if aod_batches:
                    # Take the first AOD batch (most moves that can be parallelized)
                    first_batch = aod_batches[0]
                    first_batch['type'] = 'advanced_9x9_aod_phase'
                    first_batch['phase'] = phase + 1
                    first_batch['total_phases'] = max_path_length - 1
                    individual_batches.append(first_batch)
                    
                    # Update the working lattice with moves from this batch
                    batch_moves = first_batch.get('moves', [])
                    for move in batch_moves:
                        from_pos = move['from']
                        to_pos = move['to']
                        if phase_working_lattice[from_pos[0], from_pos[1]] == 1:
                            phase_working_lattice[from_pos[0], from_pos[1]] = 0
                            phase_working_lattice[to_pos[0], to_pos[1]] = 1
                    
                    # Update the batch state to reflect the new field state
                    first_batch['state'] = phase_working_lattice.copy()
                    
                    # Remove processed moves from remaining_moves based on source positions
                    processed_sources = {move['from'] for move in batch_moves}
                    remaining_moves = [move for move in remaining_moves 
                                     if move['from'] not in processed_sources]
                    
                    batch_count = len(batch_moves)
                    batch_time = first_batch.get('time', 0)
                    print(f"    AOD sub-batch: {batch_count} moves, time: {batch_time:.6f}s")
                    print(f"    Remaining moves in phase: {len(remaining_moves)}")
                else:
                    print("    Warning: No valid AOD batches found, breaking")
                    break
            
            # Update working lattice for next phase
            working_lattice = phase_working_lattice.copy()
    
    print(f"Created {len(individual_batches)} AOD-compliant sequential batches")
    
    # Execute batches sequentially to get proper intermediate states
    if individual_batches:
        print("Executing AOD-compliant sequential batches...")
        working_lattice = lattice.copy()
        
        current_phase = 0
        for batch_idx, batch in enumerate(individual_batches):
            moves = batch.get('moves', [])
            if not moves:
                continue
            
            batch_phase = batch.get('phase', 0)
            if batch_phase != current_phase:
                current_phase = batch_phase
                print(f"\n--- PHASE {current_phase}/{batch.get('total_phases', 'unknown')} ---")
                
            print(f"Executing AOD batch {batch_idx + 1} (Phase {current_phase}): {len(moves)} moves")
            
            # Use the pre-calculated state from the batch (which includes previous moves)
            # This ensures we don't have collisions since the planning already accounted for them
            new_lattice = batch['state'].copy()
            
            for move in moves:
                from_pos = move['from']
                to_pos = move['to']
                print(f"  Executed: {from_pos} → {to_pos}")
            
            # Update batch state and working lattice
            batch['state'] = new_lattice.copy()
            working_lattice = new_lattice
            
            print(f"  AOD batch completed: {len(moves)} moves executed")
        
        # Create final assignments (pattern positions to 1)
        final_assignments = {}
        for target_pos in target_positions:
            if working_lattice[target_pos[0], target_pos[1]] == 1:
                final_assignments[target_pos] = 1
        
        execution_time = time.time() - start_time
        total_movement_time = sum(batch.get('time', 0) for batch in individual_batches)
        
        print(f"\n=== COMPLETION SUMMARY ===")
        print(f"Computation time: {execution_time:.3f}s")
        print(f"Total movement time: {total_movement_time:.6f}s")
        print(f"Pattern positions filled: {len(final_assignments)}/18")
        print(f"AOD-compliant batches executed: {len(individual_batches)}")
        
        # Check for any atoms that ended up in pattern area but aren't assigned
        pattern_area_atoms = 0
        unassigned_atoms = 0
        for target_pos in target_positions:
            if working_lattice[target_pos[0], target_pos[1]] == 1:
                pattern_area_atoms += 1
                if target_pos not in final_assignments:
                    unassigned_atoms += 1
        
        print(f"Total atoms in pattern area: {pattern_area_atoms}")
        if unassigned_atoms > 0:
            print(f"WARNING: {unassigned_atoms} unassigned atoms in pattern area!")
        
        return working_lattice, final_assignments, individual_batches
    else:
        # No moves needed
        final_assignments = {}
        for target_pos in target_positions:
            if lattice[target_pos[0], target_pos[1]] == 1:
                final_assignments[target_pos] = 1
        return lattice, final_assignments, []


def form_inner_atoms_in_9x9_patterns(lattice: np.ndarray, existing_9x9_positions: List[Tuple[int, int]] = None, target_patterns: List[str] = None) -> Tuple[np.ndarray, Dict, List]:
    """
    Forms 4 atoms within selected 9x9 pattern region(s) for more complex configurations.
    
    The method identifies the 9x9 pattern regions and places 4 atoms in the middle of each
    smaller 3x3 subregion within the specified 9x9 pattern(s):
    - Left 9x9 region: columns 1-9, rows 6-14
    - Right 9x9 region: columns 11-19, rows 6-14  
    - Inner positions: (8,3), (8,7), (12,3), (12,7) for each region
    
    Args:
        lattice: Current lattice state (assumed to have existing 9x9 patterns)
        existing_9x9_positions: Optional list of existing 9x9 pattern positions
        target_patterns: List of patterns to target - options: ['left', 'right'], defaults to ['left', 'right']
        
    Returns:
        Tuple of (final_lattice, assignments_dict, movement_batches)
    """
    import time
    
    start_time = time.time()
    rows, cols = lattice.shape
    
    # Default to both patterns if not specified
    if target_patterns is None:
        target_patterns = ['left', 'right']
    
    print(f"=== FORMING INNER ATOMS IN 9x9 PATTERNS ===")
    print(f"Target patterns: {target_patterns}")
    
    # Define the 9x9 regions (assuming 20x20 lattice)
    left_9x9_region = {
        'start_row': 6, 'end_row': 14,  # rows 6-14 (9 rows)
        'start_col': 1, 'end_col': 9    # cols 1-9 (9 cols)
    }
    right_9x9_region = {
        'start_row': 6, 'end_row': 14,  # rows 6-14 (9 rows) 
        'start_col': 11, 'end_col': 19  # cols 11-19 (9 cols)
    }
    
    # Calculate inner positions for each 9x9 region
    # These are the centers of the 3x3 subregions within each 9x9
    left_inner_positions = [
        (8, 3),   # top-left subregion center
        (8, 7),   # top-right subregion center  
        (12, 3),  # bottom-left subregion center
        (12, 7)   # bottom-right subregion center
    ]
    
    right_inner_positions = [
        (8, 13),  # top-left subregion center (offset by 10 cols)
        (8, 17),  # top-right subregion center
        (12, 13), # bottom-left subregion center  
        (12, 17)  # bottom-right subregion center
    ]
    
    # Build target positions based on selected patterns
    target_positions = []
    if 'left' in target_patterns:
        target_positions.extend(left_inner_positions)
    if 'right' in target_patterns:
        target_positions.extend(right_inner_positions)
    
    print(f"Target inner positions: {len(target_positions)} positions")
    if 'left' in target_patterns:
        print(f"Left 9x9 inner positions: {left_inner_positions}")
    if 'right' in target_patterns:
        print(f"Right 9x9 inner positions: {right_inner_positions}")
    
    # Find available atoms (exclude existing 9x9 pattern positions)
    working_field = lattice.copy()
    available_atoms = []
    
    # Get existing 9x9 positions if not provided
    if existing_9x9_positions is None:
        existing_9x9_positions = dual_9x9_pattern_positions()
    
    for i in range(rows):
        for j in range(cols):
            if working_field[i, j] == 1 and (i, j) not in existing_9x9_positions:
                available_atoms.append((i, j))
    
    print(f"Available atoms for inner positions: {len(available_atoms)}")
    
    if len(available_atoms) < len(target_positions):
        print(f"Warning: Only {len(available_atoms)} atoms available for {len(target_positions)} target positions")
    
    # Greedy assignment with pathfinding and collision avoidance
    assignments = {}
    reserved_positions = set()
    
    print(f"\n=== GREEDY ASSIGNMENT FOR INNER POSITIONS ===")
    
    for i, target_pos in enumerate(target_positions):
        print(f"\nAssigning atom to inner position {i+1}/{len(target_positions)}: {target_pos}")
        
        best_atom = None
        best_path = None
        best_cost = float('inf')
        
        # Consider all available atoms
        for atom_pos in available_atoms:
            # Skip if atom position is already taken
            if working_field[atom_pos[0], atom_pos[1]] == 0:
                continue
                
            # Find optimal path with collision avoidance
            path = find_optimal_path(working_field, atom_pos, target_pos, reserved_positions)
            
            if path:
                # Calculate path cost
                path_length = len(path) - 1
                total_distance = sum(manhattan_distance(path[j], path[j+1]) 
                                   for j in range(len(path)-1))
                
                # Cost function: prefer shorter paths
                cost = path_length * 1.0 + total_distance * 0.1
                
                if cost < best_cost:
                    best_atom = atom_pos
                    best_path = path
                    best_cost = cost
                    
                    print(f"  Candidate: {atom_pos} -> path length {path_length}, cost {cost:.2f}")
        
        if best_atom and best_path:
            assignments[best_atom] = (target_pos, best_path)
            available_atoms.remove(best_atom)
            
            # Reserve intermediate positions from this path
            for pos in best_path[1:-1]:
                reserved_positions.add(pos)
            
            # Update working field
            working_field[best_atom[0], best_atom[1]] = 0
            working_field[target_pos[0], target_pos[1]] = 1
            
            print(f"  ✅ Selected: {best_atom} -> {target_pos}")
            print(f"     Path: {' -> '.join(map(str, best_path))}")
        else:
            print(f"  ❌ No valid path found for inner position {target_pos}")
    
    # Create movement batches optimized for parallel execution
    print(f"\n=== CREATING MOVEMENT BATCHES FOR INNER POSITIONS ===")
    print(f"Assigned {len(assignments)} atoms to inner positions")
    
    # Group by columns for parallel movement (as requested)
    column_groups = {}
    for source_pos, (target_pos, path) in assignments.items():
        target_col = target_pos[1]
        if target_col not in column_groups:
            column_groups[target_col] = []
        column_groups[target_col].append((source_pos, target_pos, path))
    
    print(f"Column groups for parallel movement:")
    for col, moves in column_groups.items():
        print(f"  Column {col}: {len(moves)} moves")
    
    # Create movement batches with AOD compliance
    individual_batches = []
    working_lattice = lattice.copy()
    
    # Process movements by phases (similar to dual pattern formation)
    max_path_length = max(len(path) for _, (_, path) in assignments.items()) if assignments else 0
    print(f"Maximum path length: {max_path_length - 1} segments")
    
    for phase in range(1, max_path_length):
        print(f"\nPhase {phase}: Processing path segment {phase}")
        phase_moves = []
        
        for source_pos, (target_pos, path) in assignments.items():
            if phase < len(path):
                from_pos = path[phase - 1]
                to_pos = path[phase]
                distance = manhattan_distance(from_pos, to_pos)
                
                phase_moves.append({
                    'from': from_pos,
                    'to': to_pos,
                    'distance': distance,
                    'source_atom': source_pos
                })
                print(f"  Atom from {source_pos}: {from_pos} → {to_pos} (distance: {distance})")
        
        if phase_moves:
            # Apply proper AOD constraints using the same system as dual pattern formation
            print(f"  Checking AOD constraints for {len(phase_moves)} phase moves...")
            remaining_moves = phase_moves.copy()
            
            while remaining_moves:
                print(f"    Processing {len(remaining_moves)} remaining moves...")
                aod_moves = []
                
                # Use proper AOD constraint checking
                for move in remaining_moves[:]:
                    # Check if this move can be added to current batch
                    can_add = True
                    
                    if aod_moves:
                        # Convert to proper format for can_parallelize_moves
                        test_batch1 = [{'from': m['from'], 'to': m['to']} for m in aod_moves]
                        test_batch2 = [{'from': move['from'], 'to': move['to']}]
                        
                        if not can_parallelize_moves(working_lattice, test_batch1, test_batch2):
                            can_add = False
                    
                    if can_add:
                        aod_moves.append(move)
                        remaining_moves.remove(move)
                        print(f"      Valid move: {move['from']} -> {move['to']} (AOD compliant)")
                    else:
                        print(f"      Deferred move: {move['from']} -> {move['to']} (AOD constraint)")

                if aod_moves:
                    # Create batch with proper timing
                    batch_time = max(move['distance'] * 0.001 for move in aod_moves)
                    individual_batches.append({
                        'moves': aod_moves,
                        'time': batch_time,
                        'phase': phase
                    })
                    print(f"    AOD sub-batch: {len(aod_moves)} moves, time: {batch_time:.6f}s")
                    
                    # Update working lattice for next iteration
                    for move in aod_moves:
                        working_lattice[move['from'][0], move['from'][1]] = 0
                        working_lattice[move['to'][0], move['to'][1]] = 1
                else:
                    # No valid moves found, break to avoid infinite loop
                    print(f"    No valid moves found in this iteration, deferring {len(remaining_moves)} moves")
                    break
                
                print(f"    Remaining moves in phase: {len(remaining_moves)}")
    
    # Execute the movement batches
    print(f"Created {len(individual_batches)} AOD-compliant sequential batches")
    print(f"Executing AOD-compliant sequential batches...")
    
    final_assignments = {}
    current_lattice = lattice.copy()
    
    for i, batch in enumerate(individual_batches):
        phase = batch.get('phase', 1)
        print(f"Executing AOD batch {i+1} (Phase {phase}): {len(batch['moves'])} moves")
        
        for move in batch['moves']:
            from_pos = move['from']
            to_pos = move['to']
            
            # Execute the move
            current_lattice[from_pos[0], from_pos[1]] = 0
            current_lattice[to_pos[0], to_pos[1]] = 1
            
            print(f"  Executed: {from_pos} → {to_pos}")
        
        print(f"  AOD batch completed: {len(batch['moves'])} moves executed")
    
    # Record final assignments
    for target_pos in target_positions:
        if current_lattice[target_pos[0], target_pos[1]] == 1:
            final_assignments[target_pos] = 1
    
    execution_time = time.time() - start_time
    total_movement_time = sum(batch.get('time', 0) for batch in individual_batches)
    
    print(f"\n=== INNER POSITIONS COMPLETION SUMMARY ===")
    print(f"Computation time: {execution_time:.3f}s")
    print(f"Total movement time: {total_movement_time:.6f}s")
    print(f"Inner positions filled: {len(final_assignments)}/{len(target_positions)}")
    print(f"AOD-compliant batches executed: {len(individual_batches)}")
    
    return current_lattice, final_assignments, individual_batches


def move_atoms_under_9x9_into_columns(lattice: np.ndarray, pattern: str = 'left', source_columns: List[int] = None,
                                      num_atoms_needed: int = 4) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[Dict]]:
    """
    Move atoms that are already under (within rows of) the 9x9 region into the given source columns.

    This preparation step restricts selection to atoms whose row lies within the 9x9 region
    for the selected pattern (so we only move atoms that are "under the 9x9 configuration").

    Args:
        lattice: binary lattice
        pattern: 'left' or 'right' (determines the 9x9 region rows/cols)
        source_columns: list of columns to fill (if None, a sensible default is chosen)
        num_atoms_needed: how many atoms to ensure across the source columns (default 4)

    Returns:
        (updated_lattice, assignments, movement_batches)
    """
    print(f"=== PREPARE: move atoms under 9x9 into columns for pattern '{pattern}' -> columns {source_columns}")
    rows, cols = lattice.shape

    # Define 9x9 regions (same as in wrapper)
    left_9x9 = {'start_row': 6, 'end_row': 14, 'start_col': 1, 'end_col': 9}
    right_9x9 = {'start_row': 6, 'end_row': 14, 'start_col': 11, 'end_col': 19}

    region = left_9x9 if pattern == 'left' else right_9x9
    start_row, end_row = region['start_row'], region['end_row']

    # Default source columns (two columns near each 9x9 region)
    if source_columns is None:
        if pattern == 'left':
            source_columns = [region['start_col'] - 1 if region['start_col'] - 1 >= 0 else region['start_col'],
                              region['start_col'] + 1]
        else:
            source_columns = [region['end_col'] + 1 if region['end_col'] + 1 < cols else region['end_col'],
                              region['end_col'] - 1]

    # Collect candidate atoms that are within the 9x9 rows but NOT inside the source columns
    candidates = []
    for r in range(start_row, end_row + 1):
        for c in range(cols):
            if lattice[r, c] == 1 and c not in source_columns:
                candidates.append((r, c))

    print(f"Candidates under 9x9 rows: {len(candidates)}")

    if len(candidates) == 0:
        print("No candidate atoms under 9x9 rows to move into source columns.")
        return lattice.copy(), {}, []

    # We need to place `num_atoms_needed` atoms into the source columns (bottom-up within region rows)
    assignments: Dict[Tuple[int, int], int] = {}
    batches: List[Dict] = []
    working = lattice.copy()

    # Determine available target slots in source columns limited to the 9x9 rows
    target_slots = []
    for col in source_columns:
        for r in range(end_row, start_row - 1, -1):  # bottom-to-top
            if working[r, col] == 0:
                target_slots.append((r, col))

    # If not enough slots, return with warning
    need = max(0, num_atoms_needed - sum(1 for r in range(start_row, end_row+1) for c in source_columns if lattice[r, c] == 1))
    if need == 0:
        print("Required atoms already present in source columns under 9x9 rows.")
        return lattice.copy(), {}, []

    if len(target_slots) < need:
        print(f"Warning: Not enough free slots in source columns under 9x9. Need {need}, have {len(target_slots)}")
        need = min(need, len(target_slots))

    # Choose closest candidates to fill the target slots (greedy)
    chosen_moves = []
    candidates_sorted = sorted(candidates, key=lambda p: abs(p[1] - source_columns[0]) + abs(p[0] - (start_row+end_row)//2))
    # take first `need` candidates
    selected_candidates = candidates_sorted[:need]

    # Assign each selected candidate to a target slot (closest)
    used_slots = set()
    for cand in selected_candidates:
        best_slot = min((s for s in target_slots if s not in used_slots), key=lambda s: abs(s[0]-cand[0]) + abs(s[1]-cand[1]))
        used_slots.add(best_slot)
        # compute path and create move (simple direct vertical/horizontal path)
        path = find_optimal_path(working, cand, best_slot, set())
        if path is None:
            print(f"Could not find path to move candidate {cand} -> {best_slot}, skipping")
            continue
        chosen_moves.append({'from': cand, 'to': best_slot, 'path': path, 'distance': len(path)-1})
        # update working for next assignments
        working[cand[0], cand[1]] = 0
        working[best_slot[0], best_slot[1]] = 1
        assignments[best_slot] = 1

    # Build simple batches (attempt parallel grouping by AOD util)
    if chosen_moves:
        move_batches = []
        for mv in chosen_moves:
            move_batches.append({'moves': [mv], 'state': working.copy(), 'time': mv['distance']*1e-6, 'type': 'prepare_column_move'})
        try:
            aod_batches = merge_parallel_batches(move_batches, lattice)
        except Exception:
            aod_batches = move_batches
        batches.extend(aod_batches)

    print(f"Prepared {len(assignments)} slots in columns {source_columns} (batches: {len(batches)})")
    return working, assignments, batches


def push_4_inner_atoms_from_columns(lattice: np.ndarray, pattern: str = 'left', source_columns: List[int] = None) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[Dict]]:
    """
    Push four atoms from two prepared source columns into the inner 2x2 of the 9x9 pattern.

    Preconditions:
    - The source_columns must be provided (two columns) or chosen by default.
    - Each source column must contain at least two atoms within the 9x9 rows (we only look at those rows).
    - If the two columns' atom rows don't match, the method will attempt a 1-row column shift to align them
      (moving both atoms in that column up or down by 1) when possible.

    The method only inspects and moves atoms located in the given source columns and within the
    9x9 rows; it does not search the whole lattice.
    """
    print(f"=== PUSH: push 4 inner atoms from columns {source_columns} for pattern '{pattern}' ===")
    rows, cols = lattice.shape
    left_9x9 = {'start_row': 6, 'end_row': 14, 'start_col': 1, 'end_col': 9}
    right_9x9 = {'start_row': 6, 'end_row': 14, 'start_col': 11, 'end_col': 19}
    region = left_9x9 if pattern == 'left' else right_9x9
    start_row, end_row = region['start_row'], region['end_row']

    if source_columns is None:
        if pattern == 'left':
            source_columns = [region['start_col'] - 1 if region['start_col'] - 1 >= 0 else region['start_col'],
                              region['start_col'] + 1]
        else:
            source_columns = [region['end_col'] + 1 if region['end_col'] + 1 < cols else region['end_col'],
                              region['end_col'] - 1]

    # Gather atoms in the source columns limited to 9x9 rows
    col_atoms = {c: [] for c in source_columns}
    for c in source_columns:
        for r in range(start_row, end_row + 1):
            if lattice[r, c] == 1:
                col_atoms[c].append((r, c))

    print(f"Atoms found in source columns (within 9x9 rows): {col_atoms}")

    # Need exactly two atoms in each column (or at least two; pick closest two)
    for c in source_columns:
        if len(col_atoms[c]) < 2:
            raise ValueError(f"Not enough atoms in source column {c}. Need 2, have {len(col_atoms[c])}.")
        # keep only two (closest to center rows)
        col_atoms[c] = sorted(col_atoms[c], key=lambda p: abs(p[0] - (start_row+end_row)//2))[:2]

    # Extract rows per column and attempt to align
    rows0 = sorted([p[0] for p in col_atoms[source_columns[0]]])
    rows1 = sorted([p[0] for p in col_atoms[source_columns[1]]])

    # If rows already match (set equality), we're good
    if rows0 != rows1:
        # Try shifting first column up/down by 1 to match
        aligned = False
        for shift_col in source_columns:
            for direction in (-1, 1):
                new_rows = [r + direction for r in sorted([p[0] for p in col_atoms[shift_col]])]
                # check within bounds
                if any(r < start_row or r > end_row for r in new_rows):
                    continue
                other_col = [c for c in source_columns if c != shift_col][0]
                other_rows = sorted([p[0] for p in col_atoms[other_col]])
                if new_rows == other_rows:
                    # perform column shift: move both atoms in shift_col by direction
                    shift_moves = []
                    working = lattice.copy()
                    for atom in col_atoms[shift_col]:
                        from_pos = atom
                        to_pos = (atom[0] + direction, atom[1])
                        path = find_optimal_path(working, from_pos, to_pos, set())
                        if path is None:
                            break
                        shift_moves.append({'from': from_pos, 'to': to_pos, 'path': path, 'distance': len(path)-1})
                        # apply immediately to working
                        working[from_pos[0], from_pos[1]] = 0
                        working[to_pos[0], to_pos[1]] = 1
                    else:
                        # commit shift
                        print(f"Shifting column {shift_col} by {direction} rows to align")
                        # update lattice and col_atoms
                        lattice = working.copy()
                        col_atoms[shift_col] = [(r + direction, shift_col) for r in sorted([p[0] for p in col_atoms[shift_col]])]
                        aligned = True
                        break
            if aligned:
                break
        if not aligned:
            raise ValueError("Could not align source column atom rows with a 1-row shift; aborting push")

    # At this point both columns contain two atoms with identical row positions
    # Map top row -> top inner target, bottom row -> bottom inner target for left/right
    center_row = (start_row + end_row) // 2
    center_col = (region['start_col'] + region['end_col']) // 2
    inner_targets = [
        (center_row - 1, center_col - 1),
        (center_row - 1, center_col + 1),
        (center_row + 1, center_col - 1),
        (center_row + 1, center_col + 1)
    ]

    # Build source atom list ordered to match inner_targets by rows
    # source_columns[0] left column corresponds to inner_targets [0,2] (top-left, bottom-left)
    src0 = sorted(col_atoms[source_columns[0]])
    src1 = sorted(col_atoms[source_columns[1]])
    # order: top0, top1, bottom0, bottom1 -> map to inner_targets [0,1,2,3]
    source_order = [src0[0], src1[0], src0[1], src1[1]]

    # Compute paths for each move and execute parallel phases
    moves = []
    working = lattice.copy()
    for s, t in zip(source_order, inner_targets):
        path = find_optimal_path(working, s, t, set())
        if path is None:
            raise ValueError(f"No path found from {s} to {t} while pushing inner atoms")
        moves.append({'from': s, 'to': t, 'path': path, 'distance': len(path)-1})

    # Execute moves phase-by-phase allowing parallel moves when possible using merge_parallel_batches
    max_len = max(len(m['path']) for m in moves)
    batches = []
    for phase in range(max_len - 1):
        phase_moves = []
        for m in moves:
            if phase + 1 < len(m['path']):
                phase_moves.append({'from': m['path'][phase], 'to': m['path'][phase+1]})
        if not phase_moves:
            continue
        # create per-move batches
        indiv = [{'moves': [mv], 'state': working.copy(), 'time': 1e-6, 'type': 'push_inner'} for mv in phase_moves]
        try:
            aod = merge_parallel_batches(indiv, working)
        except Exception:
            aod = indiv
        # apply first batch and update working lattice
        if aod:
            for batch in aod:
                for mv in batch.get('moves', []):
                    f = mv['from']; to = mv['to']
                    if working[f[0], f[1]] == 1:
                        working[f[0], f[1]] = 0
                        working[to[0], to[1]] = 1
                batches.append(batch)

    # Final check: which inner positions are filled
    assignments = {}
    for t in inner_targets:
        if working[t[0], t[1]] == 1:
            assignments[t] = 1

    print(f"Pushed inner atoms; filled {len(assignments)}/4 positions")
    return working, assignments, batches
