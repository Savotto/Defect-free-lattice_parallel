# Description: Movement manager for atom rearrangement strategies.
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any

class MovementManager:
    """
    Manages atom movement with physical constraints.
    Implements multiple rearrangement strategies.
    """
    
    def __init__(self, simulator):
        """Initialize the movement manager with a reference to the simulator."""
        self.simulator = simulator
        self.target_region = None
        
    def initialize_target_region(self):
        """Calculate and initialize the target region."""
        field_height, field_width = self.simulator.field_size
        side_length = self.simulator.side_length
        
        # Center the target region
        start_row = (field_height - side_length) // 2
        start_col = (field_width - side_length) // 2
        end_row = start_row + side_length
        end_col = start_col + side_length
        
        self.target_region = (start_row, start_col, end_row, end_col)
    
    def calculate_movement_time(self, distance: float) -> float:
        """Calculate movement time based on distance and acceleration constraints."""
        # t = 2 * sqrt(d/a) using constant acceleration model
        max_acceleration = self.simulator.constraints['max_acceleration']
        site_distance = self.simulator.constraints['site_distance']
        
        # Convert distance from lattice units to micrometers
        distance_um = distance * site_distance
        
        # Convert acceleration from m/s² to μm/s²
        acceleration_um = max_acceleration * 1e6
        
        # Calculate time in seconds: t = 2 * sqrt(d/a)
        return 2 * np.sqrt(distance_um / acceleration_um)

    
    def row_wise_centering(self, show_visualization=True, parallel=True):
        """
        Row-wise centering strategy for atom rearrangement.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            parallel: Whether to move atoms within a row in parallel
            
        Returns:
            Tuple of (final_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Process each row in the target region
        total_moves_made = 0
        for row in range(target_start_row, target_end_row):
            moves_made = self.center_atoms_in_row(row, target_start_col, target_end_col, parallel)
            total_moves_made += moves_made
            
        print(f"Row-wise centering complete: {total_moves_made} total moves made")
        
        # Calculate retention rate
        target_region_array = self.simulator.field[target_start_row:target_end_row, 
                                               target_start_col:target_end_col]
        atoms_in_target = np.sum(target_region_array)
        target_size = self.simulator.side_length * self.simulator.side_length
        retention_rate = atoms_in_target / target_size if target_size > 0 else 0
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        return self.simulator.target_lattice, retention_rate, execution_time
    
    def center_atoms_in_row(self, row, target_start_col, target_end_col, parallel=True):
        """
        Center atoms in a single row, moving from both sides toward the center.
        Ensures atoms on left side only move right and atoms on right side only move left.
        Each atom moves as close to the center as possible.
        
        Args:
            row: Row index to process
            target_start_col: Starting column of target region
            target_end_col: Ending column of target region
            parallel: Whether to move atoms in parallel
        
        Returns:
            Number of atoms successfully moved
        """
        # Find all atoms in this row
        atom_cols = [col for col in range(self.simulator.field_size[1]) 
                    if self.simulator.field[row, col] == 1]
        
        if not atom_cols:
            return 0  # No atoms in this row
        
        # Calculate the center column of the target region
        center_col = (target_start_col + target_end_col) // 2
        
        # Separate atoms into left side and right side
        left_atoms = sorted([col for col in atom_cols if col < center_col])  # Ascending (left to right)
        right_atoms = sorted([col for col in atom_cols if col >= center_col], reverse=True)  # Descending (right to left)
        
        # Calculate how many atoms we need in this row for the target region
        target_width = target_end_col - target_start_col
        max_atoms_in_region = min(len(atom_cols), target_width)
        
        print(f"Row {row}: Found {len(atom_cols)} atoms - {len(left_atoms)} on left, {len(right_atoms)} on right")
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        moves_executed = 0
        
        # Generate target positions for left side (from target_start_col to center_col-1)
        left_targets = list(range(target_start_col, center_col))
        
        # Generate target positions for right side (from center_col to target_end_col-1)
        right_targets = list(range(center_col, target_end_col))
        
        if parallel:
            # Parallel movement implementation
            # We'll collect all valid moves first, then execute them simultaneously
            all_moves = []
            max_distance = 0
            
            # Process left atoms - move them rightward toward the center
            # Sort left targets from center outward (descending)
            left_targets.sort(reverse=True)
            
            for target_col in left_targets:
                # Find the rightmost atom that's to the left of this target
                # and that can move to it (path is clear)
                best_atom_col = None
                
                for atom_col in left_atoms:
                    if atom_col >= target_col or working_field[row, atom_col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move right
                    path_clear = True
                    for col in range(atom_col + 1, target_col + 1):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_col = atom_col
                        break
                
                if best_atom_col is not None:
                    # Record this move for later execution
                    from_pos = (row, best_atom_col)
                    to_pos = (row, target_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field to reflect this pending move
                    working_field[row, best_atom_col] = 0
                    working_field[row, target_col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(target_col - best_atom_col)
                    max_distance = max(max_distance, distance)
                    
                    left_atoms.remove(best_atom_col)  # Remove this atom from consideration
            
            # Process right atoms - move them leftward toward the center
            # Sort right targets from center outward (ascending)
            right_targets.sort()
            
            for target_col in right_targets:
                # Find the leftmost atom that's to the right of this target
                # and that can move to it (path is clear)
                best_atom_col = None
                
                for atom_col in right_atoms:
                    if atom_col <= target_col or working_field[row, atom_col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move left
                    path_clear = True
                    for col in range(target_col, atom_col):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_col = atom_col
                        break
                
                if best_atom_col is not None:
                    # Record this move for later execution
                    from_pos = (row, best_atom_col)
                    to_pos = (row, target_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field to reflect this pending move
                    working_field[row, best_atom_col] = 0
                    working_field[row, target_col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(target_col - best_atom_col)
                    max_distance = max(max_distance, distance)
                    
                    right_atoms.remove(best_atom_col)  # Remove this atom from consideration
            
            # Execute all moves in parallel
            if all_moves:
                # Calculate time based on maximum distance
                move_time = self.calculate_movement_time(max_distance)
                
                print(f"Moving {len(all_moves)} atoms in row {row} in parallel, max distance: {max_distance}")
                
                # Record batch move in history
                self.simulator.movement_history.append({
                    'type': 'parallel_row_move',
                    'moves': all_moves,
                    'state': working_field.copy(),
                    'time': move_time,
                    'successful': len(all_moves),
                    'failed': 0
                })
                
                moves_executed = len(all_moves)
                
                # Update simulator's field with final state
                self.simulator.field = working_field.copy()
            
        else:
            # Original sequential implementation
            # Process left atoms - move them rightward toward the center
            # Sort left targets from center outward (descending)
            left_targets.sort(reverse=True)
            
            for target_col in left_targets:
                # Find the rightmost atom that's to the left of this target
                # and that can move to it (path is clear)
                best_atom_col = None
                
                for atom_col in left_atoms:
                    if atom_col >= target_col or working_field[row, atom_col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move right
                    path_clear = True
                    for col in range(atom_col + 1, target_col + 1):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_col = atom_col
                        break
                
                if best_atom_col is not None:
                    # Execute movement
                    from_pos = (row, best_atom_col)
                    to_pos = (row, target_col)
                    
                    # Update field
                    working_field[row, best_atom_col] = 0
                    working_field[row, target_col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(target_col - best_atom_col)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving left atom from {from_pos} to {to_pos}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'row_move',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
            
            # Process right atoms - move them leftward toward the center
            # Sort right targets from center outward (ascending)
            right_targets.sort()
            
            for target_col in right_targets:
                # Find the leftmost atom that's to the right of this target
                # and that can move to it (path is clear)
                best_atom_col = None
                
                for atom_col in right_atoms:
                    if atom_col <= target_col or working_field[row, atom_col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move left
                    path_clear = True
                    for col in range(target_col, atom_col):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_col = atom_col
                        break
                
                if best_atom_col is not None:
                    # Execute movement
                    from_pos = (row, best_atom_col)
                    to_pos = (row, target_col)
                    
                    # Update field
                    working_field[row, best_atom_col] = 0
                    working_field[row, target_col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(target_col - best_atom_col)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving right atom from {from_pos} to {to_pos}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'row_move',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
        
        # Verify final state in this row
        final_atom_cols = [col for col in range(self.simulator.field_size[1]) 
                         if self.simulator.field[row, col] == 1]
        print(f"Row {row}: Final atom positions {[(row, col) for col in final_atom_cols]}")
        
        return moves_executed
    
    def column_wise_centering(self, show_visualization=True, parallel=True):
        """
        Column-wise centering strategy for atom rearrangement.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            parallel: Whether to move atoms in parallel
        
        Returns:
            Tuple of (final_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Process each column in the target region
        total_moves_made = 0
        for col in range(target_start_col, target_end_col):
            moves_made = self.center_atoms_in_column(col, target_start_row, target_end_row, parallel)
            total_moves_made += moves_made
            
        print(f"Column-wise centering complete: {total_moves_made} total moves made")
        
        # Calculate retention rate
        target_region_array = self.simulator.field[target_start_row:target_end_row, 
                                               target_start_col:target_end_col]
        atoms_in_target = np.sum(target_region_array)
        target_size = self.simulator.side_length * self.simulator.side_length
        retention_rate = atoms_in_target / target_size if target_size > 0 else 0
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        return self.simulator.target_lattice, retention_rate, execution_time
    
    def center_atoms_in_column(self, col, target_start_row, target_end_row, parallel=True):
        """
        Center atoms in a single column, moving from both sides toward the center.
        Ensures atoms on top side only move down and atoms on bottom side only move up.
        Each atom moves as close to the center as possible.
        
        Args:
            col: Column index to process
            target_start_row: Starting row of target region
            target_end_row: Ending row of target region
            parallel: Whether to move atoms in parallel
        
        Returns:
            Number of atoms successfully moved
        """
        # Find all atoms in this column
        atom_rows = [row for row in range(self.simulator.field_size[0]) 
                    if self.simulator.field[row, col] == 1]
        
        if not atom_rows:
            return 0  # No atoms in this column
        
        # Calculate the center row of the target region
        center_row = (target_start_row + target_end_row) // 2
        
        # Separate atoms into top side and bottom side
        top_atoms = sorted([row for row in atom_rows if row < center_row])  # Ascending (top to bottom)
        bottom_atoms = sorted([row for row in atom_rows if row >= center_row], reverse=True)  # Descending (bottom to top)
        
        # Calculate how many atoms we need in this column for the target region
        target_height = target_end_row - target_start_row
        max_atoms_in_region = min(len(atom_rows), target_height)
        
        print(f"Column {col}: Found {len(atom_rows)} atoms - {len(top_atoms)} on top, {len(bottom_atoms)} on bottom")
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        moves_executed = 0
        
        # Generate target positions for top side (from target_start_row to center_row-1)
        top_targets = list(range(target_start_row, center_row))
        
        # Generate target positions for bottom side (from center_row to target_end_row-1)
        bottom_targets = list(range(center_row, target_end_row))
        
        if parallel:
            # Parallel movement implementation
            # We'll collect all valid moves first, then execute them simultaneously
            all_moves = []
            max_distance = 0
            
            # Process top atoms - move them downward toward the center
            # Sort top targets from center outward (descending)
            top_targets.sort(reverse=True)
            
            for target_row in top_targets:
                # Find the lowest atom that's above this target
                # and that can move to it (path is clear)
                best_atom_row = None
                
                for atom_row in top_atoms:
                    if atom_row >= target_row or working_field[atom_row, col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move down
                    path_clear = True
                    for row in range(atom_row + 1, target_row + 1):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_row = atom_row
                        break
                
                if best_atom_row is not None:
                    # Record this move for later execution
                    from_pos = (best_atom_row, col)
                    to_pos = (target_row, col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field to reflect this pending move
                    working_field[best_atom_row, col] = 0
                    working_field[target_row, col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(target_row - best_atom_row)
                    max_distance = max(max_distance, distance)
                    
                    top_atoms.remove(best_atom_row)  # Remove this atom from consideration
            
            # Process bottom atoms - move them upward toward the center
            # Sort bottom targets from center outward (ascending)
            bottom_targets.sort()
            
            for target_row in bottom_targets:
                # Find the highest atom that's below this target
                # and that can move to it (path is clear)
                best_atom_row = None
                
                for atom_row in bottom_atoms:
                    if atom_row <= target_row or working_field[atom_row, col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move up
                    path_clear = True
                    for row in range(target_row, atom_row):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_row = atom_row
                        break
                
                if best_atom_row is not None:
                    # Record this move for later execution
                    from_pos = (best_atom_row, col)
                    to_pos = (target_row, col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field to reflect this pending move
                    working_field[best_atom_row, col] = 0
                    working_field[target_row, col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(target_row - best_atom_row)
                    max_distance = max(max_distance, distance)
                    
                    bottom_atoms.remove(best_atom_row)  # Remove this atom from consideration
            
            # Execute all moves in parallel
            if all_moves:
                # Calculate time based on maximum distance
                move_time = self.calculate_movement_time(max_distance)
                
                print(f"Moving {len(all_moves)} atoms in column {col} in parallel, max distance: {max_distance}")
                
                # Record batch move in history
                self.simulator.movement_history.append({
                    'type': 'parallel_column_move',
                    'moves': all_moves,
                    'state': working_field.copy(),
                    'time': move_time,
                    'successful': len(all_moves),
                    'failed': 0
                })
                
                moves_executed = len(all_moves)
                
                # Update simulator's field with final state
                self.simulator.field = working_field.copy()
            
        else:
            # Sequential implementation
            # Process top atoms - move them downward toward the center
            # Sort top targets from center outward (descending)
            top_targets.sort(reverse=True)
            
            for target_row in top_targets:
                # Find the lowest atom that's above this target
                # and that can move to it (path is clear)
                best_atom_row = None
                
                for atom_row in top_atoms:
                    if atom_row >= target_row or working_field[atom_row, col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move down
                    path_clear = True
                    for row in range(atom_row + 1, target_row + 1):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_row = atom_row
                        break
                
                if best_atom_row is not None:
                    # Execute movement
                    from_pos = (best_atom_row, col)
                    to_pos = (target_row, col)
                    
                    # Update field
                    working_field[best_atom_row, col] = 0
                    working_field[target_row, col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(target_row - best_atom_row)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving top atom from {from_pos} to {to_pos}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'column_move',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
            
            # Process bottom atoms - move them upward toward the center
            # Sort bottom targets from center outward (ascending)
            bottom_targets.sort()
            
            for target_row in bottom_targets:
                # Find the highest atom that's below this target
                # and that can move to it (path is clear)
                best_atom_row = None
                
                for atom_row in bottom_atoms:
                    if atom_row <= target_row or working_field[atom_row, col] == 0:
                        # Skip if atom is already at or past target, or already moved
                        continue
                    
                    # Check if path is clear to move up
                    path_clear = True
                    for row in range(target_row, atom_row):
                        if working_field[row, col] == 1:
                            path_clear = False
                            break
                    
                    if path_clear:
                        # This atom can move to the target
                        best_atom_row = atom_row
                        break
                
                if best_atom_row is not None:
                    # Execute movement
                    from_pos = (best_atom_row, col)
                    to_pos = (target_row, col)
                    
                    # Update field
                    working_field[best_atom_row, col] = 0
                    working_field[target_row, col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(target_row - best_atom_row)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving bottom atom from {from_pos} to {to_pos}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'column_move',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
        
        # Verify final state in this column
        final_atom_rows = [row for row in range(self.simulator.field_size[0]) 
                         if self.simulator.field[row, col] == 1]
        print(f"Column {col}: Final atom positions {[(row, col) for row in final_atom_rows]}")
        
        return moves_executed

    def repair_defects(self, show_visualization=True):
        """
        Moves atoms from outside the target zone to fill defects within the target zone.
        Uses an optimized approach to minimize movement cost while maximizing fill rate.
        
        Note: This operation can only be implemented sequentially due to dependencies between moves.
        
        Args:
            show_visualization: Whether to visualize the reparation
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Find all defects in the target region
        defects = []
        for row in range(target_start_row, target_end_row):
            for col in range(target_start_col, target_end_col):
                if self.simulator.field[row, col] == 0:  # Empty site = defect
                    defects.append((row, col))
        
        print(f"Found {len(defects)} defects in the target region")
        
        if not defects:
            print("No defects to repair")
            return self.simulator.field.copy(), 1.0, 0.0  # Perfect fill rate
        
        # Find all available atoms outside the target region
        available_atoms = []
        for row in range(self.simulator.field_size[0]):
            for col in range(self.simulator.field_size[1]):
                if self.simulator.field[row, col] == 1:  # Found an atom
                    # Check if it's outside the target region
                    if not (target_start_row <= row < target_end_row and 
                            target_start_col <= col < target_end_col):
                        available_atoms.append((row, col))
        
        print(f"Found {len(available_atoms)} available atoms outside target region")
        
        if not available_atoms:
            fill_rate = 1.0 - (len(defects) / (self.simulator.side_length ** 2))
            print(f"No atoms available outside target region. Fill rate: {fill_rate:.2f}")
            return self.simulator.field.copy(), fill_rate, time.time() - start_time
        
        # Calculate center of target region for prioritizing central defects first
        center_row = (target_start_row + target_end_row) // 2
        center_col = (target_start_col + target_end_col) // 2
        
        # Sort defects by distance from center (closest first)
        defects.sort(key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
        
        # Sequential implementation with optimization
        moves_executed = 0
        working_field = self.simulator.field.copy()
        
        # Create a path cache to avoid redundant path calculations
        path_cache = {}
        
        for defect_pos in defects:
            defect_row, defect_col = defect_pos
            
            # Find best atom to move to this defect
            best_atom = None
            best_path = None
            best_cost = float('inf')  # Lower is better
            
            for atom_pos in available_atoms:
                atom_row, atom_col = atom_pos
                
                # Skip if this atom is already used
                if working_field[atom_row, atom_col] == 0:
                    continue
                
                # Try to find cached path first
                cache_key = (atom_pos, defect_pos)
                if cache_key in path_cache:
                    path, cost = path_cache[cache_key]
                    if path and cost < best_cost:
                        best_atom = atom_pos
                        best_path = path
                        best_cost = cost
                    continue
                
                # Calculate direct distance (for baseline cost comparison)
                distance = abs(defect_row - atom_row) + abs(defect_col - atom_col)
                
                # Skip atoms that are too far if we already have a good candidate
                if distance > best_cost * 1.5 and best_atom is not None:
                    continue
                
                # Find optimal path using A* search (simplified here)
                path = self.find_optimal_path(working_field, atom_pos, defect_pos)
                
                if path:
                    # Calculate path cost (length + complexity penalty)
                    cost = len(path) - 1  # Number of moves (path includes start and end)
                    
                    # Store in cache
                    path_cache[cache_key] = (path, cost)
                    
                    # Update best atom if this one has lower cost
                    if cost < best_cost:
                        best_atom = atom_pos
                        best_path = path
                        best_cost = cost
            
            # If we found a suitable atom, move it to the defect
            if best_atom and best_path:
                print(f"Fixing defect at {defect_pos} using atom at {best_atom}, cost: {best_cost}")
                
                # Execute the path movements
                for i in range(1, len(best_path)):
                    from_pos = best_path[i-1]
                    to_pos = best_path[i]
                    
                    # Skip first position if we're not at the start (atom already moved)
                    if i > 1:
                        working_field[from_pos] = 0
                    else:
                        # Remove atom from starting position
                        working_field[from_pos] = 0
                    
                    # Place atom at destination
                    working_field[to_pos] = 1
                    
                    # Calculate time based on Manhattan distance
                    move_distance = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
                    move_time = self.calculate_movement_time(move_distance)
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'defect_repair_step',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                
                moves_executed += 1
                
                # Remove this atom from available atoms
                available_atoms.remove(best_atom)
                
                # Update simulator's field
                self.simulator.field = working_field.copy()
            else:
                print(f"Could not find a usable atom for defect at {defect_pos}")
        
        # Calculate fill rate (percentage of target positions filled)
        target_size = self.simulator.side_length ** 2
        remaining_defects = 0
        
        for row in range(target_start_row, target_end_row):
            for col in range(target_start_col, target_end_col):
                if self.simulator.field[row, col] == 0:
                    remaining_defects += 1
        
        fill_rate = 1.0 - (remaining_defects / target_size)
        
        print(f"Defect repair complete: {len(defects) - remaining_defects} defects filled. Fill rate: {fill_rate:.2f}")
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        return self.simulator.target_lattice, fill_rate, execution_time
    
    def find_optimal_path(self, field, start_pos, end_pos):
        """
        Finds the optimal path from start_pos to end_pos using A* search.
        Only moves along rows and columns, and cannot pass through atoms.
        Limits the maximum path length to 3 steps for efficiency.
        
        Args:
            field: Current state of the lattice field
            start_pos: Starting position (row, col)
            end_pos: Target position (row, col)
            
        Returns:
            List of positions forming the path, or None if no path found
        """
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        
        # Direct path (straight line) if start and end are in same row or column
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
                return [start_pos, end_pos]  # Direct horizontal move
                
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
                return [start_pos, end_pos]  # Direct vertical move
        
        # Try two-step paths (horizontal then vertical or vertical then horizontal)
        
        # Horizontal then Vertical
        intermediate_pos1 = (start_row, end_col)
        if field[intermediate_pos1] == 0:  # Intermediate position must be empty
            # Check horizontal path
            h_path_clear = True
            start_c = min(start_col, end_col) + 1
            end_c = max(start_col, end_col)
            for col in range(start_c, end_c):
                if field[start_row, col] == 1:
                    h_path_clear = False
                    break
            
            # Check vertical path
            v_path_clear = True
            start_r = min(start_row, end_row) + 1
            end_r = max(start_row, end_row)
            for row in range(start_r, end_r):
                if field[row, end_col] == 1:
                    v_path_clear = False
                    break
            
            if h_path_clear and v_path_clear:
                return [start_pos, intermediate_pos1, end_pos]
        
        # Vertical then Horizontal
        intermediate_pos2 = (end_row, start_col)
        if field[intermediate_pos2] == 0:  # Intermediate position must be empty
            # Check vertical path
            v_path_clear = True
            start_r = min(start_row, end_row) + 1
            end_r = max(start_row, end_row)
            for row in range(start_r, end_r):
                if field[row, start_col] == 1:
                    v_path_clear = False
                    break
            
            # Check horizontal path
            h_path_clear = True
            start_c = min(start_col, end_col) + 1
            end_c = max(start_col, end_col)
            for col in range(start_c, end_c):
                if field[end_row, col] == 1:
                    h_path_clear = False
                    break
            
            if v_path_clear and h_path_clear:
                return [start_pos, intermediate_pos2, end_pos]
        
        # Try three-step paths using an intermediate point
        # This is the most complex case - try to find a free intermediate point that allows movement
        field_height, field_width = field.shape
        
        # Only consider intermediate points that are reasonable (not too far from the direct path)
        row_min = max(0, min(start_row, end_row) - 3)
        row_max = min(field_height, max(start_row, end_row) + 3)
        col_min = max(0, min(start_col, end_col) - 3)
        col_max = min(field_width, max(start_col, end_col) + 3)
        
        for int_row in range(row_min, row_max):
            for int_col in range(col_min, col_max):
                # Skip occupied positions or start/end positions
                intermediate = (int_row, int_col)
                if (field[intermediate] == 1 or 
                    intermediate == start_pos or 
                    intermediate == end_pos):
                    continue
                
                # Try path: start -> intermediate -> end
                path1 = self.find_direct_path(field, start_pos, intermediate)
                if not path1:
                    continue
                
                path2 = self.find_direct_path(field, intermediate, end_pos)
                if not path2:
                    continue
                
                # We found a valid 3-step path: combine paths (remove duplicate intermediate)
                return path1[:-1] + path2
        
        # No valid path found within 3 steps
        return None
    
    def find_direct_path(self, field, start_pos, end_pos):
        """Helper method to find a direct path between two points."""
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
        
    def combined_filling_strategy(self, show_visualization=True, parallel=True):
        """
        An optimized comprehensive filling strategy that combines multiple methods:
        1. First applies row-wise centering to create initial structure
        2. Then applies column-wise centering to improve the structure
        3. Next iteratively:
           a. Spreads atoms outside the target zone outward from center
           b. Applies column-wise centering to utilize the repositioned atoms
           c. Continues until no further improvement
        4. Finally repairs remaining defects from the center outwards
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            parallel: Whether to move atoms in parallel during operations
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        total_movement_history = []
        self.initialize_target_region()
        
        # Step 1: Row-wise centering - creates basic structure
        print("\n--- Step 1: Performing row-wise centering ---")
        self.simulator.movement_history = []
        row_lattice, row_retention, row_time = self.row_wise_centering(
            show_visualization=False,  # Don't show animation yet
            parallel=parallel
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        print(f"Row-wise centering completed with retention rate: {row_retention:.2%}")
        
        # Get target region to analyze current state
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Count defects after row-centering
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_row = np.sum(target_region == 0)
        print(f"Defects after row-centering: {defects_after_row}")
        
        if defects_after_row == 0:
            print("Perfect fill achieved after row centering!")
            execution_time = time.time() - start_time
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 2: Column-wise centering - improves structure
        print("\n--- Step 2: Performing column-wise centering ---")
        self.simulator.movement_history = []
        col_lattice, col_retention, col_time = self.column_wise_centering(
            show_visualization=False,  # Don't show animation yet
            parallel=parallel
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        print(f"Column-wise centering completed with retention rate: {col_retention:.2%}")
        
        # Count defects after column-centering
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_col = np.sum(target_region == 0)
        print(f"Defects after column-centering: {defects_after_col}")
        
        if defects_after_col == 0:
            print("Perfect fill achieved after column centering!")
            self.simulator.movement_history = total_movement_history
            execution_time = time.time() - start_time
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 3: Iterative spread-squeeze cycle
        print("\n--- Step 3: Beginning iterative spread-squeeze cycles ---")
        center_col = (target_start_col + target_end_col) // 2
        
        # Initialize tracking variables for the iteration
        max_iterations = 5  # Prevent infinite loops in edge cases
        min_improvement = 2  # Minimum number of defects that must be fixed to continue
        previous_defects = defects_after_col
        spread_squeeze_time = 0
        spread_squeeze_moves = 0
        iteration = 0
        
        # Continue iterations until no significant improvement or max iterations reached
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} of spread-squeeze cycle ---")
            
            # Spread atoms outward
            self.simulator.movement_history = []
            _, spread_moves, spread_time = self.spread_outer_atoms(
                show_visualization=False,  # Don't show animation yet
                parallel=parallel
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            spread_squeeze_moves += spread_moves
            spread_squeeze_time += spread_time
            
            # Column-wise centering on the spread atoms
            self.simulator.movement_history = []
            _, squeeze_retention, squeeze_time = self.column_wise_centering(
                show_visualization=False,  # Don't show animation yet
                parallel=parallel
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            spread_squeeze_time += squeeze_time
            
            # Count defects after this iteration
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            # Calculate improvement
            defects_fixed = previous_defects - current_defects
            
            print(f"Iteration {iteration} results:")
            print(f"  - Defects before: {previous_defects}")
            print(f"  - Defects after: {current_defects}")
            print(f"  - Defects fixed: {defects_fixed}")
            print(f"  - Current fill rate: {1 - (current_defects / (self.simulator.side_length**2)):.2%}")
            
            # Check if we've achieved perfect fill
            if current_defects == 0:
                print("Perfect fill achieved after spread-squeeze cycle!")
                self.simulator.movement_history = total_movement_history
                execution_time = time.time() - start_time
                return self.simulator.target_lattice, 1.0, execution_time
                
            # Check if we should continue
            if defects_fixed < min_improvement:
                print(f"Stopping iterations - insufficient improvement ({defects_fixed} < {min_improvement} defects fixed)")
                break
                
            # Update for next iteration
            previous_defects = current_defects
        
        print(f"Spread-squeeze cycles completed: {iteration} iterations")
        print(f"  - Total moves: {spread_squeeze_moves}")
        print(f"  - Total time: {spread_squeeze_time:.3f} seconds")
        print(f"  - Defects fixed: {defects_after_col - current_defects}")
        
        # Final count of defects after spread-squeeze cycles
        defects_after_cycles = previous_defects
        
        # Step 4: Repair remaining defects from center outwards
        print("\n--- Step 4: Repairing remaining defects ---")
        self.simulator.movement_history = []
        final_lattice, fill_rate, repair_time = self.repair_defects(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        print(f"Defect repair completed with fill rate: {fill_rate:.2%}")
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        total_time = row_time + col_time + spread_squeeze_time + repair_time
        
        # Restore complete movement history
        self.simulator.movement_history = total_movement_history
        
        print(f"\nCombined filling strategy completed:")
        print(f"  - Total execution time: {execution_time:.3f} seconds")
        print(f"  - Total movement time: {total_time:.3f} seconds")
        print(f"  - Final fill rate: {fill_rate:.2%}")
        print(f"  - Defects fixed by row centering: {defects_after_row} positions")
        print(f"  - Additional defects fixed by first column centering: {defects_after_row - defects_after_col} positions")
        print(f"  - Additional defects fixed by spread-squeeze cycles: {defects_after_col - defects_after_cycles} positions")
        print(f"  - Additional defects fixed by repair: {defects_after_cycles - (self.simulator.side_length**2 * (1-fill_rate))} positions")
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        return self.simulator.target_lattice, fill_rate, execution_time

    def spread_outer_atoms(self, show_visualization=True, parallel=True):
        """
        Spreads atoms in rows above and below the target zone outward from the center.
        Only processes atoms that are horizontally aligned with the target zone.
        Atoms left of the horizontal center move leftward, while atoms right of the center move rightward.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            parallel: Whether to move atoms within a row in parallel
            
        Returns:
            Tuple of (final_lattice, number_of_moves, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        target_width = target_end_col - target_start_col
        center_col = (target_start_col + target_end_col) // 2
        
        # Process rows above the target region
        total_moves_made = 0
        
        # First process rows above the target zone
        for row in range(0, target_start_row):
            moves_made = self.spread_atoms_in_row(row, target_start_col, target_end_col, center_col, parallel)
            total_moves_made += moves_made
            
        # Then process rows below the target zone
        for row in range(target_end_row, self.simulator.field_size[0]):
            moves_made = self.spread_atoms_in_row(row, target_start_col, target_end_col, center_col, parallel)
            total_moves_made += moves_made
            
        print(f"Outer atom spreading complete: {total_moves_made} total moves made")
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        return self.simulator.field.copy(), total_moves_made, execution_time
    
    def spread_atoms_in_row(self, row, target_start_col, target_end_col, center_col, parallel=True):
        """
        Moves atoms in a single row outside the target zone outward from the center.
        Only processes atoms that are horizontally aligned with the target zone.
        Atoms left of the horizontal center move leftward (but not beyond left target edge),
        atoms right of center move rightward (but not beyond right target edge).
        
        Args:
            row: Row index to process
            target_start_col: Starting column of target region (left edge)
            target_end_col: Ending column of target region (right edge)
            center_col: Center column of the target region
            parallel: Whether to move atoms in parallel
            
        Returns:
            Number of atoms successfully moved
        """
        # Find all atoms in this row that are horizontally aligned with the target zone
        atom_cols = [col for col in range(target_start_col, target_end_col) 
                    if self.simulator.field[row, col] == 1]
        
        if not atom_cols:
            return 0  # No atoms in this row within target zone horizontal bounds
        
        # Split atoms by center column
        left_atoms = [col for col in atom_cols if col < center_col]
        right_atoms = [col for col in atom_cols if col >= center_col]
        
        # If no atoms to move, return
        if not left_atoms and not right_atoms:
            return 0
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        moves_executed = 0
        
        if parallel:
            # Parallel movement implementation
            all_moves = []
            max_distance = 0
            
            # Process left atoms - move them leftward (away from center)
            # Sort from leftmost to rightmost to avoid collisions
            left_atoms.sort()  # Ascending
            
            # Track new positions to avoid collisions
            new_left_positions = set()
            
            for i, col in enumerate(left_atoms):
                # Calculate new position: move as far left as possible without collision
                # but not beyond the target_start_col (left edge of target zone)
                new_col = col
                # Move left until reaching target edge or finding an obstacle
                while new_col > target_start_col and working_field[row, new_col-1] == 0 and (new_col-1) not in new_left_positions:
                    new_col -= 1
                
                # Only add move if position changed
                if new_col != col:
                    from_pos = (row, col)
                    to_pos = (row, new_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Mark this position as used
                    new_left_positions.add(new_col)
                    
                    # Update working field for dependency checking
                    working_field[row, col] = 0
                    working_field[row, new_col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(new_col - col)
                    max_distance = max(max_distance, distance)
            
            # Process right atoms - move them rightward (away from center)
            # Sort from rightmost to leftmost to avoid collisions
            right_atoms.sort(reverse=True)  # Descending
            
            # Track new positions to avoid collisions
            new_right_positions = set()
            
            for i, col in enumerate(right_atoms):
                # Calculate new position: move as far right as possible without collision
                # but not beyond the target_end_col-1 (right edge of target zone)
                new_col = col
                field_width = self.simulator.field_size[1]
                while new_col < target_end_col - 1 and working_field[row, new_col+1] == 0 and (new_col+1) not in new_right_positions:
                    new_col += 1
                
                # Only add move if position changed
                if new_col != col:
                    from_pos = (row, col)
                    to_pos = (row, new_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Mark this position as used
                    new_right_positions.add(new_col)
                    
                    # Update working field for dependency checking
                    working_field[row, col] = 0
                    working_field[row, new_col] = 1
                    
                    # Track maximum distance for time calculation
                    distance = abs(new_col - col)
                    max_distance = max(max_distance, distance)
            
            # Execute all moves in parallel
            if all_moves:
                # Calculate time based on maximum distance
                move_time = self.calculate_movement_time(max_distance)
                
                print(f"Spreading {len(all_moves)} atoms in row {row} outward in parallel, max distance: {max_distance}")
                
                # Record batch move in history
                self.simulator.movement_history.append({
                    'type': 'parallel_outward_spread',
                    'moves': all_moves,
                    'state': working_field.copy(),
                    'time': move_time,
                    'successful': len(all_moves),
                    'failed': 0
                })
                
                moves_executed = len(all_moves)
                
                # Update simulator's field with final state
                self.simulator.field = working_field.copy()
        
        else:
            # Sequential implementation
            
            # Process left atoms - move them leftward (away from center)
            # Sort from leftmost to rightmost to avoid collisions
            left_atoms.sort()  # Ascending
            
            for col in left_atoms:
                # Calculate new position: move as far left as possible without collision
                # but not beyond the target_start_col (left edge of target zone)
                new_col = col
                while new_col > target_start_col and working_field[row, new_col-1] == 0:
                    new_col -= 1
                
                # Only execute if position changed
                if new_col != col:
                    from_pos = (row, col)
                    to_pos = (row, new_col)
                    
                    # Update field
                    working_field[row, col] = 0
                    working_field[row, new_col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(new_col - col)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving atom in row {row} leftward from column {col} to {new_col}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'left_outward_spread',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
            
            # Process right atoms - move them rightward (away from center)
            # Sort from rightmost to leftmost to avoid collisions
            right_atoms.sort(reverse=True)  # Descending
            
            for col in right_atoms:
                # Calculate new position: move as far right as possible without collision
                # but not beyond the target_end_col-1 (right edge of target zone)
                new_col = col
                while new_col < target_end_col - 1 and working_field[row, new_col+1] == 0:
                    new_col += 1
                
                # Only execute if position changed
                if new_col != col:
                    from_pos = (row, col)
                    to_pos = (row, new_col)
                    
                    # Update field
                    working_field[row, col] = 0
                    working_field[row, new_col] = 1
                    
                    # Calculate time based on distance
                    distance = abs(new_col - col)
                    move_time = self.calculate_movement_time(distance)
                    
                    print(f"Moving atom in row {row} rightward from column {col} to {new_col}")
                    
                    # Record move in history
                    self.simulator.movement_history.append({
                        'type': 'right_outward_spread',
                        'moves': [{'from': from_pos, 'to': to_pos}],
                        'state': working_field.copy(),
                        'time': move_time,
                        'successful': 1,
                        'failed': 0
                    })
                    
                    moves_executed += 1
        
        # Verify final state in this row
        final_atom_cols = [col for col in range(self.simulator.field_size[1]) 
                         if self.simulator.field[row, col] == 1]
        print(f"Row {row}: Final atom positions after outward spreading {[(row, col) for col in final_atom_cols]}")
        
        return moves_executed