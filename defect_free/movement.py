import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
import heapq

class MovementManager:
    """
    Manages atom movement with physical constraints.
    Implements multiple rearrangement strategies.
    """
    
    def __init__(self, simulator):
        """Initialize the movement manager with a reference to the simulator."""
        self.simulator = simulator
        self.target_region = None
        self._movement_time_cache = {}  # Cache for movement time calculations
        
    def initialize_target_region(self):
        """Calculate and initialize the target region."""
        if self.target_region is not None:
            return  # Already initialized
            
        field_height, field_width = self.simulator.field_size
        side_length = self.simulator.side_length
        
        # Center the target region
        start_row = (field_height - side_length) // 2
        start_col = (field_width - side_length) // 2
        end_row = start_row + side_length
        end_col = start_col + side_length
        
        self.target_region = (start_row, start_col, end_row, end_col)
    
    def calculate_movement_time(self, distance: float) -> float:
        """Calculate movement time based on distance and acceleration constraints with caching."""
        # Check cache first
        if distance in self._movement_time_cache:
            return self._movement_time_cache[distance]
            
        # t = 2 * sqrt(d/a) using constant acceleration model
        max_acceleration = self.simulator.constraints['max_acceleration']
        site_distance = self.simulator.constraints['site_distance']
        
        # Convert distance from lattice units to micrometers
        distance_um = distance * site_distance
        
        # Convert acceleration from m/s² to μm/s²
        acceleration_um = max_acceleration * 1e6
        
        # Calculate time in seconds: t = 2 * sqrt(d/a)
        time_value = 2 * np.sqrt(distance_um / acceleration_um)
        
        # Cache the result
        self._movement_time_cache[distance] = time_value
        
        return time_value

    def calculate_realistic_movement_time(self, distance: float) -> float:
        """Calculate movement time with a trapezoidal velocity profile, respecting quantum speed limits."""
        # Check cache first
        cache_key = f"trapezoid_{distance}"
        if cache_key in self._movement_time_cache:
            return self._movement_time_cache[cache_key]
        
        # Physical constants
        max_acceleration = self.simulator.constraints['max_acceleration']  # m/s²
        max_velocity = self.simulator.constraints.get('max_velocity', 0.3)  # m/s (0.3 m/s from Pagano et al. 2024)
        site_distance = self.simulator.constraints['site_distance']  # μm
        settling_time = self.simulator.constraints.get('settling_time', 1e-6)  # seconds (microsecond)
        
        # Quantum speed limit constraints from research
        quantum_speed_limit = self.simulator.constraints.get('quantum_speed_limit', 10e-6)  # 10 μs (from Pagano 2024)
        quantum_speed_reference_distance = self.simulator.constraints.get('quantum_speed_reference_distance', 3.0)  # 3 μm
        
        # Convert distance from lattice units to meters
        distance_m = distance * site_distance * 1e-6
        distance_um = distance * site_distance
        
        # Quantum speed limit scaling - if we move more than the reference distance, 
        # we need to scale the minimum time appropriately
        minimum_time = (distance_um / quantum_speed_reference_distance) * quantum_speed_limit
        
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
            # We accelerate until midpoint, then decelerate
            kinematic_time = 2 * np.sqrt(distance_m / max_acceleration)
        
        # Add settling time
        kinematic_time += settling_time
        
        # Apply quantum speed limit - cannot move faster than quantum mechanics allows
        total_time = max(kinematic_time, minimum_time)
        
        # Cache and return result
        self._movement_time_cache[cache_key] = total_time
        return total_time
        
    def axis_wise_centering(self, axis='row', show_visualization=True):
        """
        Unified axis-wise centering strategy for atom rearrangement.
        
        Args:
            axis: 'row' or 'column' to specify centering direction
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Process each line (row or column) in the target region
        total_moves_made = 0
        
        if (axis == 'row'):
            for row in range(target_start_row, target_end_row):
                moves_made = self.center_atoms_in_line(
                    line_idx=row, 
                    is_row=True,
                    target_start_idx=target_start_col, 
                    target_end_idx=target_end_col
                )
                total_moves_made += moves_made
        else:  # column
            for col in range(target_start_col, target_end_col):
                moves_made = self.center_atoms_in_line(
                    line_idx=col, 
                    is_row=False,
                    target_start_idx=target_start_row, 
                    target_end_idx=target_end_row
                )
                total_moves_made += moves_made
        
        # Calculate retention rate as atoms in target / loaded atoms
        target_region_array = self.simulator.field[target_start_row:target_end_row, 
                                           target_start_col:target_end_col]
        atoms_in_target = np.sum(target_region_array)
        retention_rate = atoms_in_target / self.simulator.total_atoms if self.simulator.total_atoms > 0 else 0
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        # Calculate total physical time from movement history
        physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"{axis.capitalize()}-wise centering complete in {execution_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
        
        return self.simulator.target_lattice, retention_rate, execution_time
    
    def apply_transport_efficiency(self, moves, working_field):
        """
        Apply transport efficiency to a list of moves.
        Some atoms may be lost during transport based on the atom_loss_probability.
        
        Args:
            moves: List of move dictionaries with 'from' and 'to' positions
            working_field: Current state of the field
            
        Returns:
            Tuple of (updated_field, successful_moves, failed_moves)
        """
        # Get the transport success probability (1 - loss probability)
        success_probability = 1.0 - self.simulator.constraints.get('atom_loss_probability', 0.05)
        
        successful_moves = []
        failed_moves = []
        
        # Make a copy of the field to work with
        updated_field = working_field.copy()
        
        for move in moves:
            from_pos = move['from']
            to_pos = move['to']
            
            # Check if the atom is still at the from_position (should be, but verify)
            if updated_field[from_pos] == 0:
                continue
                
            # Apply probabilistic transport check
            if np.random.random() < success_probability:
                # Success: atom moves to new position
                updated_field[from_pos] = 0
                updated_field[to_pos] = 1
                successful_moves.append(move)
            else:
                # Failure: atom is lost during transport
                updated_field[from_pos] = 0  # Remove the atom
                failed_moves.append(move)
        
        return updated_field, successful_moves, failed_moves
    
    def center_atoms_in_line(self, line_idx, is_row, target_start_idx, target_end_idx):
        """
        Center atoms in a single row or column, moving from both sides toward the center.
        
        Args:
            line_idx: Row or column index to process
            is_row: True if processing a row, False if processing a column
            target_start_idx: Starting index of target region in the relevant dimension
            target_end_idx: Ending index of target region in the relevant dimension
        
        Returns:
            Number of atoms successfully moved
        """
        # Find all atoms in this line
        if is_row:
            atom_indices = np.where(self.simulator.field[line_idx, :] == 1)[0]
        else:
            atom_indices = np.where(self.simulator.field[:, line_idx] == 1)[0]
            
        if len(atom_indices) == 0:
            return 0  # No atoms in this line
        
        # Calculate the center of the target region
        center_idx = (target_start_idx + target_end_idx) // 2
        
        # Separate atoms into left side and right side
        left_atoms = sorted([idx for idx in atom_indices if idx < center_idx])  # Ascending
        right_atoms = sorted([idx for idx in atom_indices if idx >= center_idx], reverse=True)  # Descending
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all valid moves first, then execute them simultaneously
        all_moves = []
        max_distance = 0
        
        # Process left atoms - move them toward the center
        # Sort left targets from center outward (descending)
        left_targets = list(range(target_start_idx, center_idx))
        left_targets.sort(reverse=True)
        
        for target_idx in left_targets:
            # Find the rightmost atom that's to the left of this target
            # and that can move to it (path is clear)
            best_atom_idx = None
            
            for atom_idx in left_atoms:
                if atom_idx >= target_idx or working_field[line_idx if is_row else atom_idx, 
                                           atom_idx if is_row else line_idx] == 0:
                    # Skip if atom is already at or past target, or already moved
                    continue
                
                # Check if path is clear to move right/down
                path_clear = True
                for i in range(atom_idx + 1, target_idx + 1):
                    if working_field[line_idx if is_row else i, 
                                     i if is_row else line_idx] == 1:
                        path_clear = False
                        break
                
                if path_clear:
                    # This atom can move to the target
                    best_atom_idx = atom_idx
                    break
            
            if best_atom_idx is not None:
                # Record this move for later execution
                from_pos = (line_idx, best_atom_idx) if is_row else (best_atom_idx, line_idx)
                to_pos = (line_idx, target_idx) if is_row else (target_idx, line_idx)
                all_moves.append({'from': from_pos, 'to': to_pos})
                
                # Update working field to reflect this pending move
                working_field[from_pos] = 0
                working_field[to_pos] = 1
                
                # Track maximum distance for time calculation
                distance = abs(target_idx - best_atom_idx)
                max_distance = max(max_distance, distance)
                
                left_atoms.remove(best_atom_idx)  # Remove this atom from consideration
        
        # Process right atoms - move them toward the center
        # Sort right targets from center outward (ascending)
        right_targets = list(range(center_idx, target_end_idx))
        right_targets.sort()
        
        for target_idx in right_targets:
            # Find the leftmost atom that's to the right of this target
            # and that can move to it (path is clear)
            best_atom_idx = None
            
            for atom_idx in right_atoms:
                if atom_idx <= target_idx or working_field[line_idx if is_row else atom_idx, 
                                           atom_idx if is_row else line_idx] == 0:
                    # Skip if atom is already at or past target, or already moved
                    continue
                
                # Check if path is clear to move left/up
                path_clear = True
                for i in range(target_idx, atom_idx):
                    if working_field[line_idx if is_row else i, 
                                     i if is_row else line_idx] == 1:
                        path_clear = False
                        break
                
                if path_clear:
                    # This atom can move to the target
                    best_atom_idx = atom_idx
                    break
            
            if best_atom_idx is not None:
                # Record this move for later execution
                from_pos = (line_idx, best_atom_idx) if is_row else (best_atom_idx, line_idx)
                to_pos = (line_idx, target_idx) if is_row else (target_idx, line_idx)
                all_moves.append({'from': from_pos, 'to': to_pos})
                
                # Update working field to reflect this pending move
                working_field[from_pos] = 0
                working_field[to_pos] = 1
                
                # Track maximum distance for time calculation
                distance = abs(target_idx - best_atom_idx)
                max_distance = max(max_distance, distance)
                
                right_atoms.remove(best_atom_idx)  # Remove this atom from consideration
        
        # Execute all moves in parallel
        if all_moves:
            # Calculate time based on maximum distance
            move_time = self.calculate_realistic_movement_time(max_distance)
            
            # Apply transport efficiency to the moves
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            
            # Record batch move in history
            move_type = 'parallel_row_move' if is_row else 'parallel_column_move'
            self.simulator.movement_history.append({
                'type': move_type,
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        return len(all_moves)
    
    def row_wise_centering(self, show_visualization=True):
        """Row-wise centering strategy (calls the unified axis_wise_centering)."""
        return self.axis_wise_centering(axis='row', show_visualization=show_visualization)
    
    def column_wise_centering(self, show_visualization=True):
        """Column-wise centering strategy (calls the unified axis_wise_centering)."""
        return self.axis_wise_centering(axis='column', show_visualization=show_visualization)
        
    def spread_outer_atoms(self, show_visualization=True):
        """
        Spreads atoms in rows above and below the target zone outward from the center.
        Only processes atoms that are horizontally aligned with the target zone.
        Atoms left of the horizontal center move leftward, while atoms right of the center move rightward.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
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
            moves_made = self.spread_atoms_in_row(row, target_start_col, target_end_col, center_col)
            total_moves_made += moves_made
            
        # Then process rows below the target zone
        for row in range(target_end_row, self.simulator.field_size[0]):
            moves_made = self.spread_atoms_in_row(row, target_start_col, target_end_col, center_col)
            total_moves_made += moves_made
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        # Calculate total physical time from movement history
        physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Spread operation complete in {execution_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
        
        return self.simulator.field.copy(), total_moves_made, execution_time
    
    def spread_atoms_in_row(self, row, target_start_col, target_end_col, center_col):
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
        
        # Parallel movement implementation - always used now
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
            move_time = self.calculate_realistic_movement_time(max_distance)
            
            # Apply transport efficiency to the moves
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            
            # Record batch move in history
            self.simulator.movement_history.append({
                'type': 'parallel_outward_spread',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            moves_executed = len(successful_moves)
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        return moves_executed

    def move_corner_blocks(self, show_visualization=True):
        """
        Moves atoms from corner regions into clean zones above or below target zone.
        First checks which groups of corner blocks can move directly, and moves them in parallel:
        - If all four corners can move, move them all at once
        - Otherwise try to move upper, lower, left or right corner pairs in parallel
        - Then moves any remaining individual corners that can move directly
        - Only then attempts to clean obstacles for the remaining corner blocks.
        Preserves the shape of corner blocks by moving them as a unit.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, moves_made, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        center_col = (target_start_col + target_end_col) // 2
        
        # Get field dimensions
        field_height, field_width = self.simulator.field_size
        
        # Calculate corner block dimensions
        initial_height, initial_width = self.simulator.initial_size
        initial_side = min(initial_height, initial_width)
        side_diff = initial_side - self.simulator.side_length
        corner_width = side_diff // 2
        
        if corner_width <= 0:
            print("No corner blocks to move (initial size <= target size)")
            return self.simulator.field.copy(), 0, 0.0
        
        # Define corner regions based on the projection of the target zone
        corner_regions = {
            'upper_left': {
                'start_row': 0,
                'start_col': 0,
                'end_row': target_start_row,
                'end_col': target_start_col
            },
            'upper_right': {
                'start_row': 0,
                'start_col': target_end_col,
                'end_row': target_start_row,
                'end_col': field_width
            },
            'lower_left': {
                'start_row': target_end_row,
                'start_col': 0,
                'end_row': field_height,
                'end_col': target_start_col
            },
            'lower_right': {
                'start_row': target_end_row,
                'start_col': target_end_col,
                'end_row': field_height,
                'end_col': field_width
            }
        }
        
        # Find atoms in corner regions
        corners = {
            'upper_left': [],
            'upper_right': [],
            'lower_left': [],
            'lower_right': []
        }
        
        # Find atoms in all corners
        for corner_name, region in corner_regions.items():
            for row in range(region['start_row'], region['end_row']):
                for col in range(region['start_col'], region['end_col']):
                    if self.simulator.field[row, col] == 1:
                        corners[corner_name].append((row, col))
        
        # Count total atoms in corners
        total_corner_atoms = sum(len(atoms) for atoms in corners.values())
        print(f"Found {total_corner_atoms} atoms in corner regions")
        
        if total_corner_atoms == 0:
            print("No atoms in corner regions")
            return self.simulator.field.copy(), 0, 0.0
        
        # Calculate movement offsets for each corner
        offset_map = {
            'upper_left': (0, corner_width),     # Move right
            'upper_right': (0, -corner_width),   # Move left
            'lower_left': (0, corner_width),     # Move right
            'lower_right': (0, -corner_width)    # Move left
        }
        
        # Prepare for movement
        working_field = self.simulator.field.copy()
        total_moves_made = 0
        
        # PHASE 1: Check which corners can move directly (no obstacles)
        movable_corners = {}
        corner_obstacles = {}
        
        for corner_name, corner_atoms in corners.items():
            if not corner_atoms:
                continue  # Skip empty corners
                
            offset_row, offset_col = offset_map[corner_name]
            can_move = True
            obstacles = []
            
            # Check if the destination area is clear
            for atom_pos in corner_atoms:
                row, col = atom_pos
                new_row, new_col = row + offset_row, col + offset_col
                
                # Check if destination is within field bounds
                if (new_row < 0 or new_row >= field_height or 
                    new_col < 0 or new_col >= field_width):
                    can_move = False
                    break
                
                # Check if destination is occupied by an atom not part of this corner
                if working_field[new_row, new_col] == 1 and (new_row, new_col) not in corner_atoms:
                    can_move = False
                    obstacles.append((new_row, new_col))
            
            if can_move:
                movable_corners[corner_name] = corner_atoms
            else:
                corner_obstacles[corner_name] = obstacles
        
        # PHASE 2: Group movable corners and move them in parallel
        # Define groups based on spatial arrangement
        corner_groups = {
            'all': ['upper_left', 'upper_right', 'lower_left', 'lower_right'],
            'upper': ['upper_left', 'upper_right'],
            'lower': ['lower_left', 'lower_right'],
            'left': ['upper_left', 'lower_left'],
            'right': ['upper_right', 'lower_right']
        }
        
        # Check which groups can be moved
        movable_groups = {}
        for group_name, corner_names in corner_groups.items():
            # Check if all corners in this group can be moved
            if all(corner_name in movable_corners for corner_name in corner_names):
                movable_groups[group_name] = [
                    (corner_name, movable_corners[corner_name]) 
                    for corner_name in corner_names
                ]
        
        # Keep track of which corners have been moved
        moved_corners = set()
        
        # Choose the largest movable group - prioritize 'all' if possible
        chosen_group = None
        if 'all' in movable_groups:
            chosen_group = movable_groups['all']
            print("Moving all four corner blocks in parallel")
            moved_corners.update(['upper_left', 'upper_right', 'lower_left', 'lower_right'])
        elif any(group in movable_groups for group in ['upper', 'lower', 'left', 'right']):
            # Choose the largest available group (they should all be the same size - 2)
            for group_name in ['upper', 'lower', 'left', 'right']:
                if group_name in movable_groups:
                    chosen_group = movable_groups[group_name]
                    print(f"Moving {group_name} corner blocks in parallel")
                    # Fix: Properly extract corner_name from each tuple in chosen_group
                    moved_corners.update(corner_name for corner_name, _ in chosen_group)
                    break
        
        # Move the chosen group in parallel if one exists
        if chosen_group:
            all_moves = []
            max_distance = 0
            
            for corner_name, corner_atoms in chosen_group:
                offset_row, offset_col = offset_map[corner_name]
                direction = "right" if offset_col > 0 else "left"
                print(f"Moving {corner_name} corner block {direction} by {abs(offset_col)} positions")
                
                # Add moves for this corner
                for atom_pos in corner_atoms:
                    row, col = atom_pos
                    new_row, new_col = row + offset_row, col + offset_col
                    
                    from_pos = (row, col)
                    to_pos = (new_row, new_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field
                    working_field[row, col] = 0
                    working_field[new_row, new_col] = 1
                    
                    # Track maximum distance
                    distance = abs(offset_col)
                    max_distance = max(max_distance, distance)
            
            # Record the parallel move in history
            if all_moves:
                move_time = self.calculate_realistic_movement_time(max_distance)
                
                # Apply transport efficiency to the moves
                updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                    all_moves, self.simulator.field
                )
                
                group_type = 'all_corners' if len(moved_corners) == 4 else '_'.join(moved_corners)
                self.simulator.movement_history.append({
                    'type': f'parallel_{group_type}_move',
                    'moves': successful_moves + failed_moves,  # Record all attempted moves
                    'state': updated_field.copy(),
                    'time': move_time,
                    'successful': len(successful_moves),
                    'failed': len(failed_moves)
                })
                total_moves_made += len(successful_moves)
                
                # Update simulator's field
                self.simulator.field = updated_field.copy()
        else:
            print("No corner blocks can be moved in groups")
        
        # PHASE 3: Move any remaining individual corners that can be moved directly
        # Find all movable corners that haven't been moved yet
        remaining_movable = {name: atoms for name, atoms in movable_corners.items() 
                             if name not in moved_corners}
        
        if remaining_movable:
            print(f"\nMoving {len(remaining_movable)} remaining individual movable corners")
            
            # Process each movable corner one by one
            for corner_name, corner_atoms in remaining_movable.items():
                offset_row, offset_col = offset_map[corner_name]
                direction = "right" if offset_col > 0 else "left"
                print(f"Moving {corner_name} corner block {direction} by {abs(offset_col)} positions")
                
                corner_moves = []
                
                # Add moves for this corner
                for atom_pos in corner_atoms:
                    row, col = atom_pos
                    new_row, new_col = row + offset_row, col + offset_col
                    
                    from_pos = (row, col)
                    to_pos = (new_row, new_col)
                    corner_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field
                    working_field[row, col] = 0
                    working_field[new_row, new_col] = 1
                
                # Record the move in history
                if corner_moves:
                    move_time = self.calculate_realistic_movement_time(abs(offset_col))
                    
                    # Apply transport efficiency to the moves
                    updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                        corner_moves, self.simulator.field
                    )
                    
                    self.simulator.movement_history.append({
                        'type': f'move_{corner_name}_block',
                        'moves': successful_moves + failed_moves,  # Record all attempted moves
                        'state': updated_field.copy(),
                        'time': move_time,
                        'successful': len(successful_moves),
                        'failed': len(failed_moves)
                    })
                    total_moves_made += len(successful_moves)
                    
                    # Update simulator's field
                    self.simulator.field = updated_field.copy()
                    
                    # Mark this corner as moved
                    moved_corners.add(corner_name)
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        # Calculate total physical time from movement history
        physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Corner block movement complete in {execution_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
        
        return self.simulator.field.copy(), total_moves_made, execution_time
    

    def find_direct_path(self, field, start_pos, end_pos):
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
    
    def find_l_shaped_path(self, field, start_pos, end_pos):
        """
        Find L-shaped path with one turn (horizontal then vertical or vertical then horizontal).
        Returns the path if found, None otherwise.
        """
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        
        # Try horizontal then vertical (clockwise L)
        intermediate_pos1 = (start_row, end_col)
        if intermediate_pos1 != start_pos and intermediate_pos1 != end_pos and (
                field[intermediate_pos1] == 0 or intermediate_pos1 == end_pos):
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
        
        # Try vertical then horizontal (counter-clockwise L)
        intermediate_pos2 = (end_row, start_col)
        if intermediate_pos2 != start_pos and intermediate_pos2 != end_pos and (
                field[intermediate_pos2] == 0 or intermediate_pos2 == end_pos):
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
    
    def find_a_star_path(self, field, start_pos, end_pos, max_iterations=1000):
        """
        Implements A* search algorithm to find the shortest path.
        
        Args:
            field: Current state of the lattice field
            start_pos: Starting position (row, col)
            end_pos: Target position (row, col)
            max_iterations: Maximum number of positions to examine before giving up
            
        Returns:
            List of positions forming the path, or None if no path found
        """
        start_row, start_col = start_pos
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
                    next_pos not in closed_set):
                    
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
    
    def compress_path(self, path):
        """
        Compresses consecutive horizontal or vertical movements in a path into single steps.
        
        Args:
            path: List of positions forming the path
            
        Returns:
            Compressed path with fewer, larger steps
        """
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
                    
    def find_optimal_path(self, field, start_pos, end_pos):
        """
        Finds the optimal path from start_pos to end_pos using a tiered approach:
        1. Direct path (if possible)
        2. L-shaped path (if possible)
        3. A* search for complex paths
        
        Args:
            field: Current state of the lattice field
            start_pos: Starting position (row, col)
            end_pos: Target position (row, col)
            
        Returns:
            List of positions forming the path, or None if no path found
        """
        # Try direct path first (same row or column)
        direct_path = self.find_direct_path(field, start_pos, end_pos)
        if direct_path:
            return direct_path
        
        # Try L-shaped path (1 turn)
        l_shaped_path = self.find_l_shaped_path(field, start_pos, end_pos)
        if l_shaped_path:
            return l_shaped_path
        
        # If simpler paths fail, use A* search for complex paths
        a_star_path = self.find_a_star_path(field, start_pos, end_pos)
        return self.compress_path(a_star_path)
    
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
        defects_fixed = 0
        working_field = self.simulator.field.copy()
        
        # Create a path cache to avoid redundant path calculations
        path_cache = {}
        
        # Helper function to execute a path with transport efficiency
        def execute_path(path, working_field):
            """
            Execute a path of atom movements with transport efficiency.
            
            Args:
                path: List of positions forming the path
                working_field: Current state of the field
                
            Returns:
                (success, updated_field) - success indicates if the atom reached the final destination
            """
            if len(path) <= 1:
                return False, working_field
                
            current_field = working_field.copy()
            
            # Execute each step in the path
            for i in range(1, len(path)):
                from_pos = path[i-1]
                to_pos = path[i]
                
                # Create a move dictionary for this step
                move = {'from': from_pos, 'to': to_pos}
                
                # Apply transport efficiency
                updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                    [move], current_field
                )
                
                # If the move failed (atom lost), we can't continue this path
                if failed_moves:
                    # Calculate time based on Manhattan distance
                    move_distance = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
                    move_time = self.calculate_realistic_movement_time(move_distance)
                    
                    # Record the failed move
                    self.simulator.movement_history.append({
                        'type': 'defect_repair_step',
                        'moves': failed_moves,
                        'state': updated_field.copy(),
                        'time': move_time,
                        'successful': 0,
                        'failed': 1
                    })
                    
                    # Return failure and the updated field
                    return False, updated_field
                
                # Move was successful, update current field
                current_field = updated_field
                
                # Calculate time based on Manhattan distance
                move_distance = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
                move_time = self.calculate_realistic_movement_time(move_distance)
                
                # Record move in history
                self.simulator.movement_history.append({
                    'type': 'defect_repair_step',
                    'moves': successful_moves,
                    'state': current_field.copy(),
                    'time': move_time,
                    'successful': len(successful_moves),
                    'failed': 0
                })
            
            # If we completed the path, return success and the updated field
            return True, current_field
        
        # Process each defect
        for defect_pos in defects:
            defect_row, defect_col = defect_pos
            
            # Skip if we've already filled this defect in a previous iteration
            if working_field[defect_row, defect_col] == 1:
                continue
            
            # Find best atom to move to this defect
            best_atom = None
            best_path = None
            best_cost = float('inf')  # Lower is better
            
            # Create a copy of available_atoms to safely remove atoms during iteration
            atoms_to_consider = available_atoms.copy()
            
            for atom_pos in atoms_to_consider:
                atom_row, atom_col = atom_pos
                
                # Skip if this atom is already used (no longer in the field)
                if working_field[atom_row, atom_col] == 0:
                    available_atoms.remove(atom_pos)
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
                
                # Find optimal path using enhanced path finding (tiered approach)
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
                # Execute the path movements with transport efficiency
                success, updated_field = execute_path(best_path, working_field)
                
                # Update the working field regardless of success
                working_field = updated_field
                
                # Update simulator's field
                self.simulator.field = working_field.copy()
                
                # If successful, increment counters and update available atoms
                if success:
                    moves_executed += 1
                    defects_fixed += 1
                    
                    # Remove this atom from available atoms
                    if best_atom in available_atoms:
                        available_atoms.remove(best_atom)
                else:
                    # If transport failed, the atom is gone from both the field and our tracking
                    if best_atom in available_atoms:
                        available_atoms.remove(best_atom)
        
        # Calculate fill rate (percentage of target positions filled)
        target_size = self.simulator.side_length ** 2
        remaining_defects = 0
        
        for row in range(target_start_row, target_end_row):
            for col in range(target_start_col, target_end_col):
                if self.simulator.field[row, col] == 0:
                    remaining_defects += 1
        
        fill_rate = 1.0 - (remaining_defects / target_size)
        
        print(f"Defect repair complete: {defects_fixed} defects filled. Fill rate: {fill_rate:.2f}")
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        # Calculate total physical time from movement history
        physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Defect repair complete in {execution_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
        
        return self.simulator.target_lattice, fill_rate, execution_time

    def combined_filling_strategy(self, show_visualization=True):
        """
        An optimized comprehensive filling strategy that combines multiple methods:
        1. First applies row-wise centering to create initial structure
        2. Then applies column-wise centering to improve the structure
        3. Next iteratively:
            a. Spreads atoms outside the target zone outward from center
            b. Applies column-wise centering to utilize the repositioned atoms
            c. Continues until no further improvement
        4. Then moves corner blocks into clean zones
        5. Applies column-wise centering to move corner atoms into the target zone
        6. First repair attempt for remaining defects
        7. If defects remain, iteratively:
            a. Apply full squeezing (row-wise + column-wise centering)
            b. Repair remaining defects
            c. Continue until perfect fill or no improvement
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        total_movement_history = []
        self.initialize_target_region()
        print("\nCombined filling strategy starting...")
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Check if target zone is already defect-free
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        initial_defects = np.sum(target_region == 0)
        if initial_defects == 0:
            print("Target zone is already defect-free! No movements needed.")
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Show visualization even if no changes were made
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 1: Row-wise centering - creates basic structure
        print("\nStep 1: Applying row-wise centering...")
        row_start_time = time.time()
        self.simulator.movement_history = []
        row_lattice, row_retention, row_time = self.row_wise_centering(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_row_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Row-wise centering complete in {time.time() - row_start_time:.3f} seconds, physical time: {physical_row_time:.6f} seconds")
        
        # Get target region to analyze current state
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Count defects after row-centering
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_row = np.sum(target_region == 0)
        print(f"Defects after row-centering: {defects_after_row}")
        
        if defects_after_row == 0:
            print("Perfect arrangement achieved after row-centering!")
            self.simulator.movement_history = total_movement_history
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 2: Column-wise centering - improves structure
        print("\nStep 2: Applying column-wise centering...")
        col_start_time = time.time()
        self.simulator.movement_history = []
        col_lattice, col_retention, col_time = self.column_wise_centering(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_col_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Column-wise centering complete in {time.time() - col_start_time:.3f} seconds, physical time: {physical_col_time:.6f} seconds")
        
        # Count defects after column-centering
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_col = np.sum(target_region == 0)
        print(f"Defects after column-centering: {defects_after_col}")
        
        if defects_after_col == 0:
            print("Perfect arrangement achieved after column-centering!")
            self.simulator.movement_history = total_movement_history
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 3: Iterative spread-squeeze cycle
        print("\nStep 3: Starting iterative spread-squeeze cycles...")
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
            print(f"\nSpread-squeeze cycle {iteration}/{max_iterations}...")
            
            # Spread atoms outward
            spread_start_time = time.time()
            self.simulator.movement_history = []
            _, spread_moves, spread_time = self.spread_outer_atoms(
                show_visualization=False  # Don't show animation yet
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time from movement history
            physical_spread_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Spread phase complete: {spread_moves} atoms moved in {time.time() - spread_start_time:.3f} seconds, physical time: {physical_spread_time:.6f} seconds")
            
            spread_squeeze_moves += spread_moves
            spread_squeeze_time += spread_time
            
            # First apply row-wise centering to align atoms if atom loss probability != 0
            if self.simulator.ATOM_LOSS_PROBABILITY > 0:
                row_squeeze_start_time = time.time()
                self.simulator.movement_history = []
                _, row_squeeze_retention, row_squeeze_time = self.row_wise_centering(
                    show_visualization=False  # Don't show animation yet
                )
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)

                # Calculate total physical time from movement history
                physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Row squeeze phase complete in {time.time() - row_squeeze_start_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")

                spread_squeeze_time += row_squeeze_time
            
            # Then apply column-wise centering
            col_squeeze_start_time = time.time()
            self.simulator.movement_history = []
            _, col_squeeze_retention, col_squeeze_time = self.column_wise_centering(
                show_visualization=False  # Don't show animation yet
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time from movement history
            physical_col_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Column squeeze phase complete in {time.time() - col_squeeze_start_time:.3f} seconds, physical time: {physical_col_squeeze_time:.6f} seconds")
            
            spread_squeeze_time += col_squeeze_time
            
            # Count defects after this iteration
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            # Calculate improvement
            defects_fixed = previous_defects - current_defects
            print(f"Defects after cycle {iteration}: {current_defects} (fixed {defects_fixed} defects)")
            
            # Check if we've achieved perfect fill
            if current_defects == 0:
                print("Perfect arrangement achieved after spread-squeeze cycles!")
                self.simulator.movement_history = total_movement_history
                execution_time = time.time() - start_time
                # Add visualization before return
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, 1.0, execution_time
                
            # Check if we should continue
            if defects_fixed < min_improvement:
                print(f"Stopping iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                break
                
            # Update for next iteration
            previous_defects = current_defects
        
        # Step 4: Move corner blocks
        print("\nStep 4: Moving corner blocks into clean zones...")
        corner_start_time = time.time()
        self.simulator.movement_history = []
        
        # Move corner blocks (no longer applies column-wise centering internally)
        _, corner_moves, corner_time = self.move_corner_blocks(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_corner_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Corner block movement complete in {time.time() - corner_start_time:.3f} seconds, physical time: {physical_corner_time:.6f} seconds")
        
        # Step 5: Apply column-wise centering to move atoms from clean zones into target zone
        print("\nStep 5: Applying column-wise centering to incorporate corner blocks...")
        corner_squeeze_start_time = time.time()
        self.simulator.movement_history = []
        _, corner_squeeze_retention, corner_squeeze_time = self.column_wise_centering(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_corner_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Corner squeeze complete in {time.time() - corner_squeeze_start_time:.3f} seconds, physical time: {physical_corner_squeeze_time:.6f} seconds")
        
        # Count defects after corner block movement and squeezing
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_corner = np.sum(target_region == 0)
        print(f"Defects after corner block movements: {defects_after_corner}")
        
        # Check if we've achieved perfect fill
        if defects_after_corner == 0:
            print("Perfect arrangement achieved after corner block movements!")
            self.simulator.movement_history = total_movement_history
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 6: First repair attempt
        print(f"\nStep 6: First repair attempt for {defects_after_corner} defects...")
        repair_start_time = time.time()
        self.simulator.movement_history = []
        final_lattice, fill_rate, repair_time = self.repair_defects(
            show_visualization=False  # Don't show animation yet
        )
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_repair_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"First repair attempt complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
        
        # Count defects after initial repair
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_repair = np.sum(target_region == 0)
        print(f"Defects after first repair: {defects_after_repair}")
        
        # Check if we've achieved perfect fill
        if defects_after_repair == 0:
            print("Perfect arrangement achieved after first repair attempt!")
            self.simulator.movement_history = total_movement_history
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 7: Iterative squeeze and repair for remaining defects
        print(f"\nStep 7: Iterative squeeze and repair for remaining defects...")
        
        # Parameters for the iterative process
        max_squeeze_repair_iterations = 3
        previous_defect_count = defects_after_repair
        
        for squeeze_repair_iteration in range(max_squeeze_repair_iterations):
            print(f"\nSqueeze-repair iteration {squeeze_repair_iteration + 1}/{max_squeeze_repair_iterations}...")
            
            # Apply full squeezing to reposition atoms better
            squeeze_start_time = time.time()
            self.simulator.movement_history = []
            
            # Apply row-wise centering
            print("Applying row-wise centering...")
            _, _, _ = self.row_wise_centering(show_visualization=False)
            
            # Apply column-wise centering
            print("Applying column-wise centering...")
            _, _, _ = self.column_wise_centering(show_visualization=False)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time from movement history
            physical_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Squeezing complete in {time.time() - squeeze_start_time:.3f} seconds, physical time: {physical_squeeze_time:.6f} seconds")
            
            # Check if squeezing fixed any defects
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            defects_after_squeeze = np.sum(target_region == 0)
            defects_fixed_by_squeeze = previous_defect_count - defects_after_squeeze
            
            if defects_fixed_by_squeeze > 0:
                print(f"Squeezing fixed {defects_fixed_by_squeeze} defects directly!")
            
            # Check if we've achieved perfect fill
            if defects_after_squeeze == 0:
                print(f"Perfect arrangement achieved after squeeze iteration {squeeze_repair_iteration + 1}!")
                self.simulator.movement_history = total_movement_history
                execution_time = time.time() - start_time
                # Add visualization before return
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, 1.0, execution_time
            
            # Apply repair for remaining defects
            repair_start_time = time.time()
            self.simulator.movement_history = []
            
            # Apply direct defect repair (without internal squeezing)
            repair_lattice, repair_fill_rate, repair_time = self.repair_defects(
                show_visualization=False
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time from movement history
            physical_repair_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Repair attempt {squeeze_repair_iteration + 1} complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
            
            # Count defects after this repair iteration
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            # Calculate overall improvement
            defects_fixed = previous_defect_count - current_defects
            print(f"Defects after squeeze-repair {squeeze_repair_iteration + 1}: {current_defects} (fixed {defects_fixed} defects)")
            
            # Check if we've achieved perfect fill
            if current_defects == 0:
                print(f"Perfect arrangement achieved after squeeze-repair iteration {squeeze_repair_iteration + 1}!")
                self.simulator.movement_history = total_movement_history
                execution_time = time.time() - start_time
                # Add visualization before return
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, 1.0, execution_time
                
            # Check if we made any progress
            if defects_fixed <= 0 and squeeze_repair_iteration > 0:
                print(f"No improvement in this iteration - stopping further squeeze-repair attempts")
                break
            
            # Update for next iteration
            previous_defect_count = current_defects
            fill_rate = repair_fill_rate
        
        # Calculate final fill rate
        target_size = self.simulator.side_length ** 2
        final_defects = np.sum(self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col] == 0)
        final_fill_rate = 1.0 - (final_defects / target_size)
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        print(f"\nCombined filling strategy completed in {execution_time:.3f} seconds")
        print(f"Final fill rate: {final_fill_rate:.2%}")
        print(f"Remaining defects: {final_defects}")
        
        # Restore complete movement history
        self.simulator.movement_history = total_movement_history
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        total_physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Total physical movement time: {total_physical_time:.6f} seconds")
            
        return self.simulator.target_lattice, final_fill_rate, execution_time