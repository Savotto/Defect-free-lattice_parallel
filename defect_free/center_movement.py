"""
Center-based movement module for atom rearrangement in optical lattices.
Implements strategies that place the target zone in the center of the field.
"""
import numpy as np
import time
from defect_free.base_movement import BaseMovementManager

class CenterMovementManager(BaseMovementManager):
    """
    Implements center-based movement strategies for atom rearrangement.
    These strategies place the target zone in the center of the field and move atoms accordingly.
    """
    
    def initialize_target_region(self):
        """Calculate and initialize the center-based target region."""
        if self.target_region is not None:
            return  # Already initialized
            
        field_height, field_width = self.simulator.initial_size
        side_length = self.simulator.side_length
        
        # Center the target region
        start_row = (field_height - side_length) // 2
        start_col = (field_width - side_length) // 2
        end_row = start_row + side_length
        end_col = start_col + side_length
        
        self.target_region = (start_row, start_col, end_row, end_col)
    
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
    
    def axis_wise_centering(self, axis='row', show_visualization=True):
        """
        Unified axis-wise centering strategy for atom rearrangement.
        
        Args:
            axis: 'row' or 'column' to specify centering direction
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, execution_time)
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
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        # Calculate total physical time from movement history
        physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"{axis.capitalize()}-wise centering complete in {execution_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
        
        return self.simulator.target_lattice, execution_time
    
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
        center_col = (target_start_col + target_end_col) // 2

        total_moves_made = 0
        
        # First process rows above the target zone
        for row in range(0, target_start_row):
            moves_made = self.spread_atoms_in_row(row, target_start_col, target_end_col, center_col)
            total_moves_made += moves_made
            
        # Then process rows below the target zone
        for row in range(target_end_row, self.simulator.initial_size[0]):
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
        
        # For parallel execution, we will collect all moves first
        all_moves = []
        max_distance = 0
        
        # Process left atoms - move them leftward (away from center)
        # Sort from leftmost to rightmost to avoid collisions
        left_atoms.sort()  # Ascending
        
        # Track new positions to avoid collisions
        new_left_positions = set()
        
        for col in left_atoms:
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
        
        for col in right_atoms:
            # Calculate new position: move as far right as possible without collision
            # but not beyond the target_end_col-1 (right edge of target zone)
            new_col = col
            field_width = self.simulator.initial_size[1]
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
        
        # Get field dimensions
        field_height, field_width = self.simulator.initial_size
        
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
        
        # PHASE 1: Check which corners can move 
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

    def center_filling_strategy(self, show_visualization=True):
        """
        An optimized comprehensive filling strategy that combines multiple methods:
        1. Apply row-wise centering and column-wise centering iteratively
        2. Next iteratively:
            a. Spreads atoms outside the target zone outward from center
            b. Applies column-wise centering to utilize the repositioned atoms
            c. Continues until no further improvement
        3. Then moves corner blocks into clean zones
        4. Applies column-wise centering to move corner atoms into the target zone
        5. First repair attempt for remaining defects
        6. If defects remain, iteratively:
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
        print("\nCenter filling strategy starting...")
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Flag to indicate early completion
        early_exit = False
        
        # Check if target zone is already defect-free
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        initial_defects = np.sum(target_region == 0)
        if initial_defects == 0:
            print("Target zone is already defect-free! No movements needed.")
            self.simulator.target_lattice = self.simulator.field.copy()
            early_exit = True
        
        # Continue with algorithm if not already perfect
        if not early_exit:
            # Step 1: Iteratively apply row-wise and column-wise centering until convergence
            print("\nStep 1: Iterative row-wise and column-wise centering...")
            
            max_iterations = 10  # Prevent infinite loops
            previous_defects = initial_defects
            iteration = 0
            
            while iteration < max_iterations and not early_exit:
                iteration += 1
                print(f"\nRow-Column Centering Iteration {iteration}/{max_iterations}...")
                atoms_moved = 0
                
                # Row-wise centering
                print(f"Step 1.{iteration}: Applying row-wise centering...")
                row_start_time = time.time()
                
                self.simulator.movement_history = []
                
                self.row_wise_centering(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                row_moves_made = len(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_row_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Row-wise centering complete in {time.time() - row_start_time:.3f} seconds, physical time: {physical_row_time:.6f} seconds")
                print(f"Made {row_moves_made} moves during row-wise centering")
                
                # Update atoms_moved counter
                atoms_moved += row_moves_made
                
                # Check if target zone is full after row-centering
                target_region = self.simulator.field[target_start_row:target_end_row, 
                                                    target_start_col:target_end_col]
                defects_after_row = np.sum(target_region == 0)
                print(f"Defects after row-centering: {defects_after_row}")
                
                if defects_after_row == 0:
                    print("Perfect arrangement achieved after row-centering!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
                
                # Column-wise centering
                print(f"Step 2.{iteration}: Applying column-wise centering...")
                col_start_time = time.time()
                self.simulator.movement_history = []
                
                self.column_wise_centering(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                col_moves_made = len(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_col_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Column-wise centering complete in {time.time() - col_start_time:.3f} seconds, physical time: {physical_col_time:.6f} seconds")
                print(f"Made {col_moves_made} moves during column-wise centering")
                
                # Update atoms_moved counter
                atoms_moved += col_moves_made
                
                # Count defects after column-centering
                target_region = self.simulator.field[target_start_row:target_end_row, 
                                                    target_start_col:target_end_col]
                defects_after_col = np.sum(target_region == 0)
                print(f"Defects after column-centering: {defects_after_col}")
                
                # Check if we've achieved perfect fill
                if defects_after_col == 0:
                    print(f"Perfect arrangement achieved after iteration {iteration}!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
                
                # Check if we made progress or if we should stop iterating
                defects_fixed = previous_defects - defects_after_col
                print(f"Iteration {iteration} fixed {defects_fixed} defects. Total atoms moved: {atoms_moved}")
                
                if atoms_moved == 0:
                    print("No atoms were moved in this iteration. Stopping row-column centering.")
                    break
                    
                previous_defects = defects_after_col
            
            if not early_exit:
                print(f"Completed {iteration} iterations of row-column centering.")
                
                # Step 2: Iterative spread-squeeze cycle
                print("\nStep 2: Starting iterative spread-squeeze cycles...")
                
                # Initialize tracking variables for the iteration
                max_iterations = 8  # Prevent infinite loops in edge cases
                min_improvement = 2  # Minimum number of defects that must be fixed to continue
                previous_defects = defects_after_col
                spread_squeeze_time = 0
                spread_squeeze_moves = 0
                iteration = 0
                
                # Continue iterations until no significant improvement or max iterations reached
                while iteration < max_iterations and not early_exit:
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
                    if self.simulator.constraints.get('atom_loss_probability', 0) > 0:
                        row_squeeze_start_time = time.time()
                        self.simulator.movement_history = []
                        _, row_squeeze_time = self.row_wise_centering(
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
                    _, col_squeeze_time = self.column_wise_centering(
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
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                        
                    # Check if we should continue
                    if defects_fixed < min_improvement:
                        print(f"Stopping iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                        break
                        
                    # Update for next iteration
                    previous_defects = current_defects
            
            if not early_exit:
                # Step 3: Move corner blocks
                print("\nStep 3: Moving corner blocks into clean zones...")
                corner_start_time = time.time()
                self.simulator.movement_history = []
                
                # Move corner blocks
                self.move_corner_blocks(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_corner_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Corner block movement complete in {time.time() - corner_start_time:.3f} seconds, physical time: {physical_corner_time:.6f} seconds")
                
                # Step 4: Apply column-wise centering to move atoms from clean zones into target zone
                print("\nStep 4: Applying column-wise centering to incorporate corner blocks...")
                corner_squeeze_start_time = time.time()
                self.simulator.movement_history = []
                self.column_wise_centering(
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
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                
            if not early_exit:
                # Step 5: First repair attempt
                print(f"\nStep 5: First repair attempt for {defects_after_corner} defects...")
                repair_start_time = time.time()
                self.simulator.movement_history = []
                self.repair_defects(
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
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
            
            if not early_exit:
                # Step 6: Iterative squeeze and repair for remaining defects
                print(f"\nStep 6: Iterative squeeze and repair for remaining defects...")
                
                # Parameters for the iterative process
                max_squeeze_repair_iterations = 3
                previous_defect_count = defects_after_repair
                
                for squeeze_repair_iteration in range(max_squeeze_repair_iterations):
                    if early_exit:
                        break
                        
                    print(f"\nSqueeze-repair iteration {squeeze_repair_iteration + 1}/{max_squeeze_repair_iterations}...")
                    
                    # Apply full squeezing to reposition atoms better
                    squeeze_start_time = time.time()
                    self.simulator.movement_history = []
                    
                    # Apply row-wise centering
                    print("Applying row-wise centering...")
                    self.row_wise_centering(show_visualization=False)
                    
                    # Apply column-wise centering
                    print("Applying column-wise centering...")
                    self.column_wise_centering(show_visualization=False)
                    
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
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                    
                    # Apply repair for remaining defects
                    repair_start_time = time.time()
                    self.simulator.movement_history = []
                    
                    # Apply direct defect repair
                    self.repair_defects(
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
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                        
                    # Check if we made any progress
                    if defects_fixed <= 0 and squeeze_repair_iteration > 0:
                        print(f"No improvement in this iteration - stopping further squeeze-repair attempts")
                        break
                    
                    # Update for next iteration
                    previous_defect_count = current_defects

        # Calculate final fill rate
        target_size = self.simulator.side_length ** 2
        target_region = self.simulator.field[target_start_row:target_end_row, target_start_col:target_end_col]
        final_defects = np.sum(target_region == 0)
        final_fill_rate = 1.0 - (final_defects / target_size)
        
        # Calculate retention rate as atoms in target zone / atoms initially loaded in the lattice
        atoms_in_target = np.sum(target_region == 1)
        retention_rate = atoms_in_target / self.simulator.total_atoms if self.simulator.total_atoms > 0 else 0
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        print(f"\nCenter filling strategy completed in {execution_time:.3f} seconds")
        print(f"Final fill rate: {final_fill_rate:.2%}")
        print(f"Remaining defects: {final_defects}")
        print(f"Final retention rate: {retention_rate:.2%}")
        
        # Restore complete movement history
        self.simulator.movement_history = total_movement_history
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        total_physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Total physical movement time: {total_physical_time:.6f} seconds")
        print(f"Total time: {execution_time + total_physical_time:.6f} seconds")
            
        return self.simulator.target_lattice, final_fill_rate, execution_time