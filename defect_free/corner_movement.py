"""
Corner-based movement module for atom rearrangement in optical lattices with enhanced movement tracking.
Implements strategies that place the target zone in the top-left corner of the field.
"""
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
from defect_free.base_movement import BaseMovementManager

# Helper function to count atom moves in movement history
def count_atom_moves(movement_history):
    """Count the total number of atom moves in the movement history."""
    return sum(len(move.get('moves', [])) for move in movement_history)

class CornerMovementManager(BaseMovementManager):
    def initialize_target_region(self):
        """
        Implements corner-based movement strategies for atom rearrangement.
        These strategies place the target zone in the top-left corner of the field and move atoms accordingly.
        """
        """Initialize target region in the top-left corner of the initial atoms area."""
        if self.target_region is not None:
            return  # Already initialized
        
        start_row = 0
        start_col = 0
        side_length = self.simulator.side_length
        
        # Set target region at the top-left corner of the initial atoms
        end_row = start_row + side_length
        end_col = start_col + side_length
        
        self.target_region = (start_row, start_col, end_row, end_col)

    
    def squeeze_row_left(self, row, end_col=None):
        """
        Squeeze atoms in a row to the left edge of the target zone.
        
        Args:
            row: The row index to squeeze
            end_col: Ending column (exclusive) - defaults to field width
                
        Returns:
            Number of atoms moved
        """
        
        initial_moves = count_atom_moves(self.simulator.movement_history)
        
        if end_col is None:
            end_col = self.simulator.initial_size[1]
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the left edge of the target zone
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
                
        # Find all atoms in this row
        atom_indices = np.where(self.simulator.field[row, :end_col] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the leftmost positions starting from target_start_col
        for i, target_col in enumerate(range(target_start_col, target_start_col + len(atom_indices))):
            # Check if atom is already in the correct position
            if atom_indices[i] == target_col:
                continue
            
            # Create move
            from_pos = (row, atom_indices[i])
            to_pos = (row, target_col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(target_col - atom_indices[i])
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
                'type': 'parallel_row_left_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        final_moves = count_atom_moves(self.simulator.movement_history)
        moves_added = final_moves - initial_moves
        
        return len(all_moves)

    def squeeze_column_up(self, col, end_row=None):
        """
        Squeeze atoms in a column upward to the top edge of the target zone.
        
        Args:
            col: The column index to squeeze
            end_row: Ending row (exclusive) - defaults to field height
                
        Returns:
            Number of atoms moved
        """
        initial_moves = count_atom_moves(self.simulator.movement_history)
        
        if end_row is None:
            end_row = self.simulator.initial_size[0]
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the top edge of the target zone
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
                
        # Find all atoms in this column
        atom_indices = np.where(self.simulator.field[:end_row, col] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the topmost positions starting from target_start_row
        for i, target_row in enumerate(range(target_start_row, target_start_row + len(atom_indices))):
            # Check if atom is already in the correct position
            if atom_indices[i] == target_row:
                continue
            
            # Create move
            from_pos = (atom_indices[i], col)
            to_pos = (target_row, col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(target_row - atom_indices[i])
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
                'type': 'parallel_column_up_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        final_moves = count_atom_moves(self.simulator.movement_history)
        moves_added = final_moves - initial_moves
        
        return len(all_moves)
    
    
    def move_lower_right_corner(self, target_end_row, target_end_col):
        """
        Move the lower-right corner block leftward.
        Only moves the block if ALL atoms in the block can be moved safely.
        
        Args:
            target_end_row: End row of target zone
            target_end_col: End column of target zone
                
        Returns:
            Number of atoms successfully moved
        """
        initial_moves = count_atom_moves(self.simulator.movement_history)
        
        field_height, field_width = self.simulator.initial_size
        
        # Calculate corner block dimensions based on target size
        initial_height, initial_width = self.simulator.initial_size
        initial_side = min(initial_height, initial_width)
        side_diff = initial_side - self.simulator.side_length
        corner_width = side_diff
        
        if corner_width <= 0:
            return 0
                
        # Define lower-right corner region
        corner_start_row = target_end_row
        corner_start_col = target_end_col
        
        # Find atoms in the lower-right corner region
        corner_atoms = []
        for row in range(corner_start_row, field_height):
            for col in range(corner_start_col, field_width):
                if self.simulator.field[row, col] == 1:
                    corner_atoms.append((row, col))
        
        if not corner_atoms:
            return 0
                
        # Calculate leftward offset (move by the corner width)
        offset_col = -corner_width
        
        # Check if ALL atoms can be moved safely (no collisions)
        all_can_move = True
        working_field = self.simulator.field.copy()
        
        # First, check if any destination is outside bounds or already occupied
        for atom_pos in corner_atoms:
            row, col = atom_pos
            new_row, new_col = row, col + offset_col
            
            # Check if destination is within field bounds
            if new_col < 0:
                all_can_move = False
                break
                    
            # Check if destination is already occupied by an atom not in the corner block
            if working_field[new_row, new_col] == 1 and (new_row, new_col) not in corner_atoms:
                all_can_move = False
                break
        
        # Only proceed if all atoms can be moved
        if not all_can_move:
            return 0
        
        # Collect all moves
        all_moves = []
        max_distance = abs(offset_col)
        
        # We've verified all moves are safe, so add them all
        for atom_pos in corner_atoms:
            row, col = atom_pos
            new_row, new_col = row, col + offset_col
            
            from_pos = (row, col)
            to_pos = (new_row, new_col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
        
        # Execute all moves in parallel
        if all_moves:
            # Calculate time based on corner width
            move_time = self.calculate_realistic_movement_time(max_distance)
            
            # Apply transport efficiency to the moves
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            
            # Record batch move in history
            self.simulator.movement_history.append({
                'type': 'lower_right_corner_block_move',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
            
            # If there are any failed moves due to transport efficiency, we still consider this successful
            # because we verified physical collisions wouldn't happen
            final_moves = count_atom_moves(self.simulator.movement_history)
            moves_added = final_moves - initial_moves
            
            return len(successful_moves)
        
        return 0
    
    def squeeze_row_right(self, row, start_col=None, end_col=None):
        """
        Squeeze atoms in a row to the right edge of the target zone.
        
        Args:
            row: The row index to squeeze
            start_col: Starting column (inclusive) - defaults to 0
            end_col: Ending column (exclusive) - defaults to target_end_col
                    
        Returns:
            Number of atoms moved
        """
        initial_moves = count_atom_moves(self.simulator.movement_history)
        
        if start_col is None:
            start_col = 0
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Use target_end_col as default end_col if not specified
        if end_col is None:
            end_col = target_end_col
        
        # Find all atoms in this row between start_col and end_col
        atom_indices = np.where(self.simulator.field[row, start_col:end_col] == 1)[0]
        right_atoms = [idx + start_col for idx in atom_indices]  # Convert to absolute positions
        
        if len(right_atoms) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Sort from rightmost to leftmost to avoid collisions
        right_atoms.sort(reverse=True)  # Descending
        
        # Track new positions to avoid collisions
        new_right_positions = set()
        
        for i, col in enumerate(right_atoms):
            # Calculate new position: move as far right as possible without collision
            # but not beyond the target_end_col-1 (right edge of target zone)
            new_col = col
            field_width = self.simulator.initial_size[1]
            while new_col < end_col - 1 and working_field[row, new_col+1] == 0 and (new_col+1) not in new_right_positions:
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
                'type': 'parallel_row_right_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        final_moves = count_atom_moves(self.simulator.movement_history)
        moves_added = final_moves - initial_moves
        
        return len(all_moves)
    
    def squeeze_column_down(self, col, start_row=None, end_row=None):
        """
        Squeeze atoms in a column downward to the bottom edge of the target zone,
        ensuring no collisions occur.
        
        Args:
            col: The column index to squeeze
            start_row: Starting row (inclusive) - defaults to 0
                    
        Returns:
            Number of atoms moved
        """
        initial_moves = count_atom_moves(self.simulator.movement_history)
        
        if start_row is None:
            start_row = 0
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the bottom edge of the target zone
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
                    
        # Find all atoms in this column starting from start_row
        atom_indices = np.where(self.simulator.field[start_row:end_row, col] == 1)[0]
        atom_indices = [idx + start_row for idx in atom_indices]  # Adjust indices for the start_row offset
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Calculate bottommost position (one less than field height or target_end_row)
        bottommost_row = target_end_row - 1
        
        # Calculate target positions from bottom to top
        target_positions = list(range(bottommost_row, bottommost_row - len(atom_indices), -1))
        target_positions.reverse()  # Make ascending for clearer mapping
        
        # Sort atoms from top to bottom for clear mapping to target positions
        atom_indices.sort()  # Ascending order
        
        # Create a mapping of source positions to target positions
        moves_mapping = {}
        for atom_row, target_row in zip(atom_indices, target_positions):
            moves_mapping[atom_row] = target_row
        
        # Process the moves from bottom to top to avoid collisions
        # This ensures we move the bottommost atoms first
        sorted_atom_indices = sorted(atom_indices, reverse=True)  # Process bottom to top
        
        for atom_row in sorted_atom_indices:
            target_row = moves_mapping[atom_row]
            
            # Skip if atom is already at the target position
            if atom_row == target_row:
                continue
            
            # Skip if target position is already occupied in our working field
            if working_field[target_row, col] == 1 and target_row not in atom_indices:
                continue
                
            # Create move
            from_pos = (atom_row, col)
            to_pos = (target_row, col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(target_row - atom_row)
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
                'type': 'parallel_column_down_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        final_moves = count_atom_moves(self.simulator.movement_history)
        moves_added = final_moves - initial_moves
        
        return len(all_moves)
    
    def check_target_zone_full(self):
        """
        Check if the target zone is completely filled (no defects).
        
        Returns:
            tuple: (is_full: bool, fill_rate: float)
        """
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects = np.sum(target_region == 0)
        fill_rate = 1.0 - (defects / (self.simulator.side_length ** 2))
        return defects == 0, fill_rate

    def corner_filling_strategy(self, show_visualization=True):
        """
        A filling strategy that places the target region in the top-left corner.
        
        This method implements a different approach than the combined_filling_strategy:
        1. Places the target zone in the top-left corner
        2. Squeezes rows to the left (from row 0 to target_end_row)
        3. Squeezes columns upward (from column 0 to target_end_col)
        4. Squeezes rows right that are under the target zone but only columns up to the right edge 
        of the target zone
        5. Repeats step 3 (squeeze columns up)
        6. Iteratively repeats steps 4-5 until no more progress
        7. Squeezes atoms from the right of the target zone to the bottom edge
        8. Repeats step 2 (squeeze rows left)
        9. Repeats steps 6-8 until no more atoms or target zone full
        10. Moves lower right corner block leftward
        11. Repeats step 2 (squeeze rows left)
        12. Repeats step 3 (squeeze columns up)
        13. Repeats step 6 (iterations of right edge + squeeze up)
        14. Fill the remaining defects
        15. Repeatedly repair defects until target zone is full or no more atoms can be moved
        """
        start_time = time.time()
        total_movement_history = []
        
        # Step 1: Initialize target zone in top-left corner
        self.initialize_target_region()
        
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Check if target zone is already defect-free
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                        target_start_col:target_end_col]
        initial_defects = np.sum(target_region == 0)
        if initial_defects == 0:
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Make sure to update the movement history before returning
            self.simulator.movement_history = total_movement_history
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 2: Squeeze rows to the left - only up to target_end_row
        self.simulator.movement_history = []
        total_row_left_atoms_moved = 0
        for row in range(target_end_row):  # From row 0 to target_end_row (excluding target_end_row)
            atoms_moved = self.squeeze_row_left(row)
            total_row_left_atoms_moved += atoms_moved
        total_movement_history.extend(self.simulator.movement_history)
        
        # Check if target zone is full after left squeeze
        is_full, fill_rate = self.check_target_zone_full()
        if is_full:
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Make sure to update the movement history before returning
            self.simulator.movement_history = total_movement_history
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, fill_rate, execution_time
        
        # Step 3: Squeeze columns upward - only up to target_end_col
        self.simulator.movement_history = []
        total_col_up_atoms_moved = 0
        for col in range(target_end_col):  # From column 0 to target_end_col (excluding target_end_col)
            atoms_moved = self.squeeze_column_up(col)
            total_col_up_atoms_moved += atoms_moved
        total_movement_history.extend(self.simulator.movement_history)
        
        # Check if target zone is full after column squeeze
        is_full, fill_rate = self.check_target_zone_full()
        if is_full:
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Make sure to update the movement history before returning
            self.simulator.movement_history = total_movement_history
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, fill_rate, execution_time
        
        # Steps 4-6: Iterative right edge and column squeezing
        # Continue as long as we're making meaningful progress (more than 1 defect fixed)
        previous_defects = initial_defects
        iteration = 0
        improvement = float('inf')  # Initialize with a large value to enter the loop
        
        while improvement > 1:  # Continue while we're fixing more than 1 defect per iteration
            iteration += 1
            
            # Step 4: Squeeze rows right under target zone (only columns up to target_end_col)
            self.simulator.movement_history = []
            atoms_moved = 0
            for row in range(target_end_row, self.simulator.initial_size[0]):
                # Only move atoms in the columns up to target_end_col
                atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
            total_movement_history.extend(self.simulator.movement_history)
            
            if atoms_moved == 0:
                break
            
            # Check if target zone is full after right squeeze
            is_full, fill_rate = self.check_target_zone_full()
            if is_full:
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Make sure to update the movement history before returning
                self.simulator.movement_history = total_movement_history
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, fill_rate, execution_time
            
            # Step 5: Squeeze columns up
            self.simulator.movement_history = []
            col_up_atoms_moved = 0
            for col in range(target_end_col):  # Only up to target_end_col
                col_up_atoms_moved += self.squeeze_column_up(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Check progress
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            if current_defects == 0:
                break
            
            # Check if target zone is full after column squeeze
            is_full, fill_rate = self.check_target_zone_full()
            if is_full:
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Make sure to update the movement history before returning
                self.simulator.movement_history = total_movement_history
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, fill_rate, execution_time
            
            # Calculate improvement
            improvement = previous_defects - current_defects
            previous_defects = current_defects
        
        # Initialize variables for the second main iteration (steps 7-9)
        previous_defects = current_defects if 'current_defects' in locals() else initial_defects
        max_iterations = 5  # Maximum number of iterations for bottom edge processing
        min_improvement = 2  # Minimum number of defects that must be fixed to continue
        
        for iteration in range(max_iterations):
            
            # Step 7: Squeeze atoms from right of target zone downward
            self.simulator.movement_history = []
            atoms_moved = 0
            for col in range(target_end_col, self.simulator.initial_size[1]):
                atoms_moved += self.squeeze_column_down(col, end_row=target_end_row)
            total_movement_history.extend(self.simulator.movement_history)
            
            if atoms_moved == 0:
                break
            
            # Check if target zone is full after downward squeeze
            is_full, fill_rate = self.check_target_zone_full()
            if is_full:
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Make sure to update the movement history before returning
                self.simulator.movement_history = total_movement_history
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, fill_rate, execution_time
            
            # Step 8: Squeeze rows left (full range)
            self.simulator.movement_history = []
            left_atoms_moved = 0
            for row in range(target_end_row):  # Only up to target_end_row
                left_atoms_moved += self.squeeze_row_left(row)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Step 9: Repeat steps 4-5 (right edge and up squeezing)
            inner_atoms_moved = 0
            for inner_iter in range(3):  # Limit inner iterations
                # Squeeze right (only under target zone and up to target_end_col)
                self.simulator.movement_history = []
                inner_right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    inner_right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                inner_atoms_moved += inner_right_atoms_moved
                if inner_right_atoms_moved == 0:
                    break
                
                # Check if target zone is full after right squeeze
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    self.simulator.target_lattice = self.simulator.field.copy()
                    execution_time = time.time() - start_time
                    # Make sure to update the movement history before returning
                    self.simulator.movement_history = total_movement_history
                    if show_visualization and self.simulator.visualizer:
                        self.simulator.visualizer.animate_movements(total_movement_history)
                    return self.simulator.target_lattice, fill_rate, execution_time
                
                # Squeeze up (only up to target_end_col)
                self.simulator.movement_history = []
                inner_up_atoms_moved = 0
                for col in range(target_end_col):
                    inner_up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                
                inner_atoms_moved += inner_up_atoms_moved
                
                # Check if target zone is full after column squeeze
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    self.simulator.target_lattice = self.simulator.field.copy()
                    execution_time = time.time() - start_time
                    # Make sure to update the movement history before returning
                    self.simulator.movement_history = total_movement_history
                    if show_visualization and self.simulator.visualizer:
                        self.simulator.visualizer.animate_movements(total_movement_history)
                    return self.simulator.target_lattice, fill_rate, execution_time
            
            # Check progress
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            if current_defects == 0:
                break
            
            if previous_defects - current_defects < min_improvement:
                break
            previous_defects = current_defects
        
        # Step 10: Move lower right corner block leftward
        self.simulator.movement_history = []
        corner_atoms_moved = self.move_lower_right_corner(target_end_row, target_end_col)
        total_movement_history.extend(self.simulator.movement_history)
        
        if corner_atoms_moved > 0:
            # Step 11: Squeeze columns up (only up to target_end_col)
            self.simulator.movement_history = []
            final_up_atoms_moved = 0
            for col in range(target_end_col):
                final_up_atoms_moved += self.squeeze_column_up(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Check if target zone is full after final column squeeze
            is_full, fill_rate = self.check_target_zone_full()
            if is_full:
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Make sure to update the movement history before returning
                self.simulator.movement_history = total_movement_history
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, fill_rate, execution_time
            
            # Step 12: Squeeze rows left (up to target_end_row)
            self.simulator.movement_history = []
            final_left_atoms_moved = 0
            for row in range(target_end_row):
                final_left_atoms_moved += self.squeeze_row_left(row)
            total_movement_history.extend(self.simulator.movement_history)

            # Step 13: Final right edge and up iterations
            for final_iter in range(3):  # Limit final iterations
                atoms_moved = 0
                
                # Squeeze right (only under target zone and up to target_end_col)
                self.simulator.movement_history = []
                final_right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    final_right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                atoms_moved += final_right_atoms_moved
                if atoms_moved == 0:
                    break
                
                # Check if target zone is full after final right squeeze
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    self.simulator.target_lattice = self.simulator.field.copy()
                    execution_time = time.time() - start_time
                    # Make sure to update the movement history before returning
                    self.simulator.movement_history = total_movement_history
                    if show_visualization and self.simulator.visualizer:
                        self.simulator.visualizer.animate_movements(total_movement_history)
                    return self.simulator.target_lattice, fill_rate, execution_time
                
                # Squeeze up
                self.simulator.movement_history = []
                final_up_atoms_moved = 0
                for col in range(target_end_col):
                    final_up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                
                atoms_moved += final_up_atoms_moved
                
                # Check if target zone is full after final column squeeze
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    self.simulator.target_lattice = self.simulator.field.copy()
                    execution_time = time.time() - start_time
                    # Make sure to update the movement history before returning
                    self.simulator.movement_history = total_movement_history
                    if show_visualization and self.simulator.visualizer:
                        self.simulator.visualizer.animate_movements(total_movement_history)
                    return self.simulator.target_lattice, fill_rate, execution_time
        
        # Calculate final metrics before defect repair
        execution_time = time.time() - start_time
        is_full, fill_rate = self.check_target_zone_full()
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                        target_start_col:target_end_col]
        final_defects = np.sum(target_region == 0)

        # STEP 15: NEW ITERATIVE DEFECT REPAIR PROCESS
        if final_defects > 0:
            repair_iteration = 0
            max_repair_iterations = 10  # Safety limit to prevent infinite loops
            
            # Function to check if more repairs are needed and possible
            def check_repair_status():
                # Get target region boundaries
                target_region = self.target_region
                start_row, start_col, end_row, end_col = target_region
                
                # Count defects in target zone
                target_zone = self.simulator.field[start_row:end_row, start_col:end_col]
                defects = np.sum(target_zone == 0)
                
                # Count available atoms outside target zone
                mask = np.zeros_like(self.simulator.field, dtype=bool)
                mask[start_row:end_row, start_col:end_col] = True
                available_atoms = np.sum(self.simulator.field[~mask] == 1)
                
                return {
                    "is_full": defects == 0,
                    "available_atoms": available_atoms,
                    "defects": defects
                }
            
            # Initial repair status
            repair_status = check_repair_status()
            previous_defects = repair_status["defects"]
            
            while (not repair_status["is_full"] and 
                repair_status["available_atoms"] > 0 and 
                repair_iteration < max_repair_iterations):
                
                repair_iteration += 1
                
                # Clear movement history before repair to capture only repair movements
                self.simulator.movement_history = []
                
                # Use the sophisticated repair_defects method to fill defects
                repair_start_time = time.time()
                final_lattice, repair_fill_rate, repair_time = super().repair_defects(show_visualization=False)
                
                # Count repair atom moves
                repair_atom_moves = count_atom_moves(self.simulator.movement_history)
                
                # Add repair movements to total history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Update execution time
                execution_time += (time.time() - repair_start_time)
                
                # Calculate physical time for this repair cycle
                physical_repair_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                
                # Update repair status
                repair_status = check_repair_status()
                current_defects = repair_status["defects"]
                defects_fixed = previous_defects - current_defects
                
                # Check if we've made progress
                if defects_fixed <= 0 and repair_iteration > 1:
                    break
                    
                # Update for next iteration
                previous_defects = current_defects
                
                # If target is full, exit loop
                if repair_status["is_full"]:
                    break
            
            # Final repair results
            final_repair_status = check_repair_status()
            final_fill_rate = 1.0 - (final_repair_status["defects"] / (self.simulator.side_length ** 2))
            
            # Set final state
            self.simulator.target_lattice = self.simulator.field.copy()
            fill_rate = final_fill_rate
        else:
            # No defects to repair
            self.simulator.target_lattice = self.simulator.field.copy()
        
        # Show visualization if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(total_movement_history)
        
        # Calculate total physical time
        total_physical_time = sum(move.get('time', 0) for move in total_movement_history)
        
        # Make sure to save the movement history in the simulator for reporting
        self.simulator.movement_history = total_movement_history
        
        return self.simulator.target_lattice, fill_rate, execution_time