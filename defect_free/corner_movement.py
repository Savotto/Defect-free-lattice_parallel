"""
Corner-based movement module for atom rearrangement in optical lattices.
Implements strategies that place the target zone in the top-left corner of the field.
"""
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
from defect_free.base_movement import BaseMovementManager

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
        
        return len(all_moves)
    
    def align_atoms_with_right_edge(self, target_end_row, target_end_col):
        """
        Move atoms from below the target zone to align with the right edge of the target zone.
        
        This function identifies atoms below the target zone and moves them vertically
        to align with the bottom row of the target zone.
        
        Args:
            target_end_row: End row of target zone
            target_end_col: End column of target zone
            
        Returns:
            Number of atoms moved
        """
        field_height, field_width = self.simulator.initial_size
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # Only consider atoms below the target zone
        for row in range(target_end_row, field_height):
            for col in range(field_width):
                # Only consider atoms that can be moved to the right edge
                if self.simulator.field[row, col] == 1 and col < target_end_col:
                    # Find a free position at the right edge of the target zone
                    dest_col = target_end_col - 1
                    
                    # Only move if the destination is empty
                    if working_field[row, dest_col] == 0:
                        from_pos = (row, col)
                        to_pos = (row, dest_col)
                        all_moves.append({'from': from_pos, 'to': to_pos})
                        
                        # Update working field
                        working_field[from_pos] = 0
                        working_field[to_pos] = 1
                        
                        # Track maximum distance
                        distance = abs(dest_col - col)
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
                'type': 'align_with_right_edge',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        return len(all_moves)
    
    def align_atoms_with_bottom_edge(self, target_end_row, target_end_col):
        """
        Move atoms from right of the target zone to align with the bottom edge of the target zone.
        
        This function identifies atoms to the right of the target zone
        and moves them horizontally to align with the rightmost column of the target zone.
        
        Args:
            target_end_row: End row of target zone
            target_end_col: End column of target zone
            
        Returns:
            Number of atoms moved
        """
        field_height, field_width = self.simulator.initial_size
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # Only consider atoms to the right of the target zone
        for row in range(field_height):
            for col in range(target_end_col, field_width):
                # Only consider atoms that can be moved to the bottom edge
                if self.simulator.field[row, col] == 1 and row < target_end_row:
                    # Find a free position at the bottom edge of the target zone
                    dest_row = target_end_row - 1
                    
                    # Only move if the destination is empty
                    if working_field[dest_row, col] == 0:
                        from_pos = (row, col)
                        to_pos = (dest_row, col)
                        all_moves.append({'from': from_pos, 'to': to_pos})
                        
                        # Update working field
                        working_field[from_pos] = 0
                        working_field[to_pos] = 1
                        
                        # Track maximum distance
                        distance = abs(dest_row - row)
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
                'type': 'align_with_bottom_edge',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        return len(all_moves)
    
    def move_lower_right_corner(self, target_end_row, target_end_col):
        """
        Move the lower-right corner block leftward.
        
        Args:
            target_end_row: End row of target zone
            target_end_col: End column of target zone
            
        Returns:
            Number of atoms moved
        """
        field_height, field_width = self.simulator.initial_size
        
        # Calculate corner block dimensions based on target size
        initial_height, initial_width = self.simulator.initial_size
        initial_side = min(initial_height, initial_width)
        side_diff = initial_side - self.simulator.side_length
        corner_width = side_diff // 2
        
        if corner_width <= 0:
            print("No corner block to move (initial size <= target size)")
            return 0
            
        # Define lower-right corner region
        corner_start_row = target_end_row
        corner_start_col = target_end_col
        corner_end_row = min(corner_start_row + corner_width, field_height)
        corner_end_col = min(corner_start_col + corner_width, field_width)
        
        # Find atoms in the lower-right corner
        corner_atoms = []
        for row in range(corner_start_row, corner_end_row):
            for col in range(corner_start_col, corner_end_col):
                if self.simulator.field[row, col] == 1:
                    corner_atoms.append((row, col))
        
        if not corner_atoms:
            print("No atoms in lower-right corner block")
            return 0
            
        # Calculate leftward offset (move by the corner width)
        offset_col = -corner_width
        
        # Collect all moves
        all_moves = []
        working_field = self.simulator.field.copy()
        
        for atom_pos in corner_atoms:
            row, col = atom_pos
            new_row, new_col = row, col + offset_col
            
            # Check if destination is within field bounds
            if new_col < 0:
                continue
                
            # Check if destination is already occupied
            if working_field[new_row, new_col] == 1:
                continue
                
            from_pos = (row, col)
            to_pos = (new_row, new_col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
        
        # Execute all moves in parallel
        if all_moves:
            # Calculate time based on corner width
            move_time = self.calculate_realistic_movement_time(abs(offset_col))
            
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
            
            return len(successful_moves)
        
        return 0
    
    def squeeze_row_right(self, row, start_col=None):
        """
        Squeeze atoms in a row to the right edge of the target zone.
        
        Args:
            row: The row index to squeeze
            start_col: Starting column (inclusive) - defaults to 0
                
        Returns:
            Number of atoms moved
        """
        if start_col is None:
            start_col = 0
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the right edge of the target zone
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
                
        # Find all atoms in this row
        atom_indices = np.where(self.simulator.field[row, start_col:] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the rightmost positions starting from target_end_col - 1
        for i, target_col in enumerate(range(target_end_col - 1, target_end_col - 1- len(atom_indices), -1)):
            # Check if atom is already in the correct position
            if atom_indices[i] == target_col:
                continue
            
            # Create move
            from_pos = (row, atom_indices[i] + start_col)
            to_pos = (row, target_col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(target_col - (atom_indices[i] + start_col))
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
        
        return len(all_moves)
    
    def squeeze_column_down(self, col, start_row=None):
        """
        Squeeze atoms in a column to the bottom edge of the target zone.
        
        Args:
            col: The column index to squeeze
            start_row: Starting row (inclusive) - defaults to 0
                
        Returns:
            Number of atoms moved
        """
        if start_row is None:
            start_row = 0
        
        # Ensure target region is initialized
        if self.target_region is None:
            self.initialize_target_region()
        
        # Get the bottom edge of the target zone
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
                
        # Find all atoms in this column
        atom_indices = np.where(self.simulator.field[start_row:, col] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the bottommost positions starting from target_end_row - 1
        for i, target_row in enumerate(range(target_end_row - 1, target_end_row - 1 - len(atom_indices), -1)):
            # Check if atom is already in the correct position
            if atom_indices[i] == target_row:
                continue
            
            # Create move
            from_pos = (atom_indices[i] + start_row, col)
            to_pos = (target_row, col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(target_row - (atom_indices[i] + start_row))
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
        
        return len(all_moves)
    
    def corner_filling_strategy(self, show_visualization=True):
        """
        A filling strategy that places the target region in the top-left corner.
        
        This method implements a different approach than the combined_filling_strategy:
        1. Places the target zone in the top-left corner
        2. Squeezes rows to the left (all rows, including target zone)
        3. Squeezes columns upward (all columns, including target zone)
        4. Squeeze rows right that are under the target zone but only columns from 0 to the right edge of the target zone
        5. Repeats step 3 (squeeze columns up)
        6. Iteratively repeats steps 4-5 until no more progress
        7. Squeezes atoms from the right of the target zone to the bottom edge (column wise squeezing but down)
        8. Repeats step 2 (squeeze rows left)
        9. Repeats steps 6-8 until no more atoms or target zone full
        10. Moves lower right corner block leftward (The zone that is not touched at all yet)
        11. Repeats step 3 (squeeze columns up)
        12. Repeats step 6 (iterations of right edge + squeeze up)
        """
        start_time = time.time()
        total_movement_history = []
        
        # Step 1: Initialize target zone in top-left corner
        self.initialize_target_region()
        print("\nCorner-based filling strategy starting...")
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Check if target zone is already defect-free
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        initial_defects = np.sum(target_region == 0)
        if initial_defects == 0:
            print("Target zone is already defect-free! No movements needed.")
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 2: Squeeze rows to the left
        print("\nStep 2: Squeezing all rows to the left...")
        self.simulator.movement_history = []
        for row in range(self.simulator.initial_size[0]):
            self.squeeze_row_left(row)
        total_movement_history.extend(self.simulator.movement_history)
        
        # Step 3: Squeeze columns upward
        print("\nStep 3: Squeezing all columns upward...")
        self.simulator.movement_history = []
        for col in range(self.simulator.initial_size[1]):
            self.squeeze_column_up(col)
        total_movement_history.extend(self.simulator.movement_history)
        
        # Initialize variables for the main iteration
        max_iterations = 5
        min_improvement = 2
        previous_defects = initial_defects
        
        # Steps 4-6: Iterative right edge and column squeezing
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations} of right edge and column squeezing...")
            
            # Step 4: Squeeze rows right under target zone (only up to target_end_col)
            self.simulator.movement_history = []
            atoms_moved = 0
            for row in range(target_end_row, self.simulator.initial_size[0]):
                # Only move atoms in the columns up to target_end_col
                atoms_moved += self.squeeze_row_right(row, target_start_col)
            total_movement_history.extend(self.simulator.movement_history)
            
            if atoms_moved == 0:
                print("No atoms moved in right squeeze")
                break
                
            # Step 5: Squeeze columns up
            self.simulator.movement_history = []
            for col in range(self.simulator.initial_size[1]):
                self.squeeze_column_up(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Check progress
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            if current_defects == 0:
                break
                
            if previous_defects - current_defects < min_improvement:
                print(f"Insufficient improvement: {previous_defects - current_defects} defects fixed")
                break
            previous_defects = current_defects
        
        # Initialize variables for the second main iteration (steps 7-9)
        previous_defects = current_defects if 'current_defects' in locals() else initial_defects
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations} of bottom edge processing...")
            
            # Step 7: Squeeze atoms from right of target zone downward
            self.simulator.movement_history = []
            atoms_moved = 0
            for col in range(target_end_col, self.simulator.initial_size[1]):
                atoms_moved += self.squeeze_column_down(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            if atoms_moved == 0:
                print("No atoms moved in downward squeeze")
                break
                
            # Step 8: Squeeze rows left
            self.simulator.movement_history = []
            for row in range(self.simulator.initial_size[0]):
                self.squeeze_row_left(row)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Step 9: Repeat steps 4-5 (right edge and up squeezing)
            inner_atoms_moved = 0
            for inner_iter in range(3):  # Limit inner iterations
                # Squeeze right (only under target zone)
                self.simulator.movement_history = []
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    inner_atoms_moved += self.squeeze_row_right(row, target_start_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                if inner_atoms_moved == 0:
                    break
                    
                # Squeeze up
                self.simulator.movement_history = []
                for col in range(self.simulator.initial_size[1]):
                    self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
            
            # Check progress
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            if current_defects == 0:
                break
                
            if previous_defects - current_defects < min_improvement:
                print(f"Insufficient improvement: {previous_defects - current_defects} defects fixed")
                break
            previous_defects = current_defects
        
        # Step 10: Move lower right corner block leftward
        print("\nStep 10: Moving lower right corner block...")
        self.simulator.movement_history = []
        corner_atoms_moved = self.move_lower_right_corner(target_end_row, target_end_col)
        total_movement_history.extend(self.simulator.movement_history)
        
        if corner_atoms_moved > 0:
            # Step 11: Squeeze columns up
            print("\nStep 11: Final column squeeze up...")
            self.simulator.movement_history = []
            for col in range(self.simulator.initial_size[1]):
                self.squeeze_column_up(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Step 12: Final right edge and up iterations
            print("\nStep 12: Final right edge and up iterations...")
            for final_iter in range(3):  # Limit final iterations
                atoms_moved = 0
                
                # Squeeze right (only under target zone)
                self.simulator.movement_history = []
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    atoms_moved += self.squeeze_row_right(row, target_start_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                if atoms_moved == 0:
                    break
                    
                # Squeeze up
                self.simulator.movement_history = []
                for col in range(self.simulator.initial_size[1]):
                    self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        final_defects = np.sum(target_region == 0)
        fill_rate = 1.0 - (final_defects / (self.simulator.side_length ** 2))
        
        # Print final statistics
        print(f"\nCorner-based filling strategy completed in {execution_time:.3f} seconds")
        print(f"Final fill rate: {fill_rate:.2%}")
        print(f"Remaining defects: {final_defects}")
        
        # Set final state
        self.simulator.target_lattice = self.simulator.field.copy()
        self.simulator.movement_history = total_movement_history
        
        # Show visualization if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(total_movement_history)
        
        # Calculate total physical time
        total_physical_time = sum(move['time'] for move in total_movement_history)
        print(f"Total physical movement time: {total_physical_time:.6f} seconds")
        
        return self.simulator.target_lattice, fill_rate, execution_time