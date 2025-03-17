"""
Corner-based movement module for atom rearrangement in optical lattices.
Implements strategies that place the target zone in the top-left corner of the field.
"""
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
from defect_free.base_movement import BaseMovementManager

class CornerMovementManager(BaseMovementManager):
    """
    Implements corner-based movement strategies for atom rearrangement.
    These strategies place the target zone in the top-left corner of the field and move atoms accordingly.
    """
    def initialize_target_region(self):
        start_row, start_col = (self.field_size - self.initial_size) // 2, (self.field_size - self.initial_size) // 2
        end_row, end_col = start_row + self.side_length, start_col + self.side_length
        self.movement_manager.target_region = (start_row, start_col, end_row, end_col)

    
    def squeeze_row_left(self, row, end_col=None):
        """
        Squeeze atoms in a row to the left within the specified range.
        If end_col is not specified, uses the field width as the end column.
        
        Args:
            row: The row index to squeeze
            end_col: Ending column (exclusive) - defaults to field width
            
        Returns:
            Number of atoms moved
        """
        if end_col is None:
            end_col = self.simulator.field_size[1]
            
        # Find all atoms in this row
        atom_indices = np.where(self.simulator.field[row, :end_col] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the leftmost positions
        for i, target_col in enumerate(range(len(atom_indices))):
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
        Squeeze atoms in a column upward within the specified range.
        If end_row is not specified, uses the field height as the end row.
        
        Args:
            col: The column index to squeeze
            end_row: Ending row (exclusive) - defaults to field height
            
        Returns:
            Number of atoms moved
        """
        if end_row is None:
            end_row = self.simulator.field_size[0]
            
        # Find all atoms in this column
        atom_indices = np.where(self.simulator.field[:end_row, col] == 1)[0]
        
        if len(atom_indices) == 0:
            return 0  # No atoms to move
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        
        # We'll collect all moves to execute them in parallel
        all_moves = []
        max_distance = 0
        
        # Put atoms at the topmost positions
        for i, target_row in enumerate(range(len(atom_indices))):
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
        field_height, field_width = self.simulator.field_size
        
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
        field_height, field_width = self.simulator.field_size
        
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
        field_height, field_width = self.simulator.field_size
        
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
    
    def corner_filling_strategy(self, show_visualization=True):
        """
        A filling strategy that places the target region in the top-left corner.
        
        This method implements a different approach than the combined_filling_strategy:
        1. Places the target zone in the top-left corner
        2. Squeezes rows to the left (all rows, including target zone)
        3. Squeezes columns upward (all columns, including target zone)
        4. Aligns atoms under target zone with right edge of target zone
        5. Repeats step 2 (squeeze rows left)
        6. Iteratively repeats steps 4-5 until no more progress
        7. Aligns atoms from right of target zone with bottom edge
        8. Repeats step 2 (squeeze rows left)
        9. Repeats steps 6-8 until no more atoms or target zone full
        10. Moves lower right corner block leftward
        11. Repeats step 2 (squeeze rows left)
        12. Repeats step 6 (iterations of right edge + squeeze left)
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        total_movement_history = []
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
            # Show visualization even if no changes were made
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 1-2: Squeeze rows to the left (all rows, including target zone)
        print("\nStep 1-2: Squeezing all rows to the left...")
        row_start_time = time.time()
        self.simulator.movement_history = []
        
        # Process each row in the field
        for row in range(self.simulator.field_size[0]):
            self.squeeze_row_left(row)
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_row_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Row-wise left squeezing complete in {time.time() - row_start_time:.3f} seconds, physical time: {physical_row_time:.6f} seconds")
        
        # Count defects after row squeezing
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_row = np.sum(target_region == 0)
        print(f"Defects after row squeezing: {defects_after_row}")
        
        if defects_after_row == 0:
            print("Perfect arrangement achieved after row squeezing!")
            self.simulator.movement_history = total_movement_history
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Step 3: Squeeze columns upward (all columns, including target zone)
        print("\nStep 3: Squeezing all columns upward...")
        col_start_time = time.time()
        self.simulator.movement_history = []
        
        # Process each column in the field
        for col in range(self.simulator.field_size[1]):
            self.squeeze_column_up(col)
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time from movement history
        physical_col_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Column-wise upward squeezing complete in {time.time() - col_start_time:.3f} seconds, physical time: {physical_col_time:.6f} seconds")
        
        # Count defects after column squeezing
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        defects_after_col = np.sum(target_region == 0)
        print(f"Defects after column squeezing: {defects_after_col}")
        
        if defects_after_col == 0:
            print("Perfect arrangement achieved after column squeezing!")
            self.simulator.movement_history = total_movement_history
            self.simulator.target_lattice = self.simulator.field.copy()
            execution_time = time.time() - start_time
            # Add visualization before return
            if show_visualization and self.simulator.visualizer:
                self.simulator.visualizer.animate_movements(total_movement_history)
            return self.simulator.target_lattice, 1.0, execution_time
        
        # Steps 4-6: Iterative right edge alignment and row squeezing
        print("\nSteps 4-6: Iterative right edge alignment and row squeezing...")
        
        # Initialize tracking variables for the iteration
        max_iterations_right = 5  # Prevent infinite loops in edge cases
        min_improvement = 2  # Minimum number of defects that must be fixed to continue
        previous_defects = defects_after_col
        
        # Continue iterations until no significant improvement or max iterations reached
        for right_iteration in range(max_iterations_right):
            print(f"\nRight edge alignment cycle {right_iteration+1}/{max_iterations_right}...")
            
            # Step 4: Align atoms under target zone with right edge
            right_edge_time = time.time()
            self.simulator.movement_history = []
            right_edge_atoms_moved = self.align_atoms_with_right_edge(target_end_row, target_end_col)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time
            physical_right_edge_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Right edge alignment complete: {right_edge_atoms_moved} atoms moved in {time.time() - right_edge_time:.3f} seconds, physical time: {physical_right_edge_time:.6f} seconds")
            
            if right_edge_atoms_moved == 0:
                print("No atoms could be moved to the right edge")
                break
            
            # Step 5: Squeeze rows left again
            row_squeeze_time = time.time()
            self.simulator.movement_history = []
            
            # Process each row in the field
            for row in range(self.simulator.field_size[0]):
                self.squeeze_row_left(row)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time
            physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Row-wise left squeezing complete in {time.time() - row_squeeze_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")
            
            # Count defects after this iteration
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            # Calculate improvement
            defects_fixed = previous_defects - current_defects
            print(f"Defects after right edge cycle {right_iteration+1}: {current_defects} (fixed {defects_fixed} defects)")
            
            # Check if we've achieved perfect fill
            if current_defects == 0:
                print(f"Perfect arrangement achieved after right edge cycle {right_iteration+1}!")
                self.simulator.movement_history = total_movement_history
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Add visualization before return
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, 1.0, execution_time
                
            # Check if we should continue
            if defects_fixed < min_improvement:
                print(f"Stopping right edge iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                break
                
            # Update for next iteration
            previous_defects = current_defects
        
        # Steps 7-9: Iterative bottom edge alignment and row squeezing
        print("\nSteps 7-9: Iterative bottom edge alignment and row squeezing...")
        
        # Initialize tracking variables for the iteration
        max_iterations_bottom = 5  # Prevent infinite loops in edge cases
        previous_defects = current_defects if 'current_defects' in locals() else defects_after_col
        
        # Continue iterations until no significant improvement or max iterations reached
        for bottom_iteration in range(max_iterations_bottom):
            print(f"\nBottom edge alignment cycle {bottom_iteration+1}/{max_iterations_bottom}...")
            
            # Step 7: Align atoms right of target zone with bottom edge
            bottom_edge_time = time.time()
            self.simulator.movement_history = []
            bottom_edge_atoms_moved = self.align_atoms_with_bottom_edge(target_end_row, target_end_col)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time
            physical_bottom_edge_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Bottom edge alignment complete: {bottom_edge_atoms_moved} atoms moved in {time.time() - bottom_edge_time:.3f} seconds, physical time: {physical_bottom_edge_time:.6f} seconds")
            
            if bottom_edge_atoms_moved == 0:
                print("No atoms could be moved to the bottom edge")
                break
            
            # Step 8: Squeeze rows left again
            row_squeeze_time = time.time()
            self.simulator.movement_history = []
            
            # Process each row in the field
            for row in range(self.simulator.field_size[0]):
                self.squeeze_row_left(row)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time
            physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Row-wise left squeezing complete in {time.time() - row_squeeze_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")
            
            # Count defects after this iteration
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            
            # Calculate improvement
            defects_fixed = previous_defects - current_defects
            print(f"Defects after bottom edge cycle {bottom_iteration+1}: {current_defects} (fixed {defects_fixed} defects)")
            
            # Check if we've achieved perfect fill
            if current_defects == 0:
                print(f"Perfect arrangement achieved after bottom edge cycle {bottom_iteration+1}!")
                self.simulator.movement_history = total_movement_history
                self.simulator.target_lattice = self.simulator.field.copy()
                execution_time = time.time() - start_time
                # Add visualization before return
                if show_visualization and self.simulator.visualizer:
                    self.simulator.visualizer.animate_movements(total_movement_history)
                return self.simulator.target_lattice, 1.0, execution_time
                
            # Check if we should continue
            if defects_fixed < min_improvement:
                print(f"Stopping bottom edge iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                break
                
            # Update for next iteration
            previous_defects = current_defects
            
            # Step 9: Do another round of right edge alignment and squeezing
            # We're still in the bottom edge loop, but doing additional right edge work
            right_edge_time = time.time()
            self.simulator.movement_history = []
            right_edge_atoms_moved = self.align_atoms_with_right_edge(target_end_row, target_end_col)
            
            if right_edge_atoms_moved > 0:
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Squeeze rows left again
                self.simulator.movement_history = []
                for row in range(self.simulator.field_size[0]):
                    self.squeeze_row_left(row)
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
        
        # Step 10: Move the lower right corner block leftward
        print("\nStep 10: Moving lower right corner block leftward...")
        corner_start_time = time.time()
        self.simulator.movement_history = []
        corner_atoms_moved = self.move_lower_right_corner(target_end_row, target_end_col)
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time
        physical_corner_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Corner block movement complete: {corner_atoms_moved} atoms moved in {time.time() - corner_start_time:.3f} seconds, physical time: {physical_corner_time:.6f} seconds")
        
        # Step 11: Squeeze rows left again
        print("\nStep 11: Final row-wise left squeezing...")
        row_squeeze_time = time.time()
        self.simulator.movement_history = []
        
        # Process each row in the field
        for row in range(self.simulator.field_size[0]):
            self.squeeze_row_left(row)
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        # Calculate total physical time
        physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Row-wise left squeezing complete in {time.time() - row_squeeze_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")
        
        # Step 12: Final round of right edge alignment and squeezing
        print("\nStep 12: Final right edge alignment and squeezing...")
        right_edge_time = time.time()
        self.simulator.movement_history = []
        right_edge_atoms_moved = self.align_atoms_with_right_edge(target_end_row, target_end_col)
        
        # Save movement history
        total_movement_history.extend(self.simulator.movement_history)
        
        if right_edge_atoms_moved > 0:
            # Squeeze rows left again
            self.simulator.movement_history = []
            for row in range(self.simulator.field_size[0]):
                self.squeeze_row_left(row)
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
        
        # Count defects after all steps
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                            target_start_col:target_end_col]
        final_defects = np.sum(target_region == 0)
        
        # Final repair of any remaining defects
        if final_defects > 0:
            print(f"\nFinal repair attempt for {final_defects} remaining defects...")
            repair_start_time = time.time()
            self.simulator.movement_history = []
            final_lattice, fill_rate, repair_time = self.repair_defects(
                show_visualization=False  # Don't show animation yet
            )
            
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate total physical time from movement history
            physical_repair_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Repair complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
        else:
            # Perfect fill achieved
            fill_rate = 1.0
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        
        # Calculate final fill rate
        target_size = self.simulator.side_length ** 2
        final_defects = np.sum(self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col] == 0)
        final_fill_rate = 1.0 - (final_defects / target_size)
        
        # Print final statistics
        print(f"\nCorner-based filling strategy completed in {execution_time:.3f} seconds")
        print(f"Final fill rate: {final_fill_rate:.2%}")
        print(f"Remaining defects: {final_defects}")
        
        # Set target lattice
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Restore complete movement history
        self.simulator.movement_history = total_movement_history
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        total_physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Total physical movement time: {total_physical_time:.6f} seconds")
            
        return self.simulator.target_lattice, final_fill_rate, execution_time