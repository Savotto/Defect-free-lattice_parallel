"""
Corner-based movement module for atom rearrangement in optical lattices with enhanced movement tracking.
Implements strategies that place the target zone in the top-left corner of the field.
"""
import numpy as np
import time
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
        
        # Collect all moves
        all_moves = []
        max_distance = 0
        for i, target_col in enumerate(range(target_start_col, target_start_col + len(atom_indices))):
            if atom_indices[i] == target_col:
                continue
            from_pos = (row, atom_indices[i])
            to_pos = (row, target_col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            distance = abs(target_col - atom_indices[i])
            max_distance = max(max_distance, distance)
        
        # Execute all moves in parallel
        if all_moves:
            move_time = self.calculate_realistic_movement_time(max_distance)
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            self.simulator.movement_history.append({
                'type': 'parallel_row_left_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
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
        
        # Collect all moves
        all_moves = []
        max_distance = 0
        for i, target_row in enumerate(range(target_start_row, target_start_row + len(atom_indices))):
            if atom_indices[i] == target_row:
                continue
            from_pos = (atom_indices[i], col)
            to_pos = (target_row, col)
            all_moves.append({'from': from_pos, 'to': to_pos})
            distance = abs(target_row - atom_indices[i])
            max_distance = max(max_distance, distance)
        
        # Execute all moves in parallel
        if all_moves:
            move_time = self.calculate_realistic_movement_time(max_distance)
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            self.simulator.movement_history.append({
                'type': 'parallel_column_up_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            self.simulator.field = updated_field.copy()
        
        return len(all_moves)
    
    
    def move_lower_right_corner(self, target_end_row, target_end_col):
        """
        Move the lower-right corner block leftward.
        If the corner block is wider than the target zone, it moves the length of the target zone instead.
        
        Args:
            target_end_row: End row of target zone
            target_end_col: End column of target zone
                
        Returns:
            Number of atoms successfully moved
        """
        field_height, field_width = self.simulator.initial_size
        corner_width = min(field_height, field_width) - self.simulator.side_length
        
        if corner_width <= 0:
            return 0  # No corner block to move
        
        # Adjust the move length if the corner block is wider than the target zone
        move_length = min(corner_width, self.simulator.side_length)
        
        # Find atoms in the lower-right corner region
        corner_atoms = [
            (row, col)
            for row in range(target_end_row, field_height)
            for col in range(target_end_col, field_width)
            if self.simulator.field[row, col] == 1
        ]
        if not corner_atoms:
            return 0  # No atoms to move
        
        # Check if all moves are safe
        offset_col = -move_length
        for row, col in corner_atoms:
            new_col = col + offset_col
            if new_col < 0 or (self.simulator.field[row, new_col] == 1 and (row, new_col) not in corner_atoms):
                return 0  # Unsafe move detected
        
        # Collect all moves
        all_moves = [
            {'from': (row, col), 'to': (row, col + offset_col)}
            for row, col in corner_atoms
        ]
        
        # Execute all moves
        move_time = self.calculate_realistic_movement_time(abs(offset_col))
        updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(all_moves, self.simulator.field)
        self.simulator.movement_history.append({
            'type': 'lower_right_corner_block_move',
            'moves': successful_moves + failed_moves,  # Record all attempted moves
            'state': updated_field.copy(),
            'time': move_time,
            'successful': len(successful_moves),
            'failed': len(failed_moves)
        })
        self.simulator.field = updated_field.copy()
        
        return len(successful_moves)
    
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
        if start_col is None:
            start_col = 0
        if self.target_region is None:
            self.initialize_target_region()
        _, _, _, target_end_col = self.target_region
        if end_col is None:
            end_col = target_end_col

        # Find all atoms in this row between start_col and end_col
        atom_indices = np.where(self.simulator.field[row, start_col:end_col] == 1)[0]
        right_atoms = [idx + start_col for idx in atom_indices]
        if len(right_atoms) == 0:
            return 0  # No atoms to move

        # Create a working copy of the field
        working_field = self.simulator.field.copy()

        # Collect all moves
        all_moves = []
        max_distance = 0
        right_atoms.sort(reverse=True)  # Sort from rightmost to leftmost
        new_right_positions = set()
        for col in right_atoms:
            new_col = col
            while new_col < end_col - 1 and working_field[row, new_col + 1] == 0 and (new_col + 1) not in new_right_positions:
                new_col += 1
            if new_col != col:
                from_pos = (row, col)
                to_pos = (row, new_col)
                all_moves.append({'from': from_pos, 'to': to_pos})
                new_right_positions.add(new_col)
                working_field[row, col] = 0
                working_field[row, new_col] = 1
                distance = abs(new_col - col)
                max_distance = max(max_distance, distance)

        # Execute all moves in parallel
        if all_moves:
            move_time = self.calculate_realistic_movement_time(max_distance)
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            self.simulator.movement_history.append({
                'type': 'parallel_row_right_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            self.simulator.field = updated_field.copy()

        return len(all_moves)
    
    def squeeze_column_down(self, col, start_row=None, end_row=None):
        """
        Squeeze atoms in a column downward to the bottom edge of the target zone,
        ensuring no collisions occur.
        
        Args:
            col: The column index to squeeze
            start_row: Starting row (inclusive) - defaults to 0
            end_row: Ending row (exclusive) - defaults to field height
                    
        Returns:
            Number of atoms moved
        """
        if start_row is None:
            start_row = 0
        if self.target_region is None:
            self.initialize_target_region()
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        if end_row is None:
            end_row = self.simulator.initial_size[0]

        # Find all atoms in this column starting from start_row
        atom_indices = np.where(self.simulator.field[start_row:end_row, col] == 1)[0]
        atom_indices = [idx + start_row for idx in atom_indices]  # Adjust indices for the start_row offset
        if len(atom_indices) == 0:
            return 0  # No atoms to move

        # Create a working copy of the field
        working_field = self.simulator.field.copy()

        # Collect all moves
        all_moves = []
        max_distance = 0
        bottommost_row = target_end_row - 1
        for atom_row in sorted(atom_indices, reverse=True):  # Process bottom to top
            new_row = atom_row
            while new_row < bottommost_row and working_field[new_row + 1, col] == 0:
                new_row += 1
            if new_row != atom_row:
                from_pos = (atom_row, col)
                to_pos = (new_row, col)
                all_moves.append({'from': from_pos, 'to': to_pos})
                working_field[from_pos] = 0
                working_field[to_pos] = 1
                distance = abs(new_row - atom_row)
                max_distance = max(max_distance, distance)

        # Execute all moves in parallel
        if all_moves:
            move_time = self.calculate_realistic_movement_time(max_distance)
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            self.simulator.movement_history.append({
                'type': 'parallel_column_down_squeeze',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            self.simulator.field = updated_field.copy()

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
        
        This method implements a different approach than the center_filling_strategy:
        1. Iteratively squeezes rows to the left (from row 0 to target_end_row) 3 times
        2. Iteratively squeezes columns upward (from column 0 to target_end_col) 3 times
        3. Iteratively performs right-up squeezing:
           - Squeezes rows right under the target zone (but only up to target_end_col)
           - Squeezes columns up to fill the target zone
        4. Performs edge processing iterations:
           - Squeezes atoms from right of target zone downward
           - Squeezes rows left to fill the target zone
           - Repeats right edge and up squeezing
        5. Moves lower right corner block leftward
        6. Performs column squeeze up
        7. Executes right-up iterations
        8. Iteratively repeats steps 4 operations
        9. Performs defect repair process using optimized pathfinding for remaining defects
        
        Args:
            show_visualization: Whether to visualize the movements
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        total_movement_history = []
        self.initialize_target_region()
        print("\nCorner filling strategy starting...")
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Flag to track if early exit is needed
        early_exit = False

        # Check if target zone is already defect-free
        target_region = self.simulator.field[target_start_row:target_end_row, 
                                        target_start_col:target_end_col]
        initial_defects = np.sum(target_region == 0)
        print(f"Initial defects in target zone: {initial_defects}")
        
        if initial_defects == 0:
            print("Target zone is already defect-free! No movements needed.")
            self.simulator.target_lattice = self.simulator.field.copy()
            early_exit = True
        
        if not early_exit:
            # Steps 1 and 2: Initial row and column squeezing
            print("\nSteps 1-2: Initial row and column squeezing...")
            
            # Step 1: Squeeze rows to the left (three times)
            print("Step 1: Squeezing rows to the left (three iterations)...")
            for iteration in range(3):
                print(f"\nLeft squeeze iteration {iteration + 1}/3...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                total_row_left_atoms_moved = 0
                for row in range(target_end_row):
                    atoms_moved = self.squeeze_row_left(row)
                    total_row_left_atoms_moved += atoms_moved
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Row squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {total_row_left_atoms_moved} atoms during row squeezing")
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print(f"Perfect arrangement achieved after left squeeze iteration {iteration + 1}!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break

        if not early_exit:    
            # Step 2: Squeeze columns upward (three times)
            print("\nStep 2: Squeezing columns upward (three iterations)...")
            for iteration in range(3):
                print(f"\nUpward squeeze iteration {iteration + 1}/3...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                total_col_up_atoms_moved = 0
                for col in range(target_end_col):
                    atoms_moved = self.squeeze_column_up(col)
                    total_col_up_atoms_moved += atoms_moved
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {total_col_up_atoms_moved} atoms during column squeezing")
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print(f"Perfect arrangement achieved after upward squeeze iteration {iteration + 1}!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
        
            # Check defects after initial iterations
            target_region = self.simulator.field[target_start_row:target_end_row, target_start_col:target_end_col]
            defects_after_initial = np.sum(target_region == 0)
            print(f"Defects after initial squeezing iterations: {defects_after_initial}")

        if not early_exit:    
            # Step 3: Iterative right-up squeezing
            print("\nStep 3: Iterative right-up squeezing for atoms under target zone...")
            iteration = 0
            max_iterations = 6
            while True:
                iteration += 1
                if iteration > max_iterations:
                    print(f"Reached maximum iterations ({max_iterations})")
                    break
                    
                print(f"\nRight-Up iteration {iteration}/{max_iterations}...")
                total_atoms_moved = 0
                
                # Squeeze rows right under target zone
                print(f"Squeezing rows right under target zone...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    # Only move atoms in the columns up to target_end_col
                    right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += right_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Right squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {right_atoms_moved} atoms during right squeezing")
                
                # Squeeze columns up to fill target zone
                print(f"Squeezing columns up to fill target zone...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                up_atoms_moved = 0
                for col in range(target_end_col):  # Only up to target_end_col
                    up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += up_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {up_atoms_moved} atoms during column squeezing")
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break

                # Break if no atoms were moved
                if total_atoms_moved == 0:
                    print("No atoms moved in this iteration - breaking out of right-up loop")
                    break
            
            # Check current defects before moving to step 4
            target_region = self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col]
            current_defects = np.sum(target_region == 0)
            print(f"Defects after right-up iterations: {current_defects}")
        
        if not early_exit:
            # Step 4: Edge processing iterations
            previous_defects = current_defects
            max_iterations = 5  # Maximum number of iterations for edge processing
            min_improvement = -3  # Minimum number of defects that must be fixed to continue
            
            print(f"\nStep 4: Edge processing iterations (max {max_iterations} iterations)...")
            for iteration in range(max_iterations):
                print(f"\nEdge processing iteration {iteration+1}/{max_iterations}...")
                
                # Squeeze atoms from right of target zone downward
                print(f"Squeezing atoms from right of target zone downward...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                atoms_moved = 0
                for col in range(target_end_col, self.simulator.initial_size[1]):
                    atoms_moved += self.squeeze_column_down(col, end_row=target_end_row)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Downward squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {atoms_moved} atoms during downward squeezing")
                
                # Squeeze rows left to fill target zone
                print(f"Squeezing rows left to fill target zone...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                left_atoms_moved = 0
                for row in range(target_end_row):  # Only up to target_end_row
                    left_atoms_moved += self.squeeze_row_left(row)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Left squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {left_atoms_moved} atoms during left squeezing")
                
                # Check if target zone is full after left squeeze
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved after left squeezing!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
                
                # Repeat right-up squeezing for edge processing
                print(f"Repeating right-up squeezing...")
                inner_atoms_moved = 0
                
                # Squeeze right (only under target zone and up to target_end_col)
                step_start_time = time.time()
                self.simulator.movement_history = []
                inner_right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    inner_right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Right squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {inner_right_atoms_moved} atoms during right squeezing")
                
                inner_atoms_moved += inner_right_atoms_moved
                
                # Squeeze up (only up to target_end_col)
                step_start_time = time.time()
                self.simulator.movement_history = []
                inner_up_atoms_moved = 0
                for col in range(target_end_col):
                    inner_up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {inner_up_atoms_moved} atoms during column squeezing")
                
                inner_atoms_moved += inner_up_atoms_moved
                
                # Check progress
                target_region = self.simulator.field[target_start_row:target_end_row, 
                                                    target_start_col:target_end_col]
                current_defects = np.sum(target_region == 0)
                defect_improvement = previous_defects - current_defects
                print(f"Improvement in edge iteration {iteration+1}: {defect_improvement} defects fixed")
                    
                previous_defects = current_defects
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break

                # Check if no atoms were moved in inner iterations
                if inner_atoms_moved == 0:
                    print("No atoms moved in inner iterations, breaking edge processing loop")
                    break

                if defect_improvement < min_improvement and iteration > 0:
                    print(f"Stopping iterations: improvement ({defect_improvement}) below threshold ({min_improvement})")
                    break
        
        if not early_exit:
            # Step 5: Move lower right corner block leftward
            print("\nStep 5: Moving lower right corner block leftward...")
            step_start_time = time.time()
            self.simulator.movement_history = []
            corner_atoms_moved = self.move_lower_right_corner(target_end_row, target_end_col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate physical time for this step
            physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
            print(f"Corner block movement complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
            print(f"Moved {corner_atoms_moved} atoms in corner block")
        
            # Step 6: Squeeze columns up
            print("\nStep 6: Performing column squeeze up...")
            step_start_time = time.time()
            self.simulator.movement_history = []
            final_up_atoms_moved = 0
            for col in range(target_end_col):
                final_up_atoms_moved += self.squeeze_column_up(col)
            total_movement_history.extend(self.simulator.movement_history)
            
            # Calculate physical time for this step
            physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
            print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
            print(f"Moved {final_up_atoms_moved} atoms during column squeezing")
            
            # Check if target zone is full
            is_full, fill_rate = self.check_target_zone_full()
            if is_full:
                print("Perfect arrangement achieved after column squeezing!")
                self.simulator.target_lattice = self.simulator.field.copy()
                early_exit = True

        if not early_exit:
            # Step 7: Final right-up iterations
            print("\nStep 7: Executing final right-up iterations...")
            for final_iter in range(4):  # Limit final iterations
                print(f"Iteration {final_iter+1}/4...")
                atoms_moved = 0
                
                # Squeeze right (only under target zone and up to target_end_col)
                step_start_time = time.time()
                self.simulator.movement_history = []
                final_right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    final_right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Right squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {final_right_atoms_moved} atoms during right squeezing")
                
                atoms_moved += final_right_atoms_moved
                
                # Squeeze up
                step_start_time = time.time()
                self.simulator.movement_history = []
                final_up_atoms_moved = 0
                for col in range(target_end_col):
                    final_up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {final_up_atoms_moved} atoms during column squeezing")
                
                atoms_moved += final_up_atoms_moved
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break

                # Check if we've moved atoms this iteration
                if atoms_moved == 0:
                    print("No atoms moved in this iteration - breaking out of iterations")
                    break
        
        if not early_exit:
            # Step 8: Repeating edge processing operations
            print("\nStep 8: Repeating edge processing operations...")
            post_corner_iterations = 3  # Set the maximum number of iterations
            previous_defects = np.sum(self.simulator.field[target_start_row:target_end_row, 
                                                target_start_col:target_end_col] == 0)
            
            for iter_num in range(post_corner_iterations):
                print(f"\nIteration {iter_num+1}/{post_corner_iterations}...")
                total_atoms_moved = 0
                
                # Squeeze atoms from right of target zone downward
                print(f"Squeezing atoms from right of target zone downward...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                down_atoms_moved = 0
                for col in range(target_end_col, self.simulator.initial_size[1]):
                    down_atoms_moved += self.squeeze_column_down(col, end_row=target_end_row)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += down_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Downward squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {down_atoms_moved} atoms during downward squeezing")
                
                # Squeeze rows left
                print(f"Squeezing rows left to fill target zone...")
                step_start_time = time.time()
                self.simulator.movement_history = []
                left_atoms_moved = 0
                for row in range(target_end_row):
                    left_atoms_moved += self.squeeze_row_left(row)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += left_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Left squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {left_atoms_moved} atoms during left squeezing")
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved after left squeezing!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
                
                # Repeat right-up squeezing
                print(f"Repeating right-up squeezing...")
                
                # Squeeze right (only under target zone and up to target_end_col)
                step_start_time = time.time()
                self.simulator.movement_history = []
                right_atoms_moved = 0
                for row in range(target_end_row, self.simulator.initial_size[0]):
                    right_atoms_moved += self.squeeze_row_right(row, start_col=0, end_col=target_end_col)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += right_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Right squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {right_atoms_moved} atoms during right squeezing")
                
                # Squeeze columns up (only up to target_end_col)
                step_start_time = time.time()
                self.simulator.movement_history = []
                up_atoms_moved = 0
                for col in range(target_end_col):
                    up_atoms_moved += self.squeeze_column_up(col)
                total_movement_history.extend(self.simulator.movement_history)
                total_atoms_moved += up_atoms_moved
                
                # Calculate physical time for this step
                physical_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Column squeezing complete in {time.time() - step_start_time:.3f} seconds, physical time: {physical_time:.6f} seconds")
                print(f"Moved {up_atoms_moved} atoms during column squeezing")
                
                # Check if target zone is full
                is_full, fill_rate = self.check_target_zone_full()
                if is_full:
                    print("Perfect arrangement achieved after column squeezing!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                    break
                
                # Check if we've made progress in fixing defects
                current_defects = np.sum(self.simulator.field[target_start_row:target_end_row, 
                                                    target_start_col:target_end_col] == 0)
                defect_improvement = previous_defects - current_defects
                print(f"Improvement in iteration {iter_num+1}: {defect_improvement} defects fixed")
                previous_defects = current_defects
                
                # If no atoms were moved in this iteration, break out of the loop
                if total_atoms_moved == 0:
                    print("No atoms moved in this iteration - breaking out of loop")
                    break
        
        if not early_exit:
            # Step 9: Defect repair process
            print("\nStep 9: Starting defect repair process...")
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
            print(f"Initial repair status: {repair_status['defects']} defects, {repair_status['available_atoms']} atoms available")
            
            while (not repair_status["is_full"] and 
                repair_status["available_atoms"] > 0 and 
                repair_iteration < max_repair_iterations):
                
                repair_iteration += 1
                print(f"\nRepair iteration {repair_iteration}/{max_repair_iterations}...")
                
                # Clear movement history before repair to capture only repair movements
                self.simulator.movement_history = []
                
                # Use the sophisticated repair_defects method to fill defects
                repair_start_time = time.time()
                final_lattice, repair_fill_rate, repair_time = super().repair_defects(show_visualization=False)
                
                # Count repair atom moves
                repair_atom_moves = count_atom_moves(self.simulator.movement_history)
                
                # If no atoms were moved, perform rows left and columns up
                if repair_atom_moves == 0:
                    print("No atoms moved in repair, performing rows left and columns up...")
                    
                    # Squeeze rows to the left
                    for row in range(target_end_row):
                        self.squeeze_row_left(row)
                    
                    # Squeeze columns upward
                    for col in range(target_end_col):
                        self.squeeze_column_up(col)
                
                # Add repair movements to total history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate physical time for this repair cycle
                physical_repair_time = sum(move.get('time', 0) for move in self.simulator.movement_history)
                print(f"Repair attempt {repair_iteration} complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
                print(f"Moved {repair_atom_moves} atoms during repair")
                
                # Update repair status
                repair_status = check_repair_status()
                current_defects = repair_status["defects"]
                defects_fixed = previous_defects - current_defects
                print(f"Defects after repair iteration {repair_iteration}: {current_defects} (fixed {defects_fixed} defects)")
                    
                # Update for next iteration
                previous_defects = current_defects
                
                # If target is full, exit loop
                if repair_status["is_full"]:
                    print("Perfect arrangement achieved after repair!")
                    break
            
            # Final repair results
            final_repair_status = check_repair_status()
            final_fill_rate = 1.0 - (final_repair_status["defects"] / (self.simulator.side_length ** 2))
            print(f"Final repair status: {final_repair_status['defects']} defects remain, fill rate: {final_fill_rate:.2%}")
        
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
        print(f"\nCorner filling strategy completed in {execution_time:.3f} seconds")
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