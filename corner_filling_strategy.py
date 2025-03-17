"""
Corner-based filling strategy implementation.
These functions are used to monkey-patch the MovementManager class in corner_test.py.
"""

def initialize_corner_target_region(self):
    """Initialize target region in the top-left corner of the field."""
    if hasattr(self, 'corner_target_region') and self.corner_target_region is not None:
        return  # Already initialized
            
    # Set target region at the top-left corner
    start_row = 0
    start_col = 0
    end_row = self.simulator.side_length
    end_col = self.simulator.side_length
        
    self.corner_target_region = (start_row, start_col, end_row, end_col)

"""
Corner filling strategy with target zone placed in the top-left corner.
"""

from defect_free.movement import MovementManager

def corner_filling_strategy(movement_manager):
    """
    Execute the corner filling strategy following these steps:
    1. Places target zone in top-left corner (already done in initialization)
    2. Squeezes rows to the left (all columns)
    3. Squeezes columns upward (all rows)
    4. Align atoms under target zone with right edge
    5. Squeeze rows left again
    6. Repeat steps 4-5 until no more progress
    7. Align atoms with bottom edge from right side
    8. Do step 1 (target already in corner)
    9. Repeat 6-7-8 until no more atoms or target zone full
    10. Move low right corner block left
    11. Do step 2 (squeeze rows left)
    12. Do step 6 (edge alignment cycles)
    """
    print("\nExecuting enhanced corner filling strategy...")
    
    # Get target zone boundaries
    target_start_row, target_start_col, target_end_row, target_end_col = movement_manager.corner_target_region
    
    # Step 1: Target zone is already in top-left corner from initialization
    print("Step 1: Target zone placed in top-left corner")
    
    # Step 2: Squeeze rows to the left
    print("\nStep 2: Squeezing rows to the left...")
    for row in range(target_start_row, target_end_row):
        movement_manager.squeeze_row_left(row, target_start_col, target_end_col)
    
    # Step 3: Squeeze columns upward
    print("\nStep 3: Squeezing columns upward...")
    for col in range(target_start_col, target_end_col):
        movement_manager.squeeze_column_up(col, target_start_row, target_end_row)
    
    # Steps 4-9: Iterative edge alignment and squeezing
    max_outer_iterations = 5
    min_improvement = 1
    
    for outer_iteration in range(max_outer_iterations):
        print(f"\nOuter iteration {outer_iteration + 1}/{max_outer_iterations}")
        
        # Track defects before this iteration
        target_region = movement_manager.simulator.field[target_start_row:target_end_row, 
                                                     target_start_col:target_end_col]
        previous_defects = target_region.size - target_region.sum()
        
        # Steps 4-5: Right edge alignment cycles
        max_edge_iterations = 5
        edge_iteration = 0
        while edge_iteration < max_edge_iterations:
            print(f"\nRight edge alignment cycle {edge_iteration + 1}")
            
            # Step 4: Align atoms with right edge
            atoms_moved = movement_manager.align_atoms_with_right_edge(target_end_row, target_end_col)
            if atoms_moved == 0:
                print("No atoms to align with right edge")
                break
                
            # Step 5: Squeeze rows left
            for row in range(target_start_row, target_end_row):
                movement_manager.squeeze_row_left(row, target_start_col, target_end_col)
            
            edge_iteration += 1
        
        # Step 7: Bottom edge alignment
        print("\nStep 7: Aligning atoms with bottom edge...")
        atoms_moved = movement_manager.align_atoms_with_bottom_edge(target_end_row, target_end_col)
        if atoms_moved > 0:
            # Squeeze rows left again to incorporate newly aligned atoms
            for row in range(target_start_row, target_end_row):
                movement_manager.squeeze_row_left(row, target_start_col, target_end_col)
        
        # Step 8: Target already in corner, just check progress
        target_region = movement_manager.simulator.field[target_start_row:target_end_row, 
                                                     target_start_col:target_end_col]
        current_defects = target_region.size - target_region.sum()
        
        # Calculate improvement
        defects_fixed = previous_defects - current_defects
        print(f"Defects after iteration {outer_iteration + 1}: {current_defects} (fixed {defects_fixed})")
        
        if current_defects == 0:
            print("Target zone is full!")
            break
            
        if defects_fixed < min_improvement:
            print(f"Stopping outer iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
            break
    
    # Steps 10-12: Final corner block movement and squeezing
    if current_defects > 0:
        # Step 10: Move low right corner block left
        print("\nStep 10: Moving lower right corner block left...")
        field_height, field_width = movement_manager.simulator.initial_size
        corner_width = movement_manager.simulator.side_length // 2
        
        # Find atoms in lower right corner
        corner_atoms = []
        for row in range(target_end_row, field_height):
            for col in range(field_width - corner_width, field_width):
                if movement_manager.simulator.field[row, col] == 1:
                    corner_atoms.append((row, col))
        
        if corner_atoms:
            # Move corner block left by its width
            all_moves = []
            max_distance = corner_width
            working_field = movement_manager.simulator.field.copy()
            
            for row, col in corner_atoms:
                new_col = col - corner_width
                if new_col >= target_start_col:  # Ensure we don't move beyond target zone
                    all_moves.append({
                        'from': (row, col),
                        'to': (row, new_col)
                    })
                    working_field[row, col] = 0
                    working_field[row, new_col] = 1
            
            if all_moves:
                movement_manager.simulator.execute_parallel_moves(all_moves, max_distance)
        
        # Step 11: Squeeze rows left again
        print("\nStep 11: Final row-wise squeezing...")
        for row in range(target_start_row, target_end_row):
            movement_manager.squeeze_row_left(row, target_start_col, target_end_col)
        
        # Step 12: Final right edge alignment cycles
        print("\nStep 12: Final right edge alignment cycles...")
        max_final_iterations = 3
        for i in range(max_final_iterations):
            atoms_moved = movement_manager.align_atoms_with_right_edge(target_end_row, target_end_col)
            if atoms_moved == 0:
                break
            
            # Squeeze rows left
            for row in range(target_start_row, target_end_row):
                movement_manager.squeeze_row_left(row, target_start_col, target_end_col)
    
    # Calculate final fill rate
    target_region = movement_manager.simulator.field[target_start_row:target_end_row, 
                                                 target_start_col:target_end_col]
    final_defects = target_region.size - target_region.sum()
    fill_rate = 1.0 - (final_defects / target_region.size)
    
    print(f"\nCorner filling strategy complete. Fill rate: {fill_rate:.2%}")
    return fill_rate

def squeeze_row_left(self, row, start_col, end_col):
    """
    Squeeze atoms in a row to the left within the specified range.
    
    Args:
        row: The row index to squeeze
        start_col: Starting column (left boundary)
        end_col: Ending column (right boundary)
        
    Returns:
        Number of atoms moved
    """
    # Find all atoms in this row within the specified range
    atom_indices = []
    for col in range(start_col, end_col):
        if self.simulator.field[row, col] == 1:
            atom_indices.append(col)
    
    if len(atom_indices) == 0:
        return 0  # No atoms to move
    
    # Create a working copy of the field
    working_field = self.simulator.field.copy()
    
    # We'll collect all moves to execute them in parallel
    all_moves = []
    max_distance = 0
    
    # Put atoms at the leftmost positions
    for i, target_col in enumerate(range(start_col, start_col + len(atom_indices))):
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

def squeeze_column_up(self, col, start_row, end_row):
    """
    Squeeze atoms in a column upward within the specified range.
    
    Args:
        col: The column index to squeeze
        start_row: Starting row (top boundary)
        end_row: Ending row (bottom boundary)
        
    Returns:
        Number of atoms moved
    """
    # Find all atoms in this column within the specified range
    atom_indices = []
    for row in range(start_row, end_row):
        if self.simulator.field[row, col] == 1:
            atom_indices.append(row)
    
    if len(atom_indices) == 0:
        return 0  # No atoms to move
    
    # Create a working copy of the field
    working_field = self.simulator.field.copy()
    
    # We'll collect all moves to execute them in parallel
    all_moves = []
    max_distance = 0
    
    # Put atoms at the topmost positions
    for i, target_row in enumerate(range(start_row, start_row + len(atom_indices))):
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
    Move atoms to align with the right edge of the target zone.
    
    This function identifies atoms outside and to the right of the target zone
    and moves them horizontally to align with the right edge of the target zone.
    
    Args:
        target_end_row: End row of target zone
        target_end_col: End column of target zone
        
    Returns:
        Number of atoms moved
    """
    field_height, field_width = self.simulator.field.shape
    
    # We'll collect all moves to execute them in parallel
    all_moves = []
    max_distance = 0
    
    # Create a working copy of the field
    working_field = self.simulator.field.copy()
    
    # Process each row within the target zone height
    for row in range(0, target_end_row):
        # Check if there are any atoms to the right of the target zone in this row
        rightmost_atom_col = None
        for col in range(target_end_col, field_width):
            if self.simulator.field[row, col] == 1:
                rightmost_atom_col = col
                break
        
        if rightmost_atom_col is not None:
            # Move the atom to align with right edge
            from_pos = (row, rightmost_atom_col)
            to_pos = (row, target_end_col - 1)
            
            # Skip if destination is occupied
            if working_field[to_pos] == 1:
                continue
            
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(rightmost_atom_col - (target_end_col - 1))
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
    Move atoms to align with the bottom edge of the target zone.
    
    This function identifies atoms outside and below the target zone
    and moves them vertically to align with the bottom edge of the target zone.
    
    Args:
        target_end_row: End row of target zone
        target_end_col: End column of target zone
        
    Returns:
        Number of atoms moved
    """
    field_height, field_width = self.simulator.field.shape
    
    # We'll collect all moves to execute them in parallel
    all_moves = []
    max_distance = 0
    
    # Create a working copy of the field
    working_field = self.simulator.field.copy()
    
    # Process each column within the target zone width
    for col in range(0, target_end_col):
        # Check if there are any atoms below the target zone in this column
        bottommost_atom_row = None
        for row in range(target_end_row, field_height):
            if self.simulator.field[row, col] == 1:
                bottommost_atom_row = row
                break
        
        if bottommost_atom_row is not None:
            # Move the atom to align with bottom edge
            from_pos = (bottommost_atom_row, col)
            to_pos = (target_end_row - 1, col)
            
            # Skip if destination is occupied
            if working_field[to_pos] == 1:
                continue
            
            all_moves.append({'from': from_pos, 'to': to_pos})
            
            # Update working field
            working_field[from_pos] = 0
            working_field[to_pos] = 1
            
            # Track maximum distance
            distance = abs(bottommost_atom_row - (target_end_row - 1))
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
