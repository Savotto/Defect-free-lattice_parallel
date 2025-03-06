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

    
    def row_wise_centering(self, show_visualization=True):
        """
        Row-wise centering strategy for atom rearrangement.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
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
            moves_made = self.center_atoms_in_row(row, target_start_col, target_end_col)
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
    
    def center_atoms_in_row(self, row, target_start_col, target_end_col):
        """
        Center atoms in a single row, moving from both sides toward the center.
        Dynamically allocates target positions based on atom distribution.
        Allows parallel movement while preserving atom ordering.
        
        Args:
            row: Row index to process
            target_start_col: Starting column of target region
            target_end_col: Ending column of target region
        
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
        left_atoms = sorted([col for col in atom_cols if col < center_col], reverse=True)  # Descending (right to left)
        right_atoms = sorted([col for col in atom_cols if col >= center_col])  # Ascending (left to right)
        
        # Calculate how many atoms we need in this row for the target region
        num_atoms = len(atom_cols)
        target_width = target_end_col - target_start_col
        max_atoms_in_region = min(num_atoms, target_width)
        
        # Dynamically allocate target positions based on atom distribution
        left_count = len(left_atoms)
        right_count = len(right_atoms)
        
        # Calculate how many targets for each side
        if max_atoms_in_region % 2 == 0:  # Even number of targets
            left_targets_count = min(left_count, max_atoms_in_region // 2)
            right_targets_count = min(right_count, max_atoms_in_region // 2)
        else:  # Odd number of targets
            # Give the middle position to the side with more atoms
            if left_count >= right_count:
                left_targets_count = min(left_count, max_atoms_in_region // 2 + 1)
                right_targets_count = min(right_count, max_atoms_in_region // 2)
            else:
                left_targets_count = min(left_count, max_atoms_in_region // 2)
                right_targets_count = min(right_count, max_atoms_in_region // 2 + 1)
        
        # If one side has unused targets, give them to the other side
        if left_targets_count < min(left_count, max_atoms_in_region // 2 + (1 if max_atoms_in_region % 2 else 0)):
            unused_left = min(left_count, max_atoms_in_region // 2 + (1 if max_atoms_in_region % 2 else 0)) - left_targets_count
            right_targets_count = min(right_count, right_targets_count + unused_left)
        
        if right_targets_count < min(right_count, max_atoms_in_region // 2 + (0 if max_atoms_in_region % 2 else 0)):
            unused_right = min(right_count, max_atoms_in_region // 2 + (0 if max_atoms_in_region % 2 else 0)) - right_targets_count
            left_targets_count = min(left_count, left_targets_count + unused_right)
        
        # Generate target positions - starting from the center
        left_targets = []
        for i in range(left_targets_count):
            # Start from the center and move left
            col = center_col - 1 - i
            if col >= target_start_col:
                left_targets.append(col)
        left_targets.sort(reverse=True)  # Descending order
        
        right_targets = []
        for i in range(right_targets_count):
            # Start from the center and move right
            col = center_col + i
            if col < target_end_col:
                right_targets.append(col)
        
        print(f"Row {row}: Found {len(atom_cols)} atoms - {len(left_atoms)} on left, {len(right_atoms)} on right")
        print(f"Row {row}: Using {len(left_targets) + len(right_targets)} target positions - {len(left_targets)} on left, {len(right_targets)} on right")
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        moves_executed = 0
        
        # Handle left side - start with innermost atoms (closest to center)
        for i, atom_col in enumerate(left_atoms):
            if i >= len(left_targets):
                print(f"No more left targets available for atom at ({row}, {atom_col})")
                break
                
            target_col = left_targets[i]
            
            # Check if atom needs to move
            if atom_col == target_col:
                continue
                
            # Check if path is clear to move
            path_clear = True
            min_col = min(atom_col, target_col)
            max_col = max(atom_col, target_col)
            
            for col in range(min_col + 1, max_col):
                if working_field[row, col] == 1:
                    path_clear = False
                    print(f"Path blocked for move from ({row}, {atom_col}) to ({row}, {target_col})")
                    break
                    
            if not path_clear:
                continue
                
            # Execute movement
            from_pos = (row, atom_col)
            to_pos = (row, target_col)
            
            # Update field
            working_field[row, atom_col] = 0
            working_field[row, target_col] = 1
            
            # Calculate time based on distance
            distance = abs(target_col - atom_col)
            move_time = self.calculate_movement_time(distance)
            
            print(f"Moving left atom from {from_pos} to {to_pos}")
            
            # Record move in history
            self.simulator.movement_history.append({
                'type': 'row_move',
                'moves': [(from_pos, to_pos)],
                'state': working_field.copy(),
                'time': move_time,
                'successful': 1,
                'failed': 0
            })
            
            moves_executed += 1
        
        # Handle right side - also start with innermost atoms
        for i, atom_col in enumerate(right_atoms):
            if i >= len(right_targets):
                print(f"No more right targets available for atom at ({row}, {atom_col})")
                break
                
            target_col = right_targets[i]
            
            # Check if atom needs to move
            if atom_col == target_col:
                continue
                
            # Check if path is clear to move
            path_clear = True
            min_col = min(atom_col, target_col)
            max_col = max(atom_col, target_col)
            
            for col in range(min_col + 1, max_col):
                if working_field[row, col] == 1:
                    path_clear = False
                    print(f"Path blocked for move from ({row}, {atom_col}) to ({row}, {target_col})")
                    break
                    
            if not path_clear:
                continue
                
            # Execute movement
            from_pos = (row, atom_col)
            to_pos = (row, target_col)
            
            # Update field
            working_field[row, atom_col] = 0
            working_field[row, target_col] = 1
            
            # Calculate time based on distance
            distance = abs(target_col - atom_col)
            move_time = self.calculate_movement_time(distance)
            
            print(f"Moving right atom from {from_pos} to {to_pos}")
            
            # Record move in history
            self.simulator.movement_history.append({
                'type': 'row_move',
                'moves': [(from_pos, to_pos)],
                'state': working_field.copy(),
                'time': move_time,
                'successful': 1,
                'failed': 0
            })
            
            moves_executed += 1
        
        # Update simulator's field with final state
        self.simulator.field = working_field.copy()
        
        # Verify final state in this row
        final_atom_cols = [col for col in range(self.simulator.field_size[1]) 
                         if self.simulator.field[row, col] == 1]
        print(f"Row {row}: Final atom positions {[(row, col) for col in final_atom_cols]}")
        
        return moves_executed