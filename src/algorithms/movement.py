"""
Movement algorithms for atom rearrangement in the lattice simulator.
"""
from typing import Tuple, List, Set, Dict, Any
import numpy as np


class MovementManager:
    def __init__(self, simulator):
        """
        Initialize the MovementManager with a reference to the simulator.
        
        Args:
            simulator: The LatticeSimulator instance
        """
        self.simulator = simulator
    
    def calculate_movement_time(self, distance: float) -> float:
        """
        Calculate the minimum time required to move an atom over a given distance
        while respecting the maximum acceleration constraint.
        
        Args:
            distance: Distance to move in micrometers
            
        Returns:
            Time required for the movement in seconds
        """
        # Convert distance from micrometers to meters
        distance_m = distance * 1e-6
        
        # Using s = 0.5 * a * t^2 for constant acceleration (assuming acceleration and deceleration)
        # We need to accelerate for half the distance, then decelerate for the other half
        # So for each half: s_half = 0.5 * a * t_half^2
        # Which means t_half = sqrt(s_half / (0.5 * a))
        # Total time is 2 * t_half
        
        half_distance = distance_m / 2
        half_time = np.sqrt(half_distance / (0.5 * self.simulator.MAX_ACCELERATION))
        total_time = 2 * half_time
        
        return total_time
    
    def move_atom_with_constraints(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Move an atom from one position to another, accounting for physical constraints.
        
        Args:
            from_pos: Starting position (row, col)
            to_pos: Target position (row, col)
            
        Returns:
            Total time required for the movement in seconds
        """
        # Apply trap transfer time (once at the beginning and once at the end)
        transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
        
        # Calculate Manhattan distance in site units
        row_dist = abs(to_pos[0] - from_pos[0])
        col_dist = abs(to_pos[1] - from_pos[1])
        
        # Convert to physical distance using site distance
        physical_distance = (row_dist + col_dist) * self.simulator.SITE_DISTANCE
        
        # Calculate movement time based on acceleration constraints
        movement_time = self.calculate_movement_time(physical_distance)
        
        # Total time is trap transfer time + movement time
        total_time = transfer_time + movement_time
        
        # Update the field
        self.simulator.field[from_pos[0], from_pos[1]] = 0
        self.simulator.field[to_pos[0], to_pos[1]] = 1
        
        # Update the time counters
        self.simulator.total_transfer_time += transfer_time
        self.simulator.movement_time += movement_time
        
        return total_time
    
    def move_atom_through_path(self, start_pos: Tuple[int, int], intermediate_pos: Tuple[int, int], 
                             end_pos: Tuple[int, int]) -> float:
        """
        Move an atom through a path with an intermediate position,
        accounting for physical constraints.
        
        Args:
            start_pos: Starting position (row, col)
            intermediate_pos: Intermediate position (row, col)
            end_pos: Final position (row, col)
            
        Returns:
            Total time required for the movement in seconds
        """
        # Move to intermediate position
        first_move_time = self.move_atom_with_constraints(start_pos, intermediate_pos)
        
        # Record the movement in history
        self.simulator.movement_history.append({
            'type': 'parallel_left' if intermediate_pos[1] < start_pos[1] else 'parallel_right'
            if intermediate_pos[1] != start_pos[1] else
            'parallel_up' if intermediate_pos[0] < start_pos[0] else 'parallel_down',
            'moves': [(start_pos, intermediate_pos)],
            'state': self.simulator.field.copy(),
            'iteration': len(self.simulator.movement_history) + 1,
            'time': first_move_time
        })
        
        # Move to final position
        second_move_time = self.move_atom_with_constraints(intermediate_pos, end_pos)
        
        # Record the movement in history
        self.simulator.movement_history.append({
            'type': 'parallel_left' if end_pos[1] < intermediate_pos[1] else 'parallel_right'
            if end_pos[1] != intermediate_pos[1] else
            'parallel_up' if end_pos[0] < intermediate_pos[0] else 'parallel_down',
            'moves': [(intermediate_pos, end_pos)],
            'state': self.simulator.field.copy(),
            'iteration': len(self.simulator.movement_history) + 1,
            'time': second_move_time
        })
        
        return first_move_time + second_move_time
    
    def move_atoms_with_constraints(self) -> None:
        """Move atoms alternating between left and up movements until no more moves are possible."""
        print(f"\nStarting alternating left-up movement algorithm...")
        
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        target_end_row = start_row + self.simulator.side_length
        target_end_col = start_col + self.simulator.side_length
        
        def move_atoms_left() -> bool:
            """Move all atoms as far left as possible. Returns True if any moves were made."""
            made_moves = False
            current_state = self.simulator.field.copy()
            atoms = list(zip(*np.where(current_state == 1)))
            
            # Sort atoms by column (left to right) to prioritize leftmost movements
            atoms.sort(key=lambda pos: (pos[1], -pos[0]))  # Negative row for top priority
            
            moves = []
            total_batch_time = 0.0
            for atom_pos in atoms:
                y, x = atom_pos
                target_x = x
                
                # Find leftmost available position
                while target_x > start_col:
                    if self.simulator.field[y, target_x - 1] == 0:  # Next position is empty
                        target_x -= 1
                    else:
                        break
                
                if target_x < x:  # Found a valid left move
                    to_pos = (y, target_x)
                    # Calculate the movement time for this atom
                    movement_distance = (x - target_x) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME  # Transfer at start and end
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    self.simulator.field[y, x] = 0
                    self.simulator.field[y, target_x] = 1
                    made_moves = True
                    
                    # Track the physical time for the batch (use the longest time)
                    total_batch_time = max(total_batch_time, total_time)
            
            if moves:
                # Update the total time counters
                atoms_moved = len(moves)
                self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
                self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
                
                print(f"Moving {atoms_moved} atoms left (batch time: {total_batch_time*1000:.3f} ms)")
                self.simulator.movement_history.append({
                    'type': 'parallel_left',
                    'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                    'state': current_state.copy(),
                    'iteration': len(self.simulator.movement_history) + 1,
                    'time': total_batch_time
                })
            
            return made_moves
            
        def move_atoms_up() -> bool:
            """Move all atoms as far up as possible. Returns True if any moves were made."""
            made_moves = False
            current_state = self.simulator.field.copy()
            atoms = list(zip(*np.where(current_state == 1)))
            
            # Sort atoms by row (bottom to top) to prioritize upward movements
            atoms.sort(key=lambda pos: (pos[0], pos[1]))
            
            moves = []
            total_batch_time = 0.0
            for atom_pos in atoms:
                y, x = atom_pos
                target_y = y
                
                # Find highest available position
                while target_y > start_row:
                    if self.simulator.field[target_y - 1, x] == 0:  # Position above is empty
                        target_y -= 1
                    else:
                        break
                
                if target_y < y:  # Found a valid up move
                    to_pos = (target_y, x)
                    # Calculate the movement time for this atom
                    movement_distance = (y - target_y) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME  # Transfer at start and end
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    self.simulator.field[y, x] = 0
                    self.simulator.field[target_y, x] = 1
                    made_moves = True
                    
                    # Track the physical time for the batch (use the longest time)
                    total_batch_time = max(total_batch_time, total_time)
            
            if moves:
                # Update the total time counters
                atoms_moved = len(moves)
                self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
                self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
                
                print(f"Moving {atoms_moved} atoms up (batch time: {total_batch_time*1000:.3f} ms)")
                self.simulator.movement_history.append({
                    'type': 'parallel_up',
                    'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                    'state': current_state.copy(),
                    'iteration': len(self.simulator.movement_history) + 1,
                    'time': total_batch_time
                })
            
            return made_moves
        
        # Keep alternating between left and up movements until no more moves are possible
        iteration = 1
        total_algorithm_time = 0.0
        while True:
            print(f"\nIteration {iteration}:")
            
            # Try moving left
            left_moved = move_atoms_left()
            if left_moved:
                left_time = self.simulator.movement_history[-1]['time']
                total_algorithm_time += left_time
                print(f"Iteration {iteration}: Successfully moved atoms left (took {left_time*1000:.3f} ms)")
            
            # Try moving up
            up_moved = move_atoms_up()
            if up_moved:
                up_time = self.simulator.movement_history[-1]['time']
                total_algorithm_time += up_time
                print(f"Iteration {iteration}: Successfully moved atoms up (took {up_time*1000:.3f} ms)")
            
            # If neither movement was possible, we're done
            if not (left_moved or up_moved):
                print(f"\nNo more moves possible after {iteration} iterations")
                break
            
            iteration += 1
        
        # Check final state
        target_region = self.simulator.field[start_row:start_row+self.simulator.side_length, 
                                        start_col:start_col+self.simulator.side_length]
        final_count = np.sum(target_region)
        print(f"\nMovement complete. Total physical movement time: {total_algorithm_time*1000:.3f} ms")
        print(f"Atoms in target region: {final_count}/{self.simulator.side_length * self.simulator.side_length}")
    
    def fill_target_region(self) -> None:
        """Fill the target region to ensure a perfect lattice."""
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        target_filled = False
        attempts = 0
        max_attempts = 5  # Increased number of attempts
        
        while not target_filled and attempts < max_attempts:
            attempts += 1
            # Get current atom positions
            atoms = set(zip(*np.where(self.simulator.field == 1)))
            
            # Define target positions for perfect square
            target_positions = set()
            for i in range(self.simulator.side_length):
                for j in range(self.simulator.side_length):
                    target_positions.add((start_row + i, start_col + j))
            
            # Identify atoms already in position
            atoms_in_position = atoms & target_positions
            empty_targets = target_positions - atoms_in_position
            available_atoms = atoms - atoms_in_position
            
            if not empty_targets:  # Perfect lattice achieved
                target_filled = True
                break
                
            # For each empty target, find closest available atom
            moves_to_make = []
            used_atoms = set()
            
            # Sort empty targets prioritizing rightmost column first, then top to bottom
            sorted_targets = sorted(empty_targets, 
                                 key=lambda pos: (-pos[1], pos[0]))  # Changed sorting priority
            
            for target_pos in sorted_targets:
                # Find closest available atom considering movement constraints
                best_atom = None
                best_score = float('inf')
                
                for atom_pos in available_atoms:
                    if atom_pos not in used_atoms:
                        # Calculate manhattan distance
                        dist = abs(atom_pos[0] - target_pos[0]) + abs(atom_pos[1] - target_pos[1])
                        
                        # Add penalty for crossing other atoms
                        current_state = self.simulator.field.copy()
                        current_state[atom_pos[0], atom_pos[1]] = 0  # Remove source atom
                        
                        # Try both movement patterns
                        path1_clear = True  # Horizontal then Vertical
                        path2_clear = True  # Vertical then Horizontal
                        
                        # Check horizontal then vertical
                        intermediate1 = (atom_pos[0], target_pos[1])
                        if not self.simulator.path_finder.check_path_clear(atom_pos, intermediate1, current_state):
                            path1_clear = False
                        elif not self.simulator.path_finder.check_path_clear(intermediate1, target_pos, current_state):
                            path1_clear = False
                            
                        # Check vertical then horizontal
                        intermediate2 = (target_pos[0], atom_pos[1])
                        if not self.simulator.path_finder.check_path_clear(atom_pos, intermediate2, current_state):
                            path2_clear = False
                        elif not self.simulator.path_finder.check_path_clear(intermediate2, target_pos, current_state):
                            path2_clear = False
                        
                        # Calculate score based on distance and path clarity
                        score = dist
                        if not path1_clear:
                            score += 100
                        if not path2_clear:
                            score += 100
                        
                        # Prefer atoms that aren't in the target region already
                        if atom_pos[0] >= start_row and atom_pos[0] < start_row + self.simulator.side_length and \
                           atom_pos[1] >= start_col and atom_pos[1] < start_col + self.simulator.side_length:
                            score += 50
                            
                        if score < best_score:
                            best_score = score
                            best_atom = atom_pos
                
                if best_atom:
                    moves_to_make.append((best_atom, target_pos))
                    used_atoms.add(best_atom)
            
            # Execute moves
            for atom_pos, target_pos in moves_to_make:
                current_pos = atom_pos
                current_state = self.simulator.field.copy()
                
                # Try both movement patterns and choose the better one
                path1_ok = True
                path2_ok = True
                
                # Pattern 1: Horizontal then Vertical
                intermediate1 = (current_pos[0], target_pos[1])
                if not self.simulator.path_finder.check_path_clear(current_pos, intermediate1, current_state):
                    path1_ok = False
                elif not self.simulator.path_finder.check_path_clear(intermediate1, target_pos, current_state):
                    path1_ok = False
                    
                # Pattern 2: Vertical then Horizontal
                intermediate2 = (target_pos[0], current_pos[1])
                if not self.simulator.path_finder.check_path_clear(current_pos, intermediate2, current_state):
                    path2_ok = False
                elif not self.simulator.path_finder.check_path_clear(intermediate2, target_pos, current_state):
                    path2_ok = False
                
                # Choose the valid path, preferring the one that moves vertically first for rightmost column
                if target_pos[1] == start_col + self.simulator.side_length - 1 and path2_ok:
                    # For rightmost column, prefer vertical movement first
                    self.move_atom_through_path(current_pos, intermediate2, target_pos)
                elif path1_ok:
                    self.move_atom_through_path(current_pos, intermediate1, target_pos)
                elif path2_ok:
                    self.move_atom_through_path(current_pos, intermediate2, target_pos)
                else:
                    print(f"Warning: Could not find valid path from {current_pos} to {target_pos}")
            
            # Check if target region is complete
            target_region = self.simulator.field[start_row:start_row+self.simulator.side_length, 
                                              start_col:start_col+self.simulator.side_length]
            if np.sum(target_region) == self.simulator.side_length * self.simulator.side_length:
                target_filled = True
                print(f"Perfect lattice achieved after {attempts} filling attempts")
                break
            else:
                atoms_placed = np.sum(target_region)
                print(f"Filling attempt {attempts}: {atoms_placed}/{self.simulator.side_length * self.simulator.side_length} atoms in target")
        
        # Final verification
        target_region = self.simulator.field[start_row:start_row+self.simulator.side_length, 
                                          start_col:start_col+self.simulator.side_length]
        final_count = np.sum(target_region)
        print(f"Final target region contains {final_count}/{self.simulator.side_length * self.simulator.side_length} atoms")