"""
Movement algorithms for atom rearrangement in the lattice simulator.
"""
from typing import Tuple, List, Set, Dict, Any
import numpy as np
import time


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
    
    def move_atom_with_constraints(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[float, bool]:
        """
        Move an atom from one position to another, accounting for physical constraints.
        
        Args:
            from_pos: Starting position (row, col)
            to_pos: Target position (row, col)
            
        Returns:
            Tuple of (total_time, success) where success is False if atom was lost during transfer
        """
        # Check for trap transfer success (two transfers: one at start, one at end)
        # Each transfer has TRAP_TRANSFER_FIDELITY probability of success
        transfer_success = np.random.random() < self.simulator.TRAP_TRANSFER_FIDELITY**2
        
        if not transfer_success:
            # Atom was lost during transfer
            self.simulator.field[from_pos[0], from_pos[1]] = 0  # Remove the atom
            self.simulator.total_transfer_time += 2 * self.simulator.TRAP_TRANSFER_TIME
            return 2 * self.simulator.TRAP_TRANSFER_TIME, False
        
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
        
        return total_time, True
    
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
        first_move_time, first_success = self.move_atom_with_constraints(start_pos, intermediate_pos)
        
        if not first_success:
            return first_move_time  # Atom was lost during first move
            
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
        second_move_time, second_success = self.move_atom_with_constraints(intermediate_pos, end_pos)
        
        if not second_success:
            return first_move_time + second_move_time  # Atom was lost during second move
            
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
                    
                    # Check for trap transfer success
                    # Each movement requires two transfers (one at start, one at end)
                    transfer_success = np.random.random() < self.simulator.TRAP_TRANSFER_FIDELITY**2
                    
                    if transfer_success:
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
                    else:
                        # Atom was lost during transfer
                        self.simulator.field[y, x] = 0
                        self.simulator.total_transfer_time += 2 * self.simulator.TRAP_TRANSFER_TIME
            
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
                    
                    # Check for trap transfer success
                    # Each movement requires two transfers (one at start, one at end)
                    transfer_success = np.random.random() < self.simulator.TRAP_TRANSFER_FIDELITY**2
                    
                    if transfer_success:
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
                    else:
                        # Atom was lost during transfer
                        self.simulator.field[y, x] = 0
                        self.simulator.total_transfer_time += 2 * self.simulator.TRAP_TRANSFER_TIME
            
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

    def rearrange_atoms(self, show_visualization: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Rearrange atoms using the following procedure:
        1. Left-Up movement (standard parallel)
        2. Repeatedly move atoms under target zone to right edge then up until no more moves possible
        3. Fill remaining positions if needed
        
        Args:
            show_visualization: Whether to show the animation of the rearrangement process
        Returns:
            Tuple of (target_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        
        # Store initial atom count
        initial_atoms = self.simulator.total_atoms
        target_square_size = self.simulator.side_length * self.simulator.side_length
        
        print(f"\nRearranging {initial_atoms} atoms to form {self.simulator.side_length}x{self.simulator.side_length} square...")
        
        # Phase 1: Left-Up Movement Algorithm
        print("\nPhase 1: Left-Up Movement Algorithm")
        self.move_atoms_with_constraints()
        
        # Get target region coordinates
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        end_row = start_row + self.simulator.side_length
        end_col = start_col + self.simulator.side_length
        
        # Phase 2: Repeated Right-Up Movements
        print("\nPhase 2: Repeated Right-Up Movements")
        self.repeat_right_up_movements()
        
        # Check if target region still needs filling
        target_region = self.simulator.field[start_row:end_row, start_col:end_col]
        filled_count = np.sum(target_region)
        
        # Phase 3: Fill remaining positions if needed
        if filled_count < target_square_size:
            print(f"\nPhase 3: Filling remaining positions ({filled_count}/{target_square_size} filled)")
            self.fill_target_region()
        
        # Animate the rearrangement if visualization is enabled
        if show_visualization and hasattr(self.simulator, 'visualizer'):
            self.simulator.visualizer.animate_rearrangement()
        
        # Get final configuration
        target_region = self.simulator.field[start_row:end_row, start_col:end_col]
        atoms_in_target = np.sum(target_region)
        
        # Calculate true retention rate: atoms in target square / initial atoms
        retention_rate = atoms_in_target / initial_atoms
        
        # Print summary statistics
        print(f"\nRearrangement Summary:")
        print(f"- Initial atoms: {initial_atoms}")
        print(f"- Target atoms: {self.simulator.side_length * self.simulator.side_length}")
        print(f"- Atoms in target: {atoms_in_target}")
        print(f"- Retention rate: {retention_rate:.2%}")
        print(f"- Total movement steps: {len(self.simulator.movement_history)}")
        print(f"- Total transfer time: {self.simulator.total_transfer_time*1000:.2f} ms")
        print(f"- Total movement time: {self.simulator.movement_time*1000:.2f} ms")
        
        execution_time = time.time() - start_time
        print(f"- Total execution time: {execution_time:.3f} seconds")
        
        self.simulator.target_lattice = self.simulator.field.copy()
        return self.simulator.target_lattice, retention_rate, execution_time

    def process_atoms_below_target_parallel(self):
        """
        Process atoms that are below the target region using parallel movements:
        1. Move all atoms to the right edge of the target zone simultaneously
        2. Then move all atoms upward into the target zone simultaneously
        """
        # Get target region coordinates
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        end_row = start_row + self.simulator.side_length
        end_col = start_col + self.simulator.side_length
        
        # Define target positions for perfect square
        target_positions = set()
        for i in range(self.simulator.side_length):
            for j in range(self.simulator.side_length):
                target_positions.add((start_row + i, start_col + j))
        
        # Main processing loop
        iterations = 0
        max_iterations = 5
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\nBelow-target atoms: Iteration {iterations}")
            
            # Get current atoms and their positions
            all_atoms = set(zip(*np.where(self.simulator.field == 1)))
            atoms_in_target = all_atoms & target_positions
            empty_targets = target_positions - atoms_in_target
            
            # Check if target is full
            if not empty_targets:
                print("Target region completely filled")
                return
                
            # Find atoms below the target region
            atoms_below = set()
            for atom_pos in all_atoms - atoms_in_target:
                row, col = atom_pos
                if row >= end_row and col >= start_col and col < end_col:
                    atoms_below.add(atom_pos)
            
            if not atoms_below:
                print("No atoms below target region")
                break
                
            print(f"Found {len(atoms_below)} atoms below target, {len(empty_targets)} empty target positions")
            
            # 1. Move all atoms to the right edge simultaneously
            move_right_success = self.move_atoms_right_parallel(atoms_below, end_col)
            
            if not move_right_success:
                print("No atoms could be moved right")
                break
                
            # Get updated atom positions after right movement
            all_atoms = set(zip(*np.where(self.simulator.field == 1)))
            atoms_in_target = all_atoms & target_positions
            empty_targets = target_positions - atoms_in_target
            
            # Find atoms at the right edge that can be moved up
            atoms_at_right_edge = set()
            for atom_pos in all_atoms - atoms_in_target:
                row, col = atom_pos
                if col == end_col and row >= end_row:
                    atoms_at_right_edge.add(atom_pos)
            
            # 2. Move all atoms up into target simultaneously
            move_up_success = self.move_atoms_up_into_target_parallel(atoms_at_right_edge, empty_targets)
            
            if not move_up_success:
                print("No atoms could be moved up into target")
                break
        
        # If target still not full, try focused left-up
        target_region = self.simulator.field[start_row:end_row, start_col:end_col]
        if np.sum(target_region) < self.simulator.side_length * self.simulator.side_length:
            # Re-apply left-up movement but focused on target region
            self.focused_left_up_movement(start_row, start_col, end_row, end_col)

    def process_atoms_right_of_target_parallel(self):
        """
        Process atoms that are to the right of the target region using parallel movements:
        1. Move all atoms down to the bottom edge of the target zone simultaneously
        2. Then move all atoms left into the target zone simultaneously
        """
        # Get target region coordinates
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        end_row = start_row + self.simulator.side_length
        end_col = start_col + self.simulator.side_length
        
        # Define target positions for perfect square
        target_positions = set()
        for i in range(self.simulator.side_length):
            for j in range(self.simulator.side_length):
                target_positions.add((start_row + i, start_col + j))
        
        # Main processing loop
        iterations = 0
        max_iterations = 5
        
        while iterations < max_iterations:
            iterations += 1
            print(f"\nRight-of-target atoms: Iteration {iterations}")
            
            # Get current atoms and their positions
            all_atoms = set(zip(*np.where(self.simulator.field == 1)))
            atoms_in_target = all_atoms & target_positions
            empty_targets = target_positions - atoms_in_target
            
            # Check if target is full
            if not empty_targets:
                print("Target region completely filled")
                return
                
            # Find atoms to the right of the target region
            atoms_right = set()
            for atom_pos in all_atoms - atoms_in_target:
                row, col = atom_pos
                if col >= end_col and row >= start_row and row < end_row:
                    atoms_right.add(atom_pos)
            
            if not atoms_right:
                print("No atoms to the right of target region")
                break
                
            print(f"Found {len(atoms_right)} atoms to the right, {len(empty_targets)} empty target positions")
            
            # 1. Move all atoms down to the bottom edge simultaneously
            move_down_success = self.move_atoms_down_parallel(atoms_right, end_row - 1)
            
            if not move_down_success:
                print("No atoms could be moved down")
                break
                
            # Get updated atom positions after down movement
            all_atoms = set(zip(*np.where(self.simulator.field == 1)))
            atoms_in_target = all_atoms & target_positions
            empty_targets = target_positions - atoms_in_target
            
            # Find atoms at the bottom edge that can be moved left
            atoms_at_bottom_edge = set()
            for atom_pos in all_atoms - atoms_in_target:
                row, col = atom_pos
                if row == end_row - 1 and col >= end_col:
                    atoms_at_bottom_edge.add(atom_pos)
            
            # 2. Move all atoms left into target simultaneously
            move_left_success = self.move_atoms_left_into_target_parallel(atoms_at_bottom_edge, empty_targets)
            
            if not move_left_success:
                print("No atoms could be moved left into target")
                break

    def move_atoms_right_parallel(self, atoms_below, target_col):
        """
        Move all atoms to the right simultaneously to the specified target column.
        Returns True if any atoms were moved.
        """
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0
        
        for atom_pos in atoms_below:
            y, x = atom_pos
            
            # Check if path to right is clear
            path_clear = True
            for col in range(x + 1, target_col + 1):
                if self.simulator.field[y, col] == 1:
                    path_clear = False
                    break
            
            if path_clear:
                to_pos = (y, target_col)
                
                # Calculate movement time
                movement_distance = (target_col - x) * self.simulator.SITE_DISTANCE
                movement_time = self.calculate_movement_time(movement_distance)
                transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                total_time = movement_time + transfer_time
                
                moves.append((atom_pos, to_pos, total_time))
                total_batch_time = max(total_batch_time, total_time)
        
        # Execute all moves
        if moves:
            # First update the state
            for from_pos, to_pos, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms right in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_right',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': current_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        return False

    def move_atoms_up_into_target_parallel(self, atoms_at_edge, empty_targets):
        """
        Move all atoms at the right edge up into the target simultaneously.
        Returns True if any atoms were moved.
        """
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0
        
        # Group empty targets by column
        targets_by_column = {}
        for target_pos in empty_targets:
            col = target_pos[1]
            if col not in targets_by_column:
                targets_by_column[col] = []
            targets_by_column[col].sort(key=lambda pos: pos[0])
        
        # Sort target positions in each column from bottom to top
        for col in targets_by_column:
            targets_by_column[col].sort(key=lambda pos: pos[0])
        
        used_targets = set()
        
        for atom_pos in sorted(atoms_at_edge, key=lambda pos: pos[0]):  # Sort from top to bottom
            y, x = atom_pos
            
            # Check if there are available targets in the adjacent column
            left_col = x - 1
            if left_col in targets_by_column and targets_by_column[left_col]:
                target_pos = targets_by_column[left_col][0]  # Get the bottommost target
                
                # Check if path is clear to move up and left
                path_clear = True
                for row in range(y - 1, target_pos[0] - 1, -1):
                    if self.simulator.field[row, x] == 1:
                        path_clear = False
                        break
                
                if path_clear:
                    # Calculate movement time
                    movement_distance = (y - target_pos[0] + 1) * self.simulator.SITE_DISTANCE  # +1 for the left movement
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, target_pos, total_time))
                    total_batch_time = max(total_batch_time, total_time)
                    
                    # Remove the used target
                    targets_by_column[left_col].pop(0)
                    used_targets.add(target_pos)
                    
                    if not targets_by_column[left_col]:
                        del targets_by_column[left_col]
        
        # Execute all moves
        if moves:
            # First update the state
            for from_pos, to_pos, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms up-left into target in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_up_left',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': current_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        return False

    def move_atoms_down_parallel(self, atoms_right, target_row):
        """
        Move all atoms to the right of target down simultaneously to the specified target row.
        Returns True if any atoms were moved.
        """
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0
        
        for atom_pos in atoms_right:
            y, x = atom_pos
            
            # Skip if already at target row
            if y == target_row:
                continue
                
            # If above target row, need to move down
            if y < target_row:
                # Check if path down is clear
                path_clear = True
                for row in range(y + 1, target_row + 1):
                    if self.simulator.field[row, x] == 1:
                        path_clear = False
                        break
                
                if path_clear:
                    to_pos = (target_row, x)
                    
                    # Calculate movement time
                    movement_distance = (target_row - y) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    total_batch_time = max(total_batch_time, total_time)
        
        # Execute all moves
        if moves:
            # First update the state
            for from_pos, to_pos, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms down in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_down',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': current_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        return False

    def move_atoms_left_into_target_parallel(self, atoms_at_edge, empty_targets):
        """
        Move all atoms at the bottom edge left into the target simultaneously.
        Returns True if any atoms were moved.
        """
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0
        
        # Group empty targets by row
        targets_by_row = {}
        for target_pos in empty_targets:
            row = target_pos[0]
            if row not in targets_by_row:
                targets_by_row[row] = []
            targets_by_row[row].append(target_pos)
        
        # Sort target positions in each row from right to left
        for row in targets_by_row:
            targets_by_row[row].sort(key=lambda pos: -pos[1])  # Sort from right to left
        
        used_targets = set()
        
        for atom_pos in sorted(atoms_at_edge, key=lambda pos: pos[1]):  # Sort from left to right
            y, x = atom_pos
            
            # Check if there are available targets in this row
            if y in targets_by_row and targets_by_row[y]:
                target_pos = targets_by_row[y][0]  # Get the rightmost target
                
                # Check if path is clear to move left
                path_clear = True
                for col in range(x - 1, target_pos[1] - 1, -1):
                    if self.simulator.field[y, col] == 1:
                        path_clear = False
                        break
                
                if path_clear:
                    # Calculate movement time
                    movement_distance = (x - target_pos[1]) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, target_pos, total_time))
                    total_batch_time = max(total_batch_time, total_time)
                    
                    # Remove the used target
                    targets_by_row[y].pop(0)
                    used_targets.add(target_pos)
                    
                    if not targets_by_row[y]:
                        del targets_by_row[y]
        
        # Execute all moves
        if moves:
            # First update the state
            for from_pos, to_pos, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms left into target in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_left',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': current_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        return False

    def focused_left_up_movement(self, start_row, start_col, end_row, end_col):
        """
        Apply a focused left-up movement, but only for atoms inside the target region.
        """
        print("\nApplying focused left-up movement inside target region")
        
        # Define the focused region
        focused_field = np.zeros_like(self.simulator.field)
        focused_field[start_row:end_row, start_col:end_col] = self.simulator.field[start_row:end_row, start_col:end_col]
        
        def move_atoms_left() -> bool:
            """Move all atoms in target region as far left as possible."""
            made_moves = False
            current_state = focused_field.copy()
            atoms = []
            
            # Get atoms only in the target region
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    if self.simulator.field[row, col] == 1:
                        atoms.append((row, col))
            
            # Sort atoms by column (left to right)
            atoms.sort(key=lambda pos: (pos[1], -pos[0]))
            
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
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    self.simulator.field[y, x] = 0
                    self.simulator.field[y, target_x] = 1
                    focused_field[y, x] = 0
                    focused_field[y, target_x] = 1
                    made_moves = True
                    
                    total_batch_time = max(total_batch_time, total_time)
            
            if moves:
                atoms_moved = len(moves)
                self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
                self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
                
                print(f"Moving {atoms_moved} atoms left within target (batch time: {total_batch_time*1000:.3f} ms)")
                self.simulator.movement_history.append({
                    'type': 'parallel_left',
                    'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                    'state': current_state.copy(),
                    'iteration': len(self.simulator.movement_history) + 1,
                    'time': total_batch_time
                })
            
            return made_moves
        
        def move_atoms_up() -> bool:
            """Move all atoms in target region as far up as possible."""
            made_moves = False
            current_state = focused_field.copy()
            atoms = []
            
            # Get atoms only in the target region
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    if self.simulator.field[row, col] == 1:
                        atoms.append((row, col))
            
            # Sort atoms by row (bottom to top)
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
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    self.simulator.field[y, x] = 0
                    self.simulator.field[target_y, x] = 1
                    focused_field[y, x] = 0
                    focused_field[target_y, x] = 1
                    made_moves = True
                    
                    total_batch_time = max(total_batch_time, total_time)
            
            if moves:
                atoms_moved = len(moves)
                self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
                self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
                
                print(f"Moving {atoms_moved} atoms up within target (batch time: {total_batch_time*1000:.3f} ms)")
                self.simulator.movement_history.append({
                    'type': 'parallel_up',
                    'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                    'state': current_state.copy(),
                    'iteration': len(self.simulator.movement_history) + 1,
                    'time': total_batch_time
                })
            
            return made_moves
        
        # Alternate between left and up movements until no more moves are possible
        iteration = 1
        while True:
            print(f"Focused left-up iteration {iteration}:")
            
            left_moved = move_atoms_left()
            up_moved = move_atoms_up()
            
            if not (left_moved or up_moved):
                print(f"No more moves possible after {iteration} iterations")
                break
            
            iteration += 1

    def move_all_atoms_right(self) -> bool:
        """
        Move all atoms as far right as possible in a single parallel movement.
        This maximizes parallelism by moving every atom simultaneously to the rightmost
        available position.
        
        Returns:
            bool: True if any atoms were moved, False otherwise
        """
        print("\nMoving all atoms as far right as possible in parallel...")
        
        current_state = self.simulator.field.copy()
        field_height, field_width = self.simulator.field.shape
        moves = []
        total_batch_time = 0.0
        
        # First, for each row, calculate the rightmost possible position for each atom
        # We need to process atoms from right to left to avoid collisions
        for row in range(field_height):
            # Get all atoms in this row
            atoms_in_row = [(row, col) for col in range(field_width) if self.simulator.field[row, col] == 1]
            
            # Sort by column in descending order (right to left)
            atoms_in_row.sort(key=lambda pos: -pos[1])
            
            # Find target positions for each atom
            right_edge = field_width - 1  # Start from the rightmost position
            
            for atom_pos in atoms_in_row:
                y, x = atom_pos
                
                # Find rightmost available position (up to the current right edge)
                target_x = right_edge
                
                while target_x > x:
                    # Check if position is empty in the current state
                    if current_state[y, target_x] == 0:
                        break
                    target_x -= 1
                
                # Update right edge for next atom in this row
                right_edge = target_x - 1
                
                # If the atom can move right, add it to moves
                if target_x > x:
                    to_pos = (y, target_x)
                    
                    # Calculate movement time
                    movement_distance = (target_x - x) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    total_batch_time = max(total_batch_time, total_time)
                    
                    # Update the planning state to mark this position as filled
                    current_state[y, x] = 0
                    current_state[y, target_x] = 1
        
        # Execute all moves if any were found
        if moves:
            # Create a fresh copy for the actual moves
            execution_state = self.simulator.field.copy()
            
            # First update the state by removing all source atoms
            for from_pos, _, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
            
            # Then place all atoms at their target positions
            for _, to_pos, _ in moves:
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms right in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_right',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': execution_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        else:
            print("No atoms could be moved right")
            return False

    def move_under_atoms_to_right_edge(self) -> bool:
        """
        Moves all atoms that are under the target grid to the right edge of the target grid.
        The atoms in each row will be shifted right so the rightmost atom aligns with the target edge.
        
        Returns:
            bool: True if any atoms were moved, False otherwise
        """
        print("\nMoving atoms under target grid to the right edge...")
        
        # Get target region coordinates
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        end_row = start_row + self.simulator.side_length
        end_col = start_col + self.simulator.side_length
        
        # Create a copy of the current field state for planning
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0

        # Find all atoms under the target grid and group them by row
        atoms_by_row = {}
        for row in range(end_row, self.simulator.field_size[0]):
            row_atoms = []
            for col in range(start_col, end_col):
                if self.simulator.field[row, col] == 1:
                    row_atoms.append((row, col))
            if row_atoms:
                atoms_by_row[row] = row_atoms

        if not atoms_by_row:
            print("No atoms found under the target grid.")
            return False

        total_atoms = sum(len(atoms) for atoms in atoms_by_row.values())
        print(f"Found {total_atoms} atoms under the target grid.")
        
        # Process each row
        for row, atoms in atoms_by_row.items():
            # Sort atoms by column (right to left)
            atoms.sort(key=lambda pos: -pos[1])
            
            # Calculate how much space we need between atoms
            num_atoms = len(atoms)
            if num_atoms == 0:
                continue
                
            # Find rightmost available position
            rightmost_col = atoms[0][1]  # Column of rightmost atom
            shift_amount = end_col - 1 - rightmost_col
            
            if shift_amount <= 0:
                continue  # This row is already at or past the target edge
            
            # Check if we can move all atoms right by this amount
            can_move = True
            for atom_pos in atoms:
                _, col = atom_pos
                # Check if the path is clear
                for check_col in range(col + 1, min(col + shift_amount + 1, self.simulator.field_size[1])):
                    if current_state[row, check_col] == 1 and (row, check_col) not in atoms:
                        can_move = False
                        break
                if not can_move:
                    break
            
            if not can_move:
                continue
            
            # Move all atoms in this row
            for atom_pos in atoms:
                _, col = atom_pos
                new_col = col + shift_amount
                if new_col >= self.simulator.field_size[1]:
                    continue
                    
                from_pos = atom_pos
                to_pos = (row, new_col)
                
                # Calculate movement time
                movement_distance = shift_amount * self.simulator.SITE_DISTANCE
                movement_time = self.calculate_movement_time(movement_distance)
                transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                total_time = movement_time + transfer_time
                
                moves.append((from_pos, to_pos, total_time))
                total_batch_time = max(total_batch_time, total_time)
                
                # Update planning state
                current_state[row, col] = 0
                current_state[row, new_col] = 1
        
        # Execute all moves if any were found
        if moves:
            # Update the actual field state
            execution_state = self.simulator.field.copy()
            
            # First remove all source atoms
            for from_pos, _, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
            
            # Then place all atoms at their target positions
            for _, to_pos, _ in moves:
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms under target to right edge in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_right',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': execution_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        print("No atoms could be moved right (paths blocked or already aligned)")
        return False

    def move_under_atoms_up(self) -> bool:
        """
        Moves all atoms that are under the target grid upward in parallel into the target grid.
        Each atom will move straight up to the highest available position.
        
        Returns:
            bool: True if any atoms were moved, False otherwise
        """
        print("\nMoving atoms under target grid upward in parallel...")
        
        # Get target region coordinates
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        end_row = start_row + self.simulator.side_length
        end_col = start_col + self.simulator.side_length
        
        # Create a copy of the current field state for planning
        current_state = self.simulator.field.copy()
        moves = []
        total_batch_time = 0.0

        # Find all atoms under the target grid and group them by column
        atoms_by_column = {}
        for col in range(start_col, end_col):
            col_atoms = []
            for row in range(end_row, self.simulator.field_size[0]):
                if self.simulator.field[row, col] == 1:
                    col_atoms.append((row, col))
            if col_atoms:
                atoms_by_column[col] = col_atoms

        if not atoms_by_column:
            print("No atoms found under the target grid.")
            return False

        total_atoms = sum(len(atoms) for atoms in atoms_by_column.values())
        print(f"Found {total_atoms} atoms under the target grid.")
        
        # Process each column
        for col, atoms in atoms_by_column.items():
            # Sort atoms in column from top to bottom
            atoms.sort(key=lambda pos: pos[0])
            
            # For each atom, find the highest position it can reach
            available_row = end_row - 1  # Start from bottom of target zone
            
            for atom_pos in atoms:
                row, _ = atom_pos
                
                # Find highest empty position in this column
                target_row = available_row
                while target_row > start_row:
                    if current_state[target_row - 1, col] == 1:
                        break
                    target_row -= 1
                
                # If we can move up, add the move
                if target_row < row:
                    to_pos = (target_row, col)
                    
                    # Calculate movement time
                    movement_distance = (row - target_row) * self.simulator.SITE_DISTANCE
                    movement_time = self.calculate_movement_time(movement_distance)
                    transfer_time = 2 * self.simulator.TRAP_TRANSFER_TIME
                    total_time = movement_time + transfer_time
                    
                    moves.append((atom_pos, to_pos, total_time))
                    total_batch_time = max(total_batch_time, total_time)
                    
                    # Update planning state
                    current_state[row, col] = 0
                    current_state[target_row, col] = 1
                    
                    # Update available row for next atom in this column
                    available_row = target_row + 1
        
        # Execute all moves if any were found
        if moves:
            # Update the actual field state
            execution_state = self.simulator.field.copy()
            
            # First remove all source atoms
            for from_pos, _, _ in moves:
                self.simulator.field[from_pos[0], from_pos[1]] = 0
            
            # Then place all atoms at their target positions
            for _, to_pos, _ in moves:
                self.simulator.field[to_pos[0], to_pos[1]] = 1
            
            # Update timing counters
            atoms_moved = len(moves)
            self.simulator.total_transfer_time += atoms_moved * self.simulator.TRAP_TRANSFER_TIME * 2
            self.simulator.movement_time += total_batch_time - 2 * self.simulator.TRAP_TRANSFER_TIME
            
            # Add to movement history
            print(f"Moving {atoms_moved} atoms up in parallel (batch time: {total_batch_time*1000:.3f} ms)")
            self.simulator.movement_history.append({
                'type': 'parallel_up',
                'moves': [(from_pos, to_pos) for from_pos, to_pos, _ in moves],
                'state': execution_state.copy(),
                'iteration': len(self.simulator.movement_history) + 1,
                'time': total_batch_time
            })
            
            return True
        
        print("No atoms could be moved up (paths blocked or no atoms found)")
        return False

    def repeat_right_up_movements(self) -> bool:
        """
        Repeatedly move atoms to right edge then up until no more movements are possible.
        Each iteration consists of:
        1. Move all atoms under target zone to right edge in parallel
        2. Move all atoms under target zone upward in parallel
        
        Returns:
            bool: True if any movements were made, False otherwise
        """
        print("\nStarting repeated right-edge and upward movements...")
        
        iteration = 1
        made_moves = False
        
        while True:
            print(f"\nRight-Up Iteration {iteration}:")
            
            # Phase 1: Move to right edge
            right_moved = self.move_under_atoms_to_right_edge()
            
            # Phase 2: Move upward
            up_moved = self.move_under_atoms_up()
            
            # If neither movement was possible, we're done
            if not (right_moved or up_moved):
                print(f"No more moves possible after {iteration} iterations")
                break
            
            made_moves = True
            iteration += 1
        
        return made_moves