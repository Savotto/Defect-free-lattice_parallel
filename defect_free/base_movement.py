"""
Base movement module containing common functionality for both movement strategies.
"""
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
import heapq

class BaseMovementManager:
    """
    Base class for movement management with physical constraints.
    Contains common functionality used by both center and corner movement strategies.
    """
    
    def __init__(self, simulator):
        """Initialize the movement manager with a reference to the simulator."""
        self.simulator = simulator
        self.target_region = None
        self._movement_time_cache = {}  # Cache for movement time calculations
    
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

    # This method will be implemented by child classes
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
        for row in range(self.simulator.initial_size[0]):
            for col in range(self.simulator.initial_size[1]):
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