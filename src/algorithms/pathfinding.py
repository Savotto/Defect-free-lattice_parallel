"""
Pathfinding algorithms for the lattice simulator.
"""
from typing import List, Tuple, Dict, Set
import numpy as np


class PathFinder:
    def __init__(self, simulator):
        """
        Initialize the PathFinder with a reference to the simulator.
        
        Args:
            simulator: The LatticeSimulator instance
        """
        self.simulator = simulator
    
    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                     active_atoms: Set[Tuple[int, int]]) -> bool:
        """
        Check if moving an atom from one position to another is valid.
        
        Args:
            from_pos: Source position (row, col)
            to_pos: Destination position (row, col)
            active_atoms: Set of atom positions that are being moved and should be ignored in collision detection
            
        Returns:
            True if the move is valid, False otherwise
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check if move is within field bounds
        if not (0 <= to_row < self.simulator.field_size[0] and 0 <= to_col < self.simulator.field_size[1]):
            return False
            
        # Check if destination is occupied
        if self.simulator.field[to_row, to_col] == 1:
            return False
            
        # Check if path crosses any occupied positions
        if from_row == to_row:  # Horizontal movement
            min_col, max_col = min(from_col, to_col), max(from_col, to_col)
            for col in range(min_col + 1, max_col):
                if self.simulator.field[from_row, col] == 1 and (from_row, col) not in active_atoms:
                    return False
        elif from_col == to_col:  # Vertical movement
            min_row, max_row = min(from_row, to_row), max(from_row, to_row)
            for row in range(min_row + 1, max_row):
                if self.simulator.field[row, from_col] == 1 and (row, from_col) not in active_atoms:
                    return False
        
        return True
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                     current_state: np.ndarray) -> float:
        """
        Calculate the cost of moving an atom from one position to another.
        
        Args:
            from_pos: Source position (row, col)
            to_pos: Destination position (row, col)
            current_state: Current state of the field
        
        Returns:
            Cost value for the movement (lower is better)
        """
        y1, x1 = from_pos
        y2, x2 = to_pos
        
        # Base distance cost
        distance_cost = abs(y2 - y1) + abs(x2 - x1)
        
        # Penalty for crossing other atoms
        crossing_penalty = 0
        if y1 == y2:  # Horizontal movement
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                if current_state[y1, x] == 1:
                    crossing_penalty += 10
        elif x1 == x2:  # Vertical movement
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                if current_state[y, x1] == 1:
                    crossing_penalty += 10
                    
        # Penalty for moving away from target region
        start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
        start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
        target_penalty = 0
        if y2 < start_row or y2 >= start_row + self.simulator.side_length:
            target_penalty += 5
        if x2 < start_col or x2 >= start_col + self.simulator.side_length:
            target_penalty += 5
            
        return distance_cost + crossing_penalty + target_penalty
    
    def find_best_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                    current_state: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find the best path from source to destination using A* pathfinding.
        
        Args:
            from_pos: Source position (row, col)
            to_pos: Destination position (row, col)
            current_state: Current state of the field
            
        Returns:
            List of positions forming the path from source to destination
        """
        def manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        open_set = {from_pos}
        came_from = {}
        g_score = {from_pos: 0}
        f_score = {from_pos: manhattan_distance(from_pos, to_pos)}
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            if current == to_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(from_pos)
                return path[::-1]
                
            open_set.remove(current)
            y, x = current
            
            # Consider only orthogonal movements
            for next_pos in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]:
                if not (0 <= next_pos[0] < self.simulator.field_size[0] and 
                    0 <= next_pos[1] < self.simulator.field_size[1]):
                    continue
                    
                if current_state[next_pos[0], next_pos[1]] == 1:
                    continue
                    
                tentative_g_score = g_score[current] + self.get_movement_cost(
                    current, next_pos, current_state)
                    
                if tentative_g_score < g_score.get(next_pos, float('inf')):
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score[next_pos] = tentative_g_score + manhattan_distance(next_pos, to_pos)
                    open_set.add(next_pos)
        
        return None  # No path found
    
    def check_path_clear(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                        current_state: np.ndarray) -> bool:
        """
        Check if path between two points is clear of other atoms.
        
        Args:
            from_pos: Source position (row, col)
            to_pos: Destination position (row, col)
            current_state: Current state of the field
            
        Returns:
            True if path is clear, False otherwise
        """
        y1, x1 = from_pos
        y2, x2 = to_pos
        
        if y1 == y2:  # Horizontal movement
            min_x, max_x = min(x1, x2), max(x1, x2)
            return all(current_state[y1, x] == 0 for x in range(min_x + 1, max_x))
        elif x1 == x2:  # Vertical movement
            min_y, max_y = min(y1, y2), max(y1, y2)
            return all(current_state[y, x1] == 0 for y in range(min_y + 1, max_y))
        
        return False  # Not an orthogonal move