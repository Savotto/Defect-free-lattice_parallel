import numpy as np
from typing import Tuple, Optional, List, Set
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LatticeSimulator:
    def __init__(self, initial_size: Tuple[int, int], occupation_prob: float):
        """
        Initialize the lattice simulator with both SLM and AOD trap systems.
        
        Args:
            initial_size: Tuple of (rows, cols) for initial lattice size
            occupation_prob: Probability of a site being occupied (0 to 1)
        """
        self.initial_size = initial_size
        self.occupation_prob = occupation_prob
        self.slm_lattice = None      # SLM trap locations and atoms
        self.slm_traps = None        # Active SLM trap locations
        self.field_size = (110, 110)   # Larger field size (10m x 10m) for movement
        self.field = None            # Temporary holding space for atoms during movement
        self.active_lasers = {'rows': set(), 'cols': set()}  # Track active lasers
        self.movement_history = []    # Store movement steps for animation
        self.target_lattice = None   # Target configuration after rearrangement
        
    def generate_initial_lattice(self) -> np.ndarray:
        """Generate initial lattice with random atom placement in SLM traps."""
        # Generate initial random atom placement
        random_values = np.random.random(self.initial_size)
        self.slm_lattice = (random_values < self.occupation_prob).astype(int)
        
        # Initialize the larger field for movement
        self.field = np.zeros(self.field_size)
        # Copy initial configuration to center of field
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        self.field[start_row:start_row+self.initial_size[0], 
                  start_col:start_col+self.initial_size[1]] = self.slm_lattice
        
        # Count total atoms and calculate target size
        self.total_atoms = np.sum(self.slm_lattice)
        self.side_length = int(np.floor(np.sqrt(self.total_atoms)))
        
        return self.slm_lattice

    def is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                     active_atoms: Set[Tuple[int, int]]) -> bool:
        """Check if moving an atom from one position to another is valid."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check if move is within field bounds
        if not (0 <= to_row < self.field_size[0] and 0 <= to_col < self.field_size[1]):
            return False
            
        # Check if destination is occupied
        if self.field[to_row, to_col] == 1:
            return False
            
        # Check if path crosses any occupied positions
        if from_row == to_row:  # Horizontal movement
            min_col, max_col = min(from_col, to_col), max(from_col, to_col)
            for col in range(min_col + 1, max_col):
                if self.field[from_row, col] == 1 and (from_row, col) not in active_atoms:
                    return False
        elif from_col == to_col:  # Vertical movement
            min_row, max_row = min(from_row, to_row), max(from_row, to_row)
            for row in range(min_row + 1, max_row):
                if self.field[row, from_col] == 1 and (row, from_col) not in active_atoms:
                    return False
        
        return True
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                     current_state: np.ndarray) -> float:
        """Calculate the cost of moving an atom from one position to another."""
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
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        target_penalty = 0
        if y2 < start_row or y2 >= start_row + self.side_length:
            target_penalty += 5
        if x2 < start_col or x2 >= start_col + self.side_length:
            target_penalty += 5
            
        return distance_cost + crossing_penalty + target_penalty

    def find_best_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                    current_state: np.ndarray) -> List[Tuple[int, int]]:
        """Find the best path from source to destination using A* pathfinding."""
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
                if not (0 <= next_pos[0] < self.field_size[0] and 
                    0 <= next_pos[1] < self.field_size[1]):
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
        
        return None

    def move_atoms_with_constraints(self) -> None:
        """Move atoms alternating between left and up movements until no more moves are possible."""
        print(f"\nStarting alternating left-up movement algorithm...")
        
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        target_end_row = start_row + self.side_length
        target_end_col = start_col + self.side_length
        
        def move_atoms_left() -> bool:
            """Move all atoms as far left as possible. Returns True if any moves were made."""
            made_moves = False
            current_state = self.field.copy()
            atoms = list(zip(*np.where(current_state == 1)))
            
            # Sort atoms by column (left to right) to prioritize leftmost movements
            atoms.sort(key=lambda pos: (pos[1], -pos[0]))  # Negative row for top priority
            
            moves = []
            for atom_pos in atoms:
                y, x = atom_pos
                target_x = x
                
                # Find leftmost available position
                while target_x > start_col:
                    if self.field[y, target_x - 1] == 0:  # Next position is empty
                        target_x -= 1
                    else:
                        break
                
                if target_x < x:  # Found a valid left move
                    moves.append((atom_pos, (y, target_x)))
                    self.field[y, x] = 0
                    self.field[y, target_x] = 1
                    made_moves = True
            
            if moves:
                print(f"Moving {len(moves)} atoms left")
                self.movement_history.append({
                    'type': 'parallel_left',
                    'moves': moves,
                    'state': current_state.copy(),
                    'iteration': len(self.movement_history) + 1
                })
            
            return made_moves
            
        def move_atoms_up() -> bool:
            """Move all atoms as far up as possible. Returns True if any moves were made."""
            made_moves = False
            current_state = self.field.copy()
            atoms = list(zip(*np.where(current_state == 1)))
            
            # Sort atoms by row (bottom to top) to prioritize upward movements
            atoms.sort(key=lambda pos: (pos[0], pos[1]))
            
            moves = []
            for atom_pos in atoms:
                y, x = atom_pos
                target_y = y
                
                # Find highest available position
                while target_y > start_row:
                    if self.field[target_y - 1, x] == 0:  # Position above is empty
                        target_y -= 1
                    else:
                        break
                
                if target_y < y:  # Found a valid up move
                    moves.append((atom_pos, (target_y, x)))
                    self.field[y, x] = 0
                    self.field[target_y, x] = 1
                    made_moves = True
            
            if moves:
                print(f"Moving {len(moves)} atoms up")
                self.movement_history.append({
                    'type': 'parallel_up',
                    'moves': moves,
                    'state': current_state.copy(),
                    'iteration': len(self.movement_history) + 1
                })
            
            return made_moves
        
        # Keep alternating between left and up movements until no more moves are possible
        iteration = 1
        while True:
            print(f"\nIteration {iteration}:")
            
            # Try moving left
            left_moved = move_atoms_left()
            if left_moved:
                print(f"Iteration {iteration}: Successfully moved atoms left")
            
            # Try moving up
            up_moved = move_atoms_up()
            if up_moved:
                print(f"Iteration {iteration}: Successfully moved atoms up")
            
            # If neither movement was possible, we're done
            if not (left_moved or up_moved):
                print(f"\nNo more moves possible after {iteration} iterations")
                break
            
            iteration += 1
        
        # Check final state
        target_region = self.field[start_row:start_row+self.side_length, 
                                 start_col:start_col+self.side_length]
        final_count = np.sum(target_region)
        print(f"\nMovement complete.")
        print(f"Atoms in target region: {final_count}/{self.side_length * self.side_length}")

    def animate_rearrangement(self) -> None:
        """Animate the atom rearrangement process with parallel movements."""
        if not self.movement_history:
            print("No movements to animate")
            return
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        def update(frame):
            ax.clear()
            movement = self.movement_history[frame]
            
            # Plot grid
            ax.set_xticks(np.arange(-0.5, self.field_size[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.field_size[0], 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Calculate boundaries
            start_row = (self.field_size[0] - self.initial_size[0]) // 2
            start_col = (self.field_size[1] - self.initial_size[1]) // 2
            
            # Plot SLM traps as circles
            for i in range(self.initial_size[0]):
                for j in range(self.initial_size[1]):
                    trap = plt.Circle((start_col + j, start_row + i), 0.4,
                                    facecolor='none', edgecolor='blue',
                                    linestyle='-', linewidth=1, alpha=0.7)
                    ax.add_patch(trap)
            
            # Plot stationary atoms
            moving_from = {pos[0] for pos in movement['moves']}
            atom_positions = np.where(movement['state'] == 1)
            for y, x in zip(atom_positions[0], atom_positions[1]):
                if (y, x) not in moving_from:
                    circle = plt.Circle((x, y), 0.3,
                                     facecolor='yellow', edgecolor='black',
                                     linewidth=1)
                    ax.add_patch(circle)
            
            # Draw active AOD trap lines
            if movement['type'] == 'parallel_left':
                # Draw complete row lines for each moving atom
                for from_pos, to_pos in movement['moves']:
                    y = from_pos[0]
                    ax.axhline(y=y, color='red', linestyle='-', 
                             linewidth=1, alpha=0.3, zorder=1)
                    # Draw vertical lines for both source and destination
                    ax.axvline(x=from_pos[1], color='red', linestyle='-',
                             linewidth=1.5, alpha=0.5, zorder=2)
                    ax.axvline(x=to_pos[1], color='red', linestyle='-',
                             linewidth=1.5, alpha=0.5, zorder=2)
                    
            else:  # parallel_up
                # Draw complete column lines for each moving atom
                for from_pos, to_pos in movement['moves']:
                    x = from_pos[1]
                    ax.axvline(x=x, color='red', linestyle='-',
                             linewidth=1, alpha=0.3, zorder=1)
                    # Draw horizontal lines for both source and destination
                    ax.axhline(y=from_pos[0], color='red', linestyle='-',
                             linewidth=1.5, alpha=0.5, zorder=2)
                    ax.axhline(y=to_pos[0], color='red', linestyle='-',
                             linewidth=1.5, alpha=0.5, zorder=2)
            
            # Draw movement arrows and atoms
            for from_pos, to_pos in movement['moves']:
                # Draw movement arrow
                arrow_props = dict(arrowstyle='->,head_width=0.5,head_length=0.8',
                                 color='red', linestyle='-',
                                 linewidth=2, alpha=0.8)
                ax.annotate('', xy=(to_pos[1], to_pos[0]),
                          xytext=(from_pos[1], from_pos[0]),
                          arrowprops=arrow_props)
                
                # Draw source position (empty circle)
                source = plt.Circle((from_pos[1], from_pos[0]), 0.3,
                                  facecolor='none', edgecolor='red',
                                  linestyle='--', linewidth=2)
                ax.add_patch(source)
                
                # Draw destination position (filled circle)
                destination = plt.Circle((to_pos[1], to_pos[0]), 0.3,
                                       facecolor='yellow', edgecolor='red',
                                       linewidth=2)
                ax.add_patch(destination)
            
            # Add step information
            info_text = [
                f'Step {movement["iteration"]}: '
                f'{"Left" if movement["type"] == "parallel_left" else "Up"} Movement',
                f'Moving {len(movement["moves"])} atoms in parallel',
                f'Frame: {frame + 1}/{len(self.movement_history)}'
            ]
            
            for i, text in enumerate(info_text):
                ax.text(0.02, 0.98 - i*0.04, text,
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_xlim(-0.5, self.field_size[1] - 0.5)
            ax.set_ylim(self.field_size[0] - 0.5, -0.5)
            ax.set_title('Parallel Atom Movement Sequence')
        
        # Create animation with longer interval to see movements clearly
        anim = animation.FuncAnimation(fig, update, frames=len(self.movement_history),
                                     interval=2000, repeat=False)
        plt.show()

    def rearrange_atoms(self, show_visualization: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Rearrange atoms using AOD trap movements to create a defect-free lattice.
        Args:
            show_visualization: Whether to show the animation of the rearrangement process
        Returns:
            Tuple of (target_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        self.movement_history = []
        
        # Store initial atom count
        initial_atoms = self.total_atoms
        target_square_size = self.side_length * self.side_length
        
        print(f"\nRearranging {initial_atoms} atoms to form {self.side_length}x{self.side_length} square...")
        
        # First move atoms towards top-left using parallel movements
        self.move_atoms_with_constraints()
        
        # Then ensure perfect lattice formation
        if np.sum(self.field) >= target_square_size:  # Only if we have enough atoms
            self.fill_target_region()
        
        # Animate the rearrangement only if visualization is enabled
        if show_visualization:
            self.animate_rearrangement()
        
        # Get final configuration
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        
        # Count atoms in target square region
        target_region = self.field[start_row:start_row+self.side_length, 
                                 start_col:start_col+self.side_length]
        atoms_in_target = np.sum(target_region)
        
        # Calculate true retention rate: atoms in target square / initial atoms
        retention_rate = atoms_in_target / initial_atoms
        
        execution_time = time.time() - start_time
        self.target_lattice = self.field.copy()
        return self.target_lattice, retention_rate, execution_time

    def fill_target_region(self) -> None:
        """Fill the target region to ensure a perfect lattice."""
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        target_filled = False
        attempts = 0
        max_attempts = 5  # Increased number of attempts
        
        while not target_filled and attempts < max_attempts:
            attempts += 1
            # Get current atom positions
            atoms = set(zip(*np.where(self.field == 1)))
            
            # Define target positions for perfect square
            target_positions = set()
            for i in range(self.side_length):
                for j in range(self.side_length):
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
                        current_state = self.field.copy()
                        current_state[atom_pos[0], atom_pos[1]] = 0  # Remove source atom
                        
                        # Try both movement patterns
                        path1_clear = True  # Horizontal then Vertical
                        path2_clear = True  # Vertical then Horizontal
                        
                        # Check horizontal then vertical
                        intermediate1 = (atom_pos[0], target_pos[1])
                        if not self.check_path_clear(atom_pos, intermediate1, current_state):
                            path1_clear = False
                        elif not self.check_path_clear(intermediate1, target_pos, current_state):
                            path1_clear = False
                            
                        # Check vertical then horizontal
                        intermediate2 = (target_pos[0], atom_pos[1])
                        if not self.check_path_clear(atom_pos, intermediate2, current_state):
                            path2_clear = False
                        elif not self.check_path_clear(intermediate2, target_pos, current_state):
                            path2_clear = False
                        
                        # Calculate score based on distance and path clarity
                        score = dist
                        if not path1_clear:
                            score += 100
                        if not path2_clear:
                            score += 100
                        
                        # Prefer atoms that aren't in the target region already
                        if atom_pos[0] >= start_row and atom_pos[0] < start_row + self.side_length and \
                           atom_pos[1] >= start_col and atom_pos[1] < start_col + self.side_length:
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
                current_state = self.field.copy()
                
                # Try both movement patterns and choose the better one
                path1_ok = True
                path2_ok = True
                
                # Pattern 1: Horizontal then Vertical
                intermediate1 = (current_pos[0], target_pos[1])
                if not self.check_path_clear(current_pos, intermediate1, current_state):
                    path1_ok = False
                elif not self.check_path_clear(intermediate1, target_pos, current_state):
                    path1_ok = False
                    
                # Pattern 2: Vertical then Horizontal
                intermediate2 = (target_pos[0], current_pos[1])
                if not self.check_path_clear(current_pos, intermediate2, current_state):
                    path2_ok = False
                elif not self.check_path_clear(intermediate2, target_pos, current_state):
                    path2_ok = False
                
                # Choose the valid path, preferring the one that moves vertically first for rightmost column
                if target_pos[1] == start_col + self.side_length - 1 and path2_ok:
                    # For rightmost column, prefer vertical movement first
                    self.move_atom_through_path(current_pos, intermediate2, target_pos)
                elif path1_ok:
                    self.move_atom_through_path(current_pos, intermediate1, target_pos)
                elif path2_ok:
                    self.move_atom_through_path(current_pos, intermediate2, target_pos)
                else:
                    print(f"Warning: Could not find valid path from {current_pos} to {target_pos}")
            
            # Check if target region is complete
            target_region = self.field[start_row:start_row+self.side_length, 
                                     start_col:start_col+self.side_length]
            if np.sum(target_region) == self.side_length * self.side_length:
                target_filled = True
                print(f"Perfect lattice achieved after {attempts} filling attempts")
                break
            else:
                atoms_placed = np.sum(target_region)
                print(f"Filling attempt {attempts}: {atoms_placed}/{self.side_length * self.side_length} atoms in target")
        
        # Final verification
        target_region = self.field[start_row:start_row+self.side_length, 
                                 start_col:start_col+self.side_length]
        final_count = np.sum(target_region)
        print(f"Final target region contains {final_count}/{self.side_length * self.side_length} atoms")

    def move_atom_through_path(self, start_pos: Tuple[int, int], intermediate_pos: Tuple[int, int], 
                             end_pos: Tuple[int, int]) -> None:
        """Helper method to move an atom through a path with an intermediate position."""
        # Move to intermediate position
        self.movement_history.append({
            'type': 'parallel_left' if intermediate_pos[1] < start_pos[1] else 'parallel_right'
            if intermediate_pos[1] != start_pos[1] else
            'parallel_up' if intermediate_pos[0] < start_pos[0] else 'parallel_down',
            'moves': [(start_pos, intermediate_pos)],
            'state': self.field.copy(),
            'iteration': len(self.movement_history) + 1
        })
        self.field[start_pos[0], start_pos[1]] = 0
        self.field[intermediate_pos[0], intermediate_pos[1]] = 1
        
        # Move to final position
        self.movement_history.append({
            'type': 'parallel_left' if end_pos[1] < intermediate_pos[1] else 'parallel_right'
            if end_pos[1] != intermediate_pos[1] else
            'parallel_up' if end_pos[0] < intermediate_pos[0] else 'parallel_down',
            'moves': [(intermediate_pos, end_pos)],
            'state': self.field.copy(),
            'iteration': len(self.movement_history) + 1
        })
        self.field[intermediate_pos[0], intermediate_pos[1]] = 0
        self.field[end_pos[0], end_pos[1]] = 1

    def check_path_clear(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                        current_state: np.ndarray) -> bool:
        """Check if path between two points is clear of other atoms."""
        y1, x1 = from_pos
        y2, x2 = to_pos
        
        if y1 == y2:  # Horizontal movement
            min_x, max_x = min(x1, x2), max(x1, x2)
            return all(current_state[y1, x] == 0 for x in range(min_x + 1, max_x))
        elif x1 == x2:  # Vertical movement
            min_y, max_y = min(y1, y2), max(y1, y2)
            return all(current_state[y, x1] == 0 for y in range(min_y + 1, max_y))
        
        return False  # Not an orthogonal move

    def calculate_target_positions(self) -> np.ndarray:
        """Calculate target positions for a defect-free square lattice in top-left of the SLM region."""
        total_atoms = np.sum(self.field)
        side_length = int(np.floor(np.sqrt(total_atoms)))
        self.target_size = (side_length, side_length)
        
        # Define target region within the SLM grid:
        start_row = (self.field_size[0] - self.initial_size[0]) // 2  # SLM region offset
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        target = np.zeros_like(self.field)
        for i in range(side_length):
            for j in range(side_length):
                target[start_row + i, start_col + j] = 1
        return target
    
    def visualize_lattices(self) -> None:
        """
        Visualize the initial and final configurations side by side using the full field view.
        Shows circular SLM traps (blue circles) and atoms (yellow circles) in the larger field.
        """
        if self.slm_lattice is None:
            raise ValueError("Lattice must be generated first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.3)
        
        def plot_configuration(ax, state):
            # Plot major grid for full field
            ax.set_xticks(np.arange(0, self.field_size[1], 1))
            ax.set_yticks(np.arange(0, self.field_size[0], 1))
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Calculate SLM region boundaries
            start_row = (self.field_size[0] - self.initial_size[0]) // 2
            start_col = (self.field_size[1] - self.initial_size[1]) // 2
            
            # Plot outfield background
            outfield = plt.Rectangle((-0.5, -0.5), self.field_size[1], self.field_size[0],
                                   facecolor='lightgray', alpha=0.1)
            ax.add_patch(outfield)
            
            # Plot SLM region with different background
            slm_rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                                   self.initial_size[1], self.initial_size[0],
                                   facecolor='white', edgecolor='blue',
                                   linestyle='-', linewidth=2, alpha=0.3)
            ax.add_patch(slm_rect)
            
            # Plot all SLM traps as circles
            for i in range(self.initial_size[0]):
                for j in range(self.initial_size[1]):
                    trap = plt.Circle((start_col + j, start_row + i), 0.4,
                                    facecolor='none', edgecolor='blue',
                                    linestyle='-', linewidth=1, alpha=0.7)
                    ax.add_patch(trap)
            
            # Plot all atoms in yellow
            atom_positions = np.where(state == 1)
            for y, x in zip(atom_positions[0], atom_positions[1]):
                circle = plt.Circle((x, y), 0.3,
                                 facecolor='yellow', edgecolor='black',
                                 linewidth=1.5)
                ax.add_patch(circle)
            
            ax.set_xlim(-0.5, self.field_size[1] - 0.5)
            ax.set_ylim(self.field_size[0] - 0.5, -0.5)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        # Plot initial configuration using the field state
        plot_configuration(ax1, self.field)
        ax1.set_title('Initial Configuration', pad=20)
        
        # Plot final configuration
        if self.target_lattice is not None:
            plot_configuration(ax2, self.target_lattice)
            ax2.set_title('Final Configuration', pad=20)
        
        plt.tight_layout()
        plt.show()