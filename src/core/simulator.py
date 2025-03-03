"""
Core simulator module defining the LatticeSimulator class with initialization and constants.
"""
import numpy as np
from typing import Tuple, Optional, List, Set
import time

from ..algorithms.pathfinding import PathFinder
from ..algorithms.movement import MovementManager


class LatticeSimulator:
    # Physical constants from PowerMove research
    SITE_DISTANCE = 15.0  # Minimal spatial distance between sites in micrometers
    MAX_ACCELERATION = 2750.0  # Maximum acceleration to maintain qubit fidelity in m/s^2
    TRAP_TRANSFER_TIME = 15e-6  # Transfer time between SLM and AOD traps in seconds (15µs)
    TRAP_TRANSFER_FIDELITY = 0.999  # Fidelity of transfer between SLM and AOD traps (99.9%)
    
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
        self.field_size = (110, 110)   # Larger field size for movement
        self.field = None            # Temporary holding space for atoms during movement
        self.active_lasers = {'rows': set(), 'cols': set()}  # Track active lasers
        self.movement_history = []    # Store movement steps for animation
        self.target_lattice = None   # Target configuration after rearrangement
        self.total_transfer_time = 0.0  # Track total time spent in trap transfers
        self.movement_time = 0.0      # Track total time spent in atom movement
        
        # Initialize sub-modules
        self.path_finder = PathFinder(self)
        self.movement_manager = MovementManager(self)
    
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
    
    def rearrange_atoms(self, show_visualization: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Rearrange atoms using a three-phase approach with parallel movements.
        
        Args:
            show_visualization: Whether to show the animation of the rearrangement process
        Returns:
            Tuple of (target_lattice, retention_rate, execution_time)
        """
        # Delegate to the movement manager's new implementation
        return self.movement_manager.rearrange_atoms(show_visualization)
    
    def animate_rearrangement(self) -> None:
        """Animate the rearrangement process if visualization module is imported."""
        if hasattr(self, 'visualizer'):
            self.visualizer.animate_rearrangement()
        else:
            print("Visualization module not initialized. Please import and initialize the visualizer first.")
            
    def visualize_lattices(self) -> None:
        """Visualize initial and target lattice side by side if visualization module is imported."""
        if hasattr(self, 'visualizer'):
            self.visualizer.visualize_lattices()
        else:
            print("Visualization module not initialized. Please import and initialize the visualizer first.")