"""
Core simulator module defining the LatticeSimulator class with initialization and constants.
"""
import numpy as np
import time
from typing import Tuple, Dict, Optional
from .movement import MovementManager

class LatticeSimulator:
    """
    Simulates a quantum atom lattice with physical constraints.
    """
    # Physical constants
    SITE_DISTANCE = 15.0  # μm
    MAX_ACCELERATION = 2750.0  # m/s²
    TRAP_TRANSFER_TIME = 15e-6  # seconds (15μs)
    TRAP_TRANSFER_FIDELITY = 1.0  # 100% fidelity for now (testing)
    
    def __init__(self, 
                 initial_size: Tuple[int, int] = (50, 50),
                 occupation_prob: float = 0.5,
                 physical_constraints: Dict = None):
        """
        Initialize the lattice simulator with configurable physical constraints.
        
        Args:
            initial_size: Initial lattice dimensions (rows, columns)
            occupation_prob: Probability of atom occupation (0.0 to 1.0)
            physical_constraints: Override default physical constraints
        """
        self.initial_size = initial_size
        self.occupation_prob = occupation_prob
        self.field_size = (110, 110)  # Larger field for movement space
        
        # Initialize lattices
        self.slm_lattice = None  # Initial lattice
        self.field = None        # Working lattice during rearrangement
        self.target_lattice = None  # Final target lattice
        
        # Size of the target defect-free region
        self.side_length = min(initial_size)
        self.total_atoms = 0
        
        # Configure physical constraints
        self.constraints = {
            'site_distance': self.SITE_DISTANCE,
            'max_acceleration': self.MAX_ACCELERATION,
            'trap_transfer_time': self.TRAP_TRANSFER_TIME,
            'trap_transfer_fidelity': self.TRAP_TRANSFER_FIDELITY
        }
        
        if physical_constraints:
            self.constraints.update(physical_constraints)
            
        # Movement tracking
        self.movement_history = []
        self.movement_time = 0.0
        self.total_transfer_time = 0.0
        
        # Initialize movement manager (unified class that handles all movement strategies)
        self.movement_manager = MovementManager(self)
        
        # Visualizer will be assigned externally
        self.visualizer = None
    
    def generate_initial_lattice(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random lattice with the specified occupation probability."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate initial lattice directly in the field
        self.field = np.zeros(self.field_size, dtype=int)
        
        # Place the initial lattice at the center of the field
        start_row = (self.field_size[0] - self.initial_size[0]) // 2
        start_col = (self.field_size[1] - self.initial_size[1]) // 2
        
        # Create random distribution based on occupation probability
        initial_region = np.random.random(self.initial_size) < self.occupation_prob
        self.field[start_row:start_row+self.initial_size[0], 
                 start_col:start_col+self.initial_size[1]] = initial_region
        
        # Store the initial SLM lattice
        self.slm_lattice = self.field.copy()
        self.total_atoms = np.sum(self.field)
        
        return self.field
        
    def rearrange_for_defect_free(self, show_visualization=True):
        """
        Rearrange atoms to create a defect-free region.
        
        This method performs the complete atom rearrangement process:
        1. Determine the optimal target region based on available atoms
        2. Calculate the maximum possible defect-free square
        3. Apply row-wise centering to create the defect-free region
        
        Args:
            show_visualization: Whether to show animation
            
        Returns:
            Tuple of (target_lattice, retention_rate, execution_time)
        """
        start_time = time.time()
        
        # Reset movement tracking
        self.movement_history = []
        self.movement_time = 0.0
        
        # Determine target region size based on atom count
        total_atoms = np.sum(self.field)
        max_square_size = int(np.floor(np.sqrt(total_atoms)))
        self.side_length = min(max_square_size, self.side_length)
        
        print(f"Creating defect-free region of size {self.side_length}x{self.side_length}")
        print(f"Using {self.side_length * self.side_length} atoms out of {total_atoms} available")
        
        # Perform row-wise centering algorithm
        result = self.movement_manager.row_wise_centering(show_visualization=show_visualization)
        
        # Add execution time tracking
        execution_time = time.time() - start_time
        print(f"Total rearrangement time: {execution_time:.3f} seconds")
        
        return result