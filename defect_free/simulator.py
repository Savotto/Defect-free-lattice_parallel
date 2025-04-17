"""
Core simulator module defining the LatticeSimulator class with initialization and constants.
"""
import numpy as np
import time
from typing import Tuple, Dict, Optional
from defect_free.movement import MovementManager

class LatticeSimulator:
    """
    Simulates a quantum atom lattice with physical constraints.
    """
    # Physical constants
    SITE_DISTANCE = 5.0  # μm
    MAX_ACCELERATION = 2750.0  # m/s² (PowerMove)
    TRAP_TRANSFER_TIME = 15e-6  # seconds (15μs)
    ATOM_LOSS_PROBABILITY = 0.05 # Probability of atom loss per move
    MAX_VELOCITY = 0.1  # m/s (Parallel Assembly of Arbitrary Defect-Free Atom Arrays with a Multitweezer Algorithm)
    SETTLING_TIME = 1e-6  # seconds
    
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
            'atom_loss_probability': self.ATOM_LOSS_PROBABILITY,
            'max_velocity': self.MAX_VELOCITY,
            'settling_time': self.SETTLING_TIME
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
        
        self.strategy = 'center'  # Default strategy
    
    def generate_initial_lattice(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random lattice with the specified occupation probability."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate initial lattice directly in the field
        self.field = np.zeros(self.initial_size, dtype=int)
        
        if self.strategy == 'corner':
            # Place the initial lattice at the top-left corner of the field
            start_row, start_col = 0, 0
        else:
            # Place the initial lattice at the center of the field
            start_row = (self.initial_size[0] - self.initial_size[0]) // 2
            start_col = (self.initial_size[1] - self.initial_size[1]) // 2
        
        # Create random distribution based on occupation probability
        initial_region = np.random.random(self.initial_size) < self.occupation_prob
        self.field[start_row:start_row+self.initial_size[0], 
                 start_col:start_col+self.initial_size[1]] = initial_region
        
        # Store the initial SLM lattice
        self.slm_lattice = self.field.copy()
        self.total_atoms = np.sum(self.field)
        
        return self.field

    def calculate_max_defect_free_size(self, strategy=None) -> int:
        """
        Calculate the maximum possible size of a defect-free square lattice based on available atoms,
        with scaling of expected movements based on lattice size.
        
        Args:
            strategy: Which strategy to use for calculation ('center' or 'corner')
        
        Returns:
            The side length of the maximum possible square lattice
        """
        # Count total atoms in the field
        total_atoms = np.sum(self.field)
        
        # Get lattice dimensions
        field_height, field_width = self.initial_size
        lattice_size = max(field_height, field_width)
        
        # Base parameters for reference lattice sizes
        base_lattice_size = 30  # Reference size for scaling
        
        # Base movement counts at the reference lattice size
        if strategy == 'corner':
            base_steps = 2  # Base steps for corner strategy
        else:
            base_steps = 2  # Base steps for center strategy
        
        # Scale expected steps based on lattice size using square root scaling
        # This models the intuition that in larger lattices, atoms need to move farther
        scaling_factor = np.sqrt(lattice_size / base_lattice_size)
        
        # Calculate expected steps with scaling
        expected_steps = base_steps * scaling_factor
        
        # Add a minimum floor for very small lattices
        expected_steps = max(expected_steps, 1.0)
        
        # Calculate atom loss probability
        atom_loss_prob = self.constraints.get('atom_loss_probability', 0.0)
        
        # Transport success rate (accounting for scaled number of moves)
        transport_success_rate = (1 - atom_loss_prob) ** expected_steps
        
        # Safety factor based on strategy 
        if strategy == 'corner':
            safety_factor = 0.95  # 5% safety margin for corner strategy
        else:
            safety_factor = 0.95  # 5% safety margin for center strategy
        
        # Final calculation with all scaling factors
        max_square_size = int(np.floor(np.sqrt(total_atoms * transport_success_rate * safety_factor)))
        
        # If the calculated target is only one smaller than the initial lattice, shrink by one more
        min_dim = min(field_height, field_width)
        if max_square_size == min_dim - 1:
            max_square_size = min_dim - 2

        # Print diagnostic information
        print(f"Lattice dimensions: {field_height}x{field_width}")
        print(f"Scaling factor: {scaling_factor:.2f}")
        print(f"Expected steps per atom: {expected_steps:.2f}")
        print(f"Transport success rate: {transport_success_rate:.4f}")
        print(f"Safety factor: {safety_factor}")
        
        # Update the side_length attribute
        self.side_length = max_square_size
        
        return max_square_size

    def rearrange_for_defect_free(self, strategy='center', show_visualization=True) -> Tuple[np.ndarray, float, float]:
        """
        Rearrange atoms to create a defect-free region using the specified strategy.
        
        This method performs the complete atom rearrangement process:
        1. Determine the optimal target region based on available atoms
        2. Calculate the maximum possible defect-free square
        3. Apply the selected filling strategy to create the defect-free region
        
        Args:
            strategy: Which filling strategy to use: 'center' or 'corner'
            show_visualization: Whether to show animation
            
        Returns:
            Tuple of (target_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        
        # Reset movement tracking
        self.movement_history = []
        self.movement_time = 0.0
        
        print("Step 1: Determining optimal target region size...")
        # Use the optimal target size calculation
        self.side_length = self.calculate_max_defect_free_size(strategy=strategy)
        total_atoms = np.sum(self.field)
        
        # Calculate atom loss probability
        atom_loss_prob = self.constraints.get('atom_loss_probability', 0.0)
        
        print(f"Step 2: Calculated optimal defect-free square: {self.side_length}x{self.side_length}")
        print(f"Creating defect-free region of size {self.side_length}x{self.side_length}")
        print(f"Using {self.side_length * self.side_length} atoms out of {total_atoms} available")
        print(f"Utilization ratio: {(self.side_length * self.side_length) / total_atoms:.2%}")
        
        if atom_loss_prob > 0:
            print(f"(Accounting for {atom_loss_prob:.1%} atom loss probability per move)")
        
        # Apply the selected strategy
        print(f"Step 3: Applying {strategy} filling strategy...")
        result = self.movement_manager.rearrange_for_defect_free(
            strategy=strategy,
            show_visualization=show_visualization
        )
        
        # Add execution time tracking
        execution_time = time.time() - start_time
        print(f"Total rearrangement time: {execution_time:.3f} seconds")
        
        return result, execution_time