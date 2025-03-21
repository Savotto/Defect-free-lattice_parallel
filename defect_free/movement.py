"""
Main movement manager module that provides access to both center and corner based strategies.
"""
import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Set, Any
from defect_free.base_movement import BaseMovementManager
from defect_free.center_movement import CenterMovementManager
from defect_free.corner_movement import CornerMovementManager

class MovementManager:
    """
    Main movement manager class that provides access to both center and corner based strategies.
    """
    
    def __init__(self, simulator):
        """Initialize the movement manager with a reference to the simulator."""
        self.simulator = simulator
        self.target_region = None
        
        # Initialize strategy managers
        self.center_manager = CenterMovementManager(simulator)
        self.corner_manager = CornerMovementManager(simulator)
        
        # Default to using center strategy
        self.current_strategy = 'center'
    
    def set_strategy(self, strategy_name):
        """
        Set the active movement strategy.
        
        Args:
            strategy_name: 'center' or 'corner'
        """
        if strategy_name not in ['center', 'corner']:
            raise ValueError(f"Unknown strategy: {strategy_name}. Must be 'center' or 'corner'.")
            
        self.current_strategy = strategy_name
        print(f"Set active movement strategy to: {strategy_name}")
    
    def initialize_target_region(self):
        """
        Initialize the target region based on the current strategy.
        This will delegate to either the center or corner manager.
        """
        # Initialize both managers' target regions to ensure they're available
        self.center_manager.initialize_target_region()
        self.corner_manager.initialize_target_region()
        
        # Set the active target region based on current strategy
        if self.current_strategy == 'center':
            self.target_region = self.center_manager.target_region
        else:  # corner
            self.target_region = self.corner_manager.target_region
    
    def row_wise_centering(self, show_visualization=True):
        """Delegate to center strategy for row-wise centering."""
        return self.center_manager.row_wise_centering(show_visualization)
    
    def column_wise_centering(self, show_visualization=True):
        """Delegate to center strategy for column-wise centering."""
        return self.center_manager.column_wise_centering(show_visualization)
    
    def spread_outer_atoms(self, show_visualization=True):
        """Delegate to center strategy for spreading atoms outward."""
        return self.center_manager.spread_outer_atoms(show_visualization)
    
    def move_corner_blocks(self, show_visualization=True):
        """Delegate to center strategy for moving corner blocks."""
        return self.center_manager.move_corner_blocks(show_visualization)
    
    def repair_defects(self, show_visualization=True):
        """
        Repair defects using the active strategy.
        This will delegate to either the center or corner manager.
        """
        if self.current_strategy == 'center':
            return self.center_manager.repair_defects(show_visualization)
        else:  # corner
            return self.corner_manager.repair_defects(show_visualization)
    
    def center_filling_strategy(self, show_visualization=True):
        """
        Use the center-based filling strategy.
        This is the original strategy that places the target zone in the center.
        """
        # Always ensure we're using the center strategy for this method
        original_strategy = self.current_strategy
        self.current_strategy = 'center'
        
        # Make sure target region is initialized for center manager
        self.center_manager.initialize_target_region()
        self.target_region = self.center_manager.target_region
        
        result = self.center_manager.center_filling_strategy(show_visualization)
        
        # Ensure the target region is set in the movement manager
        self.target_region = self.center_manager.target_region
        
        # Restore original strategy
        self.current_strategy = original_strategy
        
        return result
    
    def corner_filling_strategy(self, show_visualization=True):
        """
        Use the corner-based filling strategy.
        This strategy places the target zone in the top-left corner.
        """
        # Always ensure we're using the corner strategy for this method
        original_strategy = self.current_strategy
        self.current_strategy = 'corner'
        
        # Make sure target region is initialized for corner manager
        self.corner_manager.initialize_target_region()
        self.target_region = self.corner_manager.target_region
        
        result = self.corner_manager.corner_filling_strategy(show_visualization)
        
        # Ensure the target region is set in the movement manager
        self.target_region = self.corner_manager.target_region
        
        # Restore original strategy
        self.current_strategy = original_strategy
        
        return result
    
    def rearrange_for_defect_free(self, strategy='center', show_visualization=True):
        """
        Top-level method to rearrange atoms using a specified strategy.
        
        Args:
            strategy: Which strategy to use: 'center' or 'corner'
            show_visualization: Whether to show visualization
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        # Set the active strategy
        self.set_strategy(strategy)
        
        # Initialize target regions for both managers to ensure they're available
        self.center_manager.initialize_target_region()
        self.corner_manager.initialize_target_region()
        
        # Call the appropriate strategy
        if strategy == 'center':
            print("Using center-based filling strategy")
            return self.center_filling_strategy(show_visualization)
        else:  # corner
            print("Using corner-based filling strategy")
            return self.corner_filling_strategy(show_visualization)