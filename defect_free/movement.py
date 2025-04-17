"""
Main movement manager module that provides access to both center and corner based strategies.
"""
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
        """
        if self.current_strategy == 'center':
            self.center_manager.initialize_target_region()
            self.target_region = self.center_manager.target_region
        else:
            self.corner_manager.initialize_target_region()
            self.target_region = self.corner_manager.target_region
    
    def repair_defects(self, show_visualization=True):
        """
        Repair defects using the active strategy.
        """
        if self.current_strategy == 'center':
            return self.center_manager.repair_defects(show_visualization)
        else:
            return self.corner_manager.repair_defects(show_visualization)
    
    def center_filling_strategy(self, show_visualization=True):
        """
        Use the center-based filling strategy.
        """
        original_strategy = self.current_strategy
        self.current_strategy = 'center'
        self.center_manager.initialize_target_region()
        self.target_region = self.center_manager.target_region
        result = self.center_manager.center_filling_strategy(show_visualization)
        self.target_region = self.center_manager.target_region
        self.current_strategy = original_strategy
        return result
    
    def corner_filling_strategy(self, show_visualization=True):
        """
        Use the corner-based filling strategy.
        """
        original_strategy = self.current_strategy
        self.current_strategy = 'corner'
        self.corner_manager.initialize_target_region()
        self.target_region = self.corner_manager.target_region
        result = self.corner_manager.corner_filling_strategy(show_visualization)
        self.target_region = self.corner_manager.target_region
        self.current_strategy = original_strategy
        return result
    
    def rearrange_for_defect_free(self, strategy='center', show_visualization=True):
        """
        Top-level method to rearrange atoms using a specified strategy.
        """
        lost_atoms = 0

        self.set_strategy(strategy)
        if strategy == 'center':
            self.center_manager.initialize_target_region()
            
            return self.center_filling_strategy(show_visualization)
        else:
            self.corner_manager.initialize_target_region()
            return self.corner_filling_strategy(show_visualization)
        