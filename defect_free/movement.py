"""
Main movement manager module for center-based target assembly.
"""
from defect_free.center_movement import CenterMovementManager

class MovementManager:
    """
    Main movement manager class for center-based target assembly.
    """
    
    def __init__(self, simulator):
        """Initialize the movement manager with a reference to the simulator."""
        self.simulator = simulator
        self.target_region = None
        self.target_mask = None
        
        # Initialize strategy managers
        self.center_manager = CenterMovementManager(simulator)
    
    def set_strategy(self, strategy_name):
        """
        Validate the requested movement strategy.
        
        Args:
            strategy_name: Currently only 'center'
        """
        if strategy_name != 'center':
            raise ValueError("Unknown strategy: {0}. Only 'center' is supported.".format(strategy_name))
        print("Set active movement strategy to: center")
    
    def initialize_target_region(self):
        """
        Initialize the active target region.
        """
        self.center_manager.initialize_target_region()
        self.target_region = self.center_manager.target_region
        self.target_mask = self.center_manager.target_mask

    def reset_target_definition(self):
        """Reset cached target geometry."""
        self.target_region = None
        self.target_mask = None
        self.center_manager.reset_target_definition()
    
    def repair_defects(self, show_visualization=True):
        """
        Repair defects using the center strategy.
        """
        return self.center_manager.repair_defects(show_visualization)
    
    def center_filling_strategy(self, show_visualization=True):
        """
        Use the center-based filling strategy.
        """
        self.center_manager.initialize_target_region()
        self.target_region = self.center_manager.target_region
        self.target_mask = self.center_manager.target_mask
        if self.simulator.constraints.get("reproducibility_profile"):
            final_lattice, fill_rate, execution_time, _ = (
                self.center_manager.iterative_blind_center_filling(
                    max_iterations=None,
                    min_improvement=0.0,
                    show_visualization=show_visualization,
                    use_batch_merging=True,
                )
            )
        else:
            final_lattice, fill_rate, execution_time = (
                self.center_manager.blind_center_filling_strategy(
                    show_visualization=show_visualization
                )
            )
        result = (final_lattice, fill_rate, execution_time)
        self.target_region = self.center_manager.target_region
        self.target_mask = self.center_manager.target_mask
        return result
    
    def rearrange_for_defect_free(self, strategy='center', show_visualization=True):
        """
        Top-level method to rearrange atoms using a specified strategy.
        """
        self.set_strategy(strategy)
        self.center_manager.initialize_target_region()
        return self.center_filling_strategy(show_visualization)
