"""
Visualization module for the lattice simulator.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any, Tuple


class LatticeVisualizer:
    def __init__(self, simulator):
        """
        Initialize the LatticeVisualizer with a reference to the simulator.
        
        Args:
            simulator: The LatticeSimulator instance
        """
        self.simulator = simulator
    
    def animate_rearrangement(self) -> None:
        """Animate the atom rearrangement process with parallel movements."""
        if not self.simulator.movement_history:
            print("No movements to animate")
            return
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        def update(frame):
            ax.clear()
            movement = self.simulator.movement_history[frame]
            
            # Plot grid
            ax.set_xticks(np.arange(-0.5, self.simulator.field_size[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.simulator.field_size[0], 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Calculate boundaries
            start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
            start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
            
            # Highlight target region
            target_row_end = start_row + self.simulator.side_length
            target_col_end = start_col + self.simulator.side_length
            target_rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                                      self.simulator.side_length, self.simulator.side_length,
                                      facecolor='lightgreen', edgecolor='green',
                                      linestyle='-', linewidth=2, alpha=0.2)
            ax.add_patch(target_rect)
            
            # Plot SLM traps as circles
            for i in range(self.simulator.initial_size[0]):
                for j in range(self.simulator.initial_size[1]):
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
            movement_type = movement['type']
            
            if movement_type == 'parallel_left' or movement_type == 'parallel_right':
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
                    
            elif movement_type == 'parallel_up' or movement_type == 'parallel_down':
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
                    
            elif movement_type == 'parallel_up_left' or movement_type == 'parallel_down_right':
                # Draw both vertical and horizontal lines for diagonal movements
                for from_pos, to_pos in movement['moves']:
                    # Vertical component
                    ax.axvline(x=from_pos[1], color='red', linestyle='-',
                             linewidth=1, alpha=0.3, zorder=1)
                    # Horizontal component
                    ax.axhline(y=to_pos[0], color='red', linestyle='-',
                             linewidth=1, alpha=0.3, zorder=1)
            
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
            
            # Get movement description
            movement_descriptions = {
                'parallel_left': 'Left',
                'parallel_right': 'Right',
                'parallel_up': 'Up',
                'parallel_down': 'Down',
                'parallel_up_left': 'Up-Left',
                'parallel_left_down': 'Left-Down'
            }
            move_desc = movement_descriptions.get(movement_type, movement_type.replace('parallel_', '').capitalize())
            
            # Add step information
            info_text = [
                f'Step {movement["iteration"]}: {move_desc} Movement',
                f'Moving {len(movement["moves"])} atoms in parallel',
                f'Movement time: {movement.get("time", 0)*1000:.3f} ms',
                f'Frame: {frame + 1}/{len(self.simulator.movement_history)}'
            ]
            
            for i, text in enumerate(info_text):
                ax.text(0.02, 0.98 - i*0.04, text,
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_xlim(-0.5, self.simulator.field_size[1] - 0.5)
            ax.set_ylim(self.simulator.field_size[0] - 0.5, -0.5)
            ax.set_title('Three-Phase Atom Rearrangement')
        
        # Create animation with longer interval to see movements clearly
        anim = animation.FuncAnimation(fig, update, frames=len(self.simulator.movement_history),
                                     interval=2000, repeat=False)
        plt.show()
    
    def visualize_lattices(self) -> None:
        """
        Visualize the initial and final configurations side by side using the full field view.
        Shows circular SLM traps (blue circles) and atoms (yellow circles) in the larger field.
        """
        if self.simulator.slm_lattice is None:
            raise ValueError("Lattice must be generated first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.3)
        
        def plot_configuration(ax, state):
            # Plot major grid for full field
            ax.set_xticks(np.arange(0, self.simulator.field_size[1], 1))
            ax.set_yticks(np.arange(0, self.simulator.field_size[0], 1))
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Calculate SLM region boundaries
            start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
            start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
            
            # Plot outfield background
            outfield = plt.Rectangle((-0.5, -0.5), self.simulator.field_size[1], self.simulator.field_size[0],
                                   facecolor='lightgray', alpha=0.1)
            ax.add_patch(outfield)
            
            # Plot SLM region with different background
            slm_rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                                   self.simulator.initial_size[1], self.simulator.initial_size[0],
                                   facecolor='white', edgecolor='blue',
                                   linestyle='-', linewidth=2, alpha=0.3)
            ax.add_patch(slm_rect)
            
            # Plot all SLM traps as circles
            for i in range(self.simulator.initial_size[0]):
                for j in range(self.simulator.initial_size[1]):
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
            
            ax.set_xlim(-0.5, self.simulator.field_size[1] - 0.5)
            ax.set_ylim(self.simulator.field_size[0] - 0.5, -0.5)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        # Plot initial configuration using the field state
        plot_configuration(ax1, self.simulator.field)
        ax1.set_title('Initial Configuration', pad=20)
        
        # Plot final configuration
        if self.simulator.target_lattice is not None:
            plot_configuration(ax2, self.simulator.target_lattice)
            ax2.set_title('Final Configuration', pad=20)
        
        plt.tight_layout()
        plt.show()