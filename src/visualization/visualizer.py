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
        """Animate the atom rearrangement process with both detailed and clean views."""
        if not self.simulator.movement_history:
            print("No movements to animate")
            return
            
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        plt.subplots_adjust(wspace=0.3)
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            movement = self.simulator.movement_history[frame]
            movement_type = movement['type']
            
            # Common setup for both plots
            for ax in (ax1, ax2):
                ax.set_xticks(np.arange(-0.5, self.simulator.field_size[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, self.simulator.field_size[0], 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # Calculate boundaries
                start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
                start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
                
                # Highlight target region
                target_rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                                          self.simulator.side_length, self.simulator.side_length,
                                          facecolor='lightgreen', edgecolor='green',
                                          linestyle='-', linewidth=2, alpha=0.2)
                ax.add_patch(target_rect)
            
            # Left plot: Movement details with faded elements
            # Plot SLM traps as very faint circles
            for i in range(self.simulator.initial_size[0]):
                for j in range(self.simulator.initial_size[1]):
                    trap = plt.Circle((start_col + j, start_row + i), 0.4,
                                    facecolor='none', edgecolor='blue',
                                    linestyle='-', linewidth=0.5, alpha=0.1)
                    ax1.add_patch(trap)
            
            # Plot stationary atoms
            moving_from = {pos[0] for pos in movement['moves']}
            atom_positions = np.where(movement['state'] == 1)
            for y, x in zip(atom_positions[0], atom_positions[1]):
                if (y, x) not in moving_from:
                    circle = plt.Circle((x, y), 0.3,
                                     facecolor='yellow', edgecolor='black',
                                     linewidth=1)
                    ax1.add_patch(circle)
            
            # Draw active AOD trap lines with slightly increased opacity
            if movement_type == 'parallel_left' or movement_type == 'parallel_right':
                for from_pos, to_pos in movement['moves']:
                    y = from_pos[0]
                    ax1.axhline(y=y, color='red', linestyle='-', linewidth=1, alpha=0.15)
                    ax1.axvline(x=from_pos[1], color='red', linestyle='-', linewidth=1, alpha=0.15)
                    ax1.axvline(x=to_pos[1], color='red', linestyle='-', linewidth=1, alpha=0.15)
            elif movement_type == 'parallel_up' or movement_type == 'parallel_down':
                for from_pos, to_pos in movement['moves']:
                    x = from_pos[1]
                    ax1.axvline(x=x, color='red', linestyle='-', linewidth=1, alpha=0.15)
                    ax1.axhline(y=from_pos[0], color='red', linestyle='-', linewidth=1, alpha=0.15)
                    ax1.axhline(y=to_pos[0], color='red', linestyle='-', linewidth=1, alpha=0.15)
            elif movement_type == 'parallel_up_left' or movement_type == 'parallel_down_right':
                for from_pos, to_pos in movement['moves']:
                    ax1.axvline(x=from_pos[1], color='red', linestyle='-', linewidth=1, alpha=0.15)
                    ax1.axhline(y=to_pos[0], color='red', linestyle='-', linewidth=1, alpha=0.15)
            
            # Draw movement arrows with increased visibility
            for from_pos, to_pos in movement['moves']:
                arrow_props = dict(arrowstyle='->,head_width=0.6,head_length=0.9',
                                 color='red', linestyle='-',
                                 linewidth=1.5, alpha=0.4)
                ax1.annotate('', xy=(to_pos[1], to_pos[0]),
                           xytext=(from_pos[1], from_pos[0]),
                           arrowprops=arrow_props)
                
                # Draw source and destination positions with increased visibility
                source = plt.Circle((from_pos[1], from_pos[0]), 0.3,
                                  facecolor='none', edgecolor='red',
                                  linestyle='--', linewidth=1.5, alpha=0.4)
                ax1.add_patch(source)
                
                destination = plt.Circle((to_pos[1], to_pos[0]), 0.3,
                                       facecolor='yellow', edgecolor='red',
                                       linewidth=1.5, alpha=0.8)
                ax1.add_patch(destination)
            
            # Right plot: Clean view of atom positions
            atom_positions = np.where(movement['state'] == 1)
            for y, x in zip(atom_positions[0], atom_positions[1]):
                circle = plt.Circle((x, y), 0.3,
                                 facecolor='yellow', edgecolor='black',
                                 linewidth=1)
                ax2.add_patch(circle)
            
            # Add step information
            movement_descriptions = {
                'parallel_left': 'Left',
                'parallel_right': 'Right',
                'parallel_up': 'Up',
                'parallel_down': 'Down',
                'parallel_up_left': 'Up-Left',
                'parallel_left_down': 'Left-Down'
            }
            move_desc = movement_descriptions.get(movement_type, movement_type.replace('parallel_', '').capitalize())
            
            info_text = [
                f'Step {movement["iteration"]}: {move_desc} Movement',
                f'Moving {len(movement["moves"])} atoms in parallel',
                f'Movement time: {movement.get("time", 0)*1000:.3f} ms',
                f'Frame: {frame + 1}/{len(self.simulator.movement_history)}'
            ]
            
            for i, text in enumerate(info_text):
                ax1.text(0.02, 0.98 - i*0.04, text,
                        transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7))
            
            ax2.text(0.02, 0.98, "Final Positions",
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Set plot limits and titles
            for ax in (ax1, ax2):
                ax.set_xlim(-0.5, self.simulator.field_size[1] - 0.5)
                ax.set_ylim(self.simulator.field_size[0] - 0.5, -0.5)
            
            ax1.set_title('Movement Details')
            ax2.set_title('Resulting Atom Positions')
        
        # Create animation with longer interval to see movements clearly
        anim = animation.FuncAnimation(fig, update, frames=len(self.simulator.movement_history),
                                     interval=2000, repeat=False)
        plt.show()
    
    def visualize_lattices(self) -> None:
        """
        Visualize the initial and final configurations side by side showing just atom positions.
        """
        if self.simulator.slm_lattice is None:
            raise ValueError("Lattice must be generated first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.3)
        
        def plot_configuration(ax, state):
            # Plot major grid
            ax.set_xticks(np.arange(0, self.simulator.field_size[1], 1))
            ax.set_yticks(np.arange(0, self.simulator.field_size[0], 1))
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Calculate boundaries
            start_row = (self.simulator.field_size[0] - self.simulator.initial_size[0]) // 2
            start_col = (self.simulator.field_size[1] - self.simulator.initial_size[1]) // 2
            
            # Plot target region with light background
            target_rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                                   self.simulator.side_length, self.simulator.side_length,
                                   facecolor='lightgreen', edgecolor='green',
                                   linestyle='-', linewidth=2, alpha=0.2)
            ax.add_patch(target_rect)
            
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