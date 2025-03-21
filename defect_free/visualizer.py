"""
Visualization module for atom lattice simulation and rearrangement.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Any, Tuple, Optional

class LatticeVisualizer:
    """
    Visualizes lattice states and atom movements.
    """
    
    def __init__(self, simulator):
        """
        Initialize the visualizer.
        
        Args:
            simulator: The LatticeSimulator instance
        """
        self.simulator = simulator
        
        # Default visualization settings
        self.colors = {
            'background': '#F5F5F5',      # Light gray background
            'grid': '#CCCCCC',            # Lighter gray for grid
            'atom': '#3366CC',            # Blue for atoms
            'target': '#FFD700',          # Gold for target region
            'defect': '#FF6347'           # Tomato for defects
        }
        
        # Create a custom colormap for atom density
        self.atom_cmap = LinearSegmentedColormap.from_list(
            'atom_density', ['white', self.colors['atom']], N=100)
        
        # Animation settings
        self.interval = 200  # ms between frames
        self.fig = None
        self.ani = None
    
    def plot_lattice(self, lattice: np.ndarray, title: str = 'Atom Lattice', 
                    highlight_region: Optional[Tuple[int, int, int, int]] = None,
                    show_grid: bool = True, ax=None):
        """
        Plot a single lattice state with atoms as circles.
        
        Args:
            lattice: 2D numpy array representing the lattice
            title: Plot title
            highlight_region: (start_row, start_col, end_row, end_col) to highlight
            show_grid: Whether to show grid lines
            ax: Optional matplotlib axis to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            standalone = True
        else:
            standalone = False
        
        # Set background color
        ax.set_facecolor(self.colors['background'])
        
        # Set up the plot dimensions
        rows, cols = lattice.shape
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)  # Invert y-axis to match array indexing
        
        # Draw grid if requested
        if show_grid:
            ax.grid(color=self.colors['grid'], linestyle='-', linewidth=0.5)
            
        # Draw minor gridlines
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.tick_params(which='minor', size=0)
        
        # Only show major ticks every 10 positions to avoid overcrowding
        ax.set_xticks(np.arange(0, cols, 10))
        ax.set_yticks(np.arange(0, rows, 10))
        
        # Highlight target region if specified
        if highlight_region:
            start_row, start_col, end_row, end_col = highlight_region
            rect = plt.Rectangle((start_col-0.5, start_row-0.5), 
                               end_col-start_col, end_row-start_row,
                               fill=False, edgecolor=self.colors['target'], 
                               linewidth=2, linestyle='--')
            ax.add_patch(rect)
        
        # Find all atom positions
        atom_positions = np.where(lattice == 1)
        for row, col in zip(atom_positions[0], atom_positions[1]):
            circle = plt.Circle((col, row), 0.4, 
                              facecolor=self.colors['atom'],
                              edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)
        
        # Set title
        ax.set_title(title)
        
        if standalone:
            plt.tight_layout()
            return fig
        return ax
    
    def animate_movements(self, movements: List[Dict]):
        """
        Create an animation of atom movements.
        
        Args:
            movements: List of movement records containing states and metadata
            
        Returns:
            Animation object
        """
        if not movements:
            print("No movements to animate.")
            return None
            
        # Get the initial state
        states = [self.simulator.slm_lattice.copy()]
        states.extend([move['state'] for move in movements])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Function to update the plot for each frame
        def update(frame):
            ax.clear()
            state = states[frame]
            self.plot_lattice(state, 
                            title=f"Movement Animation - Step {frame}/{len(states)-1}",
                            highlight_region=self.simulator.movement_manager.target_region,
                            ax=ax)
            
            # Draw arrows for movements in the current frame
            if frame > 0:
                move_dict = movements[frame-1]
                if 'moves' in move_dict:
                    print(f"Drawing {len(move_dict['moves'])} arrows for frame {frame}")  # Debug output
                    for move in move_dict['moves']:
                        if isinstance(move, dict) and 'from' in move and 'to' in move:
                            start_row, start_col = move['from']
                            end_row, end_col = move['to']
                            # Make the arrows more prominent
                            ax.arrow(start_col, start_row, 
                                   end_col - start_col, end_row - start_row,
                                   head_width=0.3, head_length=0.3, 
                                   fc='red', ec='red',
                                   length_includes_head=True, 
                                   alpha=0.8, linestyle='-',
                                   linewidth=2.0, zorder=10)
            
            # Add frame information at the bottom
            if frame > 0:
                move_info = movements[frame-1]
                arrow_count = len([m for m in move_info.get('moves', []) 
                                 if isinstance(m, dict) and 'from' in m and 'to' in m])
                ax.text(0.02, 0.02, 
                       f"Time: {move_info.get('time', 0)*1000:.3f} ms | "
                       f"Type: {move_info.get('type', 'unknown')} | "
                       f"Moves: {arrow_count}", 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7))
            
            return ax
            
        # Create animation
        self.ani = animation.FuncAnimation(
            fig, update, frames=len(states), interval=self.interval,
            blit=False, repeat=True)
            
        # Store reference to avoid garbage collection
        self.fig = fig
        
        plt.tight_layout()
        return self.ani
    
    def show_final_analysis(self):
        """
        Show comprehensive analysis of the final lattice state.
        
        Creates a multi-panel figure showing:
        1. Initial lattice
        2. Final lattice
        3. Defect visualization
        4. Movement statistics
        """
        if self.simulator.slm_lattice is None or self.simulator.target_lattice is None:
            print("No simulation results to analyze.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Initial lattice
        self.plot_lattice(self.simulator.slm_lattice, 
                         title="Initial Lattice State", ax=axes[0, 0])
        
        # Final lattice with target region
        self.plot_lattice(self.simulator.target_lattice,
                         title="Final Lattice State",
                         highlight_region=self.simulator.movement_manager.target_region, 
                         ax=axes[0, 1])
        
        # Defect visualization
        if self.simulator.movement_manager.target_region:
            start_row, start_col, end_row, end_col = self.simulator.movement_manager.target_region
            target = self.simulator.target_lattice[start_row:end_row, start_col:end_col]
            
            # Create mask showing defects in red
            defect_mask = np.zeros(target.shape + (3,))
            defect_mask[target == 0] = [1, 0, 0]  # Red for defects
            
            axes[1, 0].imshow(defect_mask)
            axes[1, 0].set_title("Defect Visualization")
            defect_count = np.prod(target.shape) - np.sum(target)
            axes[1, 0].text(0.5, 0.05, f"Defects: {defect_count}", 
                          transform=axes[1, 0].transAxes, 
                          ha='center', fontsize=12)
        
        # Movement statistics
        axes[1, 1].axis('off')
        
        # Calculate statistics
        total_moves = len(self.simulator.movement_history)
        total_atoms_moved = sum(len(m['moves']) for m in self.simulator.movement_history)
        total_time = self.simulator.movement_time + self.simulator.total_transfer_time
        
        # Target region stats
        if self.simulator.movement_manager.target_region:
            start_row, start_col, end_row, end_col = self.simulator.movement_manager.target_region
            target = self.simulator.target_lattice[start_row:end_row, start_col:end_col]
            initial_atoms = np.sum(self.simulator.slm_lattice)
            target_atoms = np.sum(target)
            target_size = target.shape[0] * target.shape[1]
            retention = target_atoms / target_size if target_size > 0 else 0
            
            if total_moves == 0:
                average_time_per_move = 0
            else:
                average_time_per_move = total_time * 1000 / total_moves

            # Create statistics text
            stats_text = (
                f"Movement Statistics\n"
                f"-------------------\n\n"
                f"Initial atoms: {initial_atoms}\n"
                f"Target region size: {target_size} sites\n"
                f"Atoms in target: {target_atoms}\n"
                f"Retention rate: {retention:.2%}\n\n"
                f"Total operations: {total_moves}\n"
                f"Total atoms moved: {total_atoms_moved}\n"
                f"Total movement time: {total_time*1000:.2f} ms\n"
                f"Average time per move: {average_time_per_move:.2f} ms"
            )
            
            axes[1, 1].text(0.1, 0.9, stats_text, 
                          va='top', ha='left', fontsize=12)
        
        plt.tight_layout()
        return fig
        
    def save_animation(self, filename: str, fps: int = 10):
        """
        Save the current animation to file.
        
        Args:
            filename: Output filename (e.g., 'animation.mp4', 'animation.gif')
            fps: Frames per second
        """
        if self.ani is None:
            print("No animation to save.")
            return
            
        # Determine writer based on file extension
        extension = filename.split('.')[-1].lower()
        
        if extension == 'mp4':
            writer = animation.FFMpegWriter(fps=fps)
        elif extension == 'gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            print(f"Unsupported file format: {extension}")
            return
            
        print(f"Saving animation to {filename}...")
        self.ani.save(filename, writer=writer)
        print("Animation saved successfully!")

    def visualize_movement_opportunities(self, highlight_unreachable: bool = True):
        """
        Visualize potential movement opportunities and unreachable defects.
        
        This helps diagnose issues with the movement algorithm, particularly
        when atoms aren't moving to seemingly available spots.
        
        Args:
            highlight_unreachable: Whether to highlight unreachable defects
            
        Returns:
            Matplotlib figure
        """
        if self.simulator.movement_manager.target_region is None:
            print("Target region not defined.")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot current lattice state
        self.plot_lattice(self.simulator.target_lattice, 
                         title="Movement Opportunity Analysis", 
                         highlight_region=self.simulator.movement_manager.target_region,
                         ax=ax)
        
        # Extract target region
        start_row, start_col, end_row, end_col = self.simulator.movement_manager.target_region
        target_area = self.simulator.target_lattice[start_row:end_row, start_col:end_col]
        
        # Find defects (empty spots) in the target region
        defect_positions = np.where(target_area == 0)
        defect_positions = [(start_row + r, start_col + c) for r, c in zip(defect_positions[0], defect_positions[1])]
        
        # Find atoms outside target region that could be moved
        lattice = self.simulator.target_lattice.copy()
        # Create a mask for the target region
        mask = np.zeros_like(lattice, dtype=bool)
        mask[start_row:end_row, start_col:end_col] = True
        
        # Find atoms outside target region
        outside_atoms = np.where((lattice == 1) & (~mask))
        outside_atoms = [(r, c) for r, c in zip(outside_atoms[0], outside_atoms[1])]
        
        # For each outside atom, find the closest defect and check if it's reachable
        for atom_r, atom_c in outside_atoms:
            # Find the closest defect
            distances = [(defect_r, defect_c, abs(defect_r - atom_r) + abs(defect_c - atom_c)) 
                        for defect_r, defect_c in defect_positions]
            if not distances:
                continue
                
            # Sort by distance
            distances.sort(key=lambda x: x[2])
            closest_defect = distances[0]
            defect_r, defect_c, dist = closest_defect
            
            # Determine if movement is possible (simplified check - just checking if same row or column)
            reachable = (atom_r == defect_r) or (atom_c == defect_c)
            
            # Draw a line from atom to closest defect
            line_color = 'green' if reachable else 'red'
            line_style = '-' if reachable else ':'
            ax.plot([atom_c, defect_c], [atom_r, defect_r], 
                   color=line_color, linestyle=line_style, alpha=0.6, linewidth=1.5,
                   marker='', zorder=3)
            
            # Highlight the defect
            if highlight_unreachable and not reachable:
                defect_circle = plt.Circle((defect_c, defect_r), 0.3,
                                         facecolor='none', edgecolor='red',
                                         linewidth=2, linestyle='--')
                ax.add_patch(defect_circle)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=1.5, label='Reachable path'),
            Line2D([0], [0], color='red', lw=1.5, linestyle=':', label='Unreachable path')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add information text
        info_text = (
            f"Target region defects: {len(defect_positions)}\n"
            f"Atoms outside target: {len(outside_atoms)}"
        )
        ax.text(0.02, 0.02, info_text,
               transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
