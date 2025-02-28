"""
LatticeSimulator package for simulating atom rearrangement in optical lattices.

This package provides tools for simulating and visualizing the rearrangement of atoms
in optical lattices using SLM and AOD trap systems, with realistic physical constraints.

Main components:
- core: Core simulator functionality
- algorithms: Pathfinding and atom movement algorithms
- visualization: Tools for visualizing lattice states and movements
- utils: Helper utilities and functions
"""

from .core import LatticeSimulator
from .visualization import LatticeVisualizer

__all__ = ['LatticeSimulator', 'LatticeVisualizer']
__version__ = '1.0.0'