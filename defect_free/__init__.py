"""
LatticeSimulator package for simulating atom rearrangement in optical lattices.

This package provides tools for simulating and visualizing the rearrangement of atoms
in optical lattices using SLM and AOD trap systems, with realistic physical constraints.
"""

from .simulator import LatticeSimulator
from .visualizer import LatticeVisualizer
from .movement import MovementManager

__all__ = [
    'LatticeSimulator',
    'LatticeVisualizer',
    'MovementManager',
]

__version__ = '1.0.0'