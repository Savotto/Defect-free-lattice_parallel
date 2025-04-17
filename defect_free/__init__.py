"""
LatticeSimulator package for simulating atom rearrangement in optical lattices.

This package provides tools for simulating and visualizing the rearrangement of atoms
in optical lattices using SLM and AOD trap systems, with realistic physical constraints.
"""

from .simulator import LatticeSimulator
from .visualizer import LatticeVisualizer
from .movement import MovementManager

# Import space-time visualization components
try:
    from .space_time_extractor import extract_space_time_data, export_to_json, generate_html_file
    __all__ = [
        'LatticeSimulator',
        'LatticeVisualizer',
        'MovementManager',
        'extract_space_time_data',
        'export_to_json',
        'generate_html_file',
    ]
except ImportError:
    # If space_time_extractor is not available, just import the core components
    __all__ = [
        'LatticeSimulator',
        'LatticeVisualizer',
        'MovementManager',
    ]

__version__ = '1.0.0'