"""
Helper utility functions for the lattice simulator.
"""
import numpy as np
from typing import Tuple


def calculate_min_distance(pos1: Tuple[int, int], pos2: Tuple[int, int], site_distance: float) -> float:
    """
    Calculate the minimum physical distance between two lattice sites.
    
    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)
        site_distance: Physical distance between adjacent sites in micrometers
        
    Returns:
        Physical distance in micrometers
    """
    row_dist = abs(pos2[0] - pos1[0])
    col_dist = abs(pos2[1] - pos1[1])
    
    # Manhattan distance in lattice units
    lattice_dist = row_dist + col_dist
    
    # Convert to physical distance
    physical_dist = lattice_dist * site_distance
    
    return physical_dist


def apply_physical_constraints(distance: float, max_acceleration: float) -> float:
    """
    Calculate the time required to move an atom over a given distance based on physical constraints.
    
    Args:
        distance: Distance to move in micrometers
        max_acceleration: Maximum allowed acceleration in m/s^2
        
    Returns:
        Time required for the movement in seconds
    """
    # Convert distance from micrometers to meters
    distance_m = distance * 1e-6
    
    # Using s = 0.5 * a * t^2 for constant acceleration (assuming acceleration and deceleration)
    # We need to accelerate for half the distance, then decelerate for the other half
    # So for each half: s_half = 0.5 * a * t_half^2
    # Which means t_half = sqrt(s_half / (0.5 * a))
    # Total time is 2 * t_half
    
    half_distance = distance_m / 2
    half_time = np.sqrt(half_distance / (0.5 * max_acceleration))
    total_time = 2 * half_time
    
    return total_time