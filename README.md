# Defect-free Lattice Simulator

A Python package for simulating atom rearrangement in optical lattices using SLM and AOD trap systems, with realistic physical constraints. It helps optimize and compare strategies for creating defect-free regions.

## Features

- **Physical Constraints Modeling**: Implements realistic atom movement physics including max acceleration, velocity limits, trap transfer time, and atom loss probability
- **Multiple Movement Strategies**: 
  - Center-based strategy (places target zone in the center) 
  - Corner-based strategy (places target zone in the top-left corner) 
  However, corner-based is computationally faster.
- **Advanced Algorithms**:
  - Row-wise and column-wise atom alignment
  - Sophisticated defect repair with path optimization
  - Block-based corner atom movement
  - A* pathfinding for complex moves
- **Performance Analysis**: Tools to quantify execution metrics (to be implemented)
- **Rich Visualization**: Animated atom movements and more
- **Transport Loss Modeling**: Probabilistic atom loss simulation during transport

## Project Structure

```
defect-free/
├── defect_free/                # Main package
│   ├── __init__.py
│   ├── simulator.py            # Core simulator logic
│   ├── movement.py             # Movement manager interface
│   ├── base_movement.py        # Common movement functionality
│   ├── center_movement.py      # Center-based movement strategy
│   ├── corner_movement.py      # Corner-based movement strategy
│   └── visualizer.py           # Visualization tools
├── examples/                   # Example scripts
│   ├── complete_workflow_example.py
│   ├── movement_example.py     
│   └── performance_analysis.py
├── .gitignore                  # Ignore generated files
└── README.md                   # This file
```

## Physical Model

The simulator models physical constraints of optical lattice manipulations:

- **Site Distance**: 5.0 μm (configurable)
- **Maximum Acceleration**: 2750.0 m/s² (configurable)
- **Maximum Velocity**: 0.1 m/s (configurable)
- **Settling Time**: 1 μs (configurable)
- **Atom Loss Probability**: Configurable, default 0.05

Movement timing calculations use a realistic trapezoidal velocity profile that respects both maximum acceleration and velocity limits.

## Movement Strategies

### Center-Based Strategy

1. Places the target zone in the center of the field
2. Uses row-wise and column-wise centering to align atoms
3. Iteratively spreads and squeezes outer atoms
4. Repairs remaining defects using path optimization

### Corner-Based Strategy

1. Places the target zone in the top-left corner
2. Squeezes rows left and columns up to fill the target
3. Uses right-edge squeezing for atoms below the target
4. Iteratively applies targeted filling techniques

Both strategies implement sophisticated defect repair algorithms that use optimal path finding to move atoms to remaining defect positions.

### Script Examples

Several example scripts are included in the `examples/` directory:

- `complete_workflow_example.py`: Simple end-to-end example. Can be used with both center and corner strategies.
- `movement_example.py`: Demonstrates movements. Can be run using both center and corner startegies.
- `performance_analysis.py`: Performance testing across different parameters

## Algorithm Details

### Row-wise & Column-wise Centering

Pushes atoms toward the center of their respective rows or columns to create orderly arrangements within the target zone. This creates a foundation for more complex operations.

### Defect Repair

The algorithm uses a multi-tiered pathfinding approach for optimal atom movement:
1. Attempts direct moves when possible (same row/column)
2. Uses L-shaped paths with a single turn
3. Falls back to A* search for complex paths
4. Compresses paths to reduce the number of discrete movements

### Movement Optimization

For parallel operations, atoms are grouped into batches that can be moved simultaneously without collisions. This reduces the physical execution time required.

## Visualization

The package includes a rich visualization toolkit:

- **Lattice State Visualization**: View the atom arrangement at any point
- **Movement Animation**: Animate the sequence of atom movements
- **Movement Analysis**: Analyze movement patterns and efficiency

## Performance Considerations

- **Execution Speed**: The center strategy typically requires more computation time but can achieve better fill rates in some scenarios
- **Physical Movement Time**: The corner strategy is having less complex path finding in the final filling and thus is faster in computation.
- **Atom Loss**: Both strategies are designed to be robust to atom loss during transport

## Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Future Improvements

- Center strategy could benefit from first iterating the row squeezing some times and after that iteratively squeeze columns.
- Add support for arbitrary target shapes beyond square lattices



