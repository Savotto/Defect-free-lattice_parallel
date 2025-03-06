# Defect-free Lattice Simulator

A Python package for simulating atom rearrangement in optical lattices using SLM and AOD trap systems, with realistic physical constraints.

## Project Structure

```
defect-free/
├── defect_free/             # Main package
│   ├── __init__.py
│   ├── simulator.py         # Core simulator logic
│   ├── algorithms.py        # All algorithms in one file
│   ├── visualizer.py        # Visualization code
│   └── utils.py            # Utility functions
├── tests/                  # All tests together
│   ├── __init__.py
│   ├── test_simulator.py    
│   └── test_algorithms.py  # Including row-wise centering tests
├── examples/              # Example scripts
│   └── basic_example.py   # Basic usage example
├── .gitignore            # Ignore generated files
├── setup.py              # For package installation
└── README.md            # This file
```

## Installation

To install the package in development mode:

```bash
pip install -e .
```

## Usage

Here's a basic example of using the simulator:

```python
from defect_free import LatticeSimulator, LatticeVisualizer
import matplotlib.pyplot as plt

# Initialize simulator with a 10x10 lattice and 70% occupation probability
simulator = LatticeSimulator(initial_size=(10, 10), occupation_prob=0.7)
simulator.generate_initial_lattice()

# Initialize visualizer
visualizer = LatticeVisualizer(simulator)
simulator.visualizer = visualizer

# Rearrange atoms into a perfect square lattice
simulator.rearrange_for_perfect_lattice(show_visualization=True)

plt.show()
```

For more examples, see the `examples/` directory.

## Running Tests

To run the tests:

```bash
python -m pytest tests/
```

## Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0
