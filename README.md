# Defect-free Lattice Simulator

A Python toolkit for simulating atom rearrangement in optical lattices under realistic physical constraints. It helps analyse and compare strategies for assembling defect-free regions using spatial light modulators (SLMs) and acousto-optic deflectors (AODs).

## Features

- **Physical model**: Configurable limits for acceleration, velocity, trap transfer timing, and atom loss probability.
- **Movement strategies**: Center- and corner-based algorithms that share the same simulator back-end.
- **Planning pipeline**: Row/column centering, spread-squeeze cycles, targeted defect repair, and path compression.
- **Visualization tools**: Static lattice snapshots and animated movement sequences.
- **Benchmark utilities**: Scripts for scaling studies and occupancy sweeps, with CSV/JSON/NPZ outputs.

## Project Structure

```
defect-free/
├── defect_free/
│   ├── __init__.py
│   ├── simulator.py           # Core simulator and physical timing
│   ├── movement.py            # Strategy interface
│   ├── base_movement.py       # Shared helpers for planners
│   ├── center_movement.py     # Center strategy implementation
│   ├── corner_movement.py     # Corner strategy implementation
│   └── visualizer.py          # Plotting and animation utilities
├── benchmarks/
│   ├── blind_benchmark.py
│   └── comparison_N.py
├── examples/
│   ├── complete_workflow_example.py
│   ├── movement_example.py
│   └── performance_analysis.py
├── requirements.txt
└── README.md
```

## Physical Model

The simulator enforces trapezoidal velocity profiles that honour both maximum acceleration and velocity limits. Key defaults (overridable via configuration):

- Lattice site spacing: 5.0 µm
- Maximum acceleration: 2750 m/s²
- Maximum velocity: 0.1 m/s
- Settling time: 1 µs
- Atom transport loss: configurable (default 0)

## Movement Strategies (only center is used)

### Center strategy
1. Places the target region centrally.
2. Aligns rows and columns around the target.
3. Executes spread-squeeze cycles to repair larger defects.
4. Uses targeted path planning (direct, L-shaped, A* search) for remaining vacancies.

### Corner strategy
1. Anchors the target in a lattice corner.
2. Squeezes rows left and columns up to fill the corner block.
3. Applies right-edge squeezing for atoms below the target area.
4. Finalizes with localized defect repair.

Both strategies share batching logic that groups non-conflicting moves, reducing physical execution time.

## Algorithm Reference (Center)

- **Row/column centering**: Builds dense stripes that feed later repair steps.
- **Spread-squeeze cycles**: Iteratively redistribute atoms from populated regions into vacancies.
- **Defect repair**: Progressive planner that tries straight moves, single-turn routes (L-shaped), then A* search.
- **Batch compression**: Merges compatible moves into parallel batches to minimize total operations.

## Visualization

`defect_free.visualizer` renders lattice states, movement sequences, and statistics. Can be used to generate GIFs as in `examples/movement_example.py`.

## Performance Considerations

- The center strategy achieves slightly higher fill rates but requires heavier planning.
- The corner strategy runs faster computationally, suitable for quick parameter sweeps.
- Transport loss settings can be tuned to model imperfect trap transfers.

## Dependencies

Python 3.11 (3.11.4 in my setup) with packages listed in `requirements.txt` (NumPy, Matplotlib, Pandas, SciPy, tqdm). Install them in a virtual environment before running the scripts.

## Benchmark quick-run instructions

Both benchmark entry points live in `benchmarks/` and the scripts are in the end of the file:

- `blind_benchmark.py`: Sweeps lattice sizes (10x10 to 100x100), occupation levels (0.5, 0.7, 0.9), and transport loss rates (0.0, 0.01, 0.05). It records fill/retention statistics and gives CSV summaries together with plots.
- `comparison_N.py`: Measures how the full rearrangement algorithm scales with lattice size. It repeats simulations over a size list (e.g., 50…150), fits power-law exponents for the number of move batches, and saves JSON/NPZ data. Also plots the result against literature lines (PSCA).

### Prerequisites

- Python 3.11 (I use 3.11.4)
- Dependencies installed via `pip install -r requirements.txt`

### Environment setup

1. Create and activate a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Quick sanity check

Verify the installation with a light run:

```bash
python benchmarks/comparison_N.py --initial-sizes 50 --trials 5 --output quick_check --seed 42
```

### Full L = 60…150 sweep

This preset runs lattice sizes 60,70,…,150 (step 10) with 20 trials each and stores outputs in `N_scaling_075_L60-150`.

```bash
bash run_L60_150.sh
```

Equivalent to the following command:

```bash
python benchmarks/comparison_N.py \
  --initial-sizes 60,70,80,90,100,110,120,130,140,150 \
  --occupation 0.75 \
  --loss 0.0 \
  --trials 20 \
  --seed 42 \
  --output N_scaling_075_L60-150 \
  --strategy center \
  --max-cycles 6 \
  --target-fill 1.0
```

### Outputs

- `blind_benchmark.py` writes CSV files and plots under `benchmark_results/`
- `comparison_N.py` produces figures, `complete_algorithm_results.json`, and `complete_algorithm_data.npz` in the chosen output directory

### Visualization example
- `examples/movement_example.py` simulates a 20x20 lattice with the movement strategy, and exports a GIF.


### Additional examples 

- `examples/complete_workflow_example.py` runs the full pipeline end-to-end with configurable strategies.
- `examples/performance_analysis.py` provides a template for timing studies with custom parameter sweeps.

---

