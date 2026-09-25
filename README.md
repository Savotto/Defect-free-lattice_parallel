# ATLAS: loss-aware neutral-atom rearrangement

This repository contains the reference implementation used for the Physical
Review Applied manuscript **es2026sep05_610**, “ATLAS: Efficient Atom
Rearrangement for Defect-Free Neutral-Atom Quantum Arrays Under Transport
Loss.”

ATLAS plans rearrangement on a lossless virtual lattice, merges compatible
movements into parallel AOD-safe batches, executes the batches with stochastic
transport loss, and replans from the true post-loss state.

## Paper-reproduction profile

Use the immutable profile name `es2026sep05_610` whenever reproducing reported
results:

```python
from defect_free.simulator import LatticeSimulator

simulator = LatticeSimulator(
    initial_size=(100, 100),
    occupation_prob=0.7,
    physical_constraints={"atom_loss_probability": 0.01},
    reproducibility_profile="es2026sep05_610",
)
simulator.generate_initial_lattice(seed=0)
field, fill_rate, computation_time = simulator.rearrange_for_defect_free(
    show_visualization=False
)
```

The profile freezes the choices described in the manuscript:

- centered square targets;
- the loss-aware target-sizing equation and its fitted coefficients;
- the `b5 = 0.15` correction at `p_loss >= 0.05`;
- the one-site `Delta_0` correction for zero-loss arrays with `W >= 100`;
- the original center-split row, column, and spread assignments;
- the contiguous linear greedy batch-merging pass;
- no sequential defect-repair moves in reported move and time measurements;
- termination after perfect fill or two consecutive iterations without fill
  improvement, with no additional grace iteration.

Later experimental policies are not selected by this profile and must not be
used when comparing against the manuscript.

## Installation

The reported environment used Python 3.11.4, NumPy 1.26.4, and Matplotlib
3.10.1. Create an isolated environment and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Reproduce the Monte Carlo sweep

A quick smoke run is:

```bash
python benchmarks/reproduce_es2026sep05_610.py \
  --sizes 10 \
  --occupations 0.7 \
  --loss-probs 0,0.01,0.05 \
  --samples 1 \
  --quiet-algorithm \
  --output-dir benchmark_results/smoke
```

The full paper grid is the script default: lattice widths `10,20,...,200`,
occupation probabilities `0.5,0.7,0.9`, loss probabilities `0,0.01,0.05`, and
seeds `0,...,99`:

```bash
python benchmarks/reproduce_es2026sep05_610.py \
  --quiet-algorithm \
  --output-dir benchmark_results/es2026sep05_610
```

The full sweep is computationally expensive. Results are written as raw runs,
per-iteration summaries, and combined summary tables. Computational-time values
are hardware-dependent; move counts, target sizes, fill rates, retention, and
modeled physical times are the portable quantities.

## Verification

Run the regression suite with:

```bash
python -m pytest -q
```

The paper-profile tests independently evaluate the target-sizing equation,
check every archived calibration row, and reproduce selected archived
zero-loss runs including fill rate and movement-batch count.

## Physical model

The fixed values used in the manuscript are:

| Parameter | Value |
|---|---:|
| Site spacing | 5 micrometers |
| Maximum acceleration | 2750 m/s^2 |
| Maximum velocity | 0.13 m/s |
| Pickup time | 60 microseconds |
| Drop-off time | 60 microseconds |

Transport loss is sampled independently for each attempted atom movement. A
movement batch is timed using its longest Manhattan-distance transport and a
triangular or trapezoidal velocity profile, plus pickup and drop-off time.

## Repository scope

The public paper release should contain the ATLAS implementation, the paper
reproduction script, regression fixtures, and documentation. Generated plots,
large exploratory result trees, virtual environments, caches, unrelated
planners, and local working files are intentionally excluded.

## License

The code is released under the MIT License. See `LICENSE`.
