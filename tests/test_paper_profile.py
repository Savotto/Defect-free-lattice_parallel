import csv
import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest

from defect_free.parallel_utils import merge_parallel_batches
from defect_free.simulator import LatticeSimulator


PROFILE = LatticeSimulator.PAPER_REPRODUCIBILITY_PROFILE


def simulator_with_atom_count(width, atom_count, loss):
    simulator = LatticeSimulator(
        initial_size=(width, width),
        occupation_prob=atom_count / (width * width),
        physical_constraints={"atom_loss_probability": loss},
        reproducibility_profile=PROFILE,
    )
    field = np.zeros(width * width, dtype=int)
    field[:atom_count] = 1
    simulator.field = field.reshape(width, width)
    simulator.slm_lattice = simulator.field.copy()
    simulator.total_atoms = atom_count
    return simulator


def paper_target_side(width, atom_count, loss):
    rho = atom_count / (width * width)
    coefficients = LatticeSimulator.PAPER_MOVE_FORMULA_COEFFS
    moves = math.sqrt(width) * (
        coefficients["intercept"]
        + coefficients["occupation"] * rho
        + coefficients["occupation_sq"] * rho * rho
        + coefficients["loss"] * loss
        + coefficients["occupation_loss"] * rho * loss
        + coefficients["high_loss_step_penalty"]
        * (1 if loss >= coefficients["high_loss_threshold"] else 0)
    )
    moves = max(moves, 1.0)
    side = math.floor(math.sqrt(atom_count * (1 - loss) ** moves))
    if width >= 100 and loss <= coefficients["zero_loss_probability_threshold"]:
        side -= coefficients["zero_loss_side_delta"]
    if side == width - 1:
        side = width - 2
    return side


@pytest.mark.parametrize(
    ("width", "atom_count", "loss"),
    [(40, 1117, 0.0), (80, 4473, 0.01), (80, 4473, 0.05), (100, 6997, 0.0)],
)
def test_paper_target_sizing_matches_equation(width, atom_count, loss):
    simulator = simulator_with_atom_count(width, atom_count, loss)
    assert simulator.calculate_max_defect_free_size() == paper_target_side(
        width, atom_count, loss
    )


def test_paper_profile_freezes_reported_algorithm_choices():
    simulator = simulator_with_atom_count(20, 280, 0.01)
    assert simulator.constraints["target_sizing_policy"] == "paper_move_formula"
    assert simulator.constraints["movement_policy"] == "atlas_classic"
    assert simulator.constraints["disable_defect_repair_planning"] is True
    assert simulator.constraints["atlas_classic_use_original_split_policy"] is True
    assert simulator.constraints["batch_merge_policy"] == "contiguous"
    assert simulator.constraints["near_perfect_grace_iterations"] == 0


def test_unknown_reproducibility_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown reproducibility profile"):
        LatticeSimulator(reproducibility_profile="unknown")


def test_paper_profile_rejects_behavioral_overrides():
    with pytest.raises(ValueError, match="only permits overriding"):
        LatticeSimulator(
            reproducibility_profile=PROFILE,
            physical_constraints={"batch_merge_policy": "phase_aware"},
        )


def test_contiguous_merge_policy_accepts_phase_tagged_batches():
    initial = np.zeros((2, 5), dtype=int)
    initial[0, 0] = 1
    initial[1, 0] = 1
    batches = [
        {
            "phase": "row:1",
            "moves": [{"from": (0, 0), "to": (0, 1)}],
            "state": None,
            "time": 0.1,
        },
        {
            "phase": "row:1",
            "moves": [{"from": (1, 0), "to": (1, 1)}],
            "state": None,
            "time": 0.2,
        },
    ]
    merged = merge_parallel_batches(batches, initial, policy="contiguous")
    assert len(merged) == 1
    assert len(merged[0]["moves"]) == 2
    assert merged[0]["time"] == 0.2


def test_paper_sizing_reproduces_archived_calibration_rows():
    archive = Path(__file__).parent / "data" / "paper_target_sizing_archive.csv"
    checked = 0
    with archive.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] != "move_formula_penalized":
                continue
            width = int(row["size"])
            occupation = float(row["occupation_probability"])
            loss = float(row["atom_loss_probability"])
            sample_idx = int(row["sample_idx"])
            seed = (
                777
                + width * 1_000_000
                + int(round(occupation * 1000)) * 10_000
                + int(round(loss * 10000)) * 100
                + sample_idx
            )
            simulator = LatticeSimulator(
                initial_size=(width, width),
                occupation_prob=occupation,
                physical_constraints={"atom_loss_probability": loss},
                reproducibility_profile=PROFILE,
            )
            simulator.generate_initial_lattice(seed=seed)
            assert simulator.calculate_max_defect_free_size() == int(row["target_side"])
            checked += 1
    assert checked > 0


@pytest.mark.parametrize(
    ("width", "seed", "expected_fill", "expected_moves"),
    [
        (50, 50712345, 0.996031746031746, 122),
        # The archived fill is exact; its batch count differs by one because a
        # later safety check splits one otherwise equivalent movement batch.
        (50, 50712347, 0.9970255800118977, None),
        (100, 100712345, 1.0, 227),
        (100, 100712346, 1.0, 219),
    ],
)
def test_paper_profile_reproduces_archived_zero_loss_runs(
    width, seed, expected_fill, expected_moves
):
    simulator = LatticeSimulator(
        initial_size=(width, width),
        occupation_prob=0.7,
        physical_constraints={"atom_loss_probability": 0.0},
        reproducibility_profile=PROFILE,
    )
    simulator.generate_initial_lattice(seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        simulator.calculate_max_defect_free_size()
        _, fill_rate, _, _ = (
            simulator.movement_manager.center_manager.iterative_blind_center_filling(
                max_iterations=None,
                min_improvement=0.0,
                show_visualization=False,
                use_batch_merging=True,
            )
        )
    moves = sum(item["moves"] for item in simulator.last_iteration_stats)
    assert fill_rate == pytest.approx(expected_fill)
    if expected_moves is not None:
        assert moves == expected_moves
