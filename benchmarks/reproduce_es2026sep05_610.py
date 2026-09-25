#!/usr/bin/env python3
"""Reproduce the Monte Carlo experiments in PRA manuscript es2026sep05_610."""
import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from defect_free.simulator import LatticeSimulator


ITERATION_COLUMNS = [
    "iteration",
    "computational_time_mean",
    "computational_time_std",
    "physical_time_mean",
    "physical_time_std",
    "total_time_mean",
    "total_time_std",
    "moves_mean",
    "moves_std",
    "fill_rate_mean",
    "fill_rate_std",
    "defects_mean",
    "defects_std",
    "atoms_in_target_mean",
    "atoms_in_target_std",
    "retention_rate_mean",
    "retention_rate_std",
]

SUMMARY_COLUMNS = [
    "occupation_probability",
    "atom_loss_probability",
    "mean_iterations_to_perfect_fill",
    "std_iterations_to_perfect_fill",
    "mean_final_fill_rate",
    "std_final_fill_rate",
    "mean_total_moves",
    "std_total_moves",
    "mean_final_retention_rate",
    "std_final_retention_rate",
    "mean_total_computational_time",
    "std_total_computational_time",
    "mean_total_physical_time",
    "std_total_physical_time",
]


def parse_float_list(text: str):
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def parse_int_list(text: str):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_size_sample_overrides(text: str):
    """
    Parse size->samples overrides from "10:20,30:5" format.
    """
    overrides = {}
    if not text.strip():
        return overrides
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid override '{part}'. Expected format size:samples.")
        size_text, samples_text = part.split(":", 1)
        size = int(size_text.strip())
        samples = int(samples_text.strip())
        if size <= 0 or samples <= 0:
            raise ValueError(f"Invalid override '{part}'. Size and samples must be positive.")
        overrides[size] = samples
    return overrides


def build_sizes(start: int, stop: int, step: int, extras):
    sizes = set(range(start, stop + 1, step))
    sizes.update(extras)
    return sorted(s for s in sizes if s > 0)


def mean_std(values):
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def fmt_float(x: float):
    return f"{x:.12g}"


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_single_sample(
    size: int,
    occupation: float,
    loss_prob: float,
    seed: int,
    max_iterations,
    min_improvement: float,
    quiet_algorithm: bool,
):
    simulator = LatticeSimulator(
        initial_size=(size, size),
        occupation_prob=occupation,
        physical_constraints={"atom_loss_probability": loss_prob},
        reproducibility_profile=LatticeSimulator.PAPER_REPRODUCIBILITY_PROFILE,
    )
    simulator.generate_initial_lattice(seed=seed)

    simulator.target_shape = "square"
    simulator.target_shape_params = {}
    simulator.target_mask = None
    simulator.movement_manager.reset_target_definition()
    simulator.calculate_max_defect_free_size(strategy="center")

    def _run():
        return simulator.movement_manager.center_manager.iterative_blind_center_filling(
            max_iterations=max_iterations,
            min_improvement=min_improvement,
            show_visualization=False,
            use_batch_merging=True,
        )

    if quiet_algorithm:
        with redirect_stdout(io.StringIO()):
            _, final_fill_rate, _, iterations_used = _run()
    else:
        _, final_fill_rate, _, iterations_used = _run()

    iteration_stats = list(getattr(simulator, "last_iteration_stats", []))
    total_moves = int(sum(it["moves"] for it in iteration_stats))
    total_comp = float(sum(it["computational_time"] for it in iteration_stats))
    total_phys = float(sum(it["physical_time"] for it in iteration_stats))
    final_retention = float(iteration_stats[-1]["retention_rate"]) if iteration_stats else 0.0

    iteration_to_perfect = None
    for it in iteration_stats:
        if int(it["defects"]) == 0:
            iteration_to_perfect = int(it["iteration"])
            break

    return {
        "initial_atoms": int(simulator.total_atoms),
        "target_side": int(simulator.side_length),
        "target_sites": int(simulator.side_length ** 2),
        "final_fill_rate": float(final_fill_rate),
        "iterations_used": int(iterations_used),
        "iteration_to_perfect_fill": iteration_to_perfect,
        "total_moves": total_moves,
        "total_computational_time": total_comp,
        "total_physical_time": total_phys,
        "final_retention_rate": final_retention,
        "iteration_stats": iteration_stats,
    }


def aggregate_iteration_stats(iteration_samples):
    rows = []
    max_iter = max(iteration_samples.keys()) if iteration_samples else 0
    for iteration in range(1, max_iter + 1):
        samples = iteration_samples.get(iteration, [])
        if not samples:
            continue
        rows.append(
            {
                "iteration": iteration,
                "computational_time_mean": mean_std([s["computational_time"] for s in samples])[0],
                "computational_time_std": mean_std([s["computational_time"] for s in samples])[1],
                "physical_time_mean": mean_std([s["physical_time"] for s in samples])[0],
                "physical_time_std": mean_std([s["physical_time"] for s in samples])[1],
                "total_time_mean": mean_std([s["total_time"] for s in samples])[0],
                "total_time_std": mean_std([s["total_time"] for s in samples])[1],
                "moves_mean": mean_std([s["moves"] for s in samples])[0],
                "moves_std": mean_std([s["moves"] for s in samples])[1],
                "fill_rate_mean": mean_std([s["fill_rate"] for s in samples])[0],
                "fill_rate_std": mean_std([s["fill_rate"] for s in samples])[1],
                "defects_mean": mean_std([s["defects"] for s in samples])[0],
                "defects_std": mean_std([s["defects"] for s in samples])[1],
                "atoms_in_target_mean": mean_std([s["atoms_in_target"] for s in samples])[0],
                "atoms_in_target_std": mean_std([s["atoms_in_target"] for s in samples])[1],
                "retention_rate_mean": mean_std([s["retention_rate"] for s in samples])[0],
                "retention_rate_std": mean_std([s["retention_rate"] for s in samples])[1],
            }
        )
    return rows


def aggregate_summary(sample_rows):
    by_key = {}
    for row in sample_rows:
        key = (row["mode"], row["size"], row["occupation_probability"], row["atom_loss_probability"])
        by_key.setdefault(key, []).append(row)

    combined_rows = []
    per_mode_size = {}
    for (mode, size, occ, loss), rows in sorted(by_key.items()):
        iters_perfect = [r["iteration_to_perfect_fill"] for r in rows if r["iteration_to_perfect_fill"] is not None]
        summary = {
            "occupation_probability": occ,
            "atom_loss_probability": loss,
            "mean_iterations_to_perfect_fill": mean_std(iters_perfect)[0],
            "std_iterations_to_perfect_fill": mean_std(iters_perfect)[1],
            "mean_final_fill_rate": mean_std([r["final_fill_rate"] for r in rows])[0],
            "std_final_fill_rate": mean_std([r["final_fill_rate"] for r in rows])[1],
            "mean_total_moves": mean_std([r["total_moves"] for r in rows])[0],
            "std_total_moves": mean_std([r["total_moves"] for r in rows])[1],
            "mean_final_retention_rate": mean_std([r["final_retention_rate"] for r in rows])[0],
            "std_final_retention_rate": mean_std([r["final_retention_rate"] for r in rows])[1],
            "mean_total_computational_time": mean_std([r["total_computational_time"] for r in rows])[0],
            "std_total_computational_time": mean_std([r["total_computational_time"] for r in rows])[1],
            "mean_total_physical_time": mean_std([r["total_physical_time"] for r in rows])[0],
            "std_total_physical_time": mean_std([r["total_physical_time"] for r in rows])[1],
        }
        per_mode_size.setdefault((mode, size), []).append(summary)
        combined_rows.append(
            {
                "mode": mode,
                "lattice_size": size,
                **summary,
            }
        )

    return combined_rows, per_mode_size


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce PRA manuscript es2026sep05_610 Monte Carlo experiments."
    )
    parser.add_argument("--occupations", type=str, default="0.5,0.7,0.9")
    parser.add_argument("--loss-probs", type=str, default="0,0.01,0.05")
    parser.add_argument("--samples", type=int, default=100, help="Default sample count per size/config.")
    parser.add_argument(
        "--samples-size-overrides",
        type=str,
        default="",
        help="Override samples for specific sizes. Format: '10:20,30:5'",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means unbounded.")
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--quiet-algorithm", action="store_true")
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(size) for size in range(10, 201, 10)),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: benchmark_results/square_iteration_stats_<timestamp>",
    )
    args = parser.parse_args()

    occupations = parse_float_list(args.occupations)
    loss_probs = parse_float_list(args.loss_probs)
    sample_overrides = parse_size_sample_overrides(args.samples_size_overrides)
    merge_sizes = parse_int_list(args.sizes)
    no_merge_sizes = []
    max_iterations = None if args.max_iterations <= 0 else args.max_iterations

    def samples_for_size(size: int) -> int:
        return int(sample_overrides.get(size, args.samples))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("benchmark_results") / f"square_iteration_stats_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "occupations": occupations,
        "loss_probs": loss_probs,
        "samples": args.samples,
        "samples_size_overrides": sample_overrides,
        "reproducibility_profile": LatticeSimulator.PAPER_REPRODUCIBILITY_PROFILE,
        "seed_values": [args.seed_start, args.seed_start + args.samples - 1],
        "max_iterations": max_iterations,
        "min_improvement": args.min_improvement,
        "disable_defect_repair": True,
        "merge_only": True,
        "target_sizing_policy": "paper_move_formula",
        "movement_policy": "atlas_classic",
        "assignment_policy": "original_center_split",
        "batch_merge_policy": "contiguous",
        "merge_sizes": merge_sizes,
        "no_merge_sizes": no_merge_sizes,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    raw_rows = []
    per_iter = {}
    total_runs = 0
    for sizes in (merge_sizes, no_merge_sizes):
        for size in sizes:
            total_runs += samples_for_size(size) * len(occupations) * len(loss_probs)
    run_idx = 0

    for mode, use_batch_merging, sizes in (
        ("merge", True, merge_sizes),
        ("no_merge", False, no_merge_sizes),
    ):
        print(f"\nMode: {mode}")
        for size in sizes:
            print(f"  Size {size}x{size}")
            for occ in occupations:
                for loss in loss_probs:
                    combo_key = (mode, size, occ, loss)
                    per_iter.setdefault(combo_key, {})
                    size_samples = samples_for_size(size)
                    for sample in range(size_samples):
                        run_idx += 1
                        seed = args.seed_start + sample
                        sample_result = run_single_sample(
                            size=size,
                            occupation=occ,
                            loss_prob=loss,
                            seed=seed,
                            max_iterations=max_iterations,
                            min_improvement=args.min_improvement,
                            quiet_algorithm=args.quiet_algorithm,
                        )

                        raw_rows.append(
                            {
                                "mode": mode,
                                "size": size,
                                "occupation_probability": occ,
                                "atom_loss_probability": loss,
                                "sample": sample,
                                "seed": seed,
                                "initial_atoms": sample_result["initial_atoms"],
                                "target_side": sample_result["target_side"],
                                "target_sites": sample_result["target_sites"],
                                "iteration_to_perfect_fill": sample_result["iteration_to_perfect_fill"],
                                "final_fill_rate": sample_result["final_fill_rate"],
                                "total_moves": sample_result["total_moves"],
                                "final_retention_rate": sample_result["final_retention_rate"],
                                "total_computational_time": sample_result["total_computational_time"],
                                "total_physical_time": sample_result["total_physical_time"],
                            }
                        )

                        for it in sample_result["iteration_stats"]:
                            iter_idx = int(it["iteration"])
                            per_iter[combo_key].setdefault(iter_idx, []).append(it)

                        print(
                            f"    run {run_idx}/{total_runs}: occ={occ}, loss={loss}, "
                            f"fill={sample_result['final_fill_rate']:.4f}, "
                            f"iter={sample_result['iterations_used']}"
                        )

    write_csv(
        out_dir / "raw_samples.csv",
        raw_rows,
        [
            "mode",
            "size",
            "occupation_probability",
            "atom_loss_probability",
            "sample",
            "seed",
            "initial_atoms",
            "target_side",
            "target_sites",
            "iteration_to_perfect_fill",
            "final_fill_rate",
            "total_moves",
            "final_retention_rate",
            "total_computational_time",
            "total_physical_time",
        ],
    )

    # Per-iteration stats files (same schema as your mc_iteration_stats files).
    for (mode, size, occ, loss), iteration_samples in per_iter.items():
        rows = aggregate_iteration_stats(iteration_samples)
        per_iter_dir = out_dir / "per_iteration" / mode
        filename = f"mc_iteration_stats_{size}x{size}_occ{fmt_float(occ)}_loss{fmt_float(loss)}.csv"
        write_csv(per_iter_dir / filename, rows, ITERATION_COLUMNS)

    combined_summary_rows, per_mode_size = aggregate_summary(raw_rows)

    # Combined summary with mode + size columns.
    write_csv(
        out_dir / "summary_table_all_modes_sizes.csv",
        combined_summary_rows,
        ["mode", "lattice_size"] + SUMMARY_COLUMNS,
    )

    # Separate summary_table files per mode and per size (matching the classic schema).
    for (mode, size), rows in per_mode_size.items():
        rows_sorted = sorted(rows, key=lambda r: (r["occupation_probability"], r["atom_loss_probability"]))
        write_csv(out_dir / f"summary_table_{mode}_{size}x{size}.csv", rows_sorted, SUMMARY_COLUMNS)

    print("\nDone.")
    print(f"  Output directory: {out_dir}")
    print(f"  Raw samples: {out_dir / 'raw_samples.csv'}")
    print(f"  Combined summary: {out_dir / 'summary_table_all_modes_sizes.csv'}")
    print(f"  Per-iteration directory: {out_dir / 'per_iteration'}")


if __name__ == "__main__":
    main()
