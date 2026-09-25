"""Minimal ATLAS run using the manuscript reproducibility profile."""

from defect_free.simulator import LatticeSimulator


simulator = LatticeSimulator(
    initial_size=(50, 50),
    occupation_prob=0.7,
    physical_constraints={"atom_loss_probability": 0.01},
    reproducibility_profile=LatticeSimulator.PAPER_REPRODUCIBILITY_PROFILE,
)
simulator.generate_initial_lattice(seed=0)
field, fill_rate, computation_time = simulator.rearrange_for_defect_free(
    show_visualization=False
)

print(f"target side: {simulator.side_length}")
print(f"fill rate: {fill_rate:.6f}")
print(f"retention rate: {field[simulator.target_mask].sum() / simulator.total_atoms:.6f}")
print(f"computation time: {computation_time:.6f} s")
