from collections import defaultdict
from dataclasses import dataclass
import os
import sys
import yaml
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.md import (
    get_simulation_environment,
    compute_energy_and_forces,
    compute_energy_and_forces_decomposition,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float

    @staticmethod
    def from_sequence(seq):
        return MeanStd(float(np.mean(seq)), float(np.std(seq)))


def zip_mean_stds(mean_stds: List[MeanStd]) -> Tuple[List[float], List[float]]:
    return ([ms.mean for ms in mean_stds], [ms.std for ms in mean_stds])


def get_energy_mean_std(
    state0pdbpath: str,
    positions: List[np.array],
    velocities: List[np.array],
    use_integrator_for_KE: bool = True,
    energy_breakdown: bool = False,
    parameters: str = "T1-peptides",
) -> Dict[str, List[MeanStd]]:
    """Compute means and std deviations of energies across samples for all states in a batch.

    Args:
        state0pdbpath (str): path to state0 pdb file
        positions (List[np.array]): positions along trajectory, length B list of [S, V, 3]
        velocities (List[np.array]): velocities along trajectory, length B list of [S, V, 3]
        use_integrator_for_KE: bool: whether to use the OpenMM integrator to interpolate the velocity values when computing
            kinetic energy. This can cause issues if the position values are very inaccurate, since then the force computation
            becomes inaccurate. See https://github.com/openmm/openmm/issues/1474 for details.
        parameters (str): Parameter preset name to construct force-field

    Returns:
        a dictionary containing a list of MeanStd keyed by the names of energies.
    """

    sim = get_simulation_environment(state0pdbpath, parameters)
    mean_stds: DefaultDict[str, List[MeanStd]] = defaultdict(list)

    for coord, veloc in zip(positions, velocities):
        if energy_breakdown:
            energies, _ = compute_energy_and_forces_decomposition(
                state0pdbpath, parameters, coord, veloc
            )
        else:
            potential_and_kinetic, _ = compute_energy_and_forces(
                sim,
                coord,  # [S, V, 3]
                veloc,  # [S, V, 3]
            )  # [S, 2]
            PEs = potential_and_kinetic[:, 0]  # [S]
            if use_integrator_for_KE:
                KEs = potential_and_kinetic[:, 1]  # [S]
            else:
                KEs = potential_and_kinetic[:, 2]  # [S]
            energies = {"Potential": PEs, "Kinetic": KEs}

        for key, value in energies.items():
            mean_stds[key].append(MeanStd.from_sequence(value))

        total_energies = [sum(terms) for terms in zip(*energies.values())]
        assert len(total_energies) == coord.shape[0]
        mean_stds["Total"].append(MeanStd.from_sequence(total_energies))

    return mean_stds  # Values are lists of length B


def plot_single_energy(
    means: np.array,
    stds: np.array,
    energy_type: str,
    figpath: str,
):
    traj_idx = np.arange(means.shape[0]) + 1

    plt.plot(traj_idx, means, label=energy_type)
    plt.fill_between(traj_idx, means - stds, means + stds, alpha=0.6)
    plt.xlabel("Trajectory sample")
    plt.ylabel("Energy (kJ/mol)")
    plt.title(f"{energy_type} energy")
    plt.savefig(figpath)
    plt.close()


def plot_all_energy(
    mean_stds_by_names: Dict[str, List[MeanStd]],
    output_dir: str,
    plot_name: str,
):
    """Plot of potential, kinetic, and total energy."""
    for energy_name, mean_stds in mean_stds_by_names.items():
        traj_idx = np.arange(len(mean_stds)) + 1
        means_array, stds_array = np.array(zip_mean_stds(mean_stds))
        plt.plot(traj_idx, means_array, label=energy_name)
        plt.fill_between(traj_idx, means_array - stds_array, means_array + stds_array, alpha=0.6)
    plt.xlabel("Trajectory sample")
    plt.ylabel("Energy (kJ/mol)")
    plt.title("Energies")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{plot_name}_all_energies.png"))
    plt.close()

    """Plot individual energy types"""
    for energy_name, mean_stds in mean_stds_by_names.items():
        means_array, stds_array = np.array(zip_mean_stds(mean_stds))
        plot_single_energy(
            means_array,
            stds_array,
            energy_type=energy_name,
            figpath=os.path.join(output_dir, f"{plot_name}_{energy_name}_energies.png"),
        )

    """Save energy statistics."""
    sample_energy_statistics = {}
    for energy_name, mean_stds in mean_stds_by_names.items():
        (
            sample_energy_statistics[f"{energy_name}_mean"],
            sample_energy_statistics[f"{energy_name}_std"],
        ) = zip_mean_stds(mean_stds)
    with open(os.path.join(output_dir, f"{plot_name}_energy_statistics.yaml"), "w") as outfile:
        yaml.dump(sample_energy_statistics, outfile)
