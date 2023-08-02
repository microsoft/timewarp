import torch
import numpy as np
import openmm as mm
import openmm.unit as u
import warnings

import matplotlib.pyplot as plt
import os
import sys
import argparse
from itertools import islice
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from timewarp.utils.sampling_utils import sample
from timewarp.utils.training_utils import load_model
from timewarp.utils.evaluation_utils_o2 import (
    create_o2_system,
    harm_osci_prob,
    compute_bond_length,
    sample_with_model,
)
from timewarp.datasets import RawMolDynDataset
from timewarp.dataloader import (
    moldyn_dense_collate_fn,
)
from timewarp.modules.layers.openmm_bridge import OpenmmPotentialEnergyTorch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savefile", type=str, help="Saved model config and state dict.")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for distribution comparison.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Path to directory to save images.",
    )
    parser.add_argument(
        "--sample",
        type=bool,
        default=False,
        help="Whether to sample with the model.",
    )
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(path=args.savefile).to(device)
    config = torch.load(args.savefile)["training_config"]
    step_width = config.step_width
    data_set = config.dataset

    if data_set == "O2" or data_set == "O2-CoM":
        protein = "o2"
    elif data_set == "AD-1":
        raise ValueError(
            f"So far the evaluation is not implemented for the {data_set} dataset. It is only available for the O2 dataset."
        )
        protein = "ad2"
    else:
        raise ValueError(
            f"So far the evaluation is not implemented for the {data_set} dataset. It is only available for the O2 dataset."
        )

    # load the dataset
    raw_dataset = RawMolDynDataset(data_dir=args.data_dir, step_width=step_width)
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, 100))  # Load 100 timepoints.

    if args.num_samples < 10000:
        warnings.warn("It is recommended to use 10000 samples.")

    # sample from the model
    print("Sampling ...")
    batch = batches[3]  # 81
    y_coords_model, y_velocs_model = sample(
        model=model,
        batch=batch,
        num_samples=args.num_samples,
        decorrelated=False,
        device=device,
    )  # [S, V, 3], [S, V, 3]

    # create openMM simulation
    system, topology = create_o2_system()
    integrator = mm.LangevinIntegrator(310 * u.kelvin, 1 / u.picosecond, 1 * u.femtosecond)

    simulation = mm.app.Simulation(topology, system, integrator)
    openMM_samples = []
    openMM_velocs = []

    for _ in range(args.num_samples):
        simulation.context.setPositions(batch.atom_coords.squeeze(0).cpu().numpy())
        simulation.context.setVelocities(batch.atom_velocs.squeeze(0).cpu().numpy())
        simulation.step(step_width)
        state = simulation.context.getState(getPositions=True, getEnergy=True, getVelocities=True)
        openMM_samples.append(state.getPositions(asNumpy=True)._value)
        openMM_velocs.append(state.getVelocities(asNumpy=True)._value)

    # Compute bond lengths
    bond_length_conditioning = compute_bond_length(batch.atom_coords)
    bond_length = compute_bond_length(y_coords_model)
    bond_length_openmm = compute_bond_length(np.array(openMM_samples))

    print("Plotting ...")
    plt.hist(
        bond_length_openmm,
        bins=100,
        alpha=0.5,
        density=True,
        label=f"openMM - step-width {step_width}",
    )
    plt.hist(
        bond_length, bins=100, alpha=0.5, density=True, label=f"Model - step-width {step_width}"
    )
    if protein == "o2" and step_width >= 10:
        xs = np.arange(0.09, 0.115, 0.0001)
        plt.plot(xs, harm_osci_prob(xs), label="Target dist")
    plt.axvline(
        bond_length_conditioning,
        0,
        0.9,
        color="black",
        label="Conditioning bond length",
        linewidth=5,
    )
    plt.legend()
    plt.xlabel("Distance in nm")
    plt.title("Bond length for single conditioning sample")
    plt.savefig(
        os.path.join(
            args.output_dir,
            f"{protein}_conditionial_distribution_bondlength_step_width_{step_width}.png",
        )
    )
    plt.close()

    # Two-sample Kolmogorov-Smirnov test
    ks_result, p_value = ks_2samp(bond_length, bond_length_openmm)

    print("Difference between the conditional distributions")
    print(f"KS statistic.: {ks_result}")
    print(f"p-value: {p_value}")

    if args.sample:
        print("Sampling with model...")
        integrator = mm.LangevinIntegrator(310 * u.kelvin, 1 / u.picosecond, 1 * u.femtosecond)
        openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
            system, integrator, platform_name="CUDA"
        )
        acceptance_rate, y_coords, y_velocs = sample_with_model(
            n_samples=args.num_samples,
            model=model,
            initial_batch=batch,
            target_energy=openmm_potential_energy_torch,
            system=system,
            device=device,
        )
        print(f"Acceptance rate: {acceptance_rate}")
        # Compute bond lengths
        sampled_bond_length = compute_bond_length(y_coords.cpu().numpy())

        print("Plotting ...")
        plt.hist(
            sampled_bond_length,
            bins=100,
            alpha=0.5,
            density=True,
            label=f"Model - step-width {step_width}",
        )
        if protein == "o2" and step_width >= 10:
            xs = np.arange(0.09, 0.115, 0.0001)
            plt.plot(xs, harm_osci_prob(xs), label="Target dist")
            plt.legend()
        plt.xlabel("Distance in nm")
        plt.title("Bond length for model samples")
        plt.savefig(
            os.path.join(
                args.output_dir,
                f"{protein}_bondlength_step_width_{step_width}.png",
            )
        )


if __name__ == "__main__":
    main()
