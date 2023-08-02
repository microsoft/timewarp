from pathlib import Path
import torch
import numpy as np
import openmm.unit as u

import os
import sys
import argparse
from itertools import islice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from timewarp.utils.training_utils import load_model  # noqa: F401

from timewarp.datasets import RawMolDynDataset

from simulation.md import get_simulation_environment_integrator, get_simulation_environment
from timewarp.dataloader import (
    moldyn_dense_collate_fn,
)

from utilities.training_utils import set_seed
from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
from timeit import default_timer as timer
from timewarp.model_constructor import model_constructor
from timewarp.utils.training_utils import load_or_construct_loss
from timewarp.losses import (
    wrap_or_replace_loss,
)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("savefile", type=Path, help="Saved model config and state dict.")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.", required=True)
    parser.add_argument("--protein", type=str, help="Protein name.", required=True)
    parser.add_argument(
        "--device",
        help="Device to use. If using CUDA, env variable `CUDA_VISIBLE_DEVICES=index` is the preferred way of specifying which device to use. Default: `cuda` if available, otherwise `cpu`",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for various methods.",
    )
    parser.add_argument(
        "--saving_interval",
        type=int,
        default=1000,
        help="Number of samples for various methods.",
    )
    parser.add_argument(
        "--energy_threshold",
        type=int,
        default=300,
        help="Number of samples for various methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to directory to save trajectory.",
    )
    parser.add_argument(
        "--output-dir-basename",
        type=str,
        help="Basename of the output directory. This has no effect if --output-dir is specified.",
        default="evaluation",
    )
    parser.add_argument(
        "--mh",
        action="store_true",
        help="If specified, sample from model using Metropolis-Hasting correction.",
    )
    parser.add_argument(
        "--random-velocities",
        action="store_true",
        help="If specified, velocities of the conditioning state are treated as isotropic Gaussians.",
    )
    parser.add_argument(
        "--resample-velocities",
        action="store_true",
        help="If specified, velocities for conditioning state are resampled at every MCMC iteration.",
    )
    parser.add_argument(
        "--step-width",
        type=int,
        default=10000,
        help="Step width for MD data.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If specified, additional information will be printed.",
    )
    parser.add_argument(
        "--initial_state_idx",
        type=int,
        default=1,
        help="Index of the initial state",
    )
    # Sampling related arguments.
    parser.add_argument(
        "--sample--initialize-randomly",
        action="store_true",
        help="If specified, random initialization will be used instead of a data-point.",
    )
    parser.add_argument(
        "--sample--num-proposal-steps",
        type=int,
        default=1,
        help="Number of parallel proposal steps to take per sample iteration. All states until the first accepted are added the Markov chain.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        # NOTE: This can be useful information if you want to run the script from a REPL or load it into
        # a jupyter notebook, as this can be copy-pasted and passed to `main`.
        print(f"ARGS: {argv}")

    if args.verbose:
        print(f"Using random seed {args.seed}")

    def step(x_coords, x_velocs, model, batch, num_proposal_steps=1):
        with torch.no_grad():
            y_coords, y_velocs = model.conditional_sample(
                atom_types=batch.atom_types.repeat(x_coords.shape[0], 1).to(device),
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=batch.adj_list,
                edge_batch_idx=batch.edge_batch_idx.to(device),
                masked_elements=batch.masked_elements.repeat(x_coords.shape[0], 1).to(device),
                num_samples=num_proposal_steps,
            )
            y_coords = y_coords.squeeze(0)
            y_velocs = y_velocs.squeeze(0)

        return y_coords, y_velocs

    set_seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        output_name = f"{args.output_dir_basename}--num-samples-{args.num_samples}--initial-state-idx-{args.initial_state_idx}"
        if args.mh:
            output_name += "--mh"
        if args.random_velocities:
            output_name += "--random-velocities"
        if args.resample_velocities:
            output_name += "--resample-velocities"
        if args.sample__initialize_randomly:
            output_name += "--initialize-randomly"
        if args.sample__openmm_on_proposal or args.sample__openmm_on_current:
            output_name += f"--num-openmm-steps-{args.sample__num_openmm_steps}"
            if args.sample__openmm_on_proposal:
                output_name += "--openmmm-on-proposal"
            if args.sample__openmm_on_current:
                output_name += "--openmmm-on-current"

        output_dir = args.savefile.parent / output_name
        print(f"No output directory specified; using {output_dir}")

    # load the model
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )
    print(f"Using device: {device}")

    state_dict = torch.load(args.savefile, map_location=lambda storage, loc: storage)

    config = state_dict["training_config"]
    model = model_constructor(config.model_config)
    loss_computer = load_or_construct_loss(config)
    model = wrap_or_replace_loss(model, loss_computer)
    model.load_state_dict(state_dict["module"])
    model = model.module.to(device)

    protein = args.protein
    print(f"Running sampling for {protein}...")
    state0pdbpath = os.path.join(args.data_dir, f"{protein}-traj-state0.pdb")
    parameters = "T1B-peptides"
    simulation = get_simulation_environment(state0pdbpath, parameters)
    integrator = get_simulation_environment_integrator(parameters)
    system = simulation.system
    platform_properties = (
        dict(CudaDeviceIndex=str(device.index)) if device.type == "cuda" else dict(Threads="1")
    )
    openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
        system,
        integrator,
        platform_name=device.type.upper(),
        platform_properties=platform_properties,
    )

    num_atoms = system.getNumParticles()
    masses = [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_atoms)]
    masses = torch.tensor(masses).to(device)

    # load the dataset
    raw_dataset = RawMolDynDataset(
        data_dir=args.data_dir, step_width=args.step_width, equal_data_spacing=False
    )
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, args.num_samples))
    batch = batches[args.initial_state_idx]

    num_steps = args.num_samples
    num_parallel_steps = args.sample__num_proposal_steps
    threshold = args.energy_threshold

    # chirality
    from timewarp.utils.chirality import (
        find_chirality_centers,
        compute_chirality_sign,
        check_symmetry_change,
    )

    chirality_centers = find_chirality_centers(batch.adj_list, batch.atom_types)
    reference_signs = compute_chirality_sign(batch.atom_coords, chirality_centers)

    y_coords = batch.atom_coords.to(device)
    y_velocs = batch.atom_velocs.to(device)
    trajectory_exploration = []
    energy_exploration = []
    inital_energy = openmm_potential_energy_torch(y_coords)
    energies = inital_energy.repeat(num_parallel_steps, 1)
    y_coords = y_coords.repeat(num_parallel_steps, 1, 1)
    y_velocs = y_velocs.repeat(num_parallel_steps, 1, 1)

    start = timer()

    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}")
        y_coords_new, _ = step(y_coords, y_velocs, model, batch)
        energies_new = openmm_potential_energy_torch(y_coords_new)
        changes = check_symmetry_change(y_coords_new, chirality_centers, reference_signs)
        # reject if chirality changes
        energies_new[changes] += 10000
        y_coords = torch.where(
            (energies_new - energies > threshold).unsqueeze(-1), y_coords, y_coords_new
        )
        energies = torch.where(energies_new - energies > threshold, energies, energies_new)
        trajectory_exploration.append(y_coords)
        energy_exploration.append(energies)

        y_velocs = torch.randn_like(y_coords)

    end = timer()
    duration = end - start
    trajectory_exploration = torch.cat(trajectory_exploration, axis=0)
    energy_exploration = torch.cat(energy_exploration, axis=0)

    sampled_trajectory_path = os.path.join(output_dir, f"{protein}_exploration.npz")

    np.savez(sampled_trajectory_path, positions=trajectory_exploration.cpu().numpy(), time=duration)

    return 0


if __name__ == "__main__":
    # Drop the first `argv` which is just the name of the file.
    main(sys.argv[1:])
