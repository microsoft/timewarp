from pathlib import Path
import torch
import numpy as np
import openmm.unit as u

import os
import sys
import argparse
from itertools import islice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from timewarp.utils.training_utils import load_model

from timewarp.datasets import RawMolDynDataset
from timewarp.utils.chirality import find_chirality_centers, compute_chirality_sign

from simulation.md import get_simulation_environment_integrator, get_simulation_environment
from timewarp.dataloader import (
    moldyn_dense_collate_fn,
)
from timewarp.utils.evaluation_utils import (
    sample_with_model,
)
from utilities.training_utils import set_seed
from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
from timewarp.model_constructor import model_constructor
from timewarp.utils.training_utils import load_or_construct_loss
from timewarp.losses import (
    wrap_or_replace_loss,
)
from timeit import default_timer as timer


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
        "--sample--openmm-on-current",
        action="store_true",
        help="If specified, OpenMM steps will be performed on the current state in every MCMC iteration",
    )
    parser.add_argument(
        "--sample--openmm-on-proposal",
        action="store_true",
        help="If specified, OpenMM steps will be performed on the proposed state in every MCMC iteration",
    )
    parser.add_argument(
        "--sample--num-openmm-steps",
        type=int,
        default=1,
        help="Number of OpenMM steps to take per sample iteration. Has no effect if OpenMM is not activated for sampling (see other flags).",
    )
    parser.add_argument(
        "--sample--num-proposal-steps",
        type=int,
        default=1,
        help="Number of parallel proposal steps to take per sample iteration. All states until the first accepted are added the Markov chain.",
    )
    parser.add_argument(
        "--sample--adaptive-parallelism",
        action="store_true",
        help="If specified, `num_proposal_steps` will be set adaptively depending on a smoothed average of acceptance rate, with `--sample-num-proposal-steps` now determinining the _maximum_ number of allowed steps per iteration.",
    )
    parser.add_argument(
        "--step-width",
        type=int,
        default=10000,
        help="Step width for MD data.",
    )
    parser.add_argument(
        "--conserve-chirality",
        action="store_true",
        help="If specified, step that change the chirality are rejected.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        # NOTE: This can be useful information if you want to run the script from a REPL or load it into
        # a jupyter notebook, as this can be copy-pasted and passed to `main`.
        print(f"ARGS: {argv}")

    if args.verbose:
        print(f"Using random seed {args.seed}")

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

    needs_sim = args.sample__openmm_on_proposal or args.sample__openmm_on_current

    num_iters = args.num_samples // args.saving_interval
    assert num_iters > 0, "num_samples must be larger than saving_interval."
    # traj_coords = []
    # traj_velocs = []
    # traj_chain_stats = []

    # sampled_stats_path = os.path.join(output_dir, f"{protein}_chain_stats.pkl")

    if args.conserve_chirality:
        # find potential chirality centers
        chirality_centers = find_chirality_centers(batch.adj_list, batch.atom_types)
        x_coords = batch.atom_coords
        reference_signs = compute_chirality_sign(x_coords, chirality_centers)

    # Check if chain is already started
    try:
        n_saved_iterations = len(os.listdir(output_dir))
        npz = np.load(output_dir + f"/{protein}_trajectory_model_{n_saved_iterations-1}.npz")
        batch.atom_coords = torch.from_numpy(npz["positions"][-1:])
        print("Resuming sampling")
    except FileNotFoundError:
        n_saved_iterations = 0

    for i in range(n_saved_iterations, num_iters):
        print(f"Iteration {i+1}/{num_iters}")
        start = timer()
        sampled_coords, _, _, _ = sample_with_model(
            batch,
            model,
            device,
            openmm_potential_energy_torch,
            masses,
            args.saving_interval,
            args.mh,
            random_velocs=args.random_velocities,
            resample_velocs=args.resample_velocities,
            initialize_randomly=args.sample__initialize_randomly,
            sim=simulation if needs_sim else None,
            openmm_on_current=args.sample__openmm_on_current,
            openmm_on_proposal=args.sample__openmm_on_proposal,
            num_openmm_steps=args.sample__num_openmm_steps,
            num_proposal_steps=args.sample__num_proposal_steps,
            adaptive_parallelism=args.sample__adaptive_parallelism,
            reference_signs=reference_signs if args.conserve_chirality else None,
            chirality_centers=chirality_centers if args.conserve_chirality else None,
            disable_tqdm=True,
        )
        end = timer()
        duration = end - start

        sampled_trajectory_path = os.path.join(output_dir, f"{protein}_trajectory_model_{i}.npz")

        print(f"Saving trajectory to {sampled_trajectory_path}")
        np.savez(
            sampled_trajectory_path,
            positions=sampled_coords[::10],
            # velocities=sampled_velocs[::10],  # velocities usually not needed
            time=duration,
        )
        batch.atom_coords = torch.from_numpy(sampled_coords[-1:])

    return 0


if __name__ == "__main__":
    # Drop the first `argv` which is just the name of the file.
    main(sys.argv[1:])
