import random
import glob
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import openmm.unit as u

import warnings
import matplotlib.pyplot as plt
import os
import sys
import argparse
from itertools import islice

import scipy.stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from timewarp.utils.training_utils import load_model

from timewarp.losses import unwrap_loss_wrapper, LossWrapper
from timewarp.datasets import RawMolDynDataset
from simulation.md import get_simulation_environment_integrator, get_simulation_environment
from timewarp.dataloader import (
    moldyn_dense_collate_fn,
)
from timewarp.utils.evaluation_utils import (
    sample_on_batches,
    sample_with_model,
    sample_on_single_conditional,
    plot_marginal_distribution,
    compute_internal_coordinates,
    plot_bonds,
    plot_ramachandran,
    visualize,
    plot_transitions,
    compute_kinetic_energy,
    plot_energy,
)
from utilities.training_utils import set_seed
from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
from timewarp.utils.config_utils import load_config_in_subdir
from timewarp.utils.training_utils import model_constructor
from timewarp.utils.openmm.openmm_bridge import device_to_platform_and_properties
from utilities.model_utils import load_model_state_dict
from utilities.common import StrPath, glob_only


def load_model_and_config(
    model_path: StrPath,
    config_path: Optional[StrPath] = None,
    device: Optional[torch.device] = None,
):
    """Load model and config from a model path.

    Args:
        model_path (StrPath): Path to the model.
        config_path (StrPath, optional): Path to the config. Defaults to None.
        device (torch.device, optional): Device to load the model to. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, dict]: Model and config.

    Notes:
        If `config_path` is not specified, the config in the in checkpoint at `model_path` will be used.
        `config_path` is therefore only useful if for some reason the config in the checkpoint is
        incompatible with the model. This should not happen, but if it does, you can use this argument.
    """
    model_path = Path(model_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_path is not None:
        # Load config from provided path instead of from saved checkpoint.
        print(f"Loading TrainingConfig from {config_path} instead of from checkpoint")
        config = load_config_in_subdir(config_path)
        model = LossWrapper(module=model_constructor(config.model_config), loss=None)
        model.load_state_dict(load_model_state_dict(model_path))
        model = model.to(device)
    else:
        # If `model_path` is a directory, attempt to load checkpoint from any of its decendants.
        # Allows to naturally support both runs using `deepspeed` and not.
        model_path = (
            model_path
            if os.path.isfile(model_path)
            else glob_only(os.path.join(model_path, "**", "*.pt"))
        )
        print(f"Loading TrainingConfig from checkpoint at {model_path}")
        config = torch.load(model_path)["training_config"]
        model = load_model(path=model_path).to(device)

    return model, config


def list_available_proteins(pdb_dir):
    protein_filenames = glob.glob(
        os.path.join(str(pdb_dir), "**/*-traj-state0.pdb"), recursive=True
    )
    num_chars_to_drop = len("-traj-state0.pdb")
    return sorted([os.path.split(f)[-1][:-num_chars_to_drop] for f in protein_filenames])


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("savefile", help="Saved model config and state dict.")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.", required=True)
    parser.add_argument("--protein", type=str, help="Protein name.", nargs="*")
    parser.add_argument("--config", help="Overrides the config present in `savefile`.")
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
        "--output-dir",
        type=str,
        help="Path to directory to save images.",
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
        "--no-sample",
        action="store_true",
        help="If specified, sampling with the model is not performed.",
    )
    parser.add_argument(
        "--no-conditional",
        action="store_true",
        help="If specified, evaluation of the conditional distribution is not performed.",
    )
    parser.add_argument(
        "--not-on-data",
        action="store_true",
    )
    parser.add_argument(
        "--color_model",
        type=str,
        default="orange",
        help="Color for model",
    )
    parser.add_argument(
        "--color_openMM",
        type=str,
        default="green",
        help="Color for openMM",
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
        default=0,
        help="Index of the initial state",
    )
    parser.add_argument(
        "--shuffle-proteins",
        action="store_true",
        help="If specified, the proteins are evaluted in a random order rather than in alphabetical order.",
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
        help="Number of parallel proposal steps to take per sample iteration. All states until the first accepted are added the Markov chain. Note that its behavior changes if `--sample--adaptive-parallelism` is specified; see below.",
    )
    parser.add_argument(
        "--sample--adaptive-parallelism",
        action="store_true",
        help="If specified, `num_proposal_steps` will be set adaptively depending on a smoothed average of acceptance rate, with `--sample-num-proposal-steps` now determinining the _maximum_ number of allowed steps per iteration.",
    )
    parser.add_argument(
        "--sample--ylims", help="Specifies ylims for the visaulizations of the chain stats."
    )
    # General visualization arguments.
    parser.add_argument(
        "--font-size", type=int, default=30, help="Font size used in the visualizations."
    )
    args = parser.parse_args(argv)

    if args.verbose:
        # NOTE: This can be useful information if you want to run the script from a REPL or load it into
        # a jupyter notebook, as this can be copy-pasted and passed to `main`.
        print(f"ARGS: {argv}")

    plt.rc("font", size=args.font_size)

    if args.verbose:
        print(f"Using random seed {args.seed}")

    set_seed(args.seed)

    savefile = Path(glob_only(args.savefile))

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

        output_dir = savefile.parent / output_name
        print(f"No output directory specified; using {output_dir}")

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # load the model
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )
    print(f"Using device: {device}")
    model, config = load_model_and_config(
        model_path=savefile, config_path=args.config, device=device
    )
    model = unwrap_loss_wrapper(model)

    step_width = config.step_width

    # Get the available proteins.
    proteins = args.protein if args.protein is not None else list_available_proteins(args.data_dir)
    if args.shuffle_proteins:
        random.shuffle(proteins)

    # Buffer for keeping track of the average acceptance rates for each protein.
    average_acceptance_rate_all_proteins = []
    for protein in proteins:
        print(f"Running evaluation for {protein}...")
        state0pdbpath = os.path.join(args.data_dir, f"{protein}-traj-state0.pdb")
        parameters = "alanine-dipeptide"
        simulation = get_simulation_environment(state0pdbpath, parameters)
        integrator = get_simulation_environment_integrator(parameters)
        system = simulation.system
        platform, platform_properties = device_to_platform_and_properties(device, num_threads=1)
        openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
            system,
            integrator,
            platform_name=platform,
            platform_properties=platform_properties,
        )

        num_atoms = system.getNumParticles()
        masses = [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_atoms)]
        masses = torch.tensor(masses).to(device)

        # load the dataset
        raw_dataset = RawMolDynDataset(data_dir=args.data_dir, step_width=step_width)
        pdb_names = [protein]
        raw_iterator = raw_dataset.make_iterator(pdb_names)
        batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
        batches = list(islice(batches, args.num_samples))
        batch = batches[args.initial_state_idx]

        if args.num_samples > len(batches):
            warnings.warn(
                (
                    f"The data set consists of only {len(batches)} datapoints."
                    f"Hence, some evaluation functions will only use {len(batches)}."
                )
            )

        # Some atom names for plotting
        atom_types = batches[0].atom_types.cpu().detach().numpy()[0]
        adj_list = batches[0].adj_list.numpy()

        # Stuff from data.
        print("Plotting for OpenMM...")
        coords_openmm = np.array(
            [batch.atom_coord_targets.cpu().numpy() for batch in batches]
        ).squeeze(1)

        # Ramachandran plot
        bonds_openmm, torsions_openmm = compute_internal_coordinates(
            state0pdbpath, adj_list, coords_openmm
        )
        plot_ramachandran(
            torsions=torsions_openmm,
            name="sample_on_data",
            title="openMM",
            output_dir=output_dir,
            step_width=step_width,
            protein=protein,
        )

        # Energies
        energies_openmm = (
            openmm_potential_energy_torch(torch.from_numpy(coords_openmm)).cpu().numpy()
        )

        # sample on conditioning data from the dataset
        if not args.not_on_data:
            print("Sample on conditioning data from the dataset")

            (
                y_coords_model,
                y_velocs_model,
                traj_coords,
                traj_velocs,
                traj_coords_conditioning,
                _,
                ll_reverse,
                ll_forward,
                ll_reverse_training,
                ll_forward_training,
                acceptance,
            ) = sample_on_batches(
                batches,
                model,
                device,
                openmm_potential_energy_torch,
                config.data_augmentation,
                masses,
                random_velocs=args.random_velocities,
            )
            delta_x = traj_coords - traj_coords_conditioning
            delta_x_model = y_coords_model - traj_coords_conditioning
            if args.random_velocities:
                traj_velocs = np.random.randn(*traj_velocs.shape)
            print(f"Mean acceptance for model samples conditioned on data-set: {acceptance.mean()}")
            print("Plotting marginal distribution...")
            plot_marginal_distribution(
                x_model=delta_x_model,
                v_model=y_velocs_model,
                x_traj=delta_x,
                v_traj=traj_velocs,
                color_model=args.color_model,
                color_traj=args.color_openMM,
                output_dir=output_dir,
                protein=protein,
                name="sample_on_data",
                step_width=step_width,
                atom_types=atom_types,
            )
            print("Plotting correlations...")

            # Correlations
            cor_traj = np.corrcoef(delta_x.reshape(len(batches), -1)[:, ::3].T)
            cor_model = np.corrcoef(delta_x_model.reshape(len(batches), -1)[:, ::3].T)

            # openMM
            figure = plt.figure(figsize=(16, 9))
            axes = figure.add_subplot(111)
            caxes = axes.matshow(cor_traj, interpolation="nearest")
            axes.set_xticks([])
            axes.set_yticks([])
            plt.title("Sample on data: Correlations delta x openMM")
            figure.colorbar(caxes)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_data_correlation_openmm_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            # model
            figure = plt.figure(figsize=(16, 9))
            axes = figure.add_subplot(111)
            caxes = axes.matshow(cor_model, interpolation="nearest")
            axes.set_xticks([])
            axes.set_yticks([])
            plt.title("Sample on data: Correlations delta x model")
            figure.colorbar(caxes)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_data_correlation_model_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            # Difference
            figure = plt.figure(figsize=(16, 9))
            axes = figure.add_subplot(111)
            caxes = axes.matshow(np.abs(cor_traj - cor_model), interpolation="nearest")
            plt.title("Sample on data: Correlation difference delta x")
            figure.colorbar(caxes)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_data_correlation_difference_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            print("Plotting likelihoods...")

            # Likelihood difference samples
            plt.figure(figsize=(16, 9))
            plt.hist((ll_forward - ll_reverse), bins=100, label="Model samples")
            plt.xlabel("Sample on data: Log-likelihood difference")
            plt.title("Log-likelihood difference distribution for model samples")
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_data_likelihood_difference_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            # Likelihoods
            plt.figure(figsize=(16, 9))
            plt.hist((ll_forward), alpha=0.5, density=True, bins=100, label="Model samples")
            plt.hist(
                (ll_forward_training), alpha=0.5, density=True, bins=100, label="Training samples"
            )
            plt.hist(
                (ll_reverse_training),
                alpha=0.5,
                density=True,
                bins=100,
                label="Reverse training samples",
            )
            plt.xlabel("Log-likelihood")
            plt.title("Sample on data: Log-likelihood distributions")
            plt.legend()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_data_likelihood_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            # Internal coordinates
            print("Plotting internal coordinates...")

            bonds_model, torsions_model = compute_internal_coordinates(
                state0pdbpath, adj_list, y_coords_model
            )
            bonds_traj, torsions_traj = compute_internal_coordinates(
                state0pdbpath, adj_list, traj_coords
            )
            plot_bonds(
                bonds_model=bonds_model,
                bonds_traj=bonds_traj,
                color_model=args.color_model,
                color_traj=args.color_openMM,
                output_dir=output_dir,
                protein=protein,
                name="sample_on_data",
                step_width=step_width,
                atom_types=atom_types,
                adj_list=adj_list,
            )

            # Ramachandran plot

            plot_ramachandran(
                torsions=torsions_model,
                name="sample_on_data",
                title="model",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
            )
            plot_ramachandran(
                torsions=torsions_traj,
                name="sample_on_data",
                title="openMM",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
            )

            # Energies
            print("Plotting energies...")
            energies_model = (
                openmm_potential_energy_torch(torch.from_numpy(y_coords_model)).cpu().numpy()
            )
            energies_traj = (
                openmm_potential_energy_torch(torch.from_numpy(traj_coords)).cpu().numpy()
            )
            kin_energies = compute_kinetic_energy(
                y_velocs_model, masses.cpu().numpy(), random_velocs=args.random_velocities
            )
            kin_energies_traj = compute_kinetic_energy(
                traj_velocs, masses.cpu().numpy(), random_velocs=args.random_velocities
            )

            plot_energy(
                energies_model=energies_model,
                energies_traj=energies_traj,
                name="sample_on_data",
                title="potential",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
            )
            plot_energy(
                energies_model=kin_energies,
                energies_traj=kin_energies_traj,
                name="sample_on_data",
                title="kinetic",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
            )

            if not args.no_conditional:
                # Conditional distribution
                if args.random_velocities:
                    print(
                        "Sample conditional distribution with the model and openMM with random velocities."
                    )
                else:
                    print("Sample conditional distribution with the model and openMM")

                (
                    y_coords_model_conditional,
                    y_velocs_model_conditional,
                    traj_coords_conditional,
                    traj_velocs_conditional,
                    traj_coords_conditioning_conditional,
                ) = sample_on_single_conditional(
                    batch,
                    model,
                    args.num_samples,
                    simulation,
                    step_width,
                    args.random_velocities,
                    device,
                )
                delta_x_conditional = traj_coords_conditional - traj_coords_conditioning_conditional
                delta_x_model_conditional = (
                    y_coords_model_conditional - traj_coords_conditioning_conditional
                )
                if args.random_velocities:
                    traj_velocs_conditional = np.random.randn(*traj_velocs_conditional.shape)
                print("Plotting marginal distributions...")
                plot_marginal_distribution(
                    x_model=delta_x_model_conditional,
                    v_model=y_velocs_model_conditional,
                    x_traj=delta_x_conditional,
                    v_traj=traj_velocs_conditional,
                    color_model=args.color_model,
                    color_traj=args.color_openMM,
                    output_dir=output_dir,
                    protein=protein,
                    name="sample_on_conditioning",
                    step_width=step_width,
                    atom_types=atom_types,
                    x_bg=delta_x,
                    v_bg=traj_velocs,
                )
                # Internal coordinate for conditional distribution
                print("Plotting internal coordinates...")

                bonds_model_conditional, _ = compute_internal_coordinates(
                    state0pdbpath, adj_list, y_coords_model_conditional
                )
                bonds_traj_conditional, _ = compute_internal_coordinates(
                    state0pdbpath, adj_list, traj_coords_conditional
                )
                (bonds_conditioning_conditional, _,) = compute_internal_coordinates(
                    state0pdbpath, adj_list, traj_coords_conditioning_conditional
                )

                plot_bonds(
                    bonds_model=bonds_model_conditional,
                    bonds_traj=bonds_traj_conditional,
                    color_model=args.color_model,
                    color_traj=args.color_openMM,
                    output_dir=output_dir,
                    protein=protein,
                    name="sample_on_conditioning",
                    step_width=step_width,
                    atom_types=atom_types,
                    adj_list=adj_list,
                    bonds_bg=bonds_traj,
                    bonds_intial=bonds_conditioning_conditional,
                )

                print("Plotting energies...")

                # Conditional energies
                energies_model = (
                    openmm_potential_energy_torch(torch.from_numpy(y_coords_model_conditional))
                    .cpu()
                    .numpy()
                )
                energies_traj = (
                    openmm_potential_energy_torch(torch.from_numpy(traj_coords_conditional))
                    .cpu()
                    .numpy()
                )
                kin_energies = compute_kinetic_energy(
                    y_velocs_model_conditional,
                    masses.cpu().numpy(),
                    random_velocs=args.random_velocities,
                )
                kin_energies_traj = compute_kinetic_energy(
                    traj_velocs_conditional,
                    masses.cpu().numpy(),
                    random_velocs=args.random_velocities,
                )

                plot_energy(
                    energies_model=energies_model,
                    energies_traj=energies_traj,
                    name="sample_on_conditioning",
                    title="potential",
                    output_dir=output_dir,
                    step_width=step_width,
                    protein=protein,
                )
                plot_energy(
                    energies_model=kin_energies,
                    energies_traj=kin_energies_traj,
                    name="sample_on_conditioning",
                    title="kinetic",
                    output_dir=output_dir,
                    step_width=step_width,
                    protein=protein,
                )

        if not args.no_sample:
            needs_sim = args.sample__openmm_on_proposal or args.sample__openmm_on_current
            sampled_coords, _, _, chain_stats = sample_with_model(
                batch,
                model,
                device,
                openmm_potential_energy_torch,
                masses,
                args.num_samples,
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
            )
            print(
                f"Acceptance for model samples conditioned on model samples: {chain_stats.acceptance.mean()}"
            )
            average_acceptance_rate_all_proteins.append(chain_stats.acceptance.mean())

            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            qs_labels = " / ".join([f"{100 * q}%" for q in qs])

            delta_pot_energy_quantiles = np.quantile(np.exp(-chain_stats.energies_pot_delta), qs)
            print(
                qs_labels
                + " quantiles for exp(-ΔEₚ): "
                + " / ".join(map(str, delta_pot_energy_quantiles))
            )

            delta_kin_energy_quantiles = np.quantile(np.exp(-chain_stats.energies_kin_delta), qs)
            print(
                qs_labels
                + " quantiles for exp(-ΔEₖ): "
                + " / ".join(map(str, delta_kin_energy_quantiles))
            )

            if args.random_velocities:
                # Kinetic energies are `norm(v) / 2` so we multiply by `2` to get the `norm(v)` again.
                ks = scipy_stats.ks_1samp(
                    2 * chain_stats.energies_kin, scipy_stats.chi2(df=num_atoms * 3).cdf
                )
                print(f"Kolmogorov-Smirnov test for velocities: {ks}")

                # Visualize differences in empirical distribution and target.
                # TODO : Determine xlims using CDF of `chi2` and compute actual
                # density rather than histogram.
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.hist(
                    scipy_stats.chi2(df=num_atoms * 3).rvs(10000),
                    bins=100,
                    label="Target",
                    density=True,
                    alpha=0.7,
                )
                ax.hist(
                    2 * chain_stats.energies_kin, bins=100, label="Model", density=True, alpha=0.7
                )
                plt.legend()
                random_velocity_target_path = (
                    Path(output_dir) / f"{protein}_random-velocities-target.png"
                )
                print(f"Saving histogram of velocity norms to {random_velocity_target_path}")
                plt.savefig(random_velocity_target_path)

            sampled_stats_path = os.path.join(output_dir, f"{protein}_chain_stats.pkl")
            print(f"Saving chain stats to {sampled_stats_path}")
            chain_stats.save(sampled_stats_path)

            if args.sample__ylims is not None:
                ylims = eval(args.sample__ylims)
            else:
                ylims = None

            sampled_stats_viz_path = os.path.join(output_dir, f"{protein}_chain_stats.png")
            print(f"Saving visualization of chain stats to {sampled_stats_viz_path}")
            visualize(
                chain_stats, include_transition_probs=True, include_current_energy=True, ylims=ylims
            )
            plt.savefig(sampled_stats_viz_path, bbox_inches="tight")

            sampled_stats_viz_path = os.path.join(output_dir, f"{protein}_chain_stats_delta.png")
            print(f"Saving visualization of delta chain stats to {sampled_stats_viz_path}")
            visualize(
                chain_stats,
                include_transition_probs=True,
                include_current_energy=True,
                delta=True,
                ylims=ylims,
            )
            plt.savefig(sampled_stats_viz_path, bbox_inches="tight")

            energies_pot = openmm_potential_energy_torch(torch.from_numpy(sampled_coords))

            # potential energy distribution
            plt.figure(figsize=(16, 9))

            if isinstance(energies_openmm, torch.Tensor):
                range_lb = energies_openmm.detach().cpu().numpy().min()
            else:
                range_lb = energies_openmm.min()

            plt.hist(
                energies_pot.cpu().flatten().numpy(),
                bins=100,
                color=args.color_model,
                alpha=0.5,
                density=True,
                label="model",
                range=(range_lb, 100),
            )

            # Add OpenMM for comparison.
            plt.hist(
                energies_openmm,
                bins=100,
                color=args.color_openMM,
                alpha=0.5,
                density=True,
                label="openMM",
            )

            plt.xlabel("Energy in kJ/mol")
            plt.legend()
            plt.title("Sample on model samples: Model samples potential energy distribution")
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{protein}_sample_on_samples_potential_energy_distribution_step_width_{step_width}.png",
                ),
                bbox_inches="tight",
            )

            # Internal coordinates
            print("Plotting internal coordinates...")

            bonds_model_sampled, torsions_model_sampled = compute_internal_coordinates(
                state0pdbpath, adj_list, sampled_coords
            )

            plot_bonds(
                bonds_model=bonds_model_sampled,
                bonds_traj=bonds_openmm,
                color_model=args.color_model,
                color_traj=args.color_openMM,
                output_dir=output_dir,
                protein=protein,
                name="sample_on_samples",
                step_width=step_width,
                atom_types=atom_types,
                adj_list=adj_list,
            )

            # Ramachandran plot

            plot_ramachandran(
                torsions=torsions_model_sampled,
                name="sample_on_samples",
                title="model samples",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
                show_initial_state=True,
            )

            plot_transitions(
                torsions=torsions_model_sampled,
                name="sample_on_samples",
                title="model samples",
                output_dir=output_dir,
                step_width=step_width,
                protein=protein,
            )

            if len(average_acceptance_rate_all_proteins) > 0:
                print(
                    f"Average acceptance rate over all: {np.array(average_acceptance_rate_all_proteins).mean()}"
                )
                # TODO: Improve this plot.
                plt.figure(figsize=(16, 9))
                plt.hist(average_acceptance_rate_all_proteins)
                plt.savefig(os.path.join(output_dir, "acceptance_rate.png"))

            plt.close("all")


if __name__ == "__main__":
    # Drop the first `argv` which is just the name of the file.
    main(sys.argv[1:])
