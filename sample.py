import os
import sys
import argparse
import random
from itertools import islice
from pathlib import Path
from PIL import Image  # type: ignore [import]

import torch
from omegaconf import OmegaConf

from timewarp.losses import unwrap_loss_wrapper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.datasets import RawMolDynDataset
from timewarp.utils.training_utils import load_model
from timewarp.utils.molecule_utils import bond_change_histogram
from timewarp.utils.energy_utils import get_energy_mean_std, plot_all_energy
from timewarp.utils.sampling_utils import sample_from_trajectory

from visualise.visualise import visualise
from utilities.training_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    random.seed(0)
    set_seed(0)

    # Load a model and sample from it.
    parser = argparse.ArgumentParser()
    parser.add_argument("--savefile", type=str, help="Saved model config and state dict.")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.")
    parser.add_argument("--protein", type=str, help="Protein name.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to plot.")
    parser.add_argument(
        "--output_dir", type=str, default="sample_outputs", help="Path to directory to save images."
    )
    parser.add_argument(
        "--independent", action="store_true", help="Sample every atom independently."
    )
    parser.add_argument(
        "--energy-breakdown", action="store_true", help="Breakdown the potential energy by forces."
    )
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    model = unwrap_loss_wrapper(load_model(path=args.savefile).to(device))
    config = torch.load(args.savefile)["training_config"]
    step_width = config.step_width

    # Save the config to the output folder
    with (Path(args.output_dir) / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Load the dataset
    raw_dataset = RawMolDynDataset(data_dir=args.data_dir, step_width=step_width)
    pdb_names = [args.protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, 20))  # Load max 20 timepoints.
    state0pdbpath = os.path.join(args.data_dir, f"{args.protein}-traj-state0.pdb")

    y_coords_model, y_velocs_model = sample_from_trajectory(
        model=model,
        batches=batches,
        num_samples=args.num_samples,
        decorrelated=args.independent,
        device=device,
    )

    """Make a histogram of number of bonds broken/added"""
    print("Plotting bond change histogram...")
    bond_change_histogram(
        state0pdbpath=state0pdbpath,
        data_batches=batches,
        samples=y_coords_model,
        output_dir=args.output_dir,
    )

    """Plot the energies along the trajectory."""
    print("Plotting energies...")
    # Get energies of model samples
    mean_stds_by_names = get_energy_mean_std(
        state0pdbpath=state0pdbpath,
        positions=y_coords_model,
        velocities=y_velocs_model,
        use_integrator_for_KE=False,  # Using integrator leads to inaccurate KE if coord samples are inaccurate.
        energy_breakdown=args.energy_breakdown,
    )

    # Plot model trajectory energies as a function of time
    plot_all_energy(
        mean_stds_by_names,
        output_dir=args.output_dir,
        plot_name=f"{args.protein}_model_trajectory",
    )

    # Get energies of OpenMM trajectory
    traj_coords = [
        batch.atom_coord_targets.cpu().numpy() for batch in batches
    ]  # Length B list of [1, V, 3]
    traj_velocs = [
        batch.atom_veloc_targets.cpu().numpy() for batch in batches
    ]  # Length B list of [1, V, 3]

    traj_mean_stds_by_names = get_energy_mean_std(
        state0pdbpath=state0pdbpath,
        positions=traj_coords,
        velocities=traj_velocs,
        use_integrator_for_KE=True,  # Using integrator is fine for OpenMM trajectory.
        energy_breakdown=args.energy_breakdown,
    )

    # Plot OpenMM trajectory energies as a function of time
    plot_all_energy(
        mean_stds_by_names=traj_mean_stds_by_names,
        output_dir=args.output_dir,
        plot_name=f"{args.protein}_openmm_trajectory",
    )

    """Plot rendered images of individual samples. We only plot images of molecules
    corresponding to the initial and final state of the *first* timepoint in the trajectory,
    in order to avoid generating too many images.
    """
    initial_openmm_datapoint = batches[0]  # MolDynDatapoint
    initial_model_coords_samples = y_coords_model[0]  # [S, V, 3]

    # Plot the initial state of the first timepoint in the trajectory.
    initial_state_path = os.path.join(
        args.output_dir, f"{args.protein}_initial_stepwidth_{step_width}.png"
    )
    visualise(
        state0filepath=state0pdbpath,
        positions=initial_openmm_datapoint.atom_coords[0, :, :].detach().cpu().numpy(),  # [V, 3]
        outputpngfilepath=initial_state_path,
    )

    # Plot the ground truth final state of the first timepoint.
    final_state_path = os.path.join(
        args.output_dir, f"{args.protein}_final_stepwidth_{step_width}.png"
    )
    visualise(
        state0filepath=state0pdbpath,
        positions=initial_openmm_datapoint.atom_coord_targets[0, :, :]
        .detach()
        .cpu()
        .numpy(),  # [V, 3]
        outputpngfilepath=final_state_path,
    )

    # Plot the samples from the model conditioned on the initial state of first timepoint.
    sample_filepaths = []
    for i in range(args.num_samples):
        filepath = os.path.join(
            args.output_dir, f"{args.protein}_sample_{i}_stepwidth_{step_width}.png"
        )
        print(f"Plotting sample {i} to {filepath}")
        visualise(
            state0filepath=state0pdbpath,
            positions=initial_model_coords_samples[i, :, :],  # [V, 3]
            outputpngfilepath=filepath,
        )
        sample_filepaths.append(filepath)

    # Make GIF of initial to final state.
    ground_truth_filepaths = [initial_state_path, final_state_path]
    fp_out = os.path.join(
        args.output_dir, f"{args.protein}_ground_truth_initial_to_final_step_width_{step_width}.gif"
    )
    img, *imgs = [Image.open(f) for f in ground_truth_filepaths]
    img.save(fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0)

    # Make GIF of samples.
    fp_out = os.path.join(
        args.output_dir, f"{args.protein}_conditional_samples_step_width_{step_width}.gif"
    )
    img, *imgs = [Image.open(f) for f in sample_filepaths]
    img.save(fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0)


if __name__ == "__main__":
    main()
