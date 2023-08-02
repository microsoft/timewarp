import argparse
import os
import sys

import numpy as np
from PIL import Image  # type: ignore [import]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.md import get_simulation_environment, sample
from visualise.visualise import visualise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to data directory.")
    parser.add_argument("--protein", type=str, help="Protein name.")
    parser.add_argument(
        "--simulation",
        type=str,
        help="Name of simulation preset, used to set simulation parameters.",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to plot.")
    parser.add_argument(
        "--output_dir", type=str, default="sample_outputs", help="Path to directory to save images."
    )
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load initial data
    state0pdbpath = os.path.join(args.data_dir, f"{args.protein}-traj-state0.pdb")
    npzpath = os.path.join(args.data_dir, f"{args.protein}-traj-arrays.npz")
    data = np.load(npzpath)

    # Get first frame in trajectory
    positions = data["positions"][0, :, :]  # [V, 3]
    velocities = data["velocities"][0, :, :]  # [V, 3]

    sim = get_simulation_environment(state0pdbpath, args.simulation)

    # Get conditional samples from ground truth p(y|x)
    stepwidths = [10, 100, 1000, 10000]

    sampled_pos = np.zeros(
        (
            args.num_samples,
            len(stepwidths),
            positions.shape[0],
            positions.shape[1],
        )
    )  # [S, B, V, 3]

    for i in range(args.num_samples):
        sampled_pos[i, :, :, :], _, _ = sample(
            sim,
            positions,
            velocities,
            stepwidths,
            seed=i,
        )  # [B, V, 3]

    # Visualise samples
    for step_num, stepwidth in enumerate(stepwidths):
        stepwidth_samples = sampled_pos[:, step_num, :, :]  # [S, V, 3]
        sample_filenames = [
            f"{args.protein}_openmm_stepwidth_{stepwidth}_sample_{i}_.png"
            for i in range(args.num_samples)
        ]
        sample_filepaths = [
            os.path.join(args.output_dir, filename) for filename in sample_filenames
        ]

        # Plot the samples
        for samp, sample_filepath in enumerate(sample_filepaths):
            print(f"Plotting sample {samp} for stepwidth {stepwidth}")
            visualise(
                state0filepath=state0pdbpath,
                positions=stepwidth_samples[samp, :, :],  # [V, 3]
                outputpngfilepath=sample_filepath,
            )

        # Collate into GIF
        fp_out = os.path.join(
            args.output_dir, f"{args.protein}_stepwidth_{stepwidth}_openmm_samples.gif"
        )
        img, *imgs = [Image.open(f) for f in sample_filepaths]
        img.save(fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=200, loop=0)


if __name__ == "__main__":
    main()
