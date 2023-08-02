import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualise.visualise import visualise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, help="Path to npz trajectory file.")
    parser.add_argument("--pdb", type=str, help="Path to pdb state0 file.")
    parser.add_argument("--num_frames", type=int, help="Number of frames to plot.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save images.")
    args = parser.parse_args()

    state0_suffix = "-traj-state0.pdb"
    assert args.pdb.endswith(state0_suffix)
    protein_name = os.path.basename(args.pdb)[: -len(state0_suffix)]

    # Make directory
    img_dir = os.path.join(args.output_dir, protein_name)
    os.makedirs(img_dir, exist_ok=True)

    # Plot the trajectory of an item in the trainset
    data = np.load(args.npz)
    for i in range(args.num_frames):
        step = data["step"][i]
        print(f"Plotting step {step}")
        visualise(
            state0filepath=args.pdb,
            positions=data["positions"][i, :, :],
            outputpngfilepath=os.path.join(img_dir, f"{protein_name}_step_{step}.png"),
        )
