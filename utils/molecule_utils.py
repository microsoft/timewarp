from typing import Optional, List
import os
import tempfile

import torch
import numpy as np
import matplotlib.pyplot as plt
import pymol2  # type: ignore [import]
from tqdm import tqdm

from timewarp.dataloader import DenseMolDynBatch
from visualise.visualise import writepdb


def get_centre_of_mass(
    coords: torch.Tensor,  # [batch_size, n_points, 3]
    masked_elements: Optional[torch.BoolTensor] = None,  # [batch_size, n_points]
):
    if masked_elements is not None:
        # Get inverse mask (with 1. where the elements are not masked, 0. where they are)
        inv_mask = ~masked_elements  # [B, V]
        # Set masked coordinates to zero
        coords = inv_mask.unsqueeze(-1) * coords  # [B, V, 3]
        # Get the number of elements per batch element
        num_points = inv_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]
    else:
        num_points = coords.shape[-2]  # type: ignore
    centre_of_mass = coords.sum(dim=-2, keepdim=True) / num_points
    return centre_of_mass  # [batch_size, 1, 3]


def get_bonds_from_positions(state0filepath: str, positions: np.ndarray):
    """Get a list of guessed bonds from the 3D atom positions.

    Parameters
    ----------
    state0filepath : string
        The path to the state0.pdb file specifying the topology.
    positions : np.array
        Positions, shape should be (natoms, 3).
    """
    p = pymol2.PyMOL()
    p.start()

    with tempfile.NamedTemporaryFile(prefix="bondframe", suffix=".pdb") as tmpfile:
        pdbpath = tmpfile.name
        writepdb(state0filepath, positions, pdbpath)

        p.cmd.reinitialize()
        p.cmd.load(pdbpath, "pdb")
        bonds = p.cmd.get_bonds()

    p.stop()
    return bonds


def count_changed_bonds(
    state0filepath: str, initial_positions: np.ndarray, final_positions: np.ndarray
):
    """
    Args:
        state0filepath (str): The path to the state0.pdb file specifying the initial topology.
        initial_positions (np.array): [V, 3]
        final_positions (np.array): [V, 3]

    Returns:
        [int]: Number of broken bonds
        [int]: Number of added bonds
        [float]: Intersection over union between initial and final bonds
    """

    init_bonds = set(get_bonds_from_positions(state0filepath, initial_positions))
    final_bonds = set(get_bonds_from_positions(state0filepath, final_positions))

    broken_bonds = init_bonds - final_bonds
    added_bonds = final_bonds - init_bonds

    iou = len(init_bonds.intersection(final_bonds)) / len(init_bonds.union(final_bonds))

    return len(broken_bonds), len(added_bonds), iou


def bond_change_histogram(
    state0pdbpath: str,
    data_batches: List[DenseMolDynBatch],
    samples: List[np.ndarray],
    output_dir: os.PathLike,
):
    """Make a histogram of number of bonds broken/added. This histogram is computed over both the number
    of frames (i.e., the number of initial conditions that we feed into the model), and also the number of
    samples per frame. Hence there are a total of (num_samples * num_frames) datapoints in the histogram.
    Args:
        state0pdbpath (str): The path to the state0.pdb file specifying the initial topology.
        data_batches (List[DenseMolDynBatch]): List of `frame_num` batches of data for the initial states.
        samples (List[np.array]): List of `frame_num` numpy arrays of shape [S, V, 3]. It is assumed that
            `samples` and `data_batches` come from the same molecule, and that they both have the same length,
            which is the number of frames which are used as initial conditions.
        output_dir (os.PathLike): Directory to save histogram.
    """
    assert len(data_batches) == len(samples)

    num_broken = []
    num_added = []
    iou_list = []

    num_bonds = len(
        get_bonds_from_positions(
            state0filepath=state0pdbpath, positions=data_batches[0].atom_coords[0, :, :].numpy()
        )
    )

    num_samples = samples[0].shape[0]
    for frame_num in tqdm(
        range(len(data_batches)),
        desc="Computing added / broken bond statistics",
        unit="initial state",
    ):
        for samp in range(num_samples):
            num_broken_bonds, num_added_bonds, iou = count_changed_bonds(
                state0filepath=state0pdbpath,
                initial_positions=data_batches[frame_num].atom_coords[0, :, :].numpy(),  # [V, 3]
                final_positions=samples[frame_num][samp, :, :],  # [V, 3]
            )
            num_broken.append(num_broken_bonds)
            num_added.append(num_added_bonds)
            iou_list.append(iou)

    # Make a histogram plot
    plt.hist(num_broken)
    plt.title(f"Broken bonds over all samples in trajectory. Initial bonds: {num_bonds}")
    plt.xlabel("Broken bonds")
    plt.ylabel("Samples")
    plt.savefig(os.path.join(output_dir, "broken_bonds.png"))
    plt.close()

    plt.hist(num_added)
    plt.title(f"Added bonds over all samples in trajectory. Initial bonds: {num_bonds}")
    plt.xlabel("Added bonds")
    plt.ylabel("Samples")
    plt.savefig(os.path.join(output_dir, "added_bonds.png"))
    plt.close()

    plt.hist(iou_list)
    plt.title(f"IoU between initial and final bonds. Initial bonds: {num_bonds}")
    plt.xlabel("Intersection over union")
    plt.ylabel("Samples")
    plt.savefig(os.path.join(output_dir, "iou.png"))
    plt.close()
