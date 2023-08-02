import numpy as np
import mdtraj as md
from numpy import ndarray
import os

import torch

from timewarp.datasets import RawMolDynDataset
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel

# from timewarp.sample import sample
from timewarp.utils.sampling_utils import sample_from_trajectory

from itertools import islice
from typing import Optional
from dataclasses import dataclass

from utilities.common import StrPath


@dataclass
class TorsionAngles:

    # Angles
    phi: np.ndarray
    psi: np.ndarray
    chi1: np.ndarray
    chi2: np.ndarray
    chi3: np.ndarray
    chi4: np.ndarray
    omega: np.ndarray

    # Angle indices
    phi_indices: np.ndarray
    psi_indices: np.ndarray
    chi1_indices: np.ndarray
    chi2_indices: np.ndarray
    chi3_indices: np.ndarray
    chi4_indices: np.ndarray
    omega_indices: np.ndarray


def compute_torsions(coords: ndarray, state0pdbpath: str) -> TorsionAngles:
    """
    Computes all torsion angles for a given trajectory.

    Args:
        coords (ndarray): The positional coordinates with shape [B, S, V, 3],
            where B is the number of initial conditional samples,
            S is the number of sampler per conditional sample,
            and V is the number of atoms in the molecule.
        state0pdbpath (str): Path to the state0.pdb file

    Returns:
        TorsionAngles: Dataclass with seven angle elements, i.e. one for each of the possible 7 torsion angles types.
            As well as one indices array for each angle containing all atom indices corresponding to that angle.
            Each angle element contains a np-array with shape [B, S, n_angles],
            where B is the number of initial samples n_initial_samples,
            S is the number of samples generated from a single conditioning state,
            and n_angles is the number of torsion angles of that type in the molecule.
            The angle indices have shape [n_angles, 4].
    """
    shape = coords.shape
    assert len(shape) == 4, "Shape should be [B, S, V, 3]"

    traj = md.load(state0pdbpath)
    traj.xyz = coords.reshape(-1, shape[2], shape[3])
    phi = md.compute_phi(traj)
    psi = md.compute_psi(traj)
    chi1 = md.compute_chi1(traj)
    chi2 = md.compute_chi2(traj)
    chi3 = md.compute_chi3(traj)
    chi4 = md.compute_chi4(traj)
    omega = md.compute_omega(traj)

    dihedral_angles = [phi, psi, chi1, chi2, chi3, chi4, omega]
    return TorsionAngles(
        *[angle[1].reshape(shape[0], shape[1], -1) for angle in dihedral_angles],
        *[angle[0] for angle in dihedral_angles],
    )  # For each dihedral angle: Length [B, S, n_angles]


def get_all_torsions(
    protein: str, data_dir: StrPath, step_width: int, n_initial_samples: int, targets: bool
) -> TorsionAngles:
    """
    Computes all torsion angles for a given trajectory path.

    Args:
        protein (str): Name of the protein or molecule
        data_dir (str): Path to the data directory
        step_width (int): Step width to use
        n_initial_samples (int): Number of initial (conditioning) samples
        targets (bool): If the conditioning or the target samples of the trajectory should be used.

    Returns:
        list: List with seven elements, i.e. one for each of the possible 7 torsion angles types.
            Each element contains a np-array with shape [B, S, n_angles],
            where B is the number of initial samples n_initial_samples,
            S is the number of samples generated from a single conditioning state,
            and n_angles is the number of torsion angles of that type in the molecule.
    """
    state0pdbpath = os.path.join(data_dir, f"{protein}-traj-state0.pdb")

    raw_dataset = RawMolDynDataset(data_dir=data_dir, step_width=step_width)
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = [moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator]
    batches = list(islice(batches, n_initial_samples))
    if targets:
        traj_coords = [batch.atom_coord_targets.cpu().numpy() for batch in batches]
    else:
        traj_coords = [batch.atom_coords.cpu().numpy() for batch in batches]

    return compute_torsions(np.array(traj_coords), state0pdbpath)


def get_all_torsions_model(
    protein: str,
    data_dir: StrPath,
    step_width: int,
    model: ConditionalDensityModel,
    n_initial_samples: int,
    n_samples_model: int,
    device: Optional[torch.device] = None,
) -> TorsionAngles:
    """
    Computes all torsion angles for a given trajectory path and a model.

    Args:
        protein (str): Name of the protein or molecule
        data_dir (str): Path to the data directory
        step_width (int): Step width to use
        model (ConditionalDensityModel): Model to produce samples.
        n_initial_samples (int): Number of initial (conditioning) samples
        n_samples_model (int): Number of samples per initial sample
        device (Optional[torch.device], optional): Device. Defaults to None.


    Returns:
        list: List with seven elements, i.e. one for each of the possible 7 torsion angles types.
            Each element contains a np-array with shape [B, S, n_angles],
            where B is the number of initial samples n_initial_samples,
            S is the number of samples generated from a single conditioning state,
            and n_angles is the number of torsion angles of that type in the molecule.
    """
    state0pdbpath = os.path.join(data_dir, f"{protein}-traj-state0.pdb")

    raw_dataset = RawMolDynDataset(data_dir=data_dir, step_width=step_width)
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = [moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator]
    batches = list(islice(batches, n_initial_samples))
    y_coords_model, _ = sample_from_trajectory(
        model=model,
        batches=batches,
        num_samples=n_samples_model,
        decorrelated=False,
        device=device,
    )
    return compute_torsions(np.array(y_coords_model), state0pdbpath)
