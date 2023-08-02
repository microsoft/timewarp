import torch
from timewarp.dataloader import DenseMolDynBatch
import os
import mdtraj as md

from utilities.common import StrPath
from typing import Dict, Union, Collection, Tuple


KNOWN_ELEMENTS = ["C", "H", "N", "O", "S"]
ELEMENT_VOCAB = {e: i for i, e in enumerate(KNOWN_ELEMENTS)}


def find_chirality_centers(
    adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2
) -> torch.Tensor:
    """
    Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reportet.
            Default is 2, because in this case the mirroring is a simple permutation.

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[0]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2]
        adj_types = atom_types[0][bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list)


def compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """
    Compute indicator signs for a given configuration.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    # print(coords.shape, chirality_centers.shape, chirality_centers)
    direction_vectors = (
        coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    )
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def check_symmetry_change(
    coords: torch.Tensor, chirality_centers: torch.Tensor, reference_signs: torch.Tensor
) -> torch.Tensor:
    """
    Check for a batch if the chirality changed wrt to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers
        reference_signs: List of reference sign for the chirality_centers
    Returns:
        Mask, where changes are True
    """
    perm_sign = compute_chirality_sign(coords, chirality_centers)
    return (perm_sign != reference_signs.to(coords)).any(dim=-1)


class CiralityChecker:
    def __init__(self, pdb_dirs: Union[StrPath, Collection[StrPath]]):
        self._chirality_reference_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = dict()
        self.pdb_dirs: Collection[StrPath] = (
            pdb_dirs
            if isinstance(pdb_dirs, Collection) and not isinstance(pdb_dirs, str)
            else [pdb_dirs]
        )

    def check_changes(
        self, batch: DenseMolDynBatch, coords: torch.Tensor, masked_elements: torch.Tensor
    ) -> torch.Tensor:
        """
        Check for chirality changes in a batch that might consist of different proteins.
        """
        batch_size = coords.shape[0]
        if batch.segments is not None:
            segments = batch.segments
        else:
            segments = torch.arange(batch_size + 1)
        # Make use of contiguous segments to compute for subset of
        # batch containing the same protein.
        # print(batch.segments)
        # print(batch.names, batch.adj_list)
        # print(batch.adj_list[segments[0]])
        chirality_changes = [
            self._check_chirality_single_protein(
                coords[segments[i] : segments[i + 1]],
                batch.names[segments[i]],
                batch.atom_coords[[segments[i]]],
                masked_elements[segments[i] : segments[i + 1]],
            )
            for i in range(len(segments) - 1)
        ]

        return torch.hstack(chirality_changes)  # [B, ]

    def _check_chirality_single_protein(
        self,
        coords: torch.Tensor,
        name: str,
        batch_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check for chirality changes for a single protein and add its reference to the cache if not already there.
        """
        batch_size = coords.shape[0]
        if name not in self._chirality_reference_cache:
            self.add_protein_to_cache(name, batch_coords[~mask[[0]], :].view(1, -1, 3))
        chirality_centers, reference_signs = self._chirality_reference_cache[name]

        return check_symmetry_change(
            coords[~mask, :].view(batch_size, -1, 3), chirality_centers, reference_signs
        )

    def add_protein_to_cache(
        self,
        name: str,
        batch_coords: torch.Tensor,
    ) -> None:
        """
        Add new reference sign to the cache for the given protein.
        Use here the state0pdbpath instead of the adj_list stored in the batch
        as this was not loading correctly.
        """
        state0pdbpath = None
        for pdb_dir in self.pdb_dirs:
            for (dirpath, _, _) in os.walk(str(pdb_dir)):
                # Faster to just check if the target file exists than iterating through all the files
                # in the directory to find a match.
                state0pdbpath_maybe = os.path.join(dirpath, f"{name}-traj-state0.pdb")
                if os.path.isfile(state0pdbpath_maybe):
                    state0pdbpath = state0pdbpath_maybe
                    break

        if state0pdbpath is None:
            raise ValueError(
                f"could not find PDB file for {name} in any of the provided paths {self.pdb_dirs}"
            )

        traj = md.load(state0pdbpath)
        topology = traj.topology
        atom_types = torch.tensor([[ELEMENT_VOCAB[a.element.symbol] for a in topology.atoms]])
        adj_list = torch.tensor([(b.atom1.index, b.atom2.index) for b in topology.bonds])

        chirality_centers = find_chirality_centers(adj_list, atom_types)
        reference_signs = compute_chirality_sign(batch_coords, chirality_centers)
        self._chirality_reference_cache[name] = (chirality_centers, reference_signs)
