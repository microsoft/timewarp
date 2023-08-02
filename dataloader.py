"""
Sub-module implementing the logic for loading molecular dynamics simulation data.

It allows for batching in two different ways:
 * the sparse representation suitable
for use with Graph-Neural Network (GNN)-like models, where all the elements in the
batch are collated into one big graph.
 * the dense representation suitable for us with transformer-like models. Here, the
examples of different sizes within a batch are padded to the maximum number of atoms
for any given example in the batch, and the data is stored in a dense
[batch_size, max.num.atoms ...] shaped tensor.

"""
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple, List, Iterable, Sequence, Union

import mdtraj as md  # type: ignore
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import warnings

KNOWN_ELEMENTS = ["C", "H", "N", "O", "S"]
ELEMENT_VOCAB = {e: i for i, e in enumerate(KNOWN_ELEMENTS)}


def contiguous_segments(names: Sequence[str]) -> List[int]:
    """
    Compute longest contiguous `segments` of `names`. That is, compute indices such that
    `names[segments[i] : segments[i + 1]]` only contains a single name.
    """
    results = [0]

    for (i, name) in enumerate(names):
        if name != names[results[-1]]:
            results.append(i)

    # Add endpoint so it's more natural to iterate over.
    results.append(len(names))

    return results


@dataclass
class TrajectoryInformation:
    name: str
    node_types: np.ndarray  # int32 tensor of shape [V]
    adj_list: np.ndarray  # int32 tensor of shape [E, 2]
    coord_features: List[np.ndarray]  # T float32 tensors of shape [V, 3]
    veloc_features: List[np.ndarray]  # T float32 tensors of shape [V, 3]
    force_features: List[np.ndarray]  # T float32 tensors of shape [V, 3]
    coord_targets: List[np.ndarray]  # T float32 tensors of shape [V, 3]
    veloc_targets: List[np.ndarray]  # T float32 tensors of shape [V, 3]
    force_targets: List[np.ndarray]  # T float32 tensors of shape [V, 3]


@dataclass
class MolDynDatapoint:
    """A single MOLecular DYNamics datapoint with an initial molecule state and a target state"""

    name: str
    atom_types: torch.Tensor  # int64 tensor of shape [V]
    adj_list: torch.Tensor  # int64 tensor of shape [E, 2]
    atom_coords: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_velocs: torch.Tensor  # flaot32 tensor of shape [V, 3]
    atom_forces: torch.Tensor  # flaot32 tensor of shape [V, 3]
    atom_coord_targets: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_veloc_targets: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_force_targets: torch.Tensor  # float32 tensor of shape [V, 3]

    @property
    def num_atoms(self) -> int:
        return len(self.atom_types)


@dataclass
class SparseMolDynBatch:
    """
    A batch of molecular dynamics datapoints stored in a sparse format.

    All the molecules are stored in one large graph.
    """

    names: List[str]
    atom_types: torch.Tensor  # int64 tensor of shape [V]
    adj_list: torch.Tensor  # int64 tensor of shape [E, 2]
    atom_coords: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_velocs: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_forces: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_coord_targets: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_veloc_targets: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_force_targets: torch.Tensor  # float32 tensor of shape [V, 3]
    atom_to_sample_id: torch.Tensor  # int32 tensor of shape [num_samples]

    def pin_memory(self):
        self.atom_types.pin_memory()
        self.adj_list.pin_memory()
        self.atom_coords.pin_memory()
        self.atom_velocs.pin_memory()
        self.atom_forces.pin_memory()
        self.atom_coord_targets.pin_memory()
        self.atom_veloc_targets.pin_memory()
        self.atom_force_targets.pin_memory()
        self.atom_to_sample_id.pin_memory()
        return self


@dataclass
class DenseMolDynBatch:
    """
    A batch of molecular dynamics datapoints stored in a dense format, where smaller
    elements in the batch (those with fewer molecule) are padded to match the length of
    the longest molecule.

    All the tensors will have shape [B, ...] where B is the batch-size
    """

    names: List[str]
    atom_types: torch.Tensor  # int64 tensor of shape [B, max_num_atoms]
    adj_list: torch.Tensor  # int64 tensor of shape [E, 2]
    edge_batch_idx: torch.Tensor  # int64 tensor of shape [E]
    atom_coords: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    atom_velocs: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    atom_forces: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    atom_coord_targets: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    atom_veloc_targets: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    atom_force_targets: torch.Tensor  # float32 tensor of shape [B, max_num_atoms, 3]
    masked_elements: torch.Tensor  # bool tensor of shape [B, max_num_atoms]

    def pin_memory(self):
        self.atom_types.pin_memory()
        self.adj_list.pin_memory()
        self.edge_batch_idx.pin_memory()
        self.atom_coords.pin_memory()
        self.atom_velocs.pin_memory()
        self.atom_forces.pin_memory()
        self.atom_coord_targets.pin_memory()
        self.atom_veloc_targets.pin_memory()
        self.atom_force_targets.pin_memory()
        self.masked_elements.pin_memory()
        return self

    def tofp16(self):
        return DenseMolDynBatch(
            names=self.names,
            atom_types=self.atom_types,
            atom_coords=self.atom_coords.to(torch.float16),
            atom_velocs=self.atom_velocs.to(torch.float16),
            atom_forces=self.atom_forces.to(torch.float16),
            atom_coord_targets=self.atom_coord_targets.to(torch.float16),
            atom_veloc_targets=self.atom_veloc_targets.to(torch.float16),
            atom_force_targets=self.atom_force_targets.to(torch.float16),
            adj_list=self.adj_list,
            edge_batch_idx=self.edge_batch_idx,
            masked_elements=self.masked_elements,
        )

    def todevice(self, device: torch.device):
        return DenseMolDynBatch(
            names=self.names,
            atom_types=self.atom_types.to(device),
            atom_coords=self.atom_coords.to(device),
            atom_velocs=self.atom_velocs.to(device),
            atom_forces=self.atom_forces.to(device),
            atom_coord_targets=self.atom_coord_targets.to(device),
            atom_veloc_targets=self.atom_veloc_targets.to(device),
            atom_force_targets=self.atom_force_targets.to(device),
            adj_list=self.adj_list.to(device),
            edge_batch_idx=self.edge_batch_idx.to(device),
            masked_elements=self.masked_elements.to(device),
        )

    def sort_by_name(self):
        return self.permute(torch.from_numpy(np.argsort(self.names)))

    def permute(self, permutation: torch.Tensor):
        edge_batch_idx = permutation[self.edge_batch_idx]

        return DenseMolDynBatch(
            names=[self.names[i] for i in permutation.cpu().numpy()],
            atom_types=self.atom_types[permutation],
            atom_coords=self.atom_coords[permutation],
            atom_velocs=self.atom_velocs[permutation],
            atom_forces=self.atom_forces[permutation],
            atom_coord_targets=self.atom_coord_targets[permutation],
            atom_veloc_targets=self.atom_veloc_targets[permutation],
            atom_force_targets=self.atom_force_targets[permutation],
            adj_list=self.adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=self.masked_elements[permutation],
        )

    @cached_property
    def segments(self) -> List[int]:
        return contiguous_segments(self.names)


Batch = Union[SparseMolDynBatch, DenseMolDynBatch]


class CoordDeltaTooBig(Exception):
    def __init__(self, name: str, step1: int, step2: int, delta: float):
        self._name = name
        self._step1 = step1
        self._step2 = step2
        self._delta = delta

    def __str__(self):
        return f"{self._name} trajectory has {self._delta:g} distance between steps {self._step1} and {self._step2}"


def load_pdb_trace_data(
    name: str,
    state0_file: str,
    traj_file: str,
    step_width: int = 1,
    equal_data_spacing: bool = False,
):
    topology = md.load(state0_file).topology
    traj_data = np.load(traj_file)

    node_types = np.array([ELEMENT_VOCAB[a.element.symbol] for a in topology.atoms], dtype=np.int32)
    adj_list = np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32)
    assert np.min(adj_list) >= 0
    assert np.max(adj_list) < len(node_types)

    step_to_pos_vel_force: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for step, positions, velocities, forces in zip(
        traj_data["step"], traj_data["positions"], traj_data["velocities"], traj_data["forces"]
    ):
        step_to_pos_vel_force[step] = (positions, velocities, forces)

    coord_features, veloc_features, coord_targets, veloc_targets = [], [], [], []
    force_features, force_targets = [], []
    steps = traj_data["step"][:100]
    intervals = steps[1:] - steps[:-1]
    largest_interval = intervals.max()
    # Spacing is always logarithmic
    spacing = largest_interval * 10 // 9
    if spacing <= step_width and not equal_data_spacing:
        warnings.warn(
            f"The step_width of {step_width} is larger than or equal to the spacing of {spacing} in the data. This results in an unequal spacing between conditioning-target pairs."
        )

    for step, (step_pos, step_vel, step_force) in step_to_pos_vel_force.items():
        # Skip if state is not a conditioning state
        if step % spacing != 0 and equal_data_spacing:
            continue
        next_step = step + step_width
        next_step_data = step_to_pos_vel_force.get(next_step, None)
        if next_step_data is None:
            continue
        next_step_pos, next_step_veloc, next_step_force = next_step_data
        delta_norm = np.sqrt(np.sum((step_pos - next_step_pos) ** 2))
        if delta_norm > 100:
            raise CoordDeltaTooBig(name=name, delta=delta_norm, step1=step, step2=next_step)

        coord_features.append(step_pos)
        veloc_features.append(step_vel)
        force_features.append(step_force)
        coord_targets.append(next_step_pos)
        veloc_targets.append(next_step_veloc)
        force_targets.append(next_step_force)

    return TrajectoryInformation(
        name=name,
        node_types=node_types,
        adj_list=adj_list,
        coord_features=coord_features,
        veloc_features=veloc_features,
        force_features=force_features,
        coord_targets=coord_targets,
        veloc_targets=veloc_targets,
        force_targets=force_targets,
    )


def moldyn_sparse_collate_fn(datapoints: Iterable[MolDynDatapoint]) -> SparseMolDynBatch:
    """
    Collate function to produce a SparseMolDynBatch

    Note:
    It is not recommended to return CUDA tensors when multi-processing is enables (num_workers > 0).
    For details, see https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    total_num_nodes = 0
    batch_names: List[str] = []
    batch_atom_types: List[torch.Tensor] = []
    batch_adj_list: List[torch.Tensor] = []
    batch_atom_coords: List[torch.Tensor] = []
    batch_atom_velocs: List[torch.Tensor] = []
    batch_atom_forces: List[torch.Tensor] = []
    batch_atom_coord_targets: List[torch.Tensor] = []
    batch_atom_veloc_targets: List[torch.Tensor] = []
    batch_atom_force_targets: List[torch.Tensor] = []
    batch_atom_to_sample_id: List[torch.Tensor] = []
    for sample_id, datapoint in enumerate(datapoints):
        batch_names.append(datapoint.name)
        num_nodes = datapoint.atom_types.shape[0]
        batch_atom_types.append(datapoint.atom_types)
        batch_adj_list.append(datapoint.adj_list + total_num_nodes)
        batch_atom_coords.append(datapoint.atom_coords)
        batch_atom_velocs.append(datapoint.atom_velocs)
        batch_atom_forces.append(datapoint.atom_forces)
        batch_atom_coord_targets.append(datapoint.atom_coord_targets)
        batch_atom_veloc_targets.append(datapoint.atom_veloc_targets)
        batch_atom_force_targets.append(datapoint.atom_force_targets)
        batch_atom_to_sample_id.append(
            torch.full(size=(num_nodes,), fill_value=sample_id, dtype=torch.int32)
        )
        total_num_nodes += num_nodes

    return SparseMolDynBatch(
        names=batch_names,
        atom_types=torch.cat(batch_atom_types, dim=0),
        adj_list=torch.cat(batch_adj_list, dim=0),
        atom_coords=torch.cat(batch_atom_coords, dim=0),
        atom_velocs=torch.cat(batch_atom_velocs, dim=0),
        atom_forces=torch.cat(batch_atom_forces, dim=0),
        atom_coord_targets=torch.cat(batch_atom_coord_targets, dim=0),
        atom_veloc_targets=torch.cat(batch_atom_veloc_targets, dim=0),
        atom_force_targets=torch.cat(batch_atom_force_targets, dim=0),
        atom_to_sample_id=torch.cat(batch_atom_to_sample_id, dim=0),
    )


def moldyn_dense_collate_fn(datapoints: Sequence[MolDynDatapoint], fp16=False) -> DenseMolDynBatch:
    """
    Collate function to produce a DenseMolDynBatch

    Note:
    It is not recommended to return CUDA tensors when multi-processing is enabled (num_workers > 0).
    For details, see https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    # Get the number of atoms in each element of the batch
    batch_num_atoms = torch.tensor(
        [datapoint.num_atoms for datapoint in datapoints], dtype=torch.int64
    )  # [B] int64 tensor

    # Pad all the data with per-atom features
    batch_atom_types = pad_sequence(
        [datapoint.atom_types for datapoint in datapoints], batch_first=True
    )
    batch_atom_coords = pad_sequence(
        [datapoint.atom_coords for datapoint in datapoints], batch_first=True
    )
    batch_atom_velocs = pad_sequence(
        [datapoint.atom_velocs for datapoint in datapoints], batch_first=True
    )
    batch_atom_forces = pad_sequence(
        [datapoint.atom_forces for datapoint in datapoints], batch_first=True
    )
    batch_atom_coord_targets = pad_sequence(
        [datapoint.atom_coord_targets for datapoint in datapoints],
        batch_first=True,
    )
    batch_atom_veloc_targets = pad_sequence(
        [datapoint.atom_veloc_targets for datapoint in datapoints],
        batch_first=True,
    )
    batch_atom_force_targets = pad_sequence(
        [datapoint.atom_force_targets for datapoint in datapoints],
        batch_first=True,
    )

    # Get a mask for all the padded elements with 1 indicating which entries are paddding
    masked_elements = lengths_to_mask(lengths=batch_num_atoms)

    # Collate adjacency list and add a batch-element index
    batch_adj_list = torch.cat(tuple(datapoint.adj_list for datapoint in datapoints), dim=0)
    num_edges_per_datapoint = (datapoint.adj_list.shape[0] for datapoint in datapoints)
    # Get a tensor of size [total_num_edges] where each element indicates the batch-index the correspoinding edge in
    # batch_adj_list belongs to.
    edge_batch_idx = torch.cat(
        tuple(
            i * torch.ones(size=(num_edges,), dtype=torch.int64)
            for i, num_edges in enumerate(num_edges_per_datapoint)
        ),
        dim=0,
    )

    # Collect all the element names
    batch_names = [datapoint.name for datapoint in datapoints]

    batch = DenseMolDynBatch(
        names=batch_names,
        atom_types=batch_atom_types,
        atom_coords=batch_atom_coords,
        atom_velocs=batch_atom_velocs,
        atom_forces=batch_atom_forces,
        atom_coord_targets=batch_atom_coord_targets,
        atom_veloc_targets=batch_atom_veloc_targets,
        atom_force_targets=batch_atom_force_targets,
        adj_list=batch_adj_list,
        edge_batch_idx=edge_batch_idx,
        masked_elements=masked_elements,
    )

    return batch.tofp16() if fp16 else batch


def lengths_to_mask(
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Takes a int64 lengths tensor of shape [B] and return a [B, lengths.max()] shaped tensor mask
    where mask[i][j] is:
        1 if j => lengths[i],
        0 if j < lengths[i]
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    max_len = int(lengths.max())
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype).expand(
        len(lengths), max_len
    ) >= lengths.unsqueeze(1)
    return mask
