from timewarp.modules.model_wrappers.density_model_base import (
    ConditionalDensityModel,
    ConditionalDensityModelWithForce,
)
from timewarp.dataloader import DenseMolDynBatch, Batch

from utilities.logger import TensorBoardLogger

from typing import Optional, List, Union, Tuple
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
from functools import singledispatch


@singledispatch
def get_sample(
    model: Union[ConditionalDensityModel, ConditionalDensityModelWithForce],
    batch: Batch,
    num_samples: int,
    device: Optional[torch.device] = None,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Tensor:
    raise TypeError


@get_sample.register  # type: ignore[no-redef]
def _(
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    num_samples: int,
    device: Optional[torch.device] = None,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Tuple[Tensor, Tensor]:
    y_coords, y_velocs = model.conditional_sample(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        num_samples=num_samples,
        logger=tb_logger,
    )
    return y_coords, y_velocs


@get_sample.register  # type: ignore[no-redef]
def _(
    model: ConditionalDensityModelWithForce,
    batch: DenseMolDynBatch,
    num_samples: int,
    device: Optional[str] = None,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Tuple[Tensor, Tensor]:
    y_coords, y_velocs = model.conditional_sample(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        x_forces=batch.atom_forces.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        num_samples=num_samples,
        logger=tb_logger,
    )
    return y_coords, y_velocs


def sample(
    model: Union[ConditionalDensityModel, ConditionalDensityModelWithForce],
    batch: DenseMolDynBatch,
    num_samples: int,
    decorrelated: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:  # [V, 3], [V, 3]

    assert (
        len(batch.atom_coords) == 1
    ), f"Expected batchsize of one instead of {len(batch.atom_coords)}."

    # Sample from the model one by one to avoid memory issues
    y_coords = np.zeros(
        (
            num_samples,
            batch.atom_coords.shape[-2],
            batch.atom_coords.shape[-1],
        )
    )  # [S, V, 3]
    y_velocs = np.zeros(
        (
            num_samples,
            batch.atom_velocs.shape[-2],
            batch.atom_velocs.shape[-1],
        )
    )  # [S, V, 3]

    for i in range(num_samples):
        if decorrelated:
            coords_sample, velocs_sample = get_decorrelated_sample(
                model, batch, num_samples=1, device=device
            )  # [S=1, B=1, V, 3]
        else:
            coords_sample, velocs_sample = get_sample(
                model, batch, num_samples=1, device=device
            )  # [S=1, B=1, V, 3]
        y_coords[i, :, :] = coords_sample[0, 0, :, :].detach().cpu().numpy()  # [V, 3]
        y_velocs[i, :, :] = velocs_sample[0, 0, :, :].detach().cpu().numpy()  # [V, 3]

    return y_coords, y_velocs


def get_decorrelated_sample(
    model: Union[ConditionalDensityModel, ConditionalDensityModelWithForce],
    batch: DenseMolDynBatch,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:  # [S, B, V, 3], [S, B, V, 3]
    """Sample from the model but sample every atom independently, to investigate the effect of
    correlations.
    """
    assert num_samples == 1  # TODO add support for batching over sample dimension.

    coords_sample = torch.zeros_like(batch.atom_coords)  # [B, V, 3]
    velocs_sample = torch.zeros_like(batch.atom_velocs)  # [B, V, 3]
    num_atoms = batch.atom_coords.shape[-2]
    for atom in range(num_atoms):
        with torch.no_grad():
            atom_coords_sample, atom_velocs_sample = get_sample(
                model, batch, num_samples=num_samples, device=device
            )  # [S, B, V, 3]
        coords_sample[:, atom, :] = atom_coords_sample[
            0, :, atom, :
        ]  # [B, 3], fill in one atom from the sample.
        velocs_sample[:, atom, :] = atom_velocs_sample[
            0, :, atom, :
        ]  # [B, 3], fill in one atom from the sample.

    return coords_sample[None, ...], velocs_sample[None, ...]  # [S=1, B, V, 3]


def sample_from_trajectory(
    model: Union[ConditionalDensityModel, ConditionalDensityModelWithForce],
    batches: List[DenseMolDynBatch],
    num_samples: int,
    decorrelated: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[List, List]:
    """Sample with the model from a given trajectory of conditioning states.

    Args:
        model (Union[ConditionalDensityModel, ConditionalDensityModelWithForce]): Model
        batches (List[DenseMolDynBatch]): Batches with bath size one.
        num_samples (int): Number of samples
        decorrelated (bool, optional): If true, every atom is sampled independently. Defaults to False.
        device (Optional[torch.device], optional): Device. Defaults to None.

    Returns:
        Tuple[List, List]: Sampled coordinates and velocities. Length B list of [S, V, 3]
    """

    assert (
        len(batches[0].atom_coords) == 1
    ), f"Expected batchsize of one instead of {len(batches[0].atom_coords)}."

    # Sample from the model (one by one for memory reasons).
    y_coords_model: List[np.ndarray] = []
    y_velocs_model: List[np.ndarray] = []
    for batch in tqdm(batches, desc="Sampling", unit="initial state"):
        with torch.no_grad():
            y_coords_i, y_velocs_i = sample(
                model=model,
                batch=batch,
                num_samples=num_samples,
                decorrelated=decorrelated,
                device=device,
            )  # [S, V, 3], [S, V, 3]
        y_coords_model.append(y_coords_i)  # Length B list of [S, V, 3]
        y_velocs_model.append(y_velocs_i)  # Length B list of [S, V, 3]
    return y_coords_model, y_velocs_model
