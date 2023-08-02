from typing import Optional, Union
from functools import singledispatch

from torch import Tensor
from deepspeed import DeepSpeedEngine  # type: ignore [import]
from timewarp.dataloader import Batch, DenseMolDynBatch
from timewarp.losses import LossWrapper
from timewarp.modules.model_wrappers.base import BaseModelWrapper, BaseModelWrapperWithForce
from timewarp.modules.model_wrappers.density_model_base import (
    ConditionalDensityModel,
    ConditionalDensityModelWithForce,
)
from utilities.logger import TrainingLogger


# TODO : Completely phase out this entire file. It should be possible to
# replace this entire file with what is present in `projects/timewarp/losses.py`.
@singledispatch
def get_loss(
    model: Union[DeepSpeedEngine, BaseModelWrapper, BaseModelWrapperWithForce],
    batch: Batch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    raise TypeError


@get_loss.register  # type: ignore[no-redef]
def _(
    model: DeepSpeedEngine,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    return get_loss.dispatch(model.module.__class__)(model, batch, device, tb_logger)


@get_loss.register  # type: ignore[no-redef]
def _(
    model: LossWrapper,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    return model(batch, device=device, logger=tb_logger)


@get_loss.register  # type: ignore[no-redef]
def _(
    model: BaseModelWrapper,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    loss = model(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        y_coords=batch.atom_coord_targets.to(device, non_blocking=True),
        y_velocs=batch.atom_veloc_targets.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        logger=tb_logger,
    )
    return loss


@get_loss.register  # type: ignore[no-redef]
def _(
    model: BaseModelWrapperWithForce,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    loss = model(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        x_forces=batch.atom_forces.to(device, non_blocking=True),
        y_coords=batch.atom_coord_targets.to(device, non_blocking=True),
        y_velocs=batch.atom_veloc_targets.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        logger=tb_logger,
    )
    return loss


@singledispatch
def get_log_likelihood(
    model: Union[BaseModelWrapper, BaseModelWrapperWithForce],
    batch: Batch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    raise TypeError


@get_log_likelihood.register  # type: ignore[no-redef]
def _(
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    log_likelihood = model.log_likelihood(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        y_coords=batch.atom_coord_targets.to(device, non_blocking=True),
        y_velocs=batch.atom_veloc_targets.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        logger=tb_logger,
    )
    return log_likelihood  # [B]


@get_log_likelihood.register  # type: ignore[no-redef]
def _(
    model: ConditionalDensityModelWithForce,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> Tensor:
    log_likelihood = model.log_likelihood(
        atom_types=batch.atom_types.to(device, non_blocking=True),
        x_coords=batch.atom_coords.to(device, non_blocking=True),
        x_velocs=batch.atom_velocs.to(device, non_blocking=True),
        x_forces=batch.atom_forces.to(device, non_blocking=True),
        y_coords=batch.atom_coord_targets.to(device, non_blocking=True),
        y_velocs=batch.atom_veloc_targets.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        logger=tb_logger,
    )
    return log_likelihood  # [B]
