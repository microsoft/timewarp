"""
This module contains nn.Module wrappers that take both target and input as arguments to
their `forward()` method, and output a loss scalar.

This is done to unify training/sampling interface for different kind of regression models,
e.g. flow models, point-estimate models with MSE loss, Gaussian density models, etc.
"""
from typing import Tuple, Optional
import abc

import torch
import torch.nn as nn
from torch import Tensor

from utilities.logger import TrainingLogger
from timewarp.modules.model_wrappers.base import BaseModelWrapper


class PointEstimateModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:  # both [batch_size, num_points, 3] float tensors
        """
        Returns:
            A tuple of point estimates for target coordinates and velocities respectively, of
            shape [batch_size, num_points, 3] both.
        """
        pass


class PointEstimateWrapper(BaseModelWrapper, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        model: PointEstimateModel,
    ):
        super().__init__()
        self.model = model

    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:  # [1] float tensor

        y_coords_pred, y_velocs_pred = self.model(
            atom_types, x_coords, x_velocs, adj_list, edge_batch_idx, masked_elements, logger
        )

        # Masking due to sequences in a batch being of different lengths.
        y_coords_pred_masked = (~masked_elements[:, :, None]) * y_coords_pred  # [B, V, 3]
        y_velocs_pred_masked = (~masked_elements[:, :, None]) * y_velocs_pred  # [B, V, 3]
        y_coords_masked = (~masked_elements[:, :, None]) * y_coords  # [B, V, 3]
        y_velocs_masked = (~masked_elements[:, :, None]) * y_velocs  # [B, V, 3]

        loss = nn.functional.mse_loss(
            torch.cat((y_coords_pred_masked, y_velocs_pred_masked)),
            torch.cat((y_coords_masked, y_velocs_masked)),
        )
        return loss
