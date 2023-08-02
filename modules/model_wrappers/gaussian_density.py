import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
from utilities.logger import TrainingLogger
from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel
from timewarp.utils.molecule_utils import get_centre_of_mass
from torch import Tensor, BoolTensor


class MeanLogScaleModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Take the initial state defined by: atom_types, x_coords, x_velocs,
        adj_list and edge_batch_idx as input and output means of target position
        and velocities and log standard deviations of target positions and velocities.

        Returns:
            - y_coords_mean [batch_size, num_points, 3]
            - y_velocs_mean [batch_size, num_points, 3]
            - y_coords_log_std [batch_size, num_points, 1]
            - y_velocs_log_std [batch_size, num_points, 1]

            Note: outputs one standard deviation per point.
        """
        pass


class GaussianDensityModel(ConditionalDensityModel):
    def __init__(
        self,
        *,
        mean_log_scale_model: MeanLogScaleModel,
    ):
        super().__init__()
        self.mean_log_scale_model = mean_log_scale_model

    def log_likelihood(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:
        """
        Computes the log-likelihood p(y_coords, y_velocs | x_coords, x_velocs, atom_types) assuming
        a Gaussian density where only the mean of the Gaussian depends on the conditioning variables
        x_coords, x_velocs and atom_types.
        """
        y_coords_residual = (
            y_coords - x_coords
        )  # Predict the position change (residual) rather than absolute position
        y_velocs_residual = (
            y_velocs - x_velocs
        )  # Predict the velocity change (residual) rather than absolute velocity
        # Canonicalise input position:
        centre_of_mass = get_centre_of_mass(x_coords, masked_elements=masked_elements)
        x_coords = x_coords - centre_of_mass

        # Get the means ([B, num_atoms, 3] each) and the log-st.d. ([B, num_atoms, 1] each)
        (
            y_coords_res_mean,
            y_velocs_res_mean,
            y_coords_res_log_std,
            y_velocs_res_log_std,
        ) = self.mean_log_scale_model(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )

        y_coords_std = torch.exp(y_coords_res_log_std)
        y_velocs_std = torch.exp(y_velocs_res_log_std)

        y_coords_distribution = torch.distributions.Normal(
            loc=y_coords_res_mean,
            scale=y_coords_std,
        )
        y_velocs_distribution = torch.distributions.Normal(
            loc=y_velocs_res_mean,
            scale=y_velocs_std,
        )

        log_prob_y_coords = y_coords_distribution.log_prob(y_coords_residual)  # [B]
        log_prob_y_velocs = y_velocs_distribution.log_prob(y_velocs_residual)  # [B]

        # Masking due to sequences in a batch being of different lengths.
        log_prob_y_coords = ((~masked_elements[:, :, None]) * log_prob_y_coords).sum(
            dim=(-1, -2)
        )  # [B]
        log_prob_y_velocs = ((~masked_elements[:, :, None]) * log_prob_y_velocs).sum(
            dim=(-1, -2)
        )  # [B]

        log_prob_y = log_prob_y_coords + log_prob_y_velocs  # Independent Gaussians

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("log_prob_y_coords", log_prob_y_coords.mean())
            logger.log_scalar_async("log_prob_y_velocs", log_prob_y_velocs.mean())
            logger.log_scalar_async("mean_y_coords_std", y_coords_std.mean())
            logger.log_scalar_async("mean_y_velocs_std", y_velocs_std.mean())
        return log_prob_y  # Shape [batch_size]

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
