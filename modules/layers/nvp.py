from typing import Tuple, Literal, Optional
import abc

import torch
import torch.nn as nn
from torch import Tensor
from utilities.cache import Cache

from utilities.logger import TrainingLogger
from timewarp.modules.model_wrappers.flow import LikType


class NVPCouplingLayer(nn.Module, abc.ABC):
    """
    A transformer coupling layer for molecules that is non-equivariant.
    """

    def __init__(self, transformed_vars: Literal["positions", "velocities"]):
        super().__init__()
        self.transformed_vars = transformed_vars

    def forward(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords: Tensor,  # [B, num_points, 3]
        z_velocs: Tensor,  # [B, num_points, 3]
        x_features: Tensor,  # [B, num_points, D]
        x_coords: Tensor,  # [B, num_points, 3]
        x_velocs: Tensor,  # [B, num_points, 3]
        adj_list: Tensor,  # [num_edges, 2]
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [B, num_points]
        delta_logp: LikType = None,  # [B, 1]
        reverse: bool = False,
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, LikType]:
        """
        Args:
            masked_elements: Tensor that has ones at the masked elements. I.e., elements that have
                ones are masked, and elements that have zeros are left alone. This allows us to batch
                sequences of different length in the transformer.
            delta_logp: Tensor holding the accumulated -ve log-det Jacobian terms.
                If x = f(x') where f(x') where are the previous layers of the flow, then
                this terms is equal to the difference: log p(x) - log p(x')
            reverse: If reverse is True, the layer is in "sampling mode" propagating a
                latent sample through towards the observation space.

                If reverse is False, the layer is in "density estimation mode", propagating
                an observed sample through to get the corresponding latent sample and the
                likelihood.
        Return:
            - y
            - Updated delta_logp: Subtract the log-det of Jacobian of this transformation for delta_logp,
                which is equivalent to adding log p(y) - log p(x) to delta_logp
        """
        if reverse:
            z_coords, z_velocs, logdetjac = self.flow_reverse(
                atom_types=atom_types,
                z_coords=z_coords,
                z_velocs=z_velocs,
                x_features=x_features,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                logger=logger,
                cache=cache,
            )
        else:
            z_coords, z_velocs, logdetjac = self.flow_forward(
                atom_types=atom_types,
                z_coords=z_coords,
                z_velocs=z_velocs,
                x_features=x_features,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                logger=logger,
                cache=cache,
            )

        delta_logp = delta_logp - logdetjac if delta_logp is not None else None
        return z_coords, z_velocs, delta_logp

    def flow_forward(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords,  # [B, num_points, 3]
        z_velocs,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        adj_list,
        edge_batch_idx,
        masked_elements,  # [B, num_points]
        logger,
        cache: Optional[Cache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, LikType]:
        """
        Forward pass of the flow (going in the direction from the observed space to the latent space).

        Returns:
            - Outputs of the forward transformation of x and z of shape [batch_size, dim]
            - logdetjac: The log-det Jacobian of the forward transformation of z
                of shape [B]
        """
        scale, shift = self._get_scale_and_shift(
            atom_types=atom_types,
            z_coords=z_coords,
            z_velocs=z_velocs,
            x_features=x_features,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
            cache=cache,
        )  # [B, V, 3]

        # Log-det Jacobian of the forward transform
        # Need to mask out the irrelevant terms in a batch due to different sequence length
        log_scales = torch.log(scale) * (~masked_elements[:, :, None])
        logdetjac = torch.sum(log_scales, dim=(-1, -2))  # [B]

        if self.transformed_vars == "positions":  # Transform the positions
            z_coords = z_coords * scale + shift  # [B, num_points, 3]
        elif self.transformed_vars == "velocities":  # Transform the velocities
            z_velocs = z_velocs * scale + shift  # [B, num_points, 3]

        return z_coords, z_velocs, logdetjac

    def flow_reverse(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords,  # [B, num_points, 3]
        z_velocs,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        adj_list,
        edge_batch_idx,
        masked_elements,  # [B, num_points]
        logger,
        cache: Optional[Cache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, LikType]:
        """
        Reverse pass (going in the direction from the latent space to the observed space).

        Returns:
            - Outputs of the forward transformation of x and z of shape [batch_size, dim]
            - logdetjac: The log-det Jacobian of the reverse transformation of z
                of shape [B]
        """
        scale, shift = self._get_scale_and_shift(
            atom_types=atom_types,
            z_coords=z_coords,
            z_velocs=z_velocs,
            x_features=x_features,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
            cache=cache,
        )  # [B, V, 3]

        # Log-det Jacobian of the forward transform, which has a minus sign
        # Need to mask out the irrelevant terms in a batch due to different sequence length
        log_scales = torch.log(scale) * (~masked_elements[:, :, None])
        logdetjac = -torch.sum(log_scales, dim=(-1, -2))  # [B]

        if self.transformed_vars == "positions":  # Transform the positions
            z_coords = (z_coords - shift) / scale  # [B, num_points, 3]
        elif self.transformed_vars == "velocities":  # Transform the velocities
            z_velocs = (z_velocs - shift) / scale  # [B, num_points, 3]

        return z_coords, z_velocs, logdetjac

    @abc.abstractmethod
    def _get_scale_and_shift(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords,  # [B, num_points, 3]
        z_velocs,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        adj_list,
        edge_batch_idx,
        masked_elements,  # [B, num_points]
        logger,
        cache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a scale and shift tensor based on the current x and z values.

        Returns:
            scale, shift  # [B, V, 3]
        """
        pass
