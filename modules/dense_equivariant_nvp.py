from typing import Literal
import torch

from timewarp.modules.layers.dense_equivariant_coupling_layer import ShiftModule, ScaleModule
from timewarp.modules.layers.nvp import NVPCouplingLayer


class EquivariantCouplingLayer(NVPCouplingLayer):
    """
    An equivariant coupling layer for molecules. 
    """

    def __init__(
        self,
        shift_module: ShiftModule,
        scale_module: ScaleModule,
        transformed_vars: Literal["positions", "velocities"],
    ):
        super().__init__(transformed_vars=transformed_vars)
        self.shift_module = shift_module
        self.scale_module = scale_module

    def _get_scale_and_shift(
        self,
        atom_types,  # [B, num_points]
        z_coords,  # [B, num_points, 3]
        z_velocs,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        adj_list,
        edge_batch_idx,
        masked_elements,  # [B, num_points]
        cache,
        logger,
    ):
        """Uses equivariant/invariant map to compute a shift/scale tensor based on the current x and z values.

        Returns:
            scale, shift  # [B, V, 3]
        """
        if self.transformed_vars == "positions":  # Transform the positions using the velocities
            z_untransformed = z_velocs
        elif self.transformed_vars == "velocities":  # Transform the velocities using the positions
            z_untransformed = z_coords
        else:
            raise ValueError

        log_scale = self.scale_module(
            adj_list=adj_list,
            z_untransformed=z_untransformed,
            x_features=x_features,
            x_coords=x_coords,
            x_velocs=x_velocs,
            masked_elements=masked_elements,
        )  # [B, V, 1]
        shift = self.shift_module(
            adj_list=adj_list,
            z_untransformed=z_untransformed,
            x_features=x_features,
            x_coords=x_coords,
            x_velocs=x_velocs,
            masked_elements=masked_elements,
        )  # [B, V, 3]

        scale = torch.exp(log_scale).repeat(1, 1, 3)  # [B, V, 3]

        return scale, shift  # [B, V, 3]
