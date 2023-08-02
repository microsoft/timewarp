from typing import List, Literal, Optional
import torch
from torch import Tensor

from timewarp.modules.layers.custom_attention_encoder import CustomTransformerEncoderLayer
from timewarp.modules.layers.custom_transformer_block import (
    CustomAttentionTransformerBlock,
)
from timewarp.modules.layers.nvp import NVPCouplingLayer
from utilities.cache import Cache
from utilities.logger import TrainingLogger


class CustomAttentionTransformerCouplingLayer(NVPCouplingLayer):
    """
    A transformer coupling layer for molecules that is non-equivariant, and uses *local*
    self-attention
    """

    def __init__(
        self,
        atom_embedding_dim: int,
        mlp_hidden_layer_dims: List[int],
        transformed_vars: Literal["positions", "velocities"],
        scale_transformer_encoder_layers: List[CustomTransformerEncoderLayer],
        shift_transformer_encoder_layers: List[CustomTransformerEncoderLayer],
        separate_scales_per_dimension: bool = True,
    ):
        super().__init__(transformed_vars=transformed_vars)

        self.scale_transformer = CustomAttentionTransformerBlock(
            input_dim=atom_embedding_dim + 9,
            output_dim=3 if separate_scales_per_dimension else 1,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            transformer_encoder_layers=scale_transformer_encoder_layers,
        )
        self.shift_transformer = CustomAttentionTransformerBlock(
            input_dim=atom_embedding_dim + 9,
            output_dim=3,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            transformer_encoder_layers=shift_transformer_encoder_layers,
        )

    def _get_scale_and_shift(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords: Tensor,  # [B, num_points, 3]
        z_velocs: Tensor,  # [B, num_points, 3]
        x_features: Tensor,  # [B, num_points, D]
        x_coords: Tensor,  # [B, num_points, 3]
        x_velocs: Tensor,  # [B, num_points, 3]
        adj_list: Tensor,
        edge_batch_idx: Tensor,
        masked_elements: torch.BoolTensor,  # [B, num_points]
        logger: TrainingLogger,
        cache: Optional[Cache] = None,
    ):
        """Uses Transformer encoder to compute a scale and shift tensor based on the current x and z values.
        Performs *local* self attention, with the x_coords being used to define locality.

        Returns:
            scale, shift  # [B, V, 3]
        """
        if self.transformed_vars == "positions":  # Transform the positions using the velocities
            untransformed_vars = torch.cat(
                (x_features, x_coords, x_velocs, z_velocs), dim=-1
            )  # [B, num_points, atom_embedding_dim + 9]
        elif self.transformed_vars == "velocities":  # Transform the velocities using the positions
            untransformed_vars = torch.cat(
                (x_features, x_coords, x_velocs, z_coords), dim=-1
            )  # [B, num_points, atom_embedding_dim + 9]
        else:
            raise ValueError

        # Use x_coords to define locality.
        scale = torch.exp(
            self.scale_transformer(
                input_seq=untransformed_vars,
                positions=x_coords,
                masked_elements=masked_elements,
                logger=logger,
                cache=cache,
            )
        )  # [B, num_points, 3]
        shift = self.shift_transformer(
            input_seq=untransformed_vars,
            positions=x_coords,
            masked_elements=masked_elements,
            logger=logger,
            cache=cache,
        )  # [B, num_points, 3]

        return scale, shift  # [B, V, 3]
