from typing import List, Literal
import torch

from timewarp.model_configs import (
    TransformerConfig,
    RFFPositionEncoderConfig,
)
from timewarp.modules.layers.transformer_block import TransformerBlock
from timewarp.modules.layers.nvp import NVPCouplingLayer
from timewarp.modules.layers.rff_position_encoder import RFFPositionEncoder


class TransformerCouplingLayer(NVPCouplingLayer):
    """
    A transformer coupling layer for molecules that is non-equivariant.
    """

    def __init__(
        self,
        atom_embedding_dim: int,
        transformer_hidden_dim: int,
        mlp_hidden_layer_dims: List[int],
        num_transformer_layers: int,
        transformed_vars: Literal["positions", "velocities"],
        transformer_config: TransformerConfig,
        rff_position_encoder_config: RFFPositionEncoderConfig,
    ):
        super().__init__(transformed_vars=transformed_vars)
        assert rff_position_encoder_config
        self.position_encoder = RFFPositionEncoder(
            3,
            rff_position_encoder_config.encoding_dim,
            rff_position_encoder_config.scale_mean,
            rff_position_encoder_config.scale_stddev,
        )

        self.scale_transformer = TransformerBlock(
            # Atom embeddings, x coords, x velocs, half of z, enc(x coords)
            input_dim=atom_embedding_dim + 9 + rff_position_encoder_config.encoding_dim,
            output_dim=3,
            latent_dim=transformer_hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )
        self.shift_transformer = TransformerBlock(
            # Atom embeddings, x coords, x velocs, half of z, enc(x coords)
            input_dim=atom_embedding_dim + 9 + rff_position_encoder_config.encoding_dim,
            output_dim=3,
            latent_dim=transformer_hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layer_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )

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
        """Uses Transformer encoder to compute a scale and shift tensor based on the current x and z values.

        Returns:
            scale, shift  # [B, V, 3]
        """
        x_coords_enc = self.position_encoder(x_coords)
        assert x_coords_enc.shape[0] == x_coords.shape[0]
        assert x_coords_enc.shape[1] == x_coords.shape[1]

        if self.transformed_vars == "positions":  # Transform the positions using the velocities
            untransformed_vars = torch.cat(
                (x_features, x_coords, x_velocs, z_velocs, x_coords_enc), dim=-1
            )  # [B, num_points, atom_embedding_dim + 9 + enc_dim]
        elif self.transformed_vars == "velocities":  # Transform the velocities using the positions
            untransformed_vars = torch.cat(
                (x_features, x_coords, x_velocs, z_coords, x_coords_enc), dim=-1
            )  # [B, num_points, atom_embedding_dim + 9 + enc_dim]
        else:
            raise ValueError

        scale = torch.exp(
            self.scale_transformer(untransformed_vars, masked_elements=masked_elements)
        )  # [B, num_points, 3]
        shift = self.shift_transformer(
            untransformed_vars, masked_elements=masked_elements
        )  # [B, num_points, 3]

        return scale, shift  # [B, V, 3]
