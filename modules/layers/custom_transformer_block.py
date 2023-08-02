from typing import List, Sequence

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from utilities.cache import Cache

from utilities.logger import TrainingLogger
from timewarp.modules.layers.mlp import MLP
from timewarp.modules.layers.custom_attention_encoder import CustomTransformerEncoderLayer


class CustomAttentionTransformerBlock(nn.Module):
    """A module that eats real valued vector sequences, embeds them in a higher
    dimensional space, and applies a sequence of *local* transformer encoder layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_hidden_layer_dims: List[int],
        transformer_encoder_layers: Sequence[CustomTransformerEncoderLayer],
    ):
        super().__init__()

        # Input MLP
        self.in_mlp = MLP(
            input_dim=input_dim,
            hidden_layer_dims=mlp_hidden_layer_dims,
            out_dim=transformer_encoder_layers[0].d_model,
        )

        # Transformer encoder
        self.encoder_layers = nn.ModuleList(transformer_encoder_layers)

        # Output MLP
        self.out_mlp = MLP(
            input_dim=transformer_encoder_layers[-1].d_model,
            hidden_layer_dims=mlp_hidden_layer_dims,
            out_dim=output_dim,
        )

    def forward(
        self,
        input_seq: Tensor,
        positions: Tensor,
        masked_elements: torch.BoolTensor,
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tensor:
        """
        Args:
            input_seq ([B, V, input_dim] tensor): Inputs consisting of concatenated atom
                embedding, x positions, x velocs, and either z positions or velocities.
            positions ([B, V, D] tensor): D is dimension of Euclidean space that defines locality
                Positions that are used to determine which atoms each atom attends to.
            masked_elements ([B, V] bool tensor): Tensor that has ones
                at the masked elements. I.e., elements that have ones are masked, and elements
                that have zeros are left alone.
            logger: TrainingLogger

        Returns:
            [B, V, output_dim] tensor: Output sequence
        """
        # Input embedding
        feature_seq = self.in_mlp(input_seq)  # [B, V, hidden_dim]

        # Local transformer encoder
        for layer in self.encoder_layers:
            feature_seq = layer(
                src=feature_seq,
                positions=positions,
                masked_elements=masked_elements,
                logger=logger,
                cache=cache,
            )  # [B, V, hidden_dim]

        # Output embedding
        return self.out_mlp(feature_seq)  # [B, V, output_dim]
