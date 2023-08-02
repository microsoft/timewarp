from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from timewarp.modules.layers.mlp import MLP


@dataclass
class TransformerConfig:
    n_head: int = 8
    dim_feedforward: int = 2048  # Dimension of pointwise MLP within the transformer
    dropout: float = 0.0  # Dropout causes stochasticity in likelihood computation


class TransformerBlock(nn.Module):
    """A module that eats real valued vector sequences, embeds them in a higher
    dimensional space, and applies a sequence of transformer encoder layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        mlp_hidden_layer_dims: Sequence[int],
        num_transformer_layers: int,
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        self.in_mlp = MLP(
            input_dim=input_dim, hidden_layer_dims=mlp_hidden_layer_dims, out_dim=latent_dim
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_config.n_head,
                dim_feedforward=transformer_config.dim_feedforward,
                dropout=transformer_config.dropout,  # Dropout creates stochastic likelihood during train time
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )  # input shape: [batch, seq, feature] = [B, V, feature_dim]

        self.out_mlp = MLP(
            input_dim=latent_dim, hidden_layer_dims=mlp_hidden_layer_dims, out_dim=output_dim
        )

    def forward(self, input_seq: Tensor, masked_elements: torch.BoolTensor) -> Tensor:
        """
        Args:
            input_seq ([B, V, input_dim] tensor): Inputs consisting of concatenated atom
                embedding, x positions, x velocs, and either z positions or velocities.
            masked_elements ([B, V] bool tensor): Tensor that has ones
                at the masked elements. I.e., elements that have ones are masked, and elements
                that have zeros are left alone.

        Returns:
            [B, V, output_dim] tensor: Output sequence
        """

        # Note: src_key_padding mask is used to mask out positions that are padding, i.e.,
        # after the end of the input sequence. This is not to be confused with src_mask.
        feature_seq = self.in_mlp(input_seq)  # [B, V, hidden_dim]
        out = self.transformer(
            feature_seq, src_key_padding_mask=masked_elements
        )  # [B, V, hidden_dim]

        return self.out_mlp(out)  # [B, V, output_dim]
