from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from timewarp.modules.layers.mlp import MLP


class FeatureProcessor(nn.Module):
    """Process the pointwise and relative features in a permutation equivariant and
    SE(3) invariant way, using a dense EGNN-like forward pass.
    """

    def __init__(
        self,
        *,
        input_pointwise_features_dim: int,
        input_relative_features_dim: int,
        processed_pointwise_features_dim: int,
        processed_relative_features_dim: int,
        hidden_layers_dims: Sequence[int]
    ):
        super().__init__()
        # MLPs to transform relative and pointwise features
        self._relative_features_mlp = MLP(
            input_dim=input_relative_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=processed_relative_features_dim,
        )
        self._pointwise_features_mlp = MLP(
            input_dim=input_pointwise_features_dim + processed_relative_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=processed_pointwise_features_dim,
        )

    def forward(
        self,
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_features,  # [B, V, num_pointwise_feat]
        adj_list,
        masked_elements,  # [B, V]
    ) -> Tuple[Tensor, Tensor]:
        # Concatenate features for points i and j and append to the relative_features between i, j
        num_points = pointwise_features.shape[-2]
        assert pointwise_features.ndim == 3  # Batching.
        pointwise_features_i = pointwise_features.unsqueeze(-2).expand(
            -1, -1, num_points, -1
        )  # [B, V, V, num_pointwise_feat]
        pointwise_features_j = pointwise_features.unsqueeze(-3).expand(
            -1, num_points, -1, -1
        )  # [B, V, V, num_pointwise_feat]

        relative_features = torch.cat(
            (pointwise_features_i, pointwise_features_j, relative_features), dim=-1
        )  # [B, V, V, 2 * num_pointwise_feat + num_rel_feat]
        relative_features = self._relative_features_mlp(
            relative_features
        )  # [B, V, V, num_processed_rel_feat]
        # Mask meaningless elements in the summed out dimension (due to different sequence lengths in batch).
        unmasked_elements = ~masked_elements[:, None, :, None]  # [B, 1, V, 1]
        relative_features = (
            relative_features * unmasked_elements
        )  # [B, V, V, num_processed_rel_feat]

        # Sum relative features between this and all other atoms and append to pointwise_features
        # Then pass through an MLP
        num_atoms = (~masked_elements).sum(dim=-1)  # [B]
        avg_relative_features = (
            relative_features.sum(-2) / num_atoms[:, None, None]
        )  # [B, V, num_processed_rel_feat]
        pointwise_features = self._pointwise_features_mlp(
            torch.cat(
                (pointwise_features, avg_relative_features), dim=-1
            )  # [B, V, num_pointwise_feat + num_processed_rel_feat]
        )  # [B, V, num_processed_pointwise_feat]
        return (
            relative_features,
            pointwise_features,
        )  # [B, V, V, num_processed_rel_feat], [B, V, num_processed_pointwise_feat]
