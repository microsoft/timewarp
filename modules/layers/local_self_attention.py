from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softmax

from timewarp.modules.layers.custom_self_attention import CustomSelfAttentionBase
from utilities.cache import Cache
from utilities.logger import TrainingLogger


class LocalSelfAttention(CustomSelfAttentionBase):
    """Attention module where only neighbours within a radius `max_radius` are attended to."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        value_dim: int,
        key_query_dim: int,
        max_radius: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_query_dim = key_query_dim
        self.max_radius = max_radius

        # Takes input sequence and outputs the Q, K and V matrices.
        self.qkv_proj = nn.Linear(
            input_dim, num_heads * (value_dim + 2 * key_query_dim), bias=False
        )

        # Takes multiheaded outputs and projects to outputs.
        self.output_proj = nn.Linear(
            num_heads * value_dim, output_dim, bias=False
        )  # Is bias needed here?

    def forward(
        self,
        src: Tensor,  # [B, V, input_dim],
        positions: Tensor,  # [B, V, D], D is dimension of Euclidean space that defines locality
        masked_elements: torch.BoolTensor,  # [B, V], True when an element should be masked
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tensor:  # [B, V, output_dim]
        # Get the standard shaped query, key and value matrices
        qkv = self.qkv_proj(src)  # [B, V, H * (v_dim + 2 * kq_dim)]
        qkv = qkv.reshape(
            src.shape[0], src.shape[1], self.num_heads, self.value_dim + 2 * self.key_query_dim
        )  # [B, V, H, v_dim + 2 * kq_dim]
        q, k, v = torch.split(
            qkv,
            split_size_or_sections=[self.key_query_dim, self.key_query_dim, self.value_dim],
            dim=-1,
        )  # [B, V, H, kq_dim], [B, V, H, kq_dim], [B, V, H, v_dim]

        # Get the pairwise distances between points
        # NB there are numerical precision issues if compute_mode="use_mm_for_euclid_dist",
        # see https://github.com/pytorch/pytorch/issues/42479
        distance_matrix = torch.cdist(
            positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
        )  # [B, V, V]
        # Mask out meaningless elements due to variable length sequences in batching. We do this by defining their distance
        # to every other point to be +Inf, so they will never be counted as neighbors.
        masked_distances = torch.logical_or(
            masked_elements[:, None, :], masked_elements[:, :, None]
        )
        distance_matrix = distance_matrix.masked_fill(masked_distances, math.inf)  # [B, V, V]

        # Get the maximum num. of atoms within radius `self.max_radius` of any given atom
        within_radius = distance_matrix < self.max_radius  # [B, V, V]
        max_neighbors = within_radius.sum(dim=-1).max()  # Max number of neighbours
        if logger is not None:
            logger.log_scalar_async("max_neighbors", max_neighbors)
            logger.log_scalar_async("max_neighbors_over_num_atoms", max_neighbors / src.shape[1])

        # Get idxs of all points within `self.max_radius` of each atom
        topk_distances, neighbor_idxs = torch.topk(
            distance_matrix, k=max_neighbors, dim=-1, largest=False  # type: ignore [arg-type]
        )
        neighbor_mask = (
            topk_distances > self.max_radius
        )  # True when an index is outside the max radius and should be masked
        # neighbour_mask and neighbor_idxs have shape [B, V, K]

        # For each atom, only attend over (at most K) points within radius `self.max_radius` of itself:
        k_local = get_closest(k, neighbor_idxs)  # [B, V, K, H, kq_dim]
        v_local = get_closest(v, neighbor_idxs)  # [B, V, K, H, v_dim]

        # Compute scaled dot-product attention scores in the neighbourhood
        q_broadcast = q[:, :, None, :, :]  # [B, V, 1, H, kq_dim]
        scores = (k_local * q_broadcast).sum(-1) / math.sqrt(self.key_query_dim)  # [B, V, K, H]

        # Apply the mask to scores to only consider points within distance `max_radius`
        scores = scores.masked_fill(neighbor_mask[..., None], -math.inf)  # [B, V, K, H]
        attention_weights = softmax(scores, dim=-2)  # [B, V, K, H]
        # Once again zero out the meaningless sequence elements, since the softmax can lead to NaNs
        # when all of its inputs are -inf.
        attention_weights = attention_weights.masked_fill(
            neighbor_mask[..., None], 0.0
        )  # [B, V, K, H]

        # Weight outputs with attention weights
        multihead_out = (attention_weights[..., None] * v_local).sum(dim=-3)  # [B, V, H, v_dim]

        # Combine heads with output projection
        multihead_out = multihead_out.reshape(
            multihead_out.shape[0],
            multihead_out.shape[1],
            -1,
        )  # [B, V, H * v_dim]

        return self.output_proj(multihead_out)  # [B, V, output_dim]


def get_closest(
    M: Tensor,  # [B, V, H, d]
    neighbor_idxs: Tensor,  # [B, V, K]
) -> Tensor:  # [B, V, K, H, d]
    B, seq_length, num_heads, d = M.shape
    M_expanded = M[:, None, :, :, :].expand(-1, seq_length, -1, -1, -1)  # [B, V, V, H, d]
    neighbor_idxs_expanded = neighbor_idxs[:, :, :, None, None].expand(
        -1, -1, -1, num_heads, d
    )  # [B, V, K, H, d]

    out = torch.gather(
        M_expanded,  # [B, V, V, H, d]
        dim=-3,
        index=neighbor_idxs_expanded,  # [B, V, K, H, d]
    )  # [B, V, K, H, d]
    return out
