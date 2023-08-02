import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Sequence

from timewarp.dataloader import ELEMENT_VOCAB
from timewarp.modules.model_wrappers.gaussian_density import MeanLogScaleModel
from timewarp.modules.layers.transformer_block import TransformerBlock, TransformerConfig
from utilities.logger import TrainingLogger


class FlatTransformerMeanLogScaleModel(nn.Module):
    """
    Take per-atom flat features and create mean and log-scale
    outputs using a transformer.
    """

    def __init__(
        self,
        input_dim: int,  # flat input dimension per atom
        output_dim: int,  # dimension of mean and log-scale each
        hidden_dim: int,
        num_transformer_layers: int,
        mlp_hidden_layers_dims: Sequence[int],
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        self.transformer = TransformerBlock(
            input_dim=input_dim,
            output_dim=2 * output_dim,  # mean and log-scale
            latent_dim=hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layers_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )

    def forward(
        self,
        input: Tensor,  # [batch_size, num_points, input_dim] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ):
        out = self.transformer(
            input, masked_elements=masked_elements
        )  # [B, num_points, 2*output_dim]

        out_mean, out_log_scale = torch.chunk(out, chunks=2, dim=-1)

        return out_mean, out_log_scale

    def log_likelihood(
        self,
        input: Tensor,  # [batch_size, num_points, input_dim] float tensor
        output: Tensor,  # [batch_size, num_points, output_dim] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ):
        out_mean, out_log_scale = self(input, adj_list, edge_batch_idx, masked_elements, logger)
        out_scale = torch.exp(out_log_scale)
        out_dist = torch.distributions.Normal(
            out_mean,
            out_scale,
        )  # [batch_size, num_points, output_dim]
        log_prob = out_dist.log_prob(output)  # [batch_size, num_points, output_dim]
        log_prob = torch.sum(dim=(-1, -2))  # [batch_size]

        return log_prob


class TransformerMeanLogScaleModel(MeanLogScaleModel):
    """
    A module that takes atom positions, velocities and embeddings, encodes them in a higher
    dimensional space, and applies a sequence of transformer encoder layers.
    """

    def __init__(
        self,
        atom_embedding_dim: int,
        hidden_dim: int,
        num_transformer_layers: int,
        mlp_hidden_layers_dims: Sequence[int],
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        self.atom_embedder = nn.Embedding(
            num_embeddings=len(ELEMENT_VOCAB),
            embedding_dim=atom_embedding_dim,
        )
        self.transformer = TransformerBlock(
            input_dim=atom_embedding_dim + 6,  # Atom embeddings, x coords, x velocs
            output_dim=12,  # Means and log-std of y coords and y_velocs
            latent_dim=hidden_dim,
            mlp_hidden_layer_dims=mlp_hidden_layers_dims,
            num_transformer_layers=num_transformer_layers,
            transformer_config=transformer_config,
        )

    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ):
        atom_features = self.atom_embedder(atom_types)  # [B, num_points, atom_embedding_dim]
        flat_features = torch.cat(
            (atom_features, x_coords, x_velocs), dim=-1
        )  # [B, num_points, atom_embedding_dim + 9]

        out = self.transformer(
            flat_features, masked_elements=masked_elements
        )  # [B, num_points, 12]

        y_coords_mean, y_velocs_mean, y_coords_log_std, y_velocs_log_std = torch.chunk(
            out, chunks=4, dim=-1
        )

        return y_coords_mean, y_velocs_mean, y_coords_log_std, y_velocs_log_std
