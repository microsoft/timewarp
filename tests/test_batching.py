import os
import torch
from pathlib import Path
import pytest

from torch.utils.data.dataloader import DataLoader

from timewarp.dataloader import DenseMolDynBatch, moldyn_dense_collate_fn
from timewarp.datasets import TrajectoryIterableDataset
from timewarp.utils.loss_utils import get_log_likelihood
from timewarp.model_constructor import (
    custom_transformer_nvp_constructor,
    model_constructor,
    transformer_nvp_constructor,
    equivariant_nvp_constructor,
)
from timewarp.model_configs import (
    TransformerNVPConfig,
    EquivariantNVPConfig,
    CustomAttentionTransformerNVPConfig,
)
from timewarp.modules.layers.transformer_block import TransformerConfig
from timewarp.modules.layers.custom_attention_encoder import CustomAttentionEncoderLayerConfig
from utilities.training_utils import set_seed


def get_single_datapoint(idx: int, batch: DenseMolDynBatch):
    """Extract a single datapoint from a batch."""
    return DenseMolDynBatch(
        names=batch.names[idx : idx + 1],
        atom_types=batch.atom_types[idx : idx + 1],  # [1, max_num_atoms]
        adj_list=batch.adj_list,  # TODO: support batching for sparse graph.
        edge_batch_idx=batch.edge_batch_idx,  # TODO: support batching for sparse graph.
        atom_coords=batch.atom_coords[idx : idx + 1],  # [1, max_num_atoms, 3]
        atom_velocs=batch.atom_velocs[idx : idx + 1],  # [1, max_num_atoms, 3]
        atom_forces=batch.atom_forces[idx : idx + 1],  # [1, max_num_atoms, 3]
        atom_coord_targets=batch.atom_coord_targets[idx : idx + 1],  # [1, max_num_atoms, 3]
        atom_veloc_targets=batch.atom_veloc_targets[idx : idx + 1],  # [1, max_num_atoms, 3]
        atom_force_targets=batch.atom_force_targets[idx : idx + 1],  # [1, max_num_atoms, 3]
        masked_elements=batch.masked_elements[idx : idx + 1],  # [1, max_num_atoms]
    )


def get_transformer_nvp():
    transformer_config = TransformerConfig()  # Use defaults.
    transformer_nvp_config = TransformerNVPConfig(
        atom_embedding_dim=4,
        transformer_hidden_dim=8,
        latent_mlp_hidden_dims=[8, 8],
        num_coupling_layers=2,
        num_transformer_layers=2,
        transformer_config=transformer_config,
    )
    model = transformer_nvp_constructor(transformer_nvp_config)
    return model


def get_local_transformer_nvp():
    custom_attention_encoder_config = CustomAttentionEncoderLayerConfig(
        d_model=4,
        dim_feedforward=8,
        dropout=0.0,
        num_heads=2,
        attention_type="local",
        lengthscales=None,  # Only needed for kernel attention
        max_radius=0.5,
        normalise_kernel_values=None,  # Only needed for kernel attention
    )
    local_transformer_nvp_config = CustomAttentionTransformerNVPConfig(
        atom_embedding_dim=4,
        latent_mlp_hidden_dims=[8, 8],
        num_coupling_layers=2,
        num_transformer_layers=2,
        encoder_layer_config=custom_attention_encoder_config,
    )
    model = custom_transformer_nvp_constructor(local_transformer_nvp_config)
    return model


def get_kernel_transformer_nvp():
    custom_attention_encoder_config = CustomAttentionEncoderLayerConfig(
        d_model=4,
        dim_feedforward=8,
        dropout=0.0,
        num_heads=4,
        attention_type="kernel",
        lengthscales=[0.1, 0.2, 0.5, 1.0],
        max_radius=None,  # Only needed for local attention
        normalise_kernel_values=False,
    )
    kernel_transformer_nvp_config = CustomAttentionTransformerNVPConfig(
        atom_embedding_dim=4,
        latent_mlp_hidden_dims=[8, 8],
        num_coupling_layers=2,
        num_transformer_layers=2,
        encoder_layer_config=custom_attention_encoder_config,
    )
    model = custom_transformer_nvp_constructor(kernel_transformer_nvp_config)
    return model


def get_learnable_kernel_transformer_nvp():
    custom_attention_encoder_config = CustomAttentionEncoderLayerConfig(
        d_model=4,
        dim_feedforward=8,
        dropout=0.0,
        num_heads=4,
        attention_type="learnable_kernel",
        lengthscales=[0.1, 0.2, 0.5, 1.0],
        max_radius=None,  # Only needed for local attention
        normalise_kernel_values=False,
    )
    kernel_transformer_nvp_config = CustomAttentionTransformerNVPConfig(
        atom_embedding_dim=4,
        latent_mlp_hidden_dims=[8, 8],
        num_coupling_layers=2,
        num_transformer_layers=2,
        encoder_layer_config=custom_attention_encoder_config,
    )
    model = custom_transformer_nvp_constructor(kernel_transformer_nvp_config)
    return model


def get_equivariant_nvp():
    equivariant_nvp_config = EquivariantNVPConfig(
        atom_embedding_dim=4, num_coupling_layers=2, latent_mlp_hidden_dims=[8]
    )
    model = equivariant_nvp_constructor(equivariant_nvp_config)
    return model


@pytest.mark.parametrize(
    "model_maker",
    [
        get_transformer_nvp,
        get_local_transformer_nvp,
        get_kernel_transformer_nvp,
        get_learnable_kernel_transformer_nvp,
        get_equivariant_nvp,
    ],
)
def test_batch_equal(model_maker, device: torch.device):
    set_seed(0)

    model = model_maker().to(device)

    batch_size = 4
    # Create dataset.
    datapath = os.path.join(os.path.dirname(__file__), "../testdata/output/")
    dataset = TrajectoryIterableDataset(
        data_dir=Path(datapath).expanduser().resolve(), step_width=1, shuffle=True
    )
    batch_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
    )

    with torch.no_grad():
        for _, batch in enumerate(batch_dataloader):
            # Forward pass in batch mode.
            # print(f'Num atoms: {(~ batch.masked_elements).sum(-1)}')

            log_likelihood = get_log_likelihood(model, batch=batch, device=device)  # [B]

            # Forward passes done one by one in a for loop
            for_loop_log_likelihood = torch.zeros_like(log_likelihood)
            # Construct batch via for loop with batch size 1.
            for i in range(batch.atom_types.shape[0]):
                single_point_batch = get_single_datapoint(i, batch)
                single_log_likelihood = get_log_likelihood(
                    model, batch=single_point_batch, device=device
                )  # [1]
                for_loop_log_likelihood[i] = single_log_likelihood

            assert torch.allclose(log_likelihood, for_loop_log_likelihood, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_batch_equal(get_transformer_nvp())
    test_batch_equal(get_equivariant_nvp())
