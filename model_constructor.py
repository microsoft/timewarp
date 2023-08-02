from dataclasses import asdict
from typing import List

import torch.nn as nn
from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel
from timewarp.modules.layers.rff_position_encoder import RFFPositionEncoderConfig

from timewarp.modules.transformer_nvp import TransformerCouplingLayer
from timewarp.modules.custom_transformer_nvp import (
    CustomAttentionTransformerCouplingLayer,
)
from timewarp.modules.dense_equivariant_nvp import EquivariantCouplingLayer
from timewarp.modules.layers.dense_equivariant_coupling_layer import (
    DenseEquivariantCoordShiftModule,
    DenseInvariantCoordScaleModule,
    DenseEquivariantVelocShiftModule,
    DenseInvariantVelocScaleModule,
)
from timewarp.modules.model_wrappers.conditional_vae import ConditionalVAEModel
from timewarp.modules.model_wrappers.flow import (
    ConditionalSequentialFlow,
    ConditionalFlowDensityModel,
    make_conditional_flow_density,
)
from timewarp.modules.model_wrappers.gaussian_density import GaussianDensityModel
from timewarp.modules.layers.custom_attention_encoder import (
    custom_attention_transformer_encoder_constructor,
)
from timewarp.modules.layers.kernel_attention import compute_kernel_attention_scores
from timewarp.modules.model_wrappers.point_estimate import PointEstimateWrapper
from timewarp.modules.transformer_gaussian_density import TransformerMeanLogScaleModel
from timewarp.modules.baselines import (
    InitialStateGaussian,
    LearnableLinearGaussian,
    EulerMaruyamaGaussian,
    InitialStatePointEstimate,
)
from timewarp.model_configs import (
    ModelConfig,
    TransformerNVPConfig,
    TransformerCVAEConfig,
    CustomAttentionTransformerNVPConfig,
    GaussianDensityTransformerConfig,
    EquivariantNVPConfig,
)
from timewarp.dataloader import ELEMENT_VOCAB
from utilities.cache import Cache
from utilities.common import Returns


def model_constructor(config: ModelConfig) -> nn.Module:
    if config.model_type == "transformer_nvp":
        assert config.transformer_nvp_config
        return transformer_nvp_constructor(config.transformer_nvp_config)
    elif config.model_type == "custom_attention_transformer_nvp":
        assert config.custom_transformer_nvp_config
        return custom_transformer_nvp_constructor(config.custom_transformer_nvp_config)
    elif config.model_type == "equivariant_nvp":
        assert config.equivariant_nvp_config
        return equivariant_nvp_constructor(config.equivariant_nvp_config)
    elif config.model_type == "transformer_cvae":
        assert config.transformer_cvae_config
        return transformer_cvae_constructor(config.transformer_cvae_config)
    elif config.model_type == "gaussian_density_transformer":
        assert config.gaussian_density_transformer_config
        return gaussian_density_transformer_constructor(config.gaussian_density_transformer_config)
    elif config.model_type == "initial_state_gaussian":
        return InitialStateGaussian()
    elif config.model_type == "learnable_linear_gaussian":
        return LearnableLinearGaussian()
    elif config.model_type == "euler_maruyama_gaussian":
        return EulerMaruyamaGaussian()
    elif config.model_type == "initial_state_point_estimate":
        return PointEstimateWrapper(model=InitialStatePointEstimate())
    else:
        raise NotImplementedError(f"{config.model_type} is not a recognised model.")


def transformer_cvae_constructor(
    transformer_cvae_config: TransformerCVAEConfig,
) -> ConditionalDensityModel:
    assert (
        transformer_cvae_config.num_elbo_samples >= 1
    ), "ELBO samples (num_elbo_samples) needs to be >= 1"
    assert transformer_cvae_config.elbo_estimator in {
        "elbo",
        "iwae",
    }, "ELBO estimator (elbo_estimator) must be one of 'elbo', 'iwae', 'iwae-dreg'."

    model = ConditionalVAEModel(
        atom_embedding_dim=transformer_cvae_config.atom_embedding_dim,
        latent_cvae_dim=transformer_cvae_config.latent_cvae_dim,
        num_elbo_samples=transformer_cvae_config.num_elbo_samples,
        elbo_estimator=transformer_cvae_config.elbo_estimator,
        transformer_hidden_dim=transformer_cvae_config.transformer_hidden_dim,
        num_transformer_layers=transformer_cvae_config.num_transformer_layers,
        latent_mlp_hidden_dims=transformer_cvae_config.latent_mlp_hidden_dims,
        transformer_config=transformer_cvae_config.transformer_config,
    )
    return model


def equivariant_nvp_constructor(
    equivariant_nvp_config: EquivariantNVPConfig,
) -> ConditionalFlowDensityModel:
    assert (
        equivariant_nvp_config.num_coupling_layers % 2 == 0
    ), "Real NVP should have an even number of coupling layers"

    assert equivariant_nvp_config.latent_mlp_hidden_dims is not None

    position_mod_index = equivariant_nvp_config.position_layer_index_mod_2
    assert (
        position_mod_index == 0 or position_mod_index == 1
    ), "positions_layer_index can only be 0 or 1"

    def make(cls):
        return cls(
            input_features_dim=equivariant_nvp_config.atom_embedding_dim,
            output_features_dim=equivariant_nvp_config.atom_embedding_dim,
            hidden_layers_dims=equivariant_nvp_config.latent_mlp_hidden_dims,
        )

    layers = [
        # Transform positions
        EquivariantCouplingLayer(
            shift_module=make(DenseEquivariantCoordShiftModule),
            scale_module=make(DenseInvariantCoordScaleModule),
            transformed_vars="positions",
        )
        if layer_idx % 2 == position_mod_index
        else
        # Transform velocities
        EquivariantCouplingLayer(
            shift_module=make(DenseEquivariantVelocShiftModule),
            scale_module=make(DenseInvariantVelocScaleModule),
            transformed_vars="velocities",
        )
        for layer_idx in range(equivariant_nvp_config.num_coupling_layers)
    ]

    atom_embedder = nn.Embedding(
        num_embeddings=len(ELEMENT_VOCAB),
        embedding_dim=equivariant_nvp_config.atom_embedding_dim,
    )
    flow = ConditionalSequentialFlow(layers=layers, atom_embedder=atom_embedder)
    model = make_conditional_flow_density(
        equivariant_nvp_config.conditional_flow_density, flow=flow
    )
    return model


def custom_transformer_nvp_constructor(
    config: CustomAttentionTransformerNVPConfig,
) -> ConditionalFlowDensityModel:
    assert (
        config.num_coupling_layers % 2 == 0
    ), "Real NVP should have an even number of coupling layers"

    position_mod_index = config.position_layer_index_mod_2
    assert (
        position_mod_index == 0 or position_mod_index == 1
    ), "positions_layer_index can only be 0 or 1"

    coupling_layers = [
        CustomAttentionTransformerCouplingLayer(
            atom_embedding_dim=config.atom_embedding_dim,
            mlp_hidden_layer_dims=config.latent_mlp_hidden_dims,
            transformed_vars="positions" if layer_idx % 2 == position_mod_index else "velocities",
            scale_transformer_encoder_layers=[
                custom_attention_transformer_encoder_constructor(config.encoder_layer_config)
                for _ in range(config.num_transformer_layers)
            ],
            shift_transformer_encoder_layers=[
                custom_attention_transformer_encoder_constructor(config.encoder_layer_config)
                for _ in range(config.num_transformer_layers)
            ],
            separate_scales_per_dimension=True,  # Non-rotation equivariant model.
        )
        for layer_idx in range(config.num_coupling_layers)
    ]

    atom_embedder = nn.Embedding(
        num_embeddings=len(ELEMENT_VOCAB),
        embedding_dim=config.atom_embedding_dim,
    )
    flow = ConditionalSequentialFlow(layers=coupling_layers, atom_embedder=atom_embedder)
    # TODO : Should we make `cacheable` part of the `config`?
    model = make_conditional_flow_density(
        config.conditional_flow_density,
        flow=flow,
        cache=Cache(
            cacheable={compute_kernel_attention_scores},
            keyword_transforms={compute_kernel_attention_scores: {"lengthscales": Returns(0)}},
        ),
    )
    return model


def transformer_nvp_constructor(
    transformer_nvp_config: TransformerNVPConfig,
) -> ConditionalFlowDensityModel:
    assert (
        transformer_nvp_config.num_coupling_layers % 2 == 0
    ), "Real NVP should have an even number of coupling layers"

    position_mod_index = transformer_nvp_config.position_layer_index_mod_2
    assert (
        position_mod_index == 0 or position_mod_index == 1
    ), "positions_layer_index can only be 0 or 1"

    # Default: no positional encoding
    rff_position_encoder_config = RFFPositionEncoderConfig(0, 1.0, 1.0)
    if transformer_nvp_config.rff_position_encoder_config is not None:
        rff_position_encoder_config = transformer_nvp_config.rff_position_encoder_config

    layers: List[TransformerCouplingLayer] = [
        TransformerCouplingLayer(
            atom_embedding_dim=transformer_nvp_config.atom_embedding_dim,
            transformer_hidden_dim=transformer_nvp_config.transformer_hidden_dim,
            mlp_hidden_layer_dims=transformer_nvp_config.latent_mlp_hidden_dims,
            num_transformer_layers=transformer_nvp_config.num_transformer_layers,
            transformed_vars="positions" if layer_idx % 2 == position_mod_index else "velocities",
            transformer_config=transformer_nvp_config.transformer_config,
            rff_position_encoder_config=rff_position_encoder_config,
        )
        for layer_idx in range(transformer_nvp_config.num_coupling_layers)
    ]

    atom_embedder = nn.Embedding(
        num_embeddings=len(ELEMENT_VOCAB),
        embedding_dim=transformer_nvp_config.atom_embedding_dim,
    )
    flow = ConditionalSequentialFlow(layers=layers, atom_embedder=atom_embedder)
    model = make_conditional_flow_density(
        transformer_nvp_config.conditional_flow_density, flow=flow
    )
    return model


def gaussian_density_transformer_constructor(
    gaussian_transformer_config: GaussianDensityTransformerConfig,
) -> GaussianDensityModel:
    mean_log_scale_model = TransformerMeanLogScaleModel(
        atom_embedding_dim=gaussian_transformer_config.atom_embedding_dim,
        hidden_dim=gaussian_transformer_config.latent_dim,
        num_transformer_layers=gaussian_transformer_config.num_transformer_layers,
        mlp_hidden_layers_dims=gaussian_transformer_config.latent_mlp_hidden_dims,
        transformer_config=gaussian_transformer_config.transformer_config,
    )
    density_model = GaussianDensityModel(
        mean_log_scale_model=mean_log_scale_model,
    )
    return density_model
