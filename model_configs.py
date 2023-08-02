from dataclasses import dataclass
from typing import Optional, List, Literal

from timewarp.modules.layers.transformer_block import TransformerConfig
from timewarp.modules.layers.rff_position_encoder import RFFPositionEncoderConfig
from timewarp.modules.layers.custom_attention_encoder import (
    CustomAttentionEncoderLayerConfig,
)
from timewarp.modules.model_wrappers.flow import ConditionalFlowDensityConfig

# TODO : Don't nest `ConditionalFlowDensityConfig` inside of other model configs;
# instead make it a separate field in `TrainingConfig.model_config`.

ModelTypeName = Literal[
    "transformer_cvae",
    "transformer_nvp",
    "custom_attention_transformer_nvp",
    "equivariant_nvp",
    "gaussian_density_transformer",
    "initial_state_gaussian",
    "learnable_linear_gaussian",
    "euler_maruyama_gaussian",
    "initial_state_point_estimate",
]


@dataclass
class TransformerCVAEConfig:
    atom_embedding_dim: int
    transformer_hidden_dim: int  # Dimension of latent space that the sequence elements live in
    latent_mlp_hidden_dims: List[int]  # MLP that maps from physical space to latent space and back
    num_transformer_layers: int  # Number of transformer encoder layers per coupling layer
    latent_cvae_dim: int  # Number of latent CVAE code dimensions
    num_elbo_samples: int  # Number of ELBO/IWAE/IWAE-DReG samples
    elbo_estimator: str  # One of "elbo", "iwae"
    transformer_config: TransformerConfig


@dataclass
class EquivariantNVPConfig:
    atom_embedding_dim: int
    num_coupling_layers: int  # Number of coupling layers in RealNVP
    latent_mlp_hidden_dims: List[int]  # Pointwise MLPs used in the equivariant coupling layer
    position_layer_index_mod_2: int = 0
    conditional_flow_density: ConditionalFlowDensityConfig = ConditionalFlowDensityConfig()


@dataclass
class TransformerNVPConfig:
    atom_embedding_dim: int
    transformer_hidden_dim: int  # Dimension of latent space that the sequence elements live in
    latent_mlp_hidden_dims: List[int]  # MLP that maps from physical space to latent space and back
    num_coupling_layers: int  # Number of coupling layers in RealNVP
    num_transformer_layers: int  # Number of transformer encoder layers per coupling layer
    transformer_config: TransformerConfig
    rff_position_encoder_config: Optional[RFFPositionEncoderConfig] = None
    position_layer_index_mod_2: int = 0
    conditional_flow_density: ConditionalFlowDensityConfig = ConditionalFlowDensityConfig()


@dataclass
class CustomAttentionTransformerNVPConfig:
    atom_embedding_dim: int
    latent_mlp_hidden_dims: List[int]  # MLP that maps from physical space to latent space and back
    num_coupling_layers: int  # Number of coupling layers in RealNVP
    num_transformer_layers: int  # Number of transformer encoder layers per coupling layer
    encoder_layer_config: CustomAttentionEncoderLayerConfig
    position_layer_index_mod_2: int = 0
    conditional_flow_density: ConditionalFlowDensityConfig = ConditionalFlowDensityConfig()


@dataclass
class GaussianDensityTransformerConfig:
    atom_embedding_dim: int
    latent_dim: int  # Dimension of latent space that the sequence elements live in
    latent_mlp_hidden_dims: List[int]  # MLP that maps from physical space to latent space and back
    num_transformer_layers: int  # Number of transformer encoder layers
    transformer_config: TransformerConfig


@dataclass
class ModelConfig:
    model_type: str  # Should be type ModelTypeName, not support in OmegaConf atm
    transformer_cvae_config: Optional[TransformerCVAEConfig] = None
    transformer_nvp_config: Optional[TransformerNVPConfig] = None
    custom_transformer_nvp_config: Optional[CustomAttentionTransformerNVPConfig] = None
    equivariant_nvp_config: Optional[EquivariantNVPConfig] = None
    gaussian_density_transformer_config: Optional[GaussianDensityTransformerConfig] = None
