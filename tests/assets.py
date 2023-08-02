from inspect import signature
from typing import Tuple
import torch

from timewarp.training_config import TrainingConfig
from timewarp.model_configs import (
    ModelConfig,
    TransformerNVPConfig,
    TransformerConfig,
    EquivariantNVPConfig,
    TransformerCVAEConfig,
)
from timewarp.model_constructor import model_constructor


def get_training_config(**kwargs):
    """
    Constructs a TrainingConfig using None as a default default value.
    """
    sig = signature(TrainingConfig)  # includes both positional and keyword arguments.
    return TrainingConfig(
        **{key: kwargs.get(key, param.default) for key, param in sig.parameters.items()}
    )


# --- Model constructors for the *density* models to be tested ---


def get_transformer_nvp_config(**kwargs) -> TrainingConfig:
    transformer_config = TransformerConfig()  # Use defaults.
    model_config = ModelConfig(
        model_type="transformer_nvp",
        transformer_nvp_config=TransformerNVPConfig(
            atom_embedding_dim=4,
            transformer_hidden_dim=8,
            latent_mlp_hidden_dims=[8, 8],
            num_coupling_layers=2,
            num_transformer_layers=2,
            transformer_config=transformer_config,
        ),
    )
    return get_training_config(model_config=model_config, **kwargs)


def get_equivariant_nvp_config(**kwargs) -> TrainingConfig:
    model_config = ModelConfig(
        model_type="equivariant_nvp",
        equivariant_nvp_config=EquivariantNVPConfig(
            atom_embedding_dim=4, num_coupling_layers=2, latent_mlp_hidden_dims=[8]
        ),
    )
    return get_training_config(model_config=model_config, **kwargs)


def get_transformer_cvae_config(**kwargs) -> TrainingConfig:
    transformer_config = TransformerConfig()  # Use defaults.
    model_config = ModelConfig(
        model_type="transformer_cvae",
        transformer_cvae_config=TransformerCVAEConfig(
            atom_embedding_dim=4,
            transformer_hidden_dim=8,
            latent_mlp_hidden_dims=[8, 8],
            num_transformer_layers=2,
            latent_cvae_dim=3,
            num_elbo_samples=1,
            elbo_estimator="elbo",
            transformer_config=transformer_config,
        ),
    )
    return get_training_config(model_config=model_config, **kwargs)


def get_model_config(model_name: str, **kwargs) -> Tuple[torch.nn.Module, TrainingConfig]:
    config = models[model_name](**kwargs)
    return model_constructor(config.model_config), config


models = {
    "transformer_nvp": get_transformer_nvp_config,
    "equivariant_nvp": get_equivariant_nvp_config,
    "transformer_cvae": get_transformer_cvae_config,
}
