import os

from typing import Optional, Sequence
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from utilities import unique_item, find_files, StrPath
from timewarp.model_configs import ModelConfig
from timewarp.training_config import TrainingConfig

DEFAULT_DEPRECATED_FIELDS = ["local_rank"]


def finalize_config(omega_config: DictConfig) -> TrainingConfig:
    """
    Creates an instance of TrainingConfig applying the logic in TrainingConfig.__post_init__.

    Args:
        omega_config: omegaconf's DictConfig.

    Returns:
        an instance of TrainingConfig.
    """
    config = OmegaConf.to_object(omega_config)
    assert isinstance(config, TrainingConfig)
    return config


def load_config(
    config_path: StrPath, deprecated_fields: Optional[Sequence[str]] = None
) -> DictConfig:
    """Load config from file.

    Args:
        path: location of the YAML file.
        deprecated_fields: (optional) list of fields to ignore when loading.

    Returns:
        config object, containing only the information stored in the YAML file (and default values)
    """
    deprecated_fields = deprecated_fields or DEFAULT_DEPRECATED_FIELDS
    schema = OmegaConf.structured(TrainingConfig)
    conf_yaml = OmegaConf.load(Path(config_path))
    # delete deprecated fields
    assert isinstance(conf_yaml, DictConfig)
    for key in deprecated_fields:
        if key in conf_yaml:
            del conf_yaml[key]
    config = OmegaConf.merge(schema, conf_yaml)
    assert isinstance(config, DictConfig)
    return config


def determine_config_path(path: StrPath) -> StrPath:
    if os.path.isfile(path):
        config_path = path
    else:
        config_path = unique_item(list(find_files(path, "config.yaml")))
        print(f"Found config.yaml file in {config_path}")

    return config_path


def load_config_dict_in_subdir(path: StrPath) -> DictConfig:
    """
    Loads config from a subdirectory of a path.

    Args:
        path: path to a directory to find config.yaml file.

    Note: returns finalized TrainingConfig.
    """
    return load_config(determine_config_path(path))


def load_config_in_subdir(path: StrPath) -> TrainingConfig:
    """
    Loads config from a subdirectory of a path.

    Args:
        path: path to a directory to find config.yaml file.

    Note: returns finalized TrainingConfig.
    """
    return finalize_config(load_config_dict_in_subdir(path))


def load_model_config(config: TrainingConfig) -> ModelConfig:
    """
    Loads model config from a file if config.saved_model_path is provided, otherwise return the one given in config.

    Args:
        config: TrainingConfig
    Returns:
        ModelConfig
    """
    if config.saved_model_path is not None:
        return load_config_in_subdir(config.saved_model_path).model_config
    else:
        return config.model_config


def check_saved_config(config: TrainingConfig):
    if config.saved_model_path is None:
        # Nothing to do
        return
    saved_config = load_config_in_subdir(config.saved_model_path)
    assert saved_config.step_width == config.step_width
