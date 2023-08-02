from contextlib import contextmanager
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from timewarp.loss_configs import LossConfig
from timewarp.losses import AbstractLoss, GeometricLossSchedule

from timewarp.model_configs import ModelConfig
from timewarp.training_config import TrainingConfig
from timewarp.utils.config_utils import load_config, finalize_config, load_config_in_subdir
from timewarp.utils.training_utils import (
    load_or_construct_loss,
    load_or_construct_loss_scheduler,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@contextmanager
def temporary_config_file(config_dict):
    with TemporaryDirectory() as tempdir:
        config_file = Path(tempdir) / "config.yaml"
        with open(config_file, "w") as f:
            f.write(OmegaConf.to_yaml(config_dict))

        yield config_file


def test_deprecated_fields_are_ignored():
    original_config = load_config(CONFIG_DIR / "kernel_transformer_nvp.yaml")
    d = OmegaConf.to_container(original_config)
    d["local_rank"] = -1  # set a deprecated field
    with temporary_config_file(d) as config_file:
        config = load_config(config_file)
        assert config == original_config


def test_unknown_field_raises_error():
    d = OmegaConf.to_container(load_config(CONFIG_DIR / "kernel_transformer_nvp.yaml"))
    d["foo"] = -1  # set an unknown field
    with temporary_config_file(d) as config_file:
        with pytest.raises(ConfigKeyError, match="Key 'foo' not in 'TrainingConfig'"):
            load_config(config_file)


def test_default_applied():
    config = load_config(CONFIG_DIR / "kernel_transformer_nvp.yaml")
    assert config.clip_grad_norm is None


@pytest.mark.parametrize(
    "config_name",
    [
        "equivariant_nvp",
        "gaussian_baseline",
        "gaussian_density_transformer",
        "kernel_transformer_nvp",
        "local_transformer_nvp",
        "transformer_nvp",
        "transformer_cvae",
    ],
)
def test_finalize_config(config_name: str):
    original_config = load_config(CONFIG_DIR / f"{config_name}.yaml")
    config = finalize_config(original_config)
    assert isinstance(config, TrainingConfig)
    assert isinstance(config.model_config, ModelConfig)
    assert isinstance(config.loss, LossConfig)


@pytest.mark.parametrize(
    "config_name",
    [
        "equivariant_nvp",
        "gaussian_baseline",
        "gaussian_density_transformer",
        "kernel_transformer_nvp",
        "local_transformer_nvp",
        "transformer_nvp",
        "transformer_cvae",
    ],
)
def test_load_loss(config_name: str):
    original_config = load_config(CONFIG_DIR / f"{config_name}.yaml")
    config = finalize_config(original_config)

    loss = load_or_construct_loss(config)
    assert isinstance(loss, AbstractLoss)

    if config.loss_schedule is not None:
        loss_schedule = load_or_construct_loss_scheduler(config)
        assert isinstance(loss_schedule, GeometricLossSchedule)
        assert loss_schedule.every == config.loss_schedule.every


def test_load_config_in_subdir_finalized():
    original_config = load_config(CONFIG_DIR / "kernel_transformer_nvp.yaml")
    d = OmegaConf.to_container(original_config)
    with temporary_config_file(d) as config_file:
        config = load_config_in_subdir(config_file.parent)
        assert isinstance(config, TrainingConfig)
        assert isinstance(config.model_config, ModelConfig)
