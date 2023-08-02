import os
from tempfile import TemporaryDirectory

import pytest
import torch

from utilities.model_utils import save_model
from timewarp.utils.training_utils import (
    load_model,
    load_or_construct_optimizer_lr_scheduler,
)
from timewarp.losses import LossWrapper, NegativeLogLikelihoodLoss, unwrap_loss_wrapper
from .assets import models, get_model_config, get_training_config


def check_module_params(module_1: torch.nn.Module, module_2: torch.nn.Module):
    module_2_params = dict(module_2.named_parameters())
    for name, param in module_1.named_parameters():
        assert name in module_2_params
        torch.testing.assert_allclose(module_2_params[name], param, rtol=0, atol=0)


@pytest.mark.parametrize("model_name", list(models.keys()))
def test_load_save(model_name: str, device: torch.device):
    model, config = get_model_config(
        model_name, learning_rate=1e-4, warmup_steps=1000, weight_decay=0.0
    )
    optimizer, lr_scheduler = load_or_construct_optimizer_lr_scheduler(model, config)
    with TemporaryDirectory() as dirname:
        save_model(
            os.path.join(dirname, "best_model.pt"),
            model,
            optimizer,
            lr_scheduler,
            training_config=config,
        )

        model2_wrapped = load_model(dirname)
        assert model2_wrapped.loss is None
        model2 = unwrap_loss_wrapper(model2_wrapped)
        model2_params = dict(model2.named_parameters())
        for name, param in model.named_parameters():
            assert name in model2_params
            torch.testing.assert_allclose(model2_params[name], param, rtol=0, atol=0)

        optimizer2, lr_scheduler2 = load_or_construct_optimizer_lr_scheduler(
            model2, get_training_config(saved_model_path=dirname)
        )
        assert optimizer.state_dict() == optimizer2.state_dict()
        assert lr_scheduler.state_dict() == lr_scheduler2.state_dict()

    # Wrapped in `LossWrapper`
    loss = NegativeLogLikelihoodLoss()
    model_wrapped = LossWrapper(module=model, loss=loss)
    with TemporaryDirectory() as dirname:
        save_model(
            os.path.join(dirname, "best_model.pt"),
            model_wrapped,
            optimizer,
            lr_scheduler,
            training_config=config,
        )

        model2_wrapped = load_model(dirname)
        assert model2_wrapped.loss is None
        # Check model parameters ignoring the loss.
        check_module_params(unwrap_loss_wrapper(model_wrapped), unwrap_loss_wrapper(model2_wrapped))
