import os
from typing import Callable
from pathlib import Path
import pytest

import torch
from torch.utils.data.dataloader import DataLoader


from timewarp.utils.loss_utils import get_log_likelihood
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.datasets import TrajectoryIterableDataset
from timewarp.equivariance.equivariance_transforms import (
    RandomRotation,
    RandomTranslation,
    BaseDataTransformation,
    transform_batch,
)
from utilities.training_utils import set_seed

from .assets import get_model_config


@pytest.mark.parametrize(
    "model_name, equivariance_transform_factory",
    [
        ("transformer_nvp", RandomTranslation()),
        ("equivariant_nvp", RandomTranslation()),
        ("equivariant_nvp", RandomRotation()),
    ],
)
def test_distributional_equivariance(
    model_name: str,
    equivariance_transform_factory: Callable[[int], BaseDataTransformation],
    device: torch.device,
):
    """Test that p(Ty|Tx) = p(y|x) for symmetry actions T."""
    set_seed(0)
    batch_size = 4
    # Create dataset.
    datapath = os.path.join(os.path.dirname(__file__), "../testdata/output/")
    dataset = TrajectoryIterableDataset(
        data_dir=Path(datapath).expanduser().resolve(), step_width=1, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
    )

    # Compute log-likelihood without transformation
    model, _ = get_model_config(model_name)
    model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                log_likelihood = get_log_likelihood(model, batch=batch, device=device)  # [B]

            # Compute log-likelihood with transformation
            transformed_batch = transform_batch(batch, transform=equivariance_transform_factory)
            with torch.no_grad():
                transformed_log_likelihood = get_log_likelihood(
                    model, batch=transformed_batch, device=device
                )  # [B]

            assert torch.allclose(log_likelihood, transformed_log_likelihood, rtol=1e-4, atol=1e-4)
