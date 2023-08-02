import pytest

import os
from pathlib import Path
from functools import singledispatch

import torch
import torch.nn.functional as F

from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.datasets.iterable_datasets import TrajectoryIterableDataset
from timewarp.losses import (
    AbstractLoss,
    AcceptanceLoss,
    ConvexCombinationLoss,
    EnergyLoss,
    GeometricLossSchedule,
    NegativeLogLikelihoodLoss,
    get_loss,
    loss_schedule_step,
)
from torch.utils.data.dataloader import DataLoader
from timewarp.utils.openmm.openmm_provider import OpenMMProvider
from timewarp.tests.test_batching import get_single_datapoint, get_transformer_nvp
from utilities.training_utils import set_seed


@singledispatch
def add_openmm_provider_maybe(loss: AbstractLoss, pdb_dirs, device):
    return loss


@add_openmm_provider_maybe.register
def _(loss: EnergyLoss, pdb_dirs, device):
    loss.openmm_provider = OpenMMProvider(pdb_dirs=pdb_dirs, device=device)
    return loss


@add_openmm_provider_maybe.register
def _(loss: AcceptanceLoss, pdb_dirs, device):
    loss.openmm_provider = OpenMMProvider(pdb_dirs=pdb_dirs, device=device)
    return loss


@add_openmm_provider_maybe.register
def _(loss: ConvexCombinationLoss, pdb_dirs, device):
    loss.losses = [add_openmm_provider_maybe(l, pdb_dirs, device) for l in loss.losses]
    return loss


def _get_batch_dataloader(batch_size=4):
    # Create dataset.
    batch_size = 4
    datapath = (
        Path(os.path.join(os.path.dirname(__file__), "..", "testdata", "output"))
        .expanduser()
        .resolve()
    )
    dataset = TrajectoryIterableDataset(data_dir=datapath, step_width=1, shuffle=True)
    batch_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=moldyn_dense_collate_fn,  # type: ignore
        pin_memory=True,
    )

    return batch_dataloader, datapath


@pytest.mark.parametrize(
    "loss_computer",
    [
        NegativeLogLikelihoodLoss(random_velocs=False),
    ],
)
@pytest.mark.parametrize(
    "model_constructor",
    [lambda: get_transformer_nvp()],  # Use `lambda` to avoid global allocation + same random init
)
def test_losses_batch(loss_computer, model_constructor, device: torch.device):
    set_seed(0)
    model = model_constructor().to(device)

    # Create dataset.
    batch_dataloader, datapath = _get_batch_dataloader()

    # Add the `OpenMMProvider` if needed.
    loss_computer = add_openmm_provider_maybe(loss_computer, datapath, device)

    with torch.no_grad():
        for _, batch in enumerate(batch_dataloader):
            loss = get_loss(loss_computer, model, batch, device=device)
            actual_batch_size = batch.atom_coords.size(0)

            # Forward passes done one by one in a for loop
            for_loop_loss = torch.zeros_like(loss)
            # Construct batch via for loop with batch size 1.
            for i in range(actual_batch_size):
                single_point_batch = get_single_datapoint(i, batch)
                single_loss = get_loss(loss_computer, model, single_point_batch, device=device)
                for_loop_loss += single_loss / actual_batch_size

            assert torch.allclose(loss, for_loop_loss, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "model_constructor",
    [lambda: get_transformer_nvp()],  # Use `lambda` to avoid global allocation + same random init
)
def test_convex_combination_loss(model_constructor, device: torch.device):
    set_seed(0)
    model = model_constructor().to(device)

    loss_computer_same = ConvexCombinationLoss(
        [
            NegativeLogLikelihoodLoss(random_velocs=False),
            NegativeLogLikelihoodLoss(random_velocs=False),
        ],
        torch.tensor([1 / 2, 1 / 2]),
    ).to(device)

    loss_computer_different = ConvexCombinationLoss(
        [
            NegativeLogLikelihoodLoss(random_velocs=False),
            NegativeLogLikelihoodLoss(random_velocs=True),
        ],
        torch.tensor([1 / 2, 1 / 2]),
    ).to(device)

    # Create dataset.
    batch_dataloader, _ = _get_batch_dataloader()
    with torch.no_grad():
        batch = next(iter(batch_dataloader))
        loss = get_loss(loss_computer_same, model, batch, device=device)
        loss_ = get_loss(next(iter(loss_computer_same.losses)), model, batch, device=device)
        assert torch.allclose(loss, loss_, rtol=1e-4, atol=1e-4)

        loss = get_loss(loss_computer_different, model, batch, device=device)
        loss_ = get_loss(next(iter(loss_computer_different.losses)), model, batch, device=device)
        assert (loss - loss_).norm() > 0.1


@pytest.mark.parametrize(
    "model_constructor",
    [lambda: get_transformer_nvp()],  # Use `lambda` to avoid global allocation + same random init
)
def test_energy_loss(model_constructor, device: torch.device):
    set_seed(0)
    model = model_constructor().to(device)
    batch_dataloader, datapath = _get_batch_dataloader()

    loss_computer = EnergyLoss(
        openmm_provider=OpenMMProvider(datapath, device=device, cache_size=8),
        random_velocs=False,
        num_samples=1,
    )
    loss_computer_rv = EnergyLoss(
        openmm_provider=OpenMMProvider(datapath, device=device, cache_size=8),
        random_velocs=True,
        num_samples=1,
    )
    with torch.no_grad():
        batch = next(iter(batch_dataloader))
        loss1 = get_loss(loss_computer, model, batch, device=device)
        loss2 = get_loss(loss_computer, model, batch, device=device)

        # This loss is stochastic, so we want difference between evaluations.
        assert (loss1 - loss2).norm() > 0.1

        # But if we set the seed, it should be the same.
        set_seed(42)
        loss1 = get_loss(loss_computer, model, batch, device=device)
        set_seed(42)
        loss2 = get_loss(loss_computer, model, batch, device=device)
        assert torch.allclose(loss1, loss2, atol=1e-6, rtol=1e-6)

        # Comparing with `random_velocs=True` should give different result
        # even with same random seed.
        set_seed(42)
        loss_rv = get_loss(loss_computer_rv, model, batch, device=device)
        assert (loss1 - loss_rv).norm() > 0.1


@pytest.mark.parametrize(
    "model_constructor",
    [lambda: get_transformer_nvp()],  # Use `lambda` to avoid global allocation + same random init
)
def test_acceptance_loss(model_constructor, device: torch.device):
    set_seed(0)
    model = model_constructor().to(device)
    batch_dataloader, datapath = _get_batch_dataloader()

    loss_computer = AcceptanceLoss(
        openmm_provider=OpenMMProvider(datapath, device=device, cache_size=8),
        random_velocs=False,
        num_samples=1,
        beta=0.2,
    )
    loss_computer_rv = AcceptanceLoss(
        openmm_provider=OpenMMProvider(datapath, device=device, cache_size=8),
        random_velocs=True,
        num_samples=1,
        beta=0.2,
    )
    with torch.no_grad():
        batch = next(iter(batch_dataloader))
        loss1 = get_loss(loss_computer, model, batch, device=device)
        loss2 = get_loss(loss_computer, model, batch, device=device)

        # This loss is stochastic, so we want difference between evaluations.
        assert (loss1 - loss2).norm() > 0.1

        # But if we set the seed, it should be the same.
        set_seed(42)
        loss1 = get_loss(loss_computer, model, batch, device=device)
        set_seed(42)
        loss2 = get_loss(loss_computer, model, batch, device=device)
        assert torch.allclose(loss1, loss2, atol=1e-6, rtol=1e-6)

        # Comparing with `random_velocs=True` should give different result
        # even with same random seed.
        set_seed(42)
        loss_rv = get_loss(loss_computer_rv, model, batch, device=device)
        assert (loss1 - loss_rv).norm() > 0.1


@pytest.mark.parametrize(
    "model_constructor",
    [lambda: get_transformer_nvp()],  # Use `lambda` to avoid global allocation + same random init
)
def test_acceptance_loss_high_energies(model_constructor, device: torch.device):
    set_seed(0)
    model = model_constructor().to(device)
    batch_dataloader, datapath = _get_batch_dataloader()

    loss_computer = AcceptanceLoss(
        openmm_provider=OpenMMProvider(datapath, device=device, cache_size=8),
        random_velocs=False,
        num_samples=1,
        beta=0.2,
        high_energy_threshold=200,
    )
    with torch.no_grad():
        batch = next(iter(batch_dataloader))
        loss = get_loss(loss_computer, model, batch, device=device)

        # This loss is stochastic, so we want difference between evaluations.
        assert loss == 10000


@pytest.mark.parametrize("every", [3, 5, 10])
@pytest.mark.parametrize("factor", [0.5, 0.75, 0.9])
def test_loss_schedule(every, factor, device: torch.device):
    # First has higher weighting than the other.
    pre_softmax_weights = torch.tensor([2.0, 1.0], device=device)
    weights = F.softmax(pre_softmax_weights, 0)
    loss_computer = ConvexCombinationLoss(
        [
            NegativeLogLikelihoodLoss(random_velocs=False),
            NegativeLogLikelihoodLoss(random_velocs=False),
        ],
        pre_softmax_weights=torch.clone(pre_softmax_weights),
    ).to(device)

    loss_scheduler = GeometricLossSchedule(every=every, factor=factor)
    for idx in range(1, 50 * every + 1):
        loss_schedule_step(loss_scheduler, loss_computer, idx)
        if idx % every == 0:
            # Update ground-truth weights.
            pre_softmax_weights *= factor
            weights = F.softmax(pre_softmax_weights, 0)
            assert torch.allclose(weights, loss_computer.weights)
        else:
            assert torch.allclose(weights, loss_computer.weights)

    # Should converge to (approximately) uniform weighting.
    assert torch.allclose(loss_computer.weights, torch.tensor([0.5, 0.5], device=device), atol=1e-2)


@pytest.mark.parametrize("every", [3, 5, 10])
@pytest.mark.parametrize("factor", [torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0])])
def test_loss_schedule_tensor_factor(every, factor, device: torch.device):
    factor = factor.to(device)
    # First has higher weighting than the other.
    pre_softmax_weights = torch.tensor([2.0, 1.0], device=device)
    previous_weights = F.softmax(pre_softmax_weights, 0)
    loss_computer = ConvexCombinationLoss(
        [
            NegativeLogLikelihoodLoss(random_velocs=False),
            NegativeLogLikelihoodLoss(random_velocs=False),
        ],
        pre_softmax_weights=torch.clone(pre_softmax_weights),
    ).to(device)

    loss_scheduler = GeometricLossSchedule(every=every, factor=factor, maximum=30.0)
    # Take a couple of steps which should change the weights, and make sure that
    # the values are indeed different.
    initial_num_steps_with_change = 3
    for idx in range(1, initial_num_steps_with_change * every + 1):
        loss_schedule_step(loss_scheduler, loss_computer, idx)
        if idx % every == 0:
            # Weights should have changed.
            assert not torch.allclose(previous_weights, loss_computer.weights)
        else:
            assert torch.allclose(previous_weights, loss_computer.weights)

        # Keep track of previous weights so we can check that it's changing.
        previous_weights = torch.clone(loss_computer.weights)

    # Take another 100 steps with changes so we get close to the asymptotic limit.
    num_steps_with_change = 100
    for idx in range(
        initial_num_steps_with_change * every + 1,
        (num_steps_with_change + initial_num_steps_with_change) * every + 1,
    ):
        loss_schedule_step(loss_scheduler, loss_computer, idx)

    # Should have (approximately) converged to asymptotic limit.
    asymp_limit = (factor > 1).to(dtype=loss_computer.weights.dtype)
    assert torch.allclose(loss_computer.weights, asymp_limit, atol=1e-2)
