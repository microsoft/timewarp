import dataclasses
from functools import singledispatch
from pathlib import Path
from typing import Generator, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deepspeed import DeepSpeedEngine  # type: ignore [import]

from timewarp.dataloader import Batch, DenseMolDynBatch
from timewarp.loss_configs import (
    AbstractLossConfig,
    LossConfig,
    LossScheduleConfig,
    NLLAndEnergyLossConfig,
    NLLAndAcceptanceLossConfig,
    NLLConfig,
)
from timewarp.losses import (
    AbstractLoss,
    AbstractLossSchedule,
    ConvexCombinationLoss,
    EnergyLoss,
    AcceptanceLoss,
    GeometricLossSchedule,
    LossWrapper,
    NegativeLogLikelihoodLoss,
    loss_schedule_step,
)
from timewarp.training_config import TrainingConfig
from timewarp.equivariance.equivariance_transforms import transform_batch
from timewarp.utils.openmm.openmm_provider import OpenMMProvider
from timewarp.utils.deepspeed_lr_scheduler import PlateauLR
from utilities.logger import TrainingLogger

from timewarp.utils.loss_utils import get_loss
from timewarp.model_constructor import model_constructor
from utilities import delayed_reporter

from utilities.model_utils import load_checkpoint_in_subdir, load_model as _load_model
from utilities.training_utils import Checkpointer, EpochLossTracker, get_optimizer, get_lr_scheduler
from utilities import StrPath


def all_reduce_loss(loss_value: torch.Tensor, model_engine: DeepSpeedEngine):
    loss_value.div_(model_engine.dp_world_size)
    dist.all_reduce(loss_value, group=model_engine.data_parallel_group)


def end_of_epoch_report(
    desc: str,
    run_kind: Literal["train", "valid"],
    loss_tracker: EpochLossTracker,
    config: TrainingConfig,
    tb_logger: TrainingLogger,
):
    batch_size = config.valid_batch_size if run_kind == "valid" else config.batch_size
    total_loss = loss_tracker.total_loss
    throughput = loss_tracker.throughput * batch_size
    print(
        f"  {desc} {run_kind} loss: {total_loss:g} ({loss_tracker.loss_overview}) || elapsed: {loss_tracker.elapsed:.2f} s || throughput: {throughput:.2f} samples/s"
    )
    if run_kind == "valid":
        tb_logger.log_scalar(f"{run_kind}_loss", total_loss)
    tb_logger.log_scalar(f"{run_kind}_throughput", throughput)


def run_on_dataloader(
    model,
    data_loader: Generator[Batch, None, None],
    deepspeed: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    clip_grad_norm: Optional[float] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    loss_scheduler: Optional[AbstractLossSchedule] = None,
    max_num_steps: Optional[int] = None,
    quiet: bool = False,
    run_kind: Literal["train", "valid"] = "valid",
    tb_logger: Optional[TrainingLogger] = None,
    data_augmentation: bool = False,
    compute_equivariance_discrepancy: bool = False,
    device="cpu",
    dtype: torch.dtype = torch.float32,
    checkpointer: Optional[Checkpointer] = None,
    sort_batch_by_pdb_name: bool = True,
) -> EpochLossTracker:
    """Run the given model on the provided data loader.

    Args:
        model: Model to run things on.
        data_loader: Loader that provides the data we run on.
        optimizer: Optional optimizer. If present, the given model will be trained.
        lr_scheduler: Optional learning rate scheduler around optimizer.
        max_num_steps: Optional number of steps. If not provided, will run until end of data loader.
    """
    if run_kind == "valid":
        assert optimizer is None
        model.eval()
    else:
        model.train()

    with EpochLossTracker(run_kind=run_kind, quiet=quiet) as loss_tracker:
        checkpointer = checkpointer or Checkpointer()  # default Checkpointer is a no-op.
        for batch_idx, _batch in enumerate(data_loader):
            batch = _batch
            if max_num_steps is not None and batch_idx >= max_num_steps:
                # Closing the generator writes the profile to disk if we are profiling.
                data_loader.close()
                break

            if optimizer is not None:
                optimizer.zero_grad()

            if data_augmentation:
                assert isinstance(batch, DenseMolDynBatch)
                batch = transform_batch(batch, dtype=dtype)

            if sort_batch_by_pdb_name and isinstance(batch, DenseMolDynBatch):
                batch = batch.sort_by_name()

            if run_kind == "valid":
                with torch.no_grad():
                    loss = get_loss(model, batch, device=device, tb_logger=tb_logger)
            else:
                loss = get_loss(model, batch, device=device, tb_logger=tb_logger)

            # no item() here to keep the statistics on the device
            loss_tracker.log_losses(NLL=loss)

            # Training step:
            if run_kind == "train":
                if deepspeed:
                    assert isinstance(model, DeepSpeedEngine)
                    model.backward(loss)
                    if isinstance(model.lr_scheduler, PlateauLR):
                        if batch_idx % model.lr_scheduler.logging_period == 0:
                            all_reduce_loss(loss, model)  # Average loss over all devices.
                            # includes scheduler.step(), loss needed to check best iteration.
                            model.step(lr_kwargs={"loss": loss.item()})
                        else:
                            model.step()
                    else:
                        model.step()
                    optimizer = model.optimizer
                else:
                    # not deepspeed
                    assert optimizer is not None
                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    optimizer.step()

            if tb_logger is not None:
                tb_logger.log_scalar_async("loss", loss)

                if optimizer is not None:
                    tb_logger.log_scalar("learning_rate", optimizer.param_groups[0]["lr"])

                if compute_equivariance_discrepancy:
                    assert isinstance(batch, DenseMolDynBatch)
                    with torch.no_grad():
                        discrepancy = equivariance_discrepancy(
                            model, batch, num_transforms=50, device=device, tb_logger=tb_logger
                        )
                    tb_logger.log_scalar_async("equivariance_discrepancy", discrepancy)

                tb_logger.increment_step()  # Increment at every iteration.

            if lr_scheduler is not None and not isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step()

            if loss_scheduler is not None and isinstance(model, LossWrapper):
                if tb_logger is not None:
                    loss_schedule_step(loss_scheduler, model.loss, tb_logger.step, logger=tb_logger)
                else:
                    loss_schedule_step(loss_scheduler, model.loss, batch_idx)

            # Checkpoint periodically if model_save_func is specified
            checkpointer.checkpoint(batch_idx)

            # Ensure that any oustanding reporting is flushed
            # NOTE: For larger datasets waiting until the end of each epoch before flushing
            # is non-ideal, so we flush after every batch instead.
            delayed_reporter.flush_all()

    return loss_tracker


def _model_constructor(data):
    model = model_constructor(data["training_config"].model_config)
    if not isinstance(model, LossWrapper):
        # HACK : We want a consistent return-value, and so we always return a `LossWrapper`,
        # even if the loaded `model` isn't a `LossWrapper`. This makes it easier to write code that
        # works both with and without `deepspeed`.
        model = LossWrapper(module=model, loss=None)

    return model


def load_model(path: StrPath):
    return _load_model(path, _model_constructor)


def load_optimizer_lr_scheduler(
    model: nn.Module, path: StrPath
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    data = load_checkpoint_in_subdir(Path(path))
    config = data["training_config"]
    optimizer = get_optimizer(model, config)
    optimizer.load_state_dict(data["optimizer_state_dict"])
    lr_scheduler = get_lr_scheduler(optimizer, config)
    lr_scheduler.load_state_dict(data["lr_scheduler_state_dict"])
    return optimizer, lr_scheduler


def load_or_construct_model(config: TrainingConfig) -> torch.nn.Module:
    if config.saved_model_path is not None:
        return load_model(config.saved_model_path)
    return model_constructor(config.model_config)


def load_or_construct_loss(config: TrainingConfig) -> AbstractLoss:
    if config.saved_model_path is not None and config.loss is None:
        # Use the `loss` attached to the `model` if it exists and no loss
        # has been specified in the `config`.
        model = load_model(config.saved_model_path)
        if isinstance(model, LossWrapper):
            if model.loss is not None:
                loss = model.loss
                print(f"Using loss {loss} from loaded model")
                return loss

    return loss_constructor(config.loss)


@singledispatch
def loss_constructor(config: Union[AbstractLossConfig, LossConfig]) -> AbstractLoss:
    raise NotImplementedError()


@loss_constructor.register
def _(config: LossConfig) -> AbstractLoss:
    # Check that the config is valid.
    dict_config = dataclasses.asdict(config)
    potential_configs = list(k for (k, v) in dict_config.items() if v is not None)

    if len(potential_configs) == 0:
        selected_config = NLLConfig()
    else:
        # Ensure that we have a valid config.
        assert len(potential_configs) == 1, "one loss config needs to be given"
        selected_config = getattr(config, potential_configs[0])

    # Load the thingy.
    return loss_constructor(selected_config)


@loss_constructor.register
def _(config: NLLConfig) -> NegativeLogLikelihoodLoss:
    return NegativeLogLikelihoodLoss(random_velocs=config.random_velocs)


@loss_constructor.register
def _(config: NLLAndEnergyLossConfig) -> ConvexCombinationLoss:
    openmm_provider = OpenMMProvider(**dataclasses.asdict(config.openmm_provider))

    return ConvexCombinationLoss(
        losses=[
            NegativeLogLikelihoodLoss(random_velocs=config.random_velocs),
            EnergyLoss(
                openmm_provider=openmm_provider,
                random_velocs=config.random_velocs,
                num_samples=config.num_samples,
            ),
        ],
        weights=None if config.weights is None else torch.tensor(config.weights),
        pre_softmax_weights=None
        if config.pre_softmax_weights is None
        else torch.tensor(config.pre_softmax_weights),
    )


@loss_constructor.register
def _(config: NLLAndAcceptanceLossConfig) -> ConvexCombinationLoss:
    openmm_provider = OpenMMProvider(**dataclasses.asdict(config.openmm_provider))

    return ConvexCombinationLoss(
        losses=[
            NegativeLogLikelihoodLoss(random_velocs=config.random_velocs),
            AcceptanceLoss(
                openmm_provider=openmm_provider,
                random_velocs=config.random_velocs,
                beta=config.beta,
                clamp=config.clamp,
                num_samples=config.num_samples,
                high_energy_threshold=config.high_energy_threshold,
            ),
        ],
        weights=None if config.weights is None else torch.tensor(config.weights),
        pre_softmax_weights=None
        if config.pre_softmax_weights is None
        else torch.tensor(config.pre_softmax_weights),
    )


def load_or_construct_loss_scheduler(config: TrainingConfig) -> AbstractLossSchedule:
    assert config.loss_schedule is not None
    return loss_scheduler(config.loss_schedule)


def loss_scheduler(config: LossScheduleConfig) -> AbstractLossSchedule:
    return GeometricLossSchedule(
        every=config.every,
        factor=torch.tensor(config.factor),
    )


def load_or_construct_optimizer_lr_scheduler(
    model: nn.Module, config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # Attempt to load from `saved_model_path` if provided and we're NOT doing a warm start.
    if config.saved_model_path is not None and not config.warm_start:
        try:
            optimizer, lr_scheduler = load_optimizer_lr_scheduler(model, config.saved_model_path)
        except Exception as e:
            print(
                f"Failed to load optimizer and lr_scheduler due to {e}; constructing from config instead"
            )

    if optimizer is None:
        optimizer = get_optimizer(model, config)

    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(optimizer, config)

    return optimizer, lr_scheduler


def get_gradient_norm(model: nn.Module, verbose: bool = False) -> float:
    """Get the (L2) norm of all the parameters in the model"""
    total_sq_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            if verbose:
                print(f"{name} has no gradient.")
        else:
            param_grad_norm = param.grad.data.norm(2)
            total_sq_norm += param_grad_norm.item() ** 2

    total_norm = total_sq_norm**0.5
    return total_norm


def equivariance_discrepancy(
    model: nn.Module,
    batch: DenseMolDynBatch,
    num_transforms: int,
    device: Optional[str] = None,
    tb_logger: Optional[TrainingLogger] = None,
) -> torch.Tensor:
    """
    Compute a measure of how non-equivariant a model is.

    Note: returned value is still on device to reduce device-to-host communication.
    """

    losses = torch.empty(num_transforms, device=device)
    model.eval()
    for i in range(num_transforms):
        # Apply a random transformation
        transformed_batch = transform_batch(batch)

        # Compute transformed loss
        with torch.no_grad():
            losses[i] = get_loss(model, transformed_batch, device=device, tb_logger=tb_logger)

    # Compute relative std deviation of losses
    return torch.std(losses)
