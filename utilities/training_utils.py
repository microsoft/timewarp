from collections import defaultdict
from dataclasses import dataclass
from functools import partial
import os
import sys
import time
from typing import (
    Callable,
    ContextManager,
    DefaultDict,
    Dict,
    Generator,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)
from typing_extensions import Literal, Protocol

if sys.version_info >= (3, 8, 0):
    from functools import cached_property
else:
    from cached_property import cached_property  # type: ignore


import numpy as np
import torch

try:
    from .deepspeed_utils import deepspeed_enabled
except ModuleNotFoundError:
    deepspeed_enabled = None  # type: ignore
    pass


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class RngState:
    numpy_state: Tuple[str, np.ndarray, int, int, float]
    torch_state: torch.Tensor
    torch_cuda_state: torch.Tensor

    @staticmethod
    def get_rng_state():
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        return RngState(np.random.get_state(), torch.get_rng_state(), cuda_state)

    def __eq__(self, other):
        numpy_state_equal = all(
            np.array_equal(s1, s2) if i == 1 else s1 == s2
            for i, (s1, s2) in enumerate(zip(self.numpy_state, other.numpy_state))
        )
        torch_state_equal = torch.equal(self.torch_state, other.torch_state)
        if self.torch_cuda_state is None:
            torch_cuda_state_equal = other.torch_cuda_state is None
        else:
            torch_cuda_state_equal = torch.equal(self.torch_cuda_state, other.torch_cuda_state)
        return numpy_state_equal and torch_state_equal and torch_cuda_state_equal


class EpochLossTracker(ContextManager):
    def __init__(self, run_kind: str, window_size: int = 100, quiet: bool = False):
        self._run_kind = run_kind
        self._window_size = window_size
        self._quiet = quiet
        self._step_counter = 0
        self._t0 = None
        self._finished = False
        self._losses: DefaultDict[str, torch.Tensor] = defaultdict(lambda: 0.0)  # type: ignore
        self._windowed_losses: DefaultDict[str, torch.Tensor] = defaultdict(lambda: 0.0)  # type: ignore

    @staticmethod
    def __format_loss_dict(loss_dict: Dict[str, torch.Tensor], num_steps: int) -> str:
        return ", ".join(
            f"{loss_name}: {float(loss_val) / num_steps:g}"
            for loss_name, loss_val in loss_dict.items()
        )

    @property
    def steps(self) -> int:
        return self._step_counter

    @property
    def elapsed(self) -> float:
        assert self._t0 is not None
        if not self._finished:
            self._elapsed = time.perf_counter() - self._t0
        return self._elapsed

    @property
    def throughput(self) -> float:
        return self.steps / (self.elapsed)

    def _compute_total_loss(self) -> float:
        total_loss = sum(self._losses.values()) / self._step_counter
        return float(total_loss)  # also moves to cpu if total_loss is on device.

    @cached_property
    def total_loss(self) -> float:
        """
        Returns the sum of all the loss components averaged over epoch. To cache the GPU -> CPU
        data movement, this property can only be accessed after __exit__().
        """
        assert self._finished
        return self._compute_total_loss()

    @property
    def loss_overview(self):
        return self.__format_loss_dict(self._losses, self._step_counter)

    @property
    def loss_dict(self):
        return {k: float(v) / self._step_counter for k, v in self._losses.items()}

    def log_losses(self, **kwargs: torch.Tensor) -> None:
        assert not self._finished
        for loss_name, loss_val in kwargs.items():
            self._losses[loss_name] += loss_val.detach()
            self._windowed_losses[loss_name] += loss_val.detach()

        self._step_counter += 1
        if self._step_counter % self._window_size == 0:
            if not self._quiet:
                mean_loss_str = self.__format_loss_dict(self._losses, self._step_counter)
                windowed_loss_str = self.__format_loss_dict(
                    self._windowed_losses, self._window_size
                )
                print(
                    f"   {self._run_kind} step {self._step_counter:04d}"
                    f" || Mean loss so far: {mean_loss_str}"
                    f" || This window: {windowed_loss_str}"
                    f" || elapsed: {self.elapsed:.2f} s"
                    f" || throughput: {self.throughput:.2f} steps/s"
                )
            self._windowed_losses.clear()

    def apply(self, f: Callable[[torch.Tensor], None]):
        """
        Apply function f (e.g., all_reduce) on each loss tensor.
        """
        for loss_value in self._losses.values():
            f(loss_value)
        # As this updates losses, we may need to re-compute the potentially cached total_loss value.
        # If it has been cached, the attribute has been set, and deleting it will trigger a recompute
        # on the next access:
        if hasattr(self, "total_loss"):
            delattr(self, "total_loss")

    def __enter__(self):
        assert not self._finished
        self._t0 = time.perf_counter()
        self._step_counter = 0
        return self

    def __exit__(self, *exc):
        self.elapsed  # update self._elapsed
        self._finished = True
        return False  # don't suppress exception


Batch = TypeVar("Batch")


def profiled_dataloader(
    output_folder: os.PathLike,
    schedule: Callable[[int], torch.profiler.ProfilerAction],
    data_loader: Iterator[Batch],
    tensorboard: bool = True,
) -> Generator[Batch, None, None]:
    """
    Wraps a DataLoader and adds the ability to profile according to the given profile schedule.
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_folder))
        if tensorboard
        else None,
        with_stack=True,
    ) as p:
        try:
            for batch in data_loader:
                yield batch
                p.step()
        finally:
            if not tensorboard:
                # output profile in chrome trace format (mutually exclusive with tensorboard)
                p.export_chrome_trace(os.path.join(output_folder, "profile.json"))


def asgenerator(data_loader: Iterator[Batch]) -> Generator[Batch, None, None]:
    """
    Converts a DataLoader into a generator so as to have the same interface as
    `profiled_dataloader`.
    """
    for batch in data_loader:
        yield batch


class ModelSaveFunction(Protocol):
    def __call__(self, file_name: str, **kwargs) -> None:
        ...


def best_valid_loss_controller(
    train_func: Callable[[int], EpochLossTracker],
    valid_func: Callable[[], EpochLossTracker],
    model_save_func: ModelSaveFunction,
    end_of_epoch_report_func: Callable[[str, Literal["train", "valid"], EpochLossTracker], None],
    max_num_epochs: int = 100,
    patience: int = 5,
    initial_epoch: int = 0,
    valid_first: bool = True,
) -> EpochLossTracker:
    """Controls the training loop based on max number of epochs and patience parameters.

    Args:
        train_func: callable that returns EpochLossTracker as a summary of training.
        valid_func: callable that returns EpochLossTracker as a summary of validation.
        model_save_func: callable that takes epoch and file name and saves the current model.
        end_of_epoch_report_func: function that takes care of reporting after training / validation.
        max_num_epochs: max number of epochs.
        patience: number of epochs to continue after seeing the last decrease in best validation loss.
        valid_first: if `True`, we will run through the validation dataset before the first epoch. Default: `True`.
    """
    if valid_first:
        print("== Running validation on initial model")
        best_valid_loss_tracker = valid_func()
        end_of_epoch_report_func("Initial", "valid", best_valid_loss_tracker)
    else:
        best_valid_loss_tracker = None

    model_save_func(epoch=-1, file_name="best_model.pt")

    epochs_since_best = 0
    for epoch in range(initial_epoch, max_num_epochs):
        # Train.
        print(f"== Epoch {epoch}")
        print("  = Training")
        loss_tracker = train_func(epoch)
        end_of_epoch_report_func("Mean", "train", loss_tracker)

        # Validate.
        print("  = Validation")
        new_tracker = valid_func()
        end_of_epoch_report_func("", "valid", new_tracker)
        if (
            best_valid_loss_tracker is None
            or new_tracker.total_loss < best_valid_loss_tracker.total_loss
        ):
            epochs_since_best = 0
            message = f"   New best validation result {new_tracker.total_loss:g}."

            if best_valid_loss_tracker is not None:
                message += f" (decreased from {best_valid_loss_tracker.total_loss:g})"

            print(message)

            best_valid_loss_tracker = new_tracker
            model_save_func(epoch=epoch, file_name="best_model.pt")
        else:
            epochs_since_best += 1
            print(f"   Now had {epochs_since_best} epochs since best result.")
            if epochs_since_best >= patience:
                break

    assert best_valid_loss_tracker is not None
    return best_valid_loss_tracker


class Checkpointer:
    def checkpoint(self, current_iter: int):
        pass


class PeriodicCheckpointer(Checkpointer):
    def __init__(
        self,
        min_check_point_iters: int,
        model_save_func: ModelSaveFunction,
        check_point_name: str = "latest_model.pt",
    ):
        self.min_check_point_iters = min_check_point_iters
        self.model_save_func = model_save_func
        self.check_point_name = check_point_name
        self.last_check_point_iter = 0

    def checkpoint(self, current_iter: int):
        if current_iter - self.last_check_point_iter >= self.min_check_point_iters:
            print(f"  model checkpointing at iteration {current_iter}")
            self.last_check_point_iter = current_iter
            self.model_save_func(self.check_point_name, iter=current_iter)


def get_checkpointer(
    min_check_point_iters: Optional[int] = None,
    model_save_func: Optional[ModelSaveFunction] = None,
    **kwargs,
) -> Checkpointer:
    """
    Returns a checkpointer.

    Args:
        min_check_point_iters: minimum interval to checkpoint a model during training.
        model_save_func: callable to be used to save the model and training state.
        **kwargs: additional keyword arguments for model_save_func.

    Returns:
        a PeriodicCheckpointer if min_checkpoint_iters is not None; otherwise a no-op checkpointer.
    """
    if min_check_point_iters is None:
        return Checkpointer()
    assert model_save_func is not None
    return PeriodicCheckpointer(min_check_point_iters, partial(model_save_func, **kwargs))


class TrainingConfig(Protocol):
    optimizer: str
    learning_rate: float
    warmup_steps: int
    weight_decay: float


def get_initial_learning_rate(config: TrainingConfig) -> float:
    """Convenience function to find the initial learning rate to initialize the optimizer
    when a learning rate scheduler is used.

    Args:
        config: a config object with `learning_rate` and `warmup_steps` attributes.

    Returns:
        the initial learning rate.
    """
    if deepspeed_enabled is None or not deepspeed_enabled():
        # Return the final learning rate because PyTorch LR scheduler reads the final
        # learning rate from the optimizer, and mutates the LR when it is initialized.
        return config.learning_rate
    # Here we assume linear warmup (see get_lr_scheduler)
    assert config.warmup_steps >= 0
    return (
        config.learning_rate
        if config.warmup_steps == 0
        else config.learning_rate / config.warmup_steps
    )


def get_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Gets an optimizer from TrainingConfig.

    Args:
        model: a torch model.
        config: a config object with `learning_rate`, `weight_decay` and `warmup_steps`.
    """
    assert config.optimizer == "Adam"
    return torch.optim.Adam(
        params=model.parameters(),
        lr=get_initial_learning_rate(config),
        weight_decay=config.weight_decay,
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """Gets a LR scheduler from TrainingConfig."""

    def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
        if cur_step >= warmup_steps:
            return 1.0
        return cur_step / warmup_steps

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(linear_warmup, warmup_steps=config.warmup_steps),
    )
