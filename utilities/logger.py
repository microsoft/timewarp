import abc
from typing import Callable, Optional, Sequence

import torch
from torch.types import Number
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from utilities.deepspeed_utils import deepspeed_alt
from . import delayed_reporter


class TrainingLogger(abc.ABC):
    """A base class for a training logger that maintains the number of training steps."""

    def __init__(self) -> None:
        self._step: int = 0

    def increment_step(self) -> None:
        self._step += 1

    @property
    def step(self):
        return self._step

    @abc.abstractmethod
    def _log_scalar_with_step(self, name: str, step: int, value: Number) -> None:
        """
        Logs a scalar metric with a step counter.

        Args:
            name: metric name
            step: step counter
            value: value to log
        """

    def log_scalar(self, name: str, value: Number) -> None:
        self._log_scalar_with_step(name, self.step, value)

    def log_scalar_async(self, name: str, value: torch.Tensor) -> None:
        """
        Logs a scalar tensor asynchronously.

        Note: the values are logged in the order they are scheduled using the value
        of self.step when the logging was scheduled.
        """

        def callback(cpu_value: Number, step: int = self.step):
            self._log_scalar_with_step(name, step, cpu_value)

        delayed_reporter.report_async(value, callback)

    @abc.abstractmethod
    def log_tensor(self, name: str, value: torch.Tensor) -> None:
        pass


class TensorBoardLogger(TrainingLogger):
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self.writer = writer

    def _log_scalar_with_step(self, name: str, step: int, value: Number) -> None:
        self.writer.add_scalar(name, value, global_step=step)

    def log_tensor(self, name: str, value: torch.Tensor) -> None:
        self.writer.add_histogram(name, value, global_step=self.step)


class ConditionalLogger(TrainingLogger):
    def __init__(self, logger: TrainingLogger, condition: Callable[[TrainingLogger, str], bool]):
        self.inner_logger = logger
        self.condition = condition

    @property
    def step(self):
        return self.inner_logger.step

    def _log_scalar_with_step(self, name: str, step: int, value: Number) -> None:
        raise NotImplementedError()

    def _check_condition(self, name):
        return self.condition(self.inner_logger, name)

    def increment_step(self) -> None:
        return self.inner_logger.increment_step()

    def log_scalar(self, name: str, value: Number) -> None:
        if self._check_condition(name):
            return self.inner_logger.log_scalar(name, value)

    def log_scalar_async(self, name: str, value: torch.Tensor) -> None:
        if self._check_condition(name):
            return self.inner_logger.log_scalar_async(name, value)

    def log_tensor(self, name: str, value: torch.Tensor) -> None:
        if self._check_condition(name):
            return self.inner_logger.log_tensor(name, value)


def check_period(
    period: int,
    except_names: Sequence[str] = (),
    precond: Callable[[TrainingLogger, str], bool] = lambda self, name: True,
):
    def inner(self: TrainingLogger, name: str):
        return precond(self, name) and (self.step % period == 0 or name in except_names)

    return inner


def check_rank(precond: Callable[[TrainingLogger, str], bool] = lambda self, name: True):
    def inner(self: TrainingLogger, name: str):
        return precond(self, name) and dist.get_rank() == 0

    return inner


def LeaderOnlyLogger(
    logger: TrainingLogger,
    period: Optional[int] = None,
    except_names: Sequence[str] = (),
):
    """
    Returns a logger that only logs in the process with global rank == 0.

    Args:
        logger: base logger that does the actual logging.
        period: minimum steps between logging.
        except_names: sequence of log entries to log every step.

    Returns:
        a PeriodicLogger that only logs in the process with global_rank == 0.
    """
    if period is None:
        return ConditionalLogger(logger, check_rank())
    return ConditionalLogger(logger, check_rank(check_period(period, except_names)))


@deepspeed_alt(LeaderOnlyLogger)
def PeriodicLogger(logger: TrainingLogger, period: int, except_names: Sequence[str] = ()):
    """
    Wrapper around tensorboard logger to log only periodically.

    Args:
        logger: base logger that does the actual logging.
        period: minimum steps between logging.
        except_names: sequence of log entries to log every step.

    Returns:
        a PeriodicLogger
    """

    return ConditionalLogger(logger, check_period(period, except_names))
