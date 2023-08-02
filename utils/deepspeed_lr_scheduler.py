from dataclasses import dataclass
from torch.optim import Optimizer
from typing import Optional
import math


@dataclass
class LRSchedulerConfig:
    type: str
    factor: float
    iteration_patience: int
    logging_period: int
    alpha: float


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, "optimizer") and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError(
        "{} is not a subclass of torch.optim.Optimizer".format(type(optimizer).__name__)
    )


class LRSchedulerCallable(object):
    """ "Wrap learning rate scheduler in a callable that takes in the optimizer and returns the
    scheduler object.

    Args:
        LRScheduler: Learning rate scheduler class
        scheduler_kwargs: Kwargs for the scheduler excluding the optimizer.
    """

    def __init__(
        self,
        LRScheduler,
        scheduler_kwargs: dict,
    ):
        self.LRScheduler = LRScheduler
        self.scheduler_kwargs = scheduler_kwargs

    def __call__(self, optimizer):
        return self.LRScheduler(
            optimizer=optimizer,
            **self.scheduler_kwargs,
        )


class PlateauLR(object):
    """Decrease the learning rate by a multiplicative factor if train loss doesn't decrease after a
    certain number of training iterations.

     Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr (float): Initial learning rate.
        factor (float): Factor to multiply learning rate by if loss doesn't improve.
        iteration_patience (float): Number of iterations to wait after last best loss achieved
            before multiplying learning rate by the factor.
        logging_period (int): Number of iterations between each time the loss is logged. Logging
            the loss requires an `all_reduce` operation so this is not done at every iteration.
        alpha (float): Factor for moving average calculation of loss.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        factor: float,
        iteration_patience: int,
        logging_period: int,
        alpha: float,
        moving_average_loss: Optional[float] = None,
    ):

        self.optimizer = get_torch_optimizer(optimizer)
        self.lr = lr
        self.factor = factor
        self.iteration_patience = iteration_patience
        self.logging_period = logging_period
        self.best_loss = math.inf
        self.iterations_since_best = 0
        self.moving_average_loss = moving_average_loss
        self.alpha = alpha

    def get_lr(self, loss: Optional[float] = None) -> float:
        if loss is None:
            self.iterations_since_best += 1
            return self.lr  # Loss not being logged this iteration, so don't update lr.
        elif loss > self.best_loss:
            self.iterations_since_best += 1
            print(f"Iterations since best: {self.iterations_since_best}")
            if self.iterations_since_best > self.iteration_patience:
                print(f"Reducing learning rate. Old learning rate was {self.lr}")
                self.lr = self.lr * self.factor
                self.iterations_since_best = 0  # Reset patience after reducing lr.
                print(f"New learning rate is {self.lr}")
        else:
            print(f"Loss was logged to lr scheduler, new best loss is {loss}")
            self.best_loss = loss
            self.iterations_since_best = 0

        return self.lr

    def moving_average(self, loss: Optional[float] = None) -> Optional[float]:
        if loss is None:
            return None
        else:
            if self.moving_average_loss is None:  # Don't average first logging iteration.
                self.moving_average_loss = loss
            else:
                self.moving_average_loss = (
                    self.alpha * loss + (1.0 - self.alpha) * self.moving_average_loss
                )
            return self.moving_average_loss

    def step(self, loss: Optional[float] = None) -> None:
        loss = self.moving_average(loss)
        lr = self.get_lr(loss)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "lr": self.lr,
            "best_loss": self.best_loss,
            "iterations_since_best": self.iterations_since_best,
            "logging_period": self.logging_period,
            "moving_average_loss": self.moving_average_loss,
            "alpha": self.alpha,
        }

    def load_state_dict(self, sd):
        self.lr = sd["lr"]
        self.best_loss = sd["best_loss"]
        self.iterations_since_best = sd["iterations_since_best"]
        self.logging_period = sd["logging_period"]
        self.moving_average_loss = sd["moving_average_loss"]
        self.alpha = sd["alpha"]
