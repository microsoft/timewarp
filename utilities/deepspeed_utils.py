from dataclasses import dataclass
from functools import wraps
import os
from typing import Callable, Tuple

import torch
import torch.distributed as dist

try:
    import deepspeed  # type: ignore [import]
    from deepspeed import DeepSpeedEngine
    from deepspeed.runtime.dataloader import DeepSpeedDataLoader

    deepspeed_installed = True
except ModuleNotFoundError:
    deepspeed_installed = False


@dataclass
class DeepSpeedArgs:
    local_rank: int


def deepspeed_initialize(
    *args, **kwargs
) -> Tuple[
    "DeepSpeedEngine",
    torch.optim.Optimizer,
    "DeepSpeedDataLoader",
    torch.optim.lr_scheduler._LRScheduler,
]:
    """
    Wraps deepspeed.initialize so that "args" is read out from the command line arguments.
    """
    assert deepspeed_installed
    return deepspeed.initialize(DeepSpeedArgs(local_rank=get_local_rank()), *args, **kwargs)


def get_local_rank() -> int:
    # PyTorch / DeepSpeed seems to recognize environment variable LOCAL_RANK
    # and work without installing MPI.
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    # Otherwise, use the Open MPI environment variable.
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", -1))


def deepspeed_enabled() -> bool:
    return deepspeed_installed and get_local_rank() >= 0


def deepspeed_alt(alt_func: Callable):
    """
    Decorator to provide an alternative function to be called if deepspeed is enabled.

    Args:
        alt_func: alternative function that has the same signature as the decorated function
            to be called if deepspeed is enabled.
    Returns:
        the decorated function.
    """

    def decorator(f: Callable):
        def inner(*args, **kwargs):
            if deepspeed_enabled():
                return alt_func(*args, **kwargs)
            else:
                return f(*args, **kwargs)

        return inner

    return decorator


def broadcast_from_leader(setup_func: Callable):
    """Decorator that makes a function compute something in rank 0 and broadcast the result to all ranks.

    Args:
        setup_func: a function to be called in the rank 0 process.

    Returns:
        the decorated function.
    """

    @wraps(setup_func)
    def inner(*args, **kwargs):
        if not deepspeed_enabled():
            # Not distributed
            return setup_func(*args, **kwargs)
        if not dist.is_initialized():
            deepspeed.init_distributed()
            torch.cuda.set_device(get_local_rank())
        if dist.get_rank() == 0:
            resources = [setup_func(*args, **kwargs)]
        else:
            resources = [None]
        dist.broadcast_object_list(resources, src=0)
        assert resources[0] is not None
        return resources[0]

    return inner
