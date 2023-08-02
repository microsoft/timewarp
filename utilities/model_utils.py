from functools import lru_cache
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Optional, OrderedDict

import torch

from .common import find_files, StrPath, unique_item


def save_model(
    path: StrPath,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs,
):
    data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None:
        data["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    data.update(kwargs)

    torch.save(data, path)


@lru_cache(maxsize=128)
def load_checkpoint_in_subdir(path: Path, file_name="best_model.pt", pickle_module=pickle):
    if not os.path.isfile(path):
        path = Path(unique_item(list(find_files(path, file_name))))
    return torch.load(path, pickle_module=pickle_module)


def load_model_state_dict(
    path: StrPath, file_name="best_model.pt", pickle_module=pickle
) -> OrderedDict[str, torch.Tensor]:
    data = load_checkpoint_in_subdir(Path(path), file_name, pickle_module)
    return data["model_state_dict" if "model_state_dict" in data else "module"]


def load_model(
    path: StrPath, model_constructor: Callable, file_name="best_model.pt"
) -> torch.nn.Module:
    """
    Loads a model from a path or a directory containing a model checkpoint.

    Args:
        path: path to a checkpoint file or a directory containing a subdirectory that contains the checkpoint.
        model_constructor: callable to be called on the checkpoint data and constructs the model.
        file_name: check point file name (default: "best_model.pt").

    Returns:
        loaded model.
    """
    data = load_checkpoint_in_subdir(Path(path), file_name)
    model = model_constructor(data)
    model.load_state_dict(load_model_state_dict(path, file_name))
    return model


def unflatten_state_dict(state_dict: OrderedDict[str, Any], depth: int) -> OrderedDict[str, Any]:
    """
    Converts a flattened `state_dict` into a nested structure, where the nested level is specified by `depth`.

    Args:
        state_dict: dictionary where keys are of the form "a.b.c", with `.` representing the nesting level.
        depth: integer specifying the number of levels to unflatten.
    """
    if depth <= 0:
        return state_dict

    nested_state_dict = OrderedDict()
    for (k, v) in state_dict.items():
        parent, *children = k.split(".")

        if len(children) == 0:
            # Already flat.
            nested_state_dict[parent] = v
        else:
            # Create dict holder if not already seen.
            if parent not in nested_state_dict:
                nested_state_dict[parent] = OrderedDict()

            # Re-join `children` to form the key.
            nested_state_dict[parent][".".join(children)] = v

    return OrderedDict(
        {
            k: unflatten_state_dict(v, depth - 1) if isinstance(v, OrderedDict) else v
            for (k, v) in nested_state_dict.items()
        }
    )
