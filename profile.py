import os
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timewarp.dataloader import (
    DenseMolDynBatch,
    moldyn_dense_collate_fn,
)
from timewarp.utils.dataset_utils import get_dataset
from timewarp.utils.loss_utils import get_loss
from timewarp.utils.sampling_utils import get_sample

from timewarp.model_constructor import model_constructor
from timewarp.training_config import TrainingConfig
from timewarp.train import parse_args
from utilities.training_utils import set_seed


def profile_memory(model: nn.Module, batch: DenseMolDynBatch):
    """Profile memory usage of the model."""
    model.to("cpu")  # Memory usage measured on CPU

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof:
        get_loss(model, batch, device="cpu")

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


def profile_forward_time(model: nn.Module, dataloader: DataLoader, logdir: os.PathLike):
    """Profile time taken to sample from the conditional density model."""

    # Timing should be done on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Profiler schedule parameters
    wait = 1
    warmup = 1
    active = 3
    repeat = 2

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step >= (wait + warmup + active) * repeat:
                    break
                print(f"Batch shape: {batch.atom_coords.shape}")
                get_sample(model, batch, num_samples=1, device=device)
                prof.step()


def main(config: TrainingConfig):
    set_seed(config.seed)

    # Create tensorboard output folder
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config.model_config.model_type}_{config.dataset}_batch_size_{config.batch_size}_{time_str}"
    output_folder = Path("outputs/profiling") / run_name
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset.
    train_dataset, _, *_ = get_dataset(
        dataset_name=config.dataset,
        data_dir=Path(config.data_dir).expanduser().resolve() if config.data_dir else None,
        cache_dir=Path(config.dataset_cache_dir),
        step_width=config.step_width,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
        num_workers=6,
    )

    # Create model.
    model = model_constructor(config.model_config)

    # Profile forward sampling time on GPU if available.
    profile_forward_time(model, dataloader, logdir=output_folder)

    # Profile memory use on CPU.
    batch = next(iter(dataloader))
    profile_memory(model, batch)


if __name__ == "__main__":
    config_path, args = parse_args()

    schema = OmegaConf.structured(TrainingConfig)
    conf_yaml = OmegaConf.load(config_path)
    conf_cli = OmegaConf.from_dotlist(args)
    config = OmegaConf.merge(schema, conf_yaml, conf_cli)
    main(config)  # type: ignore [arg-type]
