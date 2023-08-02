import os
import sys
import time
import json
from functools import partial
from typing import Tuple, List
from pathlib import Path
from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timewarp.losses import wrap_or_replace_loss
from timewarp.datasets import LmdbTrajectoryDataset
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.utils.config_utils import (
    check_saved_config,
    finalize_config,
    load_config_dict_in_subdir,
)
from timewarp.utils.training_utils import (
    load_or_construct_loss,
    load_or_construct_loss_scheduler,
    load_or_construct_model,
    load_or_construct_optimizer_lr_scheduler,
    end_of_epoch_report,
    run_on_dataloader,
)
from timewarp.utils.dataset_utils import get_dataset
from utilities.logger import (
    PeriodicLogger,
    TensorBoardLogger,
    TrainingLogger,
)
from timewarp.training_config import TrainingConfig
from timewarp.train_deepspeed import deepspeed_train_loop
from utilities.deepspeed_utils import (
    broadcast_from_leader,
    deepspeed_alt,
)
from utilities.model_utils import save_model
from utilities.training_utils import (
    asgenerator,
    best_valid_loss_controller,
    EpochLossTracker,
    get_checkpointer,
    profiled_dataloader,
    set_seed,
)
# from utilities.run_metadata_utils import add_git_suffix_maybe
from utilities.common import glob_only


def count_parameters(model):
    """Return the total number of trainable parameters in model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@deepspeed_alt(deepspeed_train_loop)
def train_loop(
    config: TrainingConfig,
    train_data: Dataset,
    valid_data: Dataset,
    output_folder: os.PathLike,
    tb_logger: TrainingLogger,
    quiet: bool = False,
    batch_seed: int = 0,
) -> EpochLossTracker:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tDevice: {device}")

    model = load_or_construct_model(config)
    loss_computer = load_or_construct_loss(config)

    # Wrap `model` in the `loss`.
    model = wrap_or_replace_loss(model, loss_computer)
    model = model.to(device)

    loss_scheduler = (
        None
        if config.loss_schedule is None
        else load_or_construct_loss_scheduler(config).to(device)
    )

    print(model)
    print("Total number of parameters: %d" % count_parameters(model))

    if config.num_epochs > 0:
        optimizer, lr_scheduler = load_or_construct_optimizer_lr_scheduler(model, config)
    else:
        # If we are only evaluating don't create the optimizer to save space
        optimizer, lr_scheduler = None, None  # type: ignore [assignment]

    # create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
        # if using LMDB (map-style) dataset, shuffle here; otherwise it is already shuffled.
        shuffle=config.dataset_use_lmdb,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config.valid_batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
    )

    def model_save_func(file_name: str, **kwargs):
        # TODO : Should we save model with loss or without?
        save_model(
            path=os.path.join(output_folder, file_name),
            model=model,
            training_config=config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    def train_func(epoch: int):
        # Setup profiling if requested and in the first epoch
        wrapper = (
            partial(
                profiled_dataloader,
                output_folder,
                torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=2),
            )
            if config.enable_profiler and epoch == 0
            else asgenerator
        )

        return run_on_dataloader(
            model,
            data_loader=wrapper(train_loader),  # type: ignore [operator]
            optimizer=optimizer,
            clip_grad_norm=config.clip_grad_norm,
            lr_scheduler=lr_scheduler,
            loss_scheduler=loss_scheduler,
            quiet=quiet,
            run_kind="train",
            tb_logger=tb_logger,
            data_augmentation=config.data_augmentation,
            compute_equivariance_discrepancy=config.measure_equivariance_discrepancy,
            device=device,
            checkpointer=get_checkpointer(
                config.min_check_point_iters, model_save_func, epoch=epoch
            ),
        )

    def valid_func():
        return run_on_dataloader(
            model,
            asgenerator(valid_loader),
            quiet=quiet,
            device=device,
        )

    return best_valid_loss_controller(
        train_func=train_func,
        valid_func=valid_func,
        model_save_func=model_save_func,
        end_of_epoch_report_func=partial(end_of_epoch_report, config=config, tb_logger=tb_logger),
        max_num_epochs=config.num_epochs,
        patience=config.patience,
        valid_first=config.run_valid_first,
    )


@broadcast_from_leader
def setup_output_folder_and_save_config(config: TrainingConfig) -> Path:
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config.run_prefix}{config.model_config.model_type}_stepwidth_{config.step_width}_{time_str}"

    repo_path = Path(os.path.abspath(__file__)).parent.parent.parent
    # run_name = add_git_suffix_maybe(run_name, repo_path=repo_path)

    print(f"Starting training run {run_name}")

    output_folder = Path(config.output_folder) / run_name
    os.makedirs(output_folder, exist_ok=True)
    print(f"\tOutput dir: {output_folder}")

    # Save the config to the output folder
    with (output_folder / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(config))
    return output_folder


@broadcast_from_leader
def log_best_validation_loss(
    config: TrainingConfig, output_folder: Path, best_valid_loss_tracker: EpochLossTracker
) -> int:
    best_path = os.path.join(output_folder, "best_valid_loss.json")
    with open(best_path, "w") as fp:
        save_dict = {
            "loss_dict": best_valid_loss_tracker.loss_dict,
            "step_width": config.step_width,
        }
        json.dump(save_dict, fp)
    return 0  # return something that is not None for broadcast_from_leader


@broadcast_from_leader
def generate_seed() -> int:
    """Generate a random seed that will be broadcast to all devices. This is to ensure batches are shuffled upon reallocation."""
    seed = np.random.randint(99999)
    return seed


def load_or_setup_output_folder(config: TrainingConfig) -> Path:
    """
    Returns output_folder for an existing (preempted) run; otherwise create a new one.
    """
    if config.output_folder == "outputs":
        # local run, no preemption
        return setup_output_folder_and_save_config(config)
    for run_name in os.listdir(config.output_folder):
        path = os.path.join(config.output_folder, run_name)
        if os.path.isdir(path) and run_name.startswith(config.model_config.model_type):
            output_folder = Path(config.output_folder) / run_name
            config.saved_model_path = str(output_folder)
            print(f"Resuming training run {run_name}")
            print(f"\tOutput dir: {output_folder}")
            return output_folder
    return setup_output_folder_and_save_config(config)


def close_dataset(dataset: Dataset):
    if isinstance(dataset, LmdbTrajectoryDataset):
        dataset.close_db()


def main(config: TrainingConfig) -> int:
    if config.randomise_seed:
        seed = generate_seed()
        set_seed(seed)
        print(f"Generating new seed: {seed}")
    else:
        seed = config.seed
        set_seed(seed)
        print(f"Using config seed: {seed}")

    # check the saved config is compatible with the current config.
    check_saved_config(config)

    output_folder = load_or_setup_output_folder(config)

    train_dataset, valid_dataset, *_ = get_dataset(
        dataset_name=config.dataset,
        data_dir=Path(config.data_dir).expanduser().resolve() if config.data_dir else None,
        cache_dir=Path(config.dataset_cache_dir).expanduser().resolve(),
        step_width=config.step_width,
        lmdb=config.dataset_use_lmdb,
        equal_data_spacing=config.equal_data_spacing,
    )

    # Using writer as a context manager ensures that the log is flushed at the end.
    with SummaryWriter(log_dir=output_folder) as writer:
        tb_logger = PeriodicLogger(
            logger=TensorBoardLogger(writer=writer),
            period=100,
            except_names=["valid_loss", "valid_throughput", "train_throughput"],
        )

        best_valid_loss_tracker = train_loop(
            config=config,
            train_data=train_dataset,
            valid_data=valid_dataset,
            output_folder=output_folder,
            tb_logger=tb_logger,
            batch_seed=seed,
        )

    # Make sure to close the databases if using LmdbTrajectoryDataset
    close_dataset(train_dataset)
    close_dataset(valid_dataset)

    # Save best validation loss.
    log_best_validation_loss(config, output_folder, best_valid_loss_tracker)

    return 0


def parse_args() -> Tuple[str, List[str]]:
    """
    Manually parse command-line arguments, rather than relying on a third party library.

    Returns a path to the config, and a list of overrides specified by the user.
    """
    if len(sys.argv) == 1:
        raise ValueError("A config file must be provided as first argument.")
    elif sys.argv[1] in {"--help", "-h"}:
        print(
            "Provide a yaml configuration file as first argument, and optionally specify the overrides"
            "to the configuration in a dotlist file. For instance:\n"
            "\t path/to/config.yaml learning_rate=0.01 model_config.transformer_nvp.feature_dim=10\n"
            "to override the learning rate the number of features from the command line.\n"
            "Config files with default values can be found in timewarp/configs"
        )
        exit()

    config_path = None
    overrides: List[str] = []
    for param in sys.argv[1:]:
        if config_path is None:
            config_path = param
        else:
            overrides.append(param)
    assert config_path is not None
    return glob_only(config_path), overrides


if __name__ == "__main__":
    config_path, args = parse_args()

    config = load_config_dict_in_subdir(config_path)
    conf_cli = OmegaConf.from_dotlist(args)
    config = OmegaConf.merge(config, conf_cli)  # type: ignore [assignment]
    main(finalize_config(config))
