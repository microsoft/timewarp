from functools import partial
import os

import torch
from torch.utils.data import Dataset
import deepspeed  # type: ignore [import]
from deepspeed.runtime.lr_schedules import WarmupLR

import torch.distributed as dist
from timewarp.datasets.pdb_sampler import LMDBDistributedSampler
from timewarp.losses import (
    wrap_or_replace_loss,
)

from timewarp.model_constructor import model_constructor
from timewarp.utils.config_utils import load_model_config
from utilities.logger import TrainingLogger
from timewarp.datasets import LmdbTrajectoryDataset
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.training_config import TrainingConfig
from timewarp.utils.deepspeed_lr_scheduler import PlateauLR, LRSchedulerCallable
from timewarp.utils.training_utils import (
    end_of_epoch_report,
    load_or_construct_loss,
    load_or_construct_loss_scheduler,
    run_on_dataloader,
)
from utilities import find_files
from utilities.training_utils import (
    asgenerator,
    best_valid_loss_controller,
    EpochLossTracker,
    get_checkpointer,
    get_initial_learning_rate,
)

from utilities.deepspeed_utils import deepspeed_initialize


def deepspeed_train_loop(
    config: TrainingConfig,
    train_data: Dataset,
    valid_data: Dataset,
    output_folder: os.PathLike,
    tb_logger: TrainingLogger,
    quiet: bool = False,
    batch_seed: int = 0,
) -> EpochLossTracker:
    # Construct model.
    # NOTE: For the sake of backwards compatibility, we cannot assume that all models
    # are saved wrapped in a loss. This is why we first instantiate the model, then load
    # or instantiate the loss, and then finally wrap the model in the loss.
    # In addition, the `LossWrapper` has its own `load_state_dict` which correctly
    # handles the case of loading `state_dict` for a model which was not originally
    # wrapped in a loss.
    model = model_constructor(load_model_config(config))
    # Construct or load the loss.
    loss_computer = load_or_construct_loss(config)
    model = wrap_or_replace_loss(model, loss_computer)

    # FIXME: Currently `loss_scheduler` does not work with `deepspeed`.
    assert config.loss_schedule is None, "Loss scheduler not currently supported with deepspeed."
    loss_scheduler = (
        None if config.loss_schedule is None else load_or_construct_loss_scheduler(config)
    )

    # initialize process groups
    deepspeed.init_distributed()
    local_batch_size = config.batch_size // dist.get_world_size()

    # Create learning rate scheduler callable
    if config.lr_scheduler is None or config.lr_scheduler.type == "warmup":
        assert config.warm_start, "warm_start must be true if warmup scheduler used."
        lr_scheduler = LRSchedulerCallable(
            LRScheduler=WarmupLR,
            scheduler_kwargs={
                "warmup_min_lr": 0.0,
                "warmup_max_lr": config.learning_rate,
                "warmup_type": "linear",
            },
        )
        initial_lr = get_initial_learning_rate(config)
    elif config.lr_scheduler.type == "plateau":
        assert not config.warm_start, "warm_start must be false if warmup scheduler is not used."
        lr_scheduler = LRSchedulerCallable(
            LRScheduler=PlateauLR,
            scheduler_kwargs={
                "lr": config.learning_rate,
                "factor": config.lr_scheduler.factor,
                "iteration_patience": config.lr_scheduler.iteration_patience,
                "logging_period": config.lr_scheduler.logging_period,
                "alpha": config.lr_scheduler.alpha,
            },
        )
        initial_lr = config.learning_rate
    else:
        raise ValueError()

    model_engine, _, _, _ = deepspeed_initialize(
        model=model,
        lr_scheduler=lr_scheduler,
        model_parameters=model.parameters(),
        config={
            "train_batch_size": config.batch_size,
            "train_micro_batch_size_per_gpu": local_batch_size,
            "optimizer": {
                "type": config.optimizer,
                "params": {
                    "lr": initial_lr,
                    "weight_decay": config.weight_decay,
                },
            },
            "steps_per_print": 100,
            # disable ZeRO optimization if using Lamb optimizer
            "zero_optimization": {} if config.optimizer == "Lamb" else {"stage": 1},
            # Deepspeed uses 0.0 for no clipping
            "gradient_clipping": config.clip_grad_norm or 0.0,
        },
        collate_fn=moldyn_dense_collate_fn,
    )

    # Load check point
    if config.saved_model_path is not None:
        print(f"Trying to load the latest checkpoint from {config.saved_model_path}")
        try:
            # NOTE : `deepspeed` can't load from a single model file;
            # it needs a folder for some reason.
            assert not os.path.isfile(
                config.saved_model_path
            ), 'deepspeed can only load checkpoints from a directory containing a file called "latest"'
            # Try to look for it.
            latest_file = next(find_files(config.saved_model_path, "latest"))
            print(f"Found latestfile {latest_file}")
            saved_model_path = os.path.dirname(latest_file)

            # NOTE : This works "correctly" even if we change the `loss` part of the `LossWrapper`
            # since the `LossWrapper.loss` field should have no parameters.
            if config.warm_start:
                # Only load the module weights.
                # _, client_state = model_engine.load_checkpoint(
                #     saved_model_path,
                #     tag="best_model.pt",
                #     load_module_strict=True,
                #     load_module_only=True,
                # )
                _, client_state = model_engine.load_checkpoint(
                    saved_model_path, load_module_strict=True, load_module_only=True
                )
            else:
                # Try to load everything, e.g. module weights, optimizer.
                _, client_state = model_engine.load_checkpoint(saved_model_path)

            initial_epoch = client_state["epoch"]
        except StopIteration:
            print("No checkpoint found.")
            initial_epoch = 0
    else:
        initial_epoch = 0

    # Make sure that datasets are random accessible
    assert isinstance(train_data, LmdbTrajectoryDataset)
    max_contiguous_length = (
        None
        if config.num_pdbs_per_local_batch is None
        else local_batch_size // config.num_pdbs_per_local_batch
    )
    # FIXME: Fails when there are no `pdb_names` specified, which seems like might be the case
    # if the entire dataset is for the same molecule.
    train_loader = model_engine.deepspeed_io(
        train_data,
        data_sampler=LMDBDistributedSampler(
            train_data,
            max_contiguous_length=max_contiguous_length,
            seed=batch_seed,
        ),
    )

    assert isinstance(valid_data, LmdbTrajectoryDataset)
    valid_loader = model_engine.deepspeed_io(
        valid_data,
        batch_size=config.valid_batch_size // model_engine.dp_world_size,
        data_sampler=LMDBDistributedSampler(valid_data, shuffle=False),
    )

    # Construct the losses.
    def all_reduce_loss(loss_value: torch.Tensor):
        loss_value.div_(model_engine.dp_world_size)
        dist.all_reduce(loss_value, group=model_engine.data_parallel_group)

    def model_save_func(file_name: str, **kwargs):
        model_engine.save_checkpoint(
            output_folder, tag=file_name, client_state={"training_config": config, **kwargs}
        )

    def train_func(epoch: int):
        loss_tracker = run_on_dataloader(
            model_engine,
            data_loader=asgenerator(train_loader),
            deepspeed=True,
            optimizer=None,  # model_engine takes care of the optimizer
            lr_scheduler=None,
            loss_scheduler=loss_scheduler,
            quiet=quiet,
            run_kind="train",
            tb_logger=tb_logger,
            data_augmentation=config.data_augmentation,
            compute_equivariance_discrepancy=config.measure_equivariance_discrepancy,
            device=model_engine.device,
            checkpointer=get_checkpointer(
                config.min_check_point_iters, model_save_func, epoch=epoch
            ),
        )
        loss_tracker.apply(all_reduce_loss)
        return loss_tracker

    def valid_func():
        loss_tracker = run_on_dataloader(
            model_engine,
            asgenerator(valid_loader),
            deepspeed=True,
            loss_scheduler=loss_scheduler,
            quiet=quiet,
            device=model_engine.device,
        )
        loss_tracker.apply(all_reduce_loss)
        return loss_tracker

    return best_valid_loss_controller(
        train_func=train_func,
        valid_func=valid_func,
        model_save_func=model_save_func,
        end_of_epoch_report_func=partial(end_of_epoch_report, config=config, tb_logger=tb_logger),
        max_num_epochs=config.num_epochs,
        patience=config.patience,
        initial_epoch=initial_epoch,
        valid_first=config.run_valid_first,
    )
