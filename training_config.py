import warnings

from dataclasses import dataclass
from typing import Optional

from timewarp.utils.deepspeed_lr_scheduler import LRSchedulerConfig
from timewarp.model_configs import ModelConfig
from timewarp.loss_configs import LossConfig, LossScheduleConfig
from utilities.deepspeed_utils import deepspeed_enabled


@dataclass
class TrainingConfig:
    dataset: str
    model_config: ModelConfig
    # How large of a time-step to predict into the future
    step_width: int
    batch_size: int
    num_epochs: int
    patience: int
    data_augmentation: bool
    measure_equivariance_discrepancy: bool
    use_aml_logging: bool
    loss: LossConfig = LossConfig()
    loss_schedule: Optional[LossScheduleConfig] = None
    run_prefix: str = ""
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    seed: int = 0
    # If randomise_seed is True, seed will be changed when beginning training, and at every reallocation.
    randomise_seed: bool = False
    # leaving `data_dir=None` will cause the dataset to be downloaded
    # in a subdirectory in `dataset_cache_dir`
    data_dir: Optional[str] = None
    dataset_cache_dir: str = ".data"
    dataset_use_lmdb: bool = False
    # `pdb_dir` specifies a directory where we can find corresponding PDB files.
    # This directory will be walked, and so the files can be present in any child path.
    pdb_dir: Optional[str] = None
    output_folder: str = "outputs"
    enable_profiler: bool = False
    saved_model_path: Optional[str] = None
    valid_batch_size: int = 0  # default value overridden in __post_init__
    min_check_point_iters: int = 5000
    random_velocities: bool = True
    warm_start: bool = False  # if specified, and `saved_model_path` is not `None`, only the model weights will be loaded
    # Number of molecules per batch. Useful for controlling within-batch diversity in
    # datasets containing multiple systems.
    # NOTE: This only has an effect if `dataset_use_lmdb` is `True` as it requires `LMDBDistributedSampler`.
    # TODO : Change this to `num_pdbs_per_batch` and so it instead specifies the number of
    # PDBs per full batch, independent of the number of devices we're running on. This is similar in vein to
    # `batch_size` vs. an alternative `local_batch_size`.
    num_pdbs_per_local_batch: Optional[int] = None
    # The spacing between data pairs is not equal if the step_width is
    # at least as large as the spacing in the simulated trajectory.
    # Default is False as old runs did not consider this.
    equal_data_spacing: bool = False
    # If `True`, perform a run of validation dataset before starting training.
    # This can be useful for debugging certain model + large dataset combinations.
    run_valid_first: bool = True
    lr_scheduler: Optional[LRSchedulerConfig] = None

    def __post_init__(self):
        self.valid_batch_size = self.valid_batch_size or self.batch_size
        self.dataset_use_lmdb = self.dataset_use_lmdb or deepspeed_enabled()

        if self.num_pdbs_per_local_batch is not None and not self.dataset_use_lmdb:
            warnings.warn("num_pdbs_per_local_batch only has an effect if dataset_use_lmdb is true")
