from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data.dataset import Dataset

from timewarp.datasets import TrajectoryIterableDataset, LmdbTrajectoryDataset
from utilities.downloadable_dataset import DatasetNotFoundError, ContainerBlobPath, DatasetStats


def possibly_download_trajectory_dataset(
    data_dir: os.PathLike,
    container_blob_path: ContainerBlobPath,
    expected_stats: DatasetStats,
    download: bool = True,
    lmdb: bool = False,
    **kwargs,
) -> Dataset:
    def make(cls):
        print(f"Loading dataset {cls.__qualname__} in {data_dir}")
        return cls.possibly_download_dataset(
            data_dir=data_dir,
            container_blob_path=container_blob_path,
            expected_dataset_stats=expected_stats,
            download=download,
            **kwargs,
        )

    return make(LmdbTrajectoryDataset if lmdb else TrajectoryIterableDataset)


def get_container_blob_path(
    root_blob_path: str, step_width: int, lmdb: bool, split: str = ""
) -> ContainerBlobPath:
    """
    Compose ContainerBlobPath from root_blob_path, step_width, split and lmdb flag.
    """
    parts = (
        [root_blob_path]
        + ([f"step_width_{step_width}"] if lmdb else [])
        + ([split] if len(split) > 0 else [])
    )
    return ContainerBlobPath(
        container_name="lmdb-simulated-data" if lmdb else "simulated-data",
        blob_path="/".join(parts),
    )


@dataclass(frozen=True)
class DatasetMetadata:
    root_blob_path: str
    expected_stats: Dict[str, DatasetStats]


def _get_trajectory_datasets(
    metadata: DatasetMetadata,
    step_width: int,
    splits: Optional[List[str]] = None,
    data_dir: Optional[os.PathLike] = None,
    cache_dir: Optional[os.PathLike] = None,
    lmdb: bool = False,
    shuffle: Optional[bool] = None,
    **kwargs,
) -> Tuple[Dataset, ...]:
    if splits is None:
        splits = list(metadata.expected_stats.keys())
    if not data_dir:
        assert cache_dir is not None
        cb = get_container_blob_path(metadata.root_blob_path, step_width, lmdb)
        data_dir = Path(cache_dir) / cb.container_name / cb.blob_path
        kwargs.setdefault("download", True)
    else:
        # When data_dir is specified, dont' try to download so as not to require interactive
        # sign-in in a remote run.
        kwargs["download"] = False

    def shuffle_kwargs(split: str):
        # don't specify shuffle if using lmdb (instead it is shuffled by the sampler)
        return {} if lmdb else dict(shuffle=shuffle if shuffle is not None else (split == "train"))

    try:
        return tuple(
            possibly_download_trajectory_dataset(
                data_dir=Path(data_dir) / split,
                container_blob_path=get_container_blob_path(
                    metadata.root_blob_path, step_width, lmdb, split
                ),
                expected_stats=metadata.expected_stats[split],
                step_width=step_width,
                lmdb=lmdb,
                **shuffle_kwargs(split),
                **kwargs,
            )
            for split in splits
        )
    except DatasetNotFoundError as e:
        raise RuntimeError(
            f"Could not find {e.cls} data in {data_dir}. Try `data_dir=None`, then the data will be automatically downloaded to {cache_dir}."
        ) from e


_datasets = {
    # Datasets created with the amber-99 force-field #
    # Short trajectories of small proteins (100-800 atoms)
    "T1": DatasetMetadata(
        root_blob_path="trajectory-data/T1-peptides/sim-amber99-10ns",
        expected_stats={
            "train": DatasetStats(num_files=100, num_points=1786),
            "val": DatasetStats(num_files=40, num_points=665),
            "test": DatasetStats(num_files=49, num_points=874),
        },
    ),
    # Same as T1, but longer trajectories
    "T1-large": DatasetMetadata(
        root_blob_path="trajectory-data/T1-peptides-large",
        expected_stats={
            "train": DatasetStats(num_files=71, num_points=710000),
            "val": DatasetStats(num_files=34, num_points=340000),
            "test": DatasetStats(num_files=46, num_points=460000),
        },
    ),
    # Short trajectory of alanine dipeptide (misses phi transition)
    "AD-1": DatasetMetadata(
        root_blob_path="trajectory-data/AD-1",
        expected_stats={
            "train": DatasetStats(num_files=1, num_points=10000),
            "test": DatasetStats(num_files=1, num_points=10000),
        },
    ),
    # Long trajectory of alanine dipeptide
    "AD-2": DatasetMetadata(
        root_blob_path="trajectory-data/AD-2",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Datasets created with the amber-14 force-field #
    # Same as T1-large
    "T1-large-like-T1B": DatasetMetadata(
        root_blob_path="trajectory-data/T1-large-like-T1B",
        expected_stats={
            "train": DatasetStats(num_files=64),
            "val": DatasetStats(num_files=29),
            "test": DatasetStats(num_files=41),
        },
    ),
    # More peptides than T-1-large and longer trajectories
    "T1B": DatasetMetadata(
        root_blob_path="trajectory-data/T1B-peptides",
        expected_stats={
            "train": DatasetStats(num_files=80),
            "val": DatasetStats(num_files=29),
            "test": DatasetStats(num_files=41),
        },
    ),
    # Short trajectories of dipeptides
    "2AA-1": DatasetMetadata(
        root_blob_path="trajectory-data/2AA-1",
        expected_stats={
            "train": DatasetStats(num_files=200),
            "val": DatasetStats(num_files=80),
            "test": DatasetStats(num_files=100),
        },
    ),
    # Same as 2AA-1, but longer trajectories.
    # Note: States are NOT saved in the usual logarithmic way.
    # For timewarp use 2AA-1-big
    "2AA-1-large": DatasetMetadata(
        root_blob_path="trajectory-data/2AA-1-large",
        expected_stats={
            "train": DatasetStats(num_files=200),
            "val": DatasetStats(num_files=80),
            "test": DatasetStats(num_files=100),
        },
    ),
    # Same as 2AA-1, but longer trajectories.
    # States are saved every 2000 MD steps (no intermediate steps)
    "2AA-1-big": DatasetMetadata(
        root_blob_path="trajectory-data/2AA-1-big",
        expected_stats={
            "train": DatasetStats(num_files=200),
            "val": DatasetStats(num_files=80),
            "test": DatasetStats(num_files=100),
        },
    ),
    # 2AA dataset that also contain the missing peptides
    # from the other 2AA datasets.
    "2AA-complete": DatasetMetadata(
        root_blob_path="trajectory-data/2AA-complete",
        expected_stats={
            "train": DatasetStats(num_files=208),
            "val": DatasetStats(num_files=88),
            "test": DatasetStats(num_files=104),
        },
    ),
    # Short trajectories of tetrapeptides
    "4AA": DatasetMetadata(
        root_blob_path="trajectory-data/4AA",
        expected_stats={
            "train": DatasetStats(num_files=1500, num_points=1996 * 1500),
            "val": DatasetStats(num_files=400, num_points=1996 * 400),
            "test": DatasetStats(num_files=433, num_points=1996 * 433),
        },
    ),
    # Long trajectories of tetrapeptides
    "4AA-large": DatasetMetadata(
        root_blob_path="trajectory-data/4AA-large",
        expected_stats={
            "train": DatasetStats(num_files=1456),
            "val": DatasetStats(num_files=372),
            "test": DatasetStats(num_files=416),
        },
    ),
    # Same simulation length as 4AA-large, but larger spacing
    "4AA-big": DatasetMetadata(
        root_blob_path="trajectory-data/4AA-big",
        expected_stats={
            "train": DatasetStats(num_files=1464),
            "val": DatasetStats(num_files=374),
            "test": DatasetStats(num_files=416),
        },
    ),
    # Same as 4AA-big, but 8 fewer tetrapeptides
    "4AA-big2": DatasetMetadata(
        root_blob_path="trajectory-data/4AA-big2",
        expected_stats={
            "train": DatasetStats(num_files=1456),
            "val": DatasetStats(num_files=372),
            "test": DatasetStats(num_files=372),
        },
    ),
    # Same as 4AA-large, but with repeated phi transition pairs
    "4AA-biased": DatasetMetadata(
        root_blob_path="trajectory-data/4AA-biased",
        expected_stats={
            "train": DatasetStats(num_files=1464),
            "val": DatasetStats(num_files=378),
            "test": DatasetStats(num_files=416),
        },
    ),
    # Some very long trajectories for the 4AA dataset for evaluations
    "4AA-huge": DatasetMetadata(
        root_blob_path="trajectory-data/4AA-huge",
        expected_stats={
            "train": DatasetStats(num_files=1500),
            "val": DatasetStats(num_files=379),
            "test": DatasetStats(num_files=433),
        },
    ),
    # Long trajectory for alanine dipeptide
    "AD-3": DatasetMetadata(
        root_blob_path="trajectory-data/AD-3",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Long trajectory for alanine dipeptide at 600K
    "AD-600": DatasetMetadata(
        root_blob_path="trajectory-data/AD-600",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Long trajectory for alanine dipeptide at 400K
    "AD-400": DatasetMetadata(
        root_blob_path="trajectory-data/AD-400",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # 10K conditioning states with 100 target states each
    "AD-3conditional10000": DatasetMetadata(
        root_blob_path="trajectory-data/AD-3conditional10000",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Same as AD-3, but with repeated phi transition pairs (50x)
    "AD-3-biased": DatasetMetadata(
        root_blob_path="trajectory-data/AD-3-biased",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Same as AD-3, but with repeated phi transition pairs (10x)
    "AD-3-biased-10": DatasetMetadata(
        root_blob_path="trajectory-data/AD-3-biased-10",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Same as AD-3-biased-10, but also with each phi transition reversed
    "AD-3-biased-reversed": DatasetMetadata(
        root_blob_path="trajectory-data/AD-3-biased-reversed",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Long trajectory for single tetrapetide LAKS
    "LAKS": DatasetMetadata(
        root_blob_path="trajectory-data/LAKS",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Oxygen molecule
    "O2": DatasetMetadata(
        root_blob_path="trajectory-data/O2",
        expected_stats={
            "train": DatasetStats(num_files=1),
            "test": DatasetStats(num_files=1),
        },
    ),
    # Short test trajectory
    "test": DatasetMetadata(
        root_blob_path="trajectory-data/None",
        expected_stats={"smallest_molecule": DatasetStats(num_files=1)},
    ),
}


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    if dataset_name not in _datasets:
        raise ValueError(
            f"Unknown dataset {dataset_name}. Available datasets are: {list(_datasets.keys())}"
        )
    return _datasets[dataset_name]


def get_dataset(
    dataset_name: str,
    data_dir: Optional[os.PathLike] = None,
    cache_dir: Optional[os.PathLike] = None,
    **kwargs,
) -> Tuple[Dataset, ...]:
    """
    Get dataset splits.

    Args:
        dataset_name: name of the dataset (T1, T1-large, AD-1, etc)
        data_dir: where the specified dataset is stored locally. If `data_dir` is specified
            but the dataset is not found, an error will be raised. If `data_dir` is not specified,
            it is downloaded in a subdirectory under `cache_dir`.
        cache_dir: where to store the dataset locally. If the dataset is not found in the
            expected subdirectory under `cache_dir`, the dataset will be automatically
            downloaded.
    """
    metadata = get_dataset_metadata(dataset_name)
    return _get_trajectory_datasets(
        metadata=metadata, data_dir=data_dir, cache_dir=cache_dir, **kwargs
    )
