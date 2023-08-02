from dataclasses import dataclass, field
from functools import cached_property
import os
from typing import Optional, Iterator, List, Sequence, Set, Any

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm  # type: ignore [import]

from timewarp.dataloader import (
    load_pdb_trace_data,
    CoordDeltaTooBig,
    MolDynDatapoint,
    TrajectoryInformation,
)
from utilities import StrPath, approximately_equal_partition
# from utilities.blob_storage import BlobClient, BlobNotFoundError
from utilities.downloadable_dataset import DownloadableDataset, DatasetStats


def get_pdb_names(data_dir: StrPath):
    # Collect the pdb names for simplicity:
    state0_suffix = "-traj-state0.pdb"
    pdb_names = []
    for f in os.listdir(data_dir):
        if f.endswith(state0_suffix):
            pdb_names.append(f[: -len(state0_suffix)])
    return sorted(pdb_names)


# The base dataclasses cannot have any default attributes if the subclass has attributes without a default (python < 3.10).
# To fix this, we need to introduce multiple base classes to inherit from in correct order.
@dataclass(frozen=True)
class _RawMolDynDatasetBase:
    data_dir: StrPath
    step_width: int


@dataclass(frozen=True)
class _RawMolDynDatasetDefaultBase:
    equal_data_spacing: bool = field(default=False)


@dataclass(frozen=True)
class RawMolDynDataset(_RawMolDynDatasetDefaultBase, _RawMolDynDatasetBase):
    @cached_property
    def pdb_names(self) -> Sequence[str]:
        pdb_names = get_pdb_names(self.data_dir)
        print(f"I: Found {len(pdb_names)} trace files in {self.data_dir}.")
        return tuple(pdb_names)

    def pdb_file_name(self, pdb_name: str):
        return f"{self.data_dir}/{pdb_name}-traj-state0.pdb"

    def npz_file_name(self, pdb_name: str):
        return f"{self.data_dir}/{pdb_name}-traj-arrays.npz"

    def _gracefully_load_pdb_trace_data(self, pdb_name: str) -> Optional[TrajectoryInformation]:
        """
        Returns a simulation trajectory corresponding to a pdb name. If the file doesn't exist or
        other error occurs, returns None.
        """
        try:
            return load_pdb_trace_data(
                pdb_name,
                self.pdb_file_name(pdb_name),
                self.npz_file_name(pdb_name),
                step_width=self.step_width,
                equal_data_spacing=self.equal_data_spacing,
            )
        except FileNotFoundError:
            print(f"W: {pdb_name} data not fully present.")
        except CoordDeltaTooBig as e:
            print(f"W: {e}.")
        except Exception as e:
            raise RuntimeError(
                f"Got unexpected exception while trying to load {self.pdb_file_name(pdb_name)} / {self.npz_file_name(pdb_name)}"
            ) from e
        return None

    def make_iterator(self, pdb_names: Sequence[str]) -> Iterator[MolDynDatapoint]:
        """
        Creates an iterator over a subset of trajectory data.

        Args:
            pdb_names: PDB names to iterate over.
        """
        with tqdm(
            unit="samples", bar_format="Loading trajectories {percentage:3.0f}% |{r_bar}"
        ) as progress:
            samples_seen_so_far = 0
            for i, pdb_name in enumerate(pdb_names):
                traj_info = self._gracefully_load_pdb_trace_data(pdb_name)
                if traj_info is None:
                    continue
                atom_types = torch.tensor(traj_info.node_types, dtype=torch.int64)
                adj_list = torch.tensor(traj_info.adj_list, dtype=torch.int64)
                # adjust the total depending on the number of samples per trajectory
                samples_seen_so_far += len(traj_info.coord_features)
                progress.total = samples_seen_so_far * len(pdb_names) // (i + 1)
                for (
                    coord_feats,
                    veloc_feats,
                    force_feats,
                    coord_targets,
                    veloc_targets,
                    force_targets,
                ) in zip(
                    traj_info.coord_features,
                    traj_info.veloc_features,
                    traj_info.force_features,
                    traj_info.coord_targets,
                    traj_info.veloc_targets,
                    traj_info.force_targets,
                ):
                    # TODO: support subsampling of traces here
                    yield MolDynDatapoint(
                        name=traj_info.name,
                        atom_types=atom_types,
                        adj_list=adj_list,
                        atom_coords=torch.tensor(coord_feats),
                        atom_velocs=torch.tensor(veloc_feats),
                        atom_forces=torch.tensor(force_feats),
                        atom_coord_targets=torch.tensor(coord_targets),
                        atom_veloc_targets=torch.tensor(veloc_targets),
                        atom_force_targets=torch.tensor(force_targets),
                    )
                    progress.update()


@dataclass(frozen=True)
class _IncrementalRawMolDynDatasetBase:
    downloader: Any


@dataclass(frozen=True)
class _IncrementalRawMolDynDatasetDefaultBase:
    download_attempted: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class IncrementalRawMolDynDataset(
    _IncrementalRawMolDynDatasetDefaultBase, RawMolDynDataset, _IncrementalRawMolDynDatasetBase
):
    @cached_property
    def pdb_names(self):
        suffix = "-traj-state0.pdb"
        result = [fn[: -len(suffix)] for fn in self.downloader.list_files() if fn.endswith(suffix)]
        print(
            f"I: Found {len(result)} trace files in blob storage {self.downloader.container_name}:/{self.downloader.blob_root}."
        )
        return result

    def _gracefully_load_pdb_trace_data(self, pdb_name: str) -> Optional[TrajectoryInformation]:
        if pdb_name not in self.download_attempted:
            try:
                self.downloader.download_file_if_missing(
                    os.path.basename(self.pdb_file_name(pdb_name))
                )
                self.downloader.download_file_if_missing(
                    os.path.basename(self.npz_file_name(pdb_name))
                )
            except BlobNotFoundError:
                # This happens due to some missing npz files.
                pass
            # Don't try to download again if the above failed
            self.download_attempted.add(pdb_name)
        return super()._gracefully_load_pdb_trace_data(pdb_name)


@dataclass(frozen=True)
class TrajectoryIterableDatasetIterator:
    raw_iterator: Iterator[MolDynDatapoint]
    num_traces_per_chunk: int
    shuffle: bool = field(default=False)
    skip_incomplete_chunks: bool = field(default=False)
    # This will hold the data when we are running:
    loaded_datapoints: List[MolDynDatapoint] = field(default_factory=list, repr=False)

    def __load_next_task_chunk(self) -> None:
        try:
            loaded_molecule_names: Set[str] = set()
            while len(loaded_molecule_names) < self.num_traces_per_chunk:
                datapoint = next(self.raw_iterator)
                loaded_molecule_names.add(datapoint.name)
                self.loaded_datapoints.append(datapoint)

        except StopIteration:
            # raw_iterator ran out before we accumulated a complete chunk
            if self.skip_incomplete_chunks:
                # We've reached the end of the task iterator - avoid chunks of data made up of fewer
                # tasks and clear out the bits we have:
                self.loaded_datapoints.clear()
            if len(self.loaded_datapoints) == 0:
                # We really don't have any data left:
                raise StopIteration
        finally:
            if self.shuffle:
                np.random.shuffle(self.loaded_datapoints)  # type: ignore

    def __next__(self) -> MolDynDatapoint:
        # First make sure that we have data in our buffer - if there's no more, this will bubble up
        # a StopIteration:
        if len(self.loaded_datapoints) == 0:
            self.__load_next_task_chunk()

        return self.loaded_datapoints.pop(0)

    def __iter__(self):
        return self


@dataclass(frozen=True)
class TrajectoryIterableDataset(DownloadableDataset[DatasetStats], IterableDataset):
    num_traces_per_chunk: int = field(default=16)
    step_width: int = field(default=1)
    shuffle: bool = field(default=False)
    equal_data_spacing: bool = field(default=False)

    @cached_property
    def raw_dataset(self):
        if self.downloader:
            return IncrementalRawMolDynDataset(
                data_dir=self.data_dir,
                step_width=self.step_width,
                equal_data_spacing=self.equal_data_spacing,
                downloader=self.downloader,
            )
        return RawMolDynDataset(
            data_dir=self.data_dir,
            step_width=self.step_width,
            equal_data_spacing=self.equal_data_spacing,
        )

    @staticmethod
    def check_dir(data_dir: StrPath, expected_dataset_stats: DatasetStats):
        return len(get_pdb_names(data_dir)) == expected_dataset_stats.num_files

    def pdb_names_for_worker(self, worker_id: int, num_workers: int):
        pdb_names = self.raw_dataset.pdb_names
        return (
            pdb_names
            if num_workers == 1
            else approximately_equal_partition(pdb_names, num_workers)[worker_id]
        )

    def make_iterator_for_pdb_names(self, pdb_names: Sequence[str]):
        return TrajectoryIterableDatasetIterator(
            raw_iterator=self.raw_dataset.make_iterator(pdb_names),
            num_traces_per_chunk=self.num_traces_per_chunk,
            shuffle=self.shuffle,
        )

    def make_iterator_for_worker(self, worker_id: int, num_workers: int):
        pdb_names = self.pdb_names_for_worker(worker_id, num_workers)
        if num_workers > 1:
            # This resets the seed of each worker to the `current seed` + `worker id`.
            # As the current seed is copied from the main process, and we reset it to the step
            # number at the beginning of each epoch, this means that we get a different order
            # in each run.
            state = np.random.get_state(legacy=True)
            assert isinstance(state, tuple)
            np.random.seed(state[1][0] + worker_id)

        return self.make_iterator_for_pdb_names(pdb_names)

    def __iter__(self):
        # If we are using several workers, we want to split the data files to load between
        # the different workers:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return one full iterator
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return self.make_iterator_for_worker(worker_id, num_workers)
