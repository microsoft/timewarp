from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lmdb  # type: ignore [import]
import numpy as np
from torch.utils.data.dataset import Dataset

from timewarp.dataloader import MolDynDatapoint
from utilities import StrPath
from utilities.downloadable_dataset import DownloadableDataset, DatasetStats
from utilities.lmdb_utils import (
    get_envs,
    get_indices,
    get_length,
    lmdb_get,
    Metadata,
)


class StepWidth(Metadata):
    name = "step_width"
    is_frozen = True

    def __init__(self, value: int):
        self._value = value


class NumAtoms(Metadata[MolDynDatapoint]):
    name = "num_atoms"

    def __init__(self, value: Optional[List[int]] = None):
        self._value = value or []

    def check(self, num_points: int):
        assert len(self.value) == num_points

    def update(self, index: int, sample: MolDynDatapoint):
        self._value.append(sample.num_atoms)


class PdbIndices(Metadata[MolDynDatapoint]):
    name = "pdb_indices"

    def __init__(self, value: Optional[Dict[str, List[int]]] = None):
        self._value = value or defaultdict(list)

    def check(self, num_points: int):
        assert sorted(sum(self._value.values(), [])) == list(range(num_points))

    def update(self, index: int, sample: MolDynDatapoint):
        self._value[sample.name].append(index)


def init_all_metadata(step_width: int):
    return [StepWidth(step_width), NumAtoms(), PdbIndices()]


# Based on https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/datasets/trajectory_lmdb.py
@dataclass(frozen=True)
class LmdbTrajectoryDataset(DownloadableDataset[DatasetStats], Dataset):
    step_width: int = field(default=1)
    envs: List[lmdb.Environment] = field(default_factory=list, repr=False)
    cum_lengths: List[int] = field(default_factory=list, repr=False)
    num_atoms: List[int] = field(default_factory=list, repr=False)
    pdb_indices: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    equal_data_spacing: bool = field(default=False)

    def __post_init__(self):
        if self.downloader is not None:
            self.download_all()

        lengths = []
        for env in get_envs(self.data_dir):
            with env.begin() as txn:
                stored_step_width = lmdb_get(txn, "step_width")
                num_atoms = lmdb_get(txn, "num_atoms", [])
                pdb_indices = lmdb_get(txn, "pdb_indices", {})

            if stored_step_width != self.step_width:
                raise ValueError(
                    f"Expected step_width={self.step_width}, but LMDB file {env.path()} contains step_width={stored_step_width}."
                )

            self.envs.append(env)

            # store number of atoms in each data point
            self.num_atoms.extend(num_atoms)

            # store the mapping from pdb_name to indices
            start_index = sum(lengths)
            self.pdb_indices.update(
                {
                    pdb_name: [start_index + i for i in indices]
                    for pdb_name, indices in pdb_indices.items()
                }
            )

            # update the lengths
            lengths.append(get_length(env))

        if len(self.envs) == 0:
            raise ValueError(f"No LMDB files found in {self.data_dir}.")

        assert len(self.cum_lengths) == 0
        self.cum_lengths.extend(np.cumsum(lengths).tolist())

        assert len(self.num_atoms) == len(self)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, index: int) -> MolDynDatapoint:
        db_index, el_index = get_indices(self.cum_lengths, index)
        with self.envs[db_index].begin() as txn:
            return lmdb_get(txn, str(el_index))

    @staticmethod
    def check_dir(data_dir: StrPath, expected_dataset_stats: DatasetStats):
        total_num_points = 0
        for env in get_envs(data_dir):
            total_num_points += get_length(env)
            env.close()
        # num_points is not a fixed value for a data_set
        # e.g. if the step_width is larger than the sapcing, the number of training pairs will be smaller.
        if expected_dataset_stats.num_points:
            return total_num_points == expected_dataset_stats.num_points
        # Return True if the number of points is not specified in the data_set
        else:
            return True

    def close_db(self):
        for env in self.envs:
            env.close()
