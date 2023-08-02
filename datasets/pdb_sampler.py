import math
from typing import TypeVar, Optional, Iterator, Sequence

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

from timewarp.datasets.lmdb_dataset import LmdbTrajectoryDataset


T_co = TypeVar("T_co", covariant=True)


# TODO : Move?
def chunks_iterator(a: Sequence, length_per_chunk: int):
    return (a[i : i + length_per_chunk] for i in range(0, len(a), length_per_chunk))


class LMDBDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset based on PDB files.

    This is in effect very similar to :class:`torch.utils.data.distributed.DistributedSampler`
    but with a couple of exceptions:
    1. The provided `dataset` has to be an instance of :class:`LmdbTrajectoryDataset`.
    2. The splitting of the dataset is not performed based on the indices of the data,
       but instead based on the proteins available the dataset.

    Depending on the `rank` and `num_replicas`, the iterator produced by an instance of this
    class will only return examples from a subset of the proteins available in the dataset.

    If `shuffle` is `True`, by default the data subset will be flatten and shuffled. This means
    that taking, say, 5 examples from its iterator will likely result in data examples from
    5 different proteins, as expected. Depending on what sort of computations one wants to perform
    on the data, such a high diversity of proteins can be determinental for computational performance.
    In these cases, it can be useful to specify the `max_continuous_length` which provides some control
    over the diversity in the produced sequence of examples. If we set, say, `max_continugous_length=2`
    and then take, say, 5 examples from its iterator, we will now only see examples for 3 different
    proteins: `[protein_1, protein_1, protein_2, protein_2, protein_3]`.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        max_contiguous_length (int, optional): if specified and :attr:`shuffle=True`,
            its iterator will produce a sequence containing single-protein subsequences
            of (at most) this length.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    """

    def __init__(
        self,
        dataset: LmdbTrajectoryDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_contiguous_length: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.max_contiguous_length = max_contiguous_length

        # NOTE : We sort, just in case the ordering is different across workers for some reason.
        # TODO : Is `sorted(...)` unnecessary?
        pdb_names = sorted(dataset.pdb_indices.keys())
        num_pdb_names = len(pdb_names)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and num_pdb_names % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_pdb_names = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (num_pdb_names - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_pdb_names = math.ceil(num_pdb_names / self.num_replicas)  # type: ignore[arg-type]

        self.total_num_pdb_names = self.num_pdb_names * self.num_replicas

        # Determine which keys correspond to this `rank`.
        self.pdb_names = pdb_names[self.rank : self.total_num_pdb_names : self.num_replicas]

        # Flatten the indices used in this worker.
        self.local_indices = [idx for name in self.pdb_names for idx in dataset.pdb_indices[name]]

        print(
            f"LMDBDistributedSampler: rank={rank} largest molecule={max([dataset.num_atoms[idx] for idx in self.local_indices])} pdb_names={self.pdb_names}"
        )

        # Shuffling related.
        self.shuffle = shuffle
        self.seed = seed

        # NOTE : Assumes there's an equal number of samples for every `pdb_name`!!!
        # This is only needed to define `num_samples`, which in turn is used to define `__len__`.
        self.num_samples_per_pdb_name = len(dataset.pdb_indices[self.pdb_names[0]])
        assert all(
            len(dataset.pdb_indices[name]) == self.num_samples_per_pdb_name
            for name in self.pdb_names[1:]
        ), "currently only an equal number of examples for protein is supported"
        self.num_samples = self.num_samples_per_pdb_name * len(self.pdb_names)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if self.max_contiguous_length is None:
                # Here we just shuffle the flatten data indices.
                indices = [
                    self.local_indices[i]
                    for i in torch.randperm(len(self.local_indices), generator=g).tolist()
                ]
            else:
                # 1. Shuffle per-PDB examples.
                pdb_indices = [
                    [
                        self.dataset.pdb_indices[name][i]
                        for i in torch.randperm(
                            len(self.dataset.pdb_indices[name]), generator=g
                        ).tolist()
                    ]
                    for name in self.pdb_names
                ]

                # 2. Split the per-PDB indices into chunks of length `self.max_contiguous_length`.
                pdb_indices_segmented = [
                    chunk
                    for segment in pdb_indices
                    for chunk in chunks_iterator(segment, self.max_contiguous_length)
                ]

                # 3. Shuffle segments.
                pdb_segment_indices = torch.randperm(
                    len(pdb_indices_segmented), generator=g
                ).tolist()

                # 4. Flatten the shuffled segments.
                indices = [
                    idx
                    for segment_idx in pdb_segment_indices
                    for idx in pdb_indices_segmented[segment_idx]
                ]
        else:
            indices = [i for i in self.local_indices]  # copy

        if not self.drop_last:
            # Add extra samples to the data to make evenly divisible.
            padding_size = self.num_samples - len(indices)
            indices += indices[:padding_size]
        else:
            # Remove tail of the data to make evenly divisible.
            indices = indices[: self.num_samples]

        # subsample
        return iter(indices)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
