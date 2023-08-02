from torch.utils.data.dataloader import DataLoader
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.datasets.lmdb_dataset import LmdbTrajectoryDataset

from timewarp.datasets.pdb_sampler import LMDBDistributedSampler
from timewarp.utils.dataset_utils import get_dataset
from utilities.training_utils import set_seed


def _get_dataloader(dataset, sampler, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=moldyn_dense_collate_fn,  # type: ignore
        pin_memory=True,
        sampler=sampler,
    )


def test_lmdb_distributed_sampler():
    set_seed(0)

    # Let's use `T1` here since it's already used by other tests, e.g. `test_lmdb_dataset.py`.
    dataset_name = "T1"
    batch_size = 8

    # Create dataset. Use `test` since it's the smallest.
    (dataset,) = get_dataset(
        dataset_name,
        cache_dir=".data",
        step_width=1000,
        lmdb=True,
        splits=["val"],
    )

    assert isinstance(dataset, LmdbTrajectoryDataset)

    # No partitioning.
    sampler = LMDBDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    sampler_shuffled = LMDBDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    assert set(sampler.pdb_names) == set(dataset.pdb_indices.keys())

    batch_dataloader = _get_dataloader(dataset, sampler, batch_size)
    batch_dataloader_shuffled = _get_dataloader(dataset, sampler_shuffled, batch_size)

    # These two batches should have non-empty intersection.
    batch, batch_shuffled = next(zip(iter(batch_dataloader), iter(batch_dataloader_shuffled)))
    assert len(set(batch_shuffled.names) - set(batch.names)) > 0

    # Control diversity within a batch.
    max_contiguous_length = 2
    sampler = LMDBDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        max_contiguous_length=max_contiguous_length,
        shuffle=True,
    )
    batch_dataloader = _get_dataloader(dataset, sampler, batch_size)

    # Every protein should (with almost prob 1) have at most `max_contiguous_length` examples
    # in this batch, so the number of different proteins for which we see examples should
    # be `batch_size // max_contiguous_length`.
    batch = next(iter(batch_dataloader))
    assert len(set(batch.names)) == batch_size // max_contiguous_length

    # Maximum partitioning.
    sampler = LMDBDistributedSampler(
        dataset, num_replicas=len(dataset.pdb_indices), rank=0, shuffle=False
    )
    sampler_shuffled = LMDBDistributedSampler(
        dataset, num_replicas=len(dataset.pdb_indices), rank=0, shuffle=True
    )
    assert len(sampler.pdb_names) == 1
    assert len(sampler_shuffled.pdb_names) == 1

    batch_dataloader = _get_dataloader(dataset, sampler, batch_size)
    batch_dataloader_shuffled = _get_dataloader(dataset, sampler_shuffled, batch_size)

    # These batches should only contain a single protein and it should be the same.
    batch, batch_shuffled = next(zip(iter(batch_dataloader), iter(batch_dataloader_shuffled)))
    assert len(set(batch_shuffled.names) - set(batch.names)) == 0
