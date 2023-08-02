from torch.utils.data.dataset import IterableDataset
import torch.distributed as dist

from .common import unique_item
from .training_utils import RngState


def check_rng_state():
    """
    Asserts that all ranks have the same RNG state.
    """
    if not dist.is_initialized():
        # not distributed
        return
    if dist.get_rank() == 0:
        objects = [RngState.get_rng_state()]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    assert RngState.get_rng_state() == unique_item(objects)


class DistributedIterableDataset(IterableDataset):
    """
    Wrapper for IterableDataset in a data-parallel setting.

    Args:
        dataset: an iterable dataset.
        num_replicas: number of data parallel replicas.
        rank: rank within the replicas.

    Note:
        This class does not take care of shuffling. It assumes that all
        ranks shuffles the data in exactly the same way.
    """

    def __init__(self, dataset: IterableDataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        check_rng_state()
        current_batch = []
        current_sample = None
        for sample in self.dataset:
            current_batch.append(sample)
            if len(current_batch) == self.num_replicas:
                current_sample = current_batch[self.rank]
                yield current_sample
                current_batch.clear()

        if len(current_batch) > 0 and current_sample is not None:
            # all ranks have yielded at least one sample.
            # repeat the last sample if the batch is not filled.
            yield current_batch[self.rank] if self.rank < len(current_batch) else current_sample
