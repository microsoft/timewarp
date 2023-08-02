import torch
import sys
import os
from typing import List, Union, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from timewarp.equivariance.equivariance_utils import (
    random_rotation_matrix,
    random_translation_vector,
    random_permutation,
)
from timewarp.dataloader import DenseMolDynBatch


class BaseDataTransformation:
    """
    A base data transformation that transforms all types of attributes with
    the identity transform
    """

    def transform_pointwise_feature(self, pointwise_feature: torch.Tensor) -> torch.Tensor:
        return pointwise_feature

    def transform_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return coord

    def transform_veloc(self, veloc: torch.Tensor) -> torch.Tensor:
        return veloc

    def transform_adjacency_list(self, adj_list: torch.Tensor) -> torch.Tensor:
        return adj_list

    def __add__(self, other: "BaseDataTransformation") -> "CompositionTransformation":
        return CompositionTransformation(transforms=[self, other])


class CompositionTransformation:
    """
    A composition of multiple transforms that applies the transforms in sequence.
    """

    def __init__(self, transforms: List[BaseDataTransformation]):
        self.transforms = transforms

    def transform_pointwise_feature(self, pointwise_feature: torch.Tensor) -> torch.Tensor:
        trans_feature = pointwise_feature

        for transform in self.transforms:
            trans_feature = transform.transform_pointwise_feature(trans_feature)

        return trans_feature

    def transform_coord(self, coord: torch.Tensor) -> torch.Tensor:
        trans_coord = coord

        for transform in self.transforms:
            trans_coord = transform.transform_coord(trans_coord)

        return trans_coord

    def transform_veloc(self, veloc: torch.Tensor) -> torch.Tensor:
        trans_veloc = veloc

        for transform in self.transforms:
            trans_veloc = transform.transform_veloc(trans_veloc)

        return trans_veloc

    def transform_adjacency_list(self, adj_list: torch.Tensor) -> torch.Tensor:
        trans_adj_list = adj_list

        for transform in self.transforms:
            trans_adj_list = transform.transform_adjacency_list(adj_list)

        return trans_adj_list

    def __add__(
        self, other: Union[BaseDataTransformation, "CompositionTransformation"]
    ) -> "CompositionTransformation":
        # TODO: Once moved to Python 3.8 use @functools.singledispatchmethod
        if isinstance(other, BaseDataTransformation):
            return CompositionTransformation(transforms=self.transforms + [other])
        elif isinstance(other, CompositionTransformation):
            return CompositionTransformation(transforms=self.transforms + other.transforms)


class Permutation(BaseDataTransformation):
    # TODO: Support batching
    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation
        self.inv_permutation = torch.argsort(permutation)

    def transform_pointwise_feature(self, pointwise_feature: torch.Tensor) -> torch.Tensor:
        if pointwise_feature.ndim > 2:
            raise NotImplementedError(
                f"Permutation transform doesn't work with shape {pointwise_feature.shape}"
            )
        return pointwise_feature[self.inv_permutation]

    def transform_coord(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.ndim > 2:
            raise NotImplementedError(
                f"Permutation transform doesn't work with shape {coord.shape}"
            )
        return coord[self.inv_permutation]

    def transform_veloc(self, veloc: torch.Tensor) -> torch.Tensor:
        if veloc.ndim > 2:
            raise NotImplementedError(
                f"Permutation transform doesn't work with shape {veloc.shape}"
            )
        return veloc[self.inv_permutation]

    def transform_adjacency_list(self, adj_list: torch.Tensor) -> torch.Tensor:
        return self.permutation[adj_list]


class Rotation(BaseDataTransformation):
    def __init__(self, rotation_matrix: torch.Tensor):
        self.rotation_matrix = rotation_matrix

    def transform_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return (self.rotation_matrix @ coord.transpose(-1, -2)).transpose(-1, -2)

    def transform_veloc(self, veloc: torch.Tensor) -> torch.Tensor:
        return (self.rotation_matrix @ veloc.transpose(-1, -2)).transpose(-1, -2)


class Translation(BaseDataTransformation):
    def __init__(self, translation_vector: torch.Tensor):
        self.translation_vector = translation_vector

    def transform_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return coord + self.translation_vector


class RandomPermutation(Permutation):
    # TODO: Support batching
    def __init__(self, num_points: int, device: Optional[str] = None):
        super().__init__(permutation=random_permutation(num_points, device=device))


class RandomRotation(Rotation):
    def __init__(self, device: Optional[str] = None, dtype=torch.float32):
        super().__init__(rotation_matrix=random_rotation_matrix(device=device, dtype=dtype))


class RandomTranslation(Translation):
    def __init__(self, device: Optional[str] = None, dtype=torch.float32):
        super().__init__(translation_vector=random_translation_vector(device=device, dtype=dtype))


def transform_batch(batch: DenseMolDynBatch, transform=None, dtype=torch.float32):
    if transform is None:
        # Apply a random rotation + translation transformation
        transform = RandomTranslation(dtype=dtype) + RandomRotation(dtype=dtype)

    transformed_batch = DenseMolDynBatch(
        names=batch.names,
        atom_types=transform.transform_pointwise_feature(batch.atom_types),
        adj_list=transform.transform_adjacency_list(batch.adj_list),  # TODO: Support sparse graph
        edge_batch_idx=batch.edge_batch_idx,
        atom_coords=transform.transform_coord(batch.atom_coords),
        atom_velocs=transform.transform_veloc(batch.atom_velocs),
        atom_forces=transform.transform_veloc(
            batch.atom_forces
        ),  # Forces transform like velocities
        atom_coord_targets=transform.transform_coord(batch.atom_coord_targets),
        atom_veloc_targets=transform.transform_veloc(batch.atom_veloc_targets),
        atom_force_targets=transform.transform_veloc(
            batch.atom_force_targets
        ),  # Forces transform like velocities
        masked_elements=batch.masked_elements,  # TODO: support permutation
    )
    return transformed_batch
