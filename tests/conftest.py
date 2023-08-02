from typing import List

import pytest
import torch

from timewarp.dataloader import MolDynDatapoint


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_datapoints() -> List[MolDynDatapoint]:
    # A point with 3 atoms
    point1 = MolDynDatapoint(
        name="point1",
        atom_types=torch.tensor([0, 2, 0]),
        adj_list=torch.tensor([[0, 1], [1, 0]]),
        atom_coords=torch.randn(3, 3),
        atom_velocs=torch.randn(3, 3),
        atom_forces=torch.randn(3, 3),
        atom_coord_targets=torch.randn(3, 3),
        atom_veloc_targets=torch.randn(3, 3),
        atom_force_targets=torch.randn(3, 3),
    )
    # A point with 5 atoms
    point2 = MolDynDatapoint(
        name="point2",
        atom_types=torch.tensor([0, 2, 0, 2, 1]),
        adj_list=torch.tensor([[0, 2]]),
        atom_coords=torch.randn(5, 3),
        atom_velocs=torch.randn(5, 3),
        atom_forces=torch.randn(5, 3),
        atom_coord_targets=torch.randn(5, 3),
        atom_veloc_targets=torch.randn(5, 3),
        atom_force_targets=torch.randn(5, 3),
    )
    return [point1, point2]
