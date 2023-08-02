import os

import torch
import numpy as np
from itertools import islice
from timewarp.datasets import RawMolDynDataset
from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.utils.chirality import (
    find_chirality_centers,
    compute_chirality_sign,
    check_symmetry_change,
    CiralityChecker,
)
from utilities.training_utils import set_seed
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from timewarp.datasets import TrajectoryIterableDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_chirality_changes():
    # Load test protein 1hgv
    testdata_dir = os.path.dirname(os.path.realpath(__file__))
    testdata_dir = os.path.join(testdata_dir, "..", "testdata", "output")
    pdb_name = "1hgv"
    traj_path = os.path.join(testdata_dir, f"{pdb_name}-traj-arrays.npz")
    traj_coords = np.load(traj_path)["positions"]
    # Load the input data.
    raw_dataset = RawMolDynDataset(data_dir=testdata_dir, step_width=1000)
    raw_iterator = raw_dataset.make_iterator([pdb_name])
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, 2))

    # Compute reference chirality centers from initial state
    batch = batches[0]
    chirality_centers = find_chirality_centers(batch.adj_list, batch.atom_types)
    reference_signs = compute_chirality_sign(batch.atom_coords, chirality_centers)

    # There are 54 chirality centers in 1hgv
    assert len(chirality_centers) == 54
    assert reference_signs.shape == torch.Size([1, 54])

    # There should be no chirality changes in an MD trajectory
    changes = check_symmetry_change(
        torch.from_numpy(traj_coords), chirality_centers, reference_signs
    )
    assert torch.equal(changes, torch.zeros(traj_coords.shape[0], dtype=bool))


# TODO test the chirality checker with the test batch!!


def test_chirality_checker():
    set_seed(0)

    batch_size = 4
    # Create dataset.
    datapath = os.path.join(os.path.dirname(__file__), "../testdata/output/")
    dataset = TrajectoryIterableDataset(
        data_dir=Path(datapath).expanduser().resolve(), step_width=1, shuffle=True
    )
    chirality_checker = CiralityChecker(datapath)

    batch_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=moldyn_dense_collate_fn,
        pin_memory=True,
    )

    with torch.no_grad():
        for _, batch in enumerate(batch_dataloader):
            chirality_changes = chirality_checker.check_changes(
                batch, batch.atom_coord_targets, batch.masked_elements
            )
            # There should be no chirality change between the conditioning and target states in the batches
            assert torch.equal(
                chirality_changes, torch.zeros((batch.atom_coords.shape[0]), dtype=bool)
            )

    # The reference chirality signs are stored for both proteins in the test dataset
    assert len(chirality_checker._chirality_reference_cache.keys()) == 2
    # There are 54 chirality centers in 1hgv
    chirality_centers, reference_signs = chirality_checker._chirality_reference_cache["1hgv"]
    assert len(chirality_centers) == 54
    assert reference_signs.shape == torch.Size([1, 54])

    with torch.no_grad():
        for _, batch in enumerate(batch_dataloader):
            chirality_changes = chirality_checker.check_changes(
                batch, -batch.atom_coord_targets, batch.masked_elements
            )
            # The chirality changes if we mirror the configuration
            assert torch.equal(
                chirality_changes, torch.ones((batch.atom_coords.shape[0]), dtype=bool)
            )
