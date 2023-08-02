import pytest
import dataclasses
import torch
import numpy as np
import os
from pathlib import Path

from timewarp.dataloader import DenseMolDynBatch, moldyn_dense_collate_fn, load_pdb_trace_data


AVAILABLE_DEVICES = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]


def _batch_compare(batch_left: DenseMolDynBatch, batch_right: DenseMolDynBatch) -> bool:
    return (
        batch_left.names == batch_right.names
        and torch.all(batch_left.atom_types == batch_right.atom_types)
        and torch.all(batch_left.adj_list == batch_right.adj_list)
        and torch.all(batch_left.edge_batch_idx == batch_right.edge_batch_idx)
        and torch.all(batch_left.atom_coords == batch_right.atom_coords)
        and torch.all(batch_left.atom_velocs == batch_right.atom_velocs)
        and torch.all(batch_left.atom_forces == batch_right.atom_forces)
        and torch.all(batch_left.atom_coord_targets == batch_right.atom_coord_targets)
        and torch.all(batch_left.atom_veloc_targets == batch_right.atom_veloc_targets)
        and torch.all(batch_left.atom_force_targets == batch_right.atom_force_targets)
        and torch.all(batch_left.masked_elements == batch_right.masked_elements)
    )


def test_dense_collate_function(dummy_datapoints):
    batch = moldyn_dense_collate_fn(dummy_datapoints)
    # Check the shapes of the elements in the batch
    max_num_points = 5
    batch_size = len(dummy_datapoints)
    assert batch.atom_types.shape == (batch_size, max_num_points)
    assert batch.adj_list.shape == (3, 2)
    assert batch.atom_coords.shape == (batch_size, max_num_points, 3)
    assert batch.atom_velocs.shape == (batch_size, max_num_points, 3)
    assert batch.atom_forces.shape == (batch_size, max_num_points, 3)
    assert batch.atom_coord_targets.shape == (batch_size, max_num_points, 3)
    assert batch.atom_veloc_targets.shape == (batch_size, max_num_points, 3)
    assert batch.atom_force_targets.shape == (batch_size, max_num_points, 3)

    assert batch.masked_elements.shape == (batch_size, max_num_points)
    # Check the mask is as expected
    assert torch.all(batch.masked_elements == torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]]))
    # Check the adjacency list is as expected
    assert torch.all(
        batch.adj_list
        == torch.cat(tuple(datapoint.adj_list for datapoint in dummy_datapoints), dim=0)
    )


def test_dense_mol_dyn_batch_todevice(dummy_datapoints):
    for device in map(torch.device, AVAILABLE_DEVICES):
        batch = moldyn_dense_collate_fn(dummy_datapoints).todevice(device)

        for (k, v) in dataclasses.asdict(batch).items():
            if k == "names":
                continue

            assert v.device.type == device.type


def test_dense_mol_dyn_batch_permute(dummy_datapoints, device: torch.device):
    batch = moldyn_dense_collate_fn(dummy_datapoints).todevice(device)
    batch_size = len(batch.names)

    # `batch.permute(...)`
    # Swap the two elements in the batch.
    permutation = torch.tensor([1, 0], device=device)
    batch_permuted = batch.permute(permutation)
    assert not _batch_compare(batch, batch_permuted)

    # Certain properties we want to check explicitly.
    for i in range(batch_size):
        i_permuted = permutation.cpu().numpy()[i]

        # Make sure the connectivity is preserved appropriately.
        (indices,) = torch.where(batch.edge_batch_idx == i)
        (indices_permuted,) = torch.where(batch_permuted.edge_batch_idx == i_permuted)
        assert torch.all(batch.adj_list[indices] == batch_permuted.adj_list[indices_permuted])

    # Applying same permutation twice is identity.
    batch_same = batch_permuted.permute(permutation)
    assert _batch_compare(batch, batch_same)


def test_dense_mol_dyn_batch_segments(dummy_datapoints, device: torch.device):
    batch = moldyn_dense_collate_fn([*dummy_datapoints, dummy_datapoints[0]]).todevice(device)
    batch_size = len(batch.names)

    # Before sorting.
    assert len(batch.segments) == batch_size + 1
    # Endpoints.
    assert max(batch.segments) == batch_size
    assert min(batch.segments) == 0
    # Should all be different.
    s0 = batch.segments[0]
    assert all(s != s0 for s in batch.segments[1:])

    # After sorting.
    batch_sorted = batch.sort_by_name()
    assert len(batch_sorted.segments) == batch_size - 1 + 1
    # Endpoints.
    assert max(batch_sorted.segments) == batch_size
    assert min(batch_sorted.segments) == 0
    # Should all be different.
    s0 = batch_sorted.segments[0]
    assert all(s != s0 for s in batch_sorted.segments[1:])


@pytest.mark.parametrize("step_width", [1, 1000, 1e6, 1e7, 1e8])
@pytest.mark.parametrize("protein", ["1hgv", "2olx"])
@pytest.mark.parametrize("equal_data_spacing", [True, False])
@pytest.mark.filterwarnings("ignore:load_pdb_trace_data")
def test_load_pdb_trace_data(step_width, protein, equal_data_spacing):
    # The spacing, i.e. number of MD steps between saving steps, is 1e6 for T1
    DATA_DIR = (
        Path(os.path.join(os.path.dirname(__file__), "../testdata/output/")).expanduser().resolve()
    )
    state0pdbpath = os.path.join(DATA_DIR, f"{protein}-traj-state0.pdb")
    trajectory_path = os.path.join(DATA_DIR, f"{protein}-traj-arrays.npz")
    traj_raw_data = np.load(trajectory_path)["positions"]
    traj_info = load_pdb_trace_data(
        protein, state0pdbpath, trajectory_path, step_width, equal_data_spacing=equal_data_spacing
    )

    # The number of conditioning-target pairs depends on the step_width
    # If the step_width is a multiple of the spacing, non conditioning states are used as conditioning states
    # if equal_data_spacing is not enforced.
    if step_width == 1e6:
        num_conditioning_states = 19 if equal_data_spacing else 133
    # If the step_width is larger than the spacing, some conditioning states do not have a target.
    elif step_width == 1e7:
        num_conditioning_states = 10 if equal_data_spacing else 70
    # If the step_width is too large there are no pairs
    elif step_width == 1e8:
        num_conditioning_states = 0
    else:
        num_conditioning_states = 19

    assert traj_info.name == protein
    assert len(traj_info.coord_features) == num_conditioning_states

    # The T1 dataset is missing the first conditioning state in the raw data.
    # The first conditioning state is the seventh in the MD trajectory, as seven states are saved in each simulation iteration.
    if step_width < 1e6:
        np.testing.assert_array_equal(traj_info.coord_features[0], traj_raw_data[6])
        np.testing.assert_array_equal(traj_info.coord_features[1], traj_raw_data[6 + 7])
        np.testing.assert_array_equal(
            traj_info.coord_targets[0], traj_raw_data[6 + int(np.log10(step_width)) + 1]
        )
    elif step_width == 1e6:
        # If not equal_data_spacing, the firt pair is between the first and
        idx = 6 + 7 if equal_data_spacing else 7
        np.testing.assert_array_equal(traj_info.coord_targets[0], traj_raw_data[idx])
    elif step_width == 1e7:
        # The step_width is 10 times higher than the spacing
        idx = 6 + 7 * 10 if equal_data_spacing else 7 * 10
        np.testing.assert_array_equal(traj_info.coord_targets[0], traj_raw_data[idx])
