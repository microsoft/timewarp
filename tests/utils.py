import os

import pytest
import torch
import torch.nn as nn

from timewarp.dataloader import ELEMENT_VOCAB, load_pdb_trace_data
from timewarp.modules.gnn.message_passing import PointCloudState  # type: ignore [import]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atom_embedding_dim() -> int:
    return 16


@pytest.fixture
def atom_embedder():
    atom_embedder_module = nn.Embedding(
        num_embeddings=len(ELEMENT_VOCAB),
        embedding_dim=atom_embedding_dim(),
    ).to(device)
    return atom_embedder_module


def load_molecule_input_data():
    testdata_dir = os.path.dirname(os.path.realpath(__file__))
    testdata_dir = os.path.join(testdata_dir, "..", "testdata", "output")
    pdb_name = "1hgv"

    # Load the input data.
    traj_info = load_pdb_trace_data(
        pdb_name,
        f"{testdata_dir}/{pdb_name}-traj-state0.pdb",
        f"{testdata_dir}/{pdb_name}-traj-arrays.npz",
        step_width=1,
    )
    atom_types = torch.tensor(traj_info.node_types, dtype=torch.int64).to(
        device
    )  # int64 tensor of shape [V]

    adj_list = torch.tensor(traj_info.adj_list, dtype=torch.int64).to(
        device
    )  # int32 tensor of shape [E, 2]
    initial_coords = torch.tensor(traj_info.coord_features[0]).to(device)  # [V, 3]
    initial_velocs = torch.tensor(traj_info.veloc_features[0]).to(device)  # [V, 3]

    return atom_types, adj_list, initial_coords, initial_velocs


def load_2atom_data():
    num_atoms = 2

    atom_types = torch.randint(high=len(ELEMENT_VOCAB), size=(num_atoms,)).to(device)

    adj_list = torch.tensor([[0, 1]]).to(device)
    initial_coords = torch.randn(num_atoms, 3).to(device)
    initial_velocs = torch.randn(num_atoms, 3).to(device)

    return atom_types, adj_list, initial_coords, initial_velocs


def load_5atom_cyclic_graph_data():
    num_atoms = 5
    atom_types = torch.randint(high=len(ELEMENT_VOCAB), size=(num_atoms,)).to(device)

    adj_list = [
        [0, 1],
        [1, 3],
        [2, 1],
        [3, 2],
        [3, 4],
    ]
    adj_list = torch.tensor(adj_list).to(device)
    initial_coords = torch.randn(num_atoms, 3).to(device)
    initial_velocs = torch.randn(num_atoms, 3).to(device)

    return atom_types, adj_list, initial_coords, initial_velocs


class PointCloudPreprocessingWrapper(nn.Module):
    """
    Some models take PointCloudState as input. Define a wrapper
    to convert between that representation, and the representation used by other
    models.
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, atom_features, adj_list, initial_coords, initial_velocs):
        # Prepare the input data.
        input_point_cloud = PointCloudState(
            features=atom_features, coords=initial_coords, velocities=initial_velocs
        )

        # Pass the input data through the EGNN.
        predictions = self.mod(
            x=input_point_cloud,
            adj_list=adj_list,
            edge_features=torch.zeros(
                size=(adj_list.shape[0], 0), dtype=torch.float32, device=adj_list.device
            ),
        )

        output_features = predictions.features  # [V, D]
        output_coords = predictions.coords  # [V, 3]
        output_velocs = predictions.velocities  # [V, 3]

        return output_features, output_coords, output_velocs
