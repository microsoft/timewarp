import torch

from timewarp.modules.layers.local_self_attention import get_closest
from utilities.training_utils import set_seed


def test_get_closest():
    set_seed(0)
    max_radius = 3.0

    # Define shapes
    B = 2
    V = 3
    H = 5
    d = 7

    # Make random positions in Euclidean space
    positions = torch.randn(B, V, d)

    # Get the pairwise distances between points
    distance_matrix = torch.cdist(positions, positions)  # [B, V, V]

    # Get the maximum num. of atoms within radius `self.max_radius` of any given atom
    within_radius = distance_matrix < max_radius  # [B, V, V]
    max_neighbors: int = within_radius.sum(dim=-1).max()  # Max number of neighbours

    # Get idxs of all points within `self.max_radius` of each atom
    topk_distances, neighbor_idxs = torch.topk(
        distance_matrix, k=max_neighbors, dim=-1, largest=False
    )
    # topk_distances and neighbor_idxs have shape [B, V, K]

    # Random query matrix
    Q = torch.randn((B, V, H, d))

    # Get local version
    Q_local = get_closest(Q, neighbor_idxs)  # [B, V, K, H, d]

    # Test entries manually
    for batch in range(B):
        for atom in range(V):
            for head in range(H):
                for neighbor in range(max_neighbors):
                    assert torch.allclose(
                        Q_local[batch, atom, neighbor, head, :],
                        Q[batch, neighbor_idxs[batch, atom, neighbor], head, :],
                    )
