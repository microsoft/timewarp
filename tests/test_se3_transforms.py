import torch

from timewarp.equivariance.equivariance_transforms import Permutation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_permute_adjacency__simple_example():
    # 0 -> 1   -> 3 -> 4
    #       <\   /
    #          2<
    adj_list = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [2, 1],
            [3, 2],
            [3, 4],
        ]
    ).to(device)
    permutation = torch.tensor([3, 0, 2, 4, 1])
    permutation_transform = Permutation(permutation)
    man_permuted_adj_list = [
        [3, 0],
        [0, 4],
        [2, 0],
        [4, 2],
        [4, 1],
    ]
    perm_adj_list = permutation_transform.transform_adjacency_list(adj_list)
    assert man_permuted_adj_list == perm_adj_list.tolist()


def test_apply_permutation():
    # coords should be:
    # [[0.1, 0.2, 0.3],
    #  [0.4, 0.5, 0.6],
    #        ...
    coords = torch.reshape(0.1 * torch.arange(3 * 5) + 0.1, shape=(5, 3)).to(device)
    adj_list = [
        [0, 1],
        [1, 3],
        [2, 1],
        [3, 2],
        [3, 4],
    ]
    adj_list = torch.tensor(adj_list).to(device)

    # Manually specify a permutation
    permutation = torch.tensor([0, 3, 1, 2, 4]).to(device)
    permutation_transform = Permutation(permutation)

    permuted_coords = permutation_transform.transform_coord(coords)
    permuted_adj_list = permutation_transform.transform_adjacency_list(adj_list)

    manually_permuted_coords = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [0.4, 0.5, 0.6],
            [1.3, 1.4, 1.5],
        ]
    ).to(device)
    manually_permuted_adj_list = [
        [0, 3],
        [3, 2],
        [1, 3],
        [2, 1],
        [2, 4],
    ]
    assert torch.allclose(manually_permuted_coords, permuted_coords, rtol=1e-4, atol=1e-4)
    assert manually_permuted_adj_list == permuted_adj_list.tolist()
