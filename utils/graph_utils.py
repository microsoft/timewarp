import torch


def dense_to_sparse(adj_matrix: torch.Tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj_matrix (Tensor): The dense adjacency matrix.

    Return:
        * adj_list - A [num_edges, 2] tensor of edges where adj_list[i]
            has the indices of the source and target of the an edge
        * edge_type - a [num_edges] size tensor with the int type of each edge
    """
    # Assert there is at most one batching dimension
    assert adj_matrix.dim() >= 2 and adj_matrix.dim() <= 3
    # Assert this is a valid adjacency matrix
    assert adj_matrix.size(-1) == adj_matrix.size(-2)

    index = adj_matrix.nonzero(as_tuple=True)
    edge_type = adj_matrix[index]

    if len(index) == 3:
        batch = index[0] * adj_matrix.size(-1)
        index = (batch + index[1], batch + index[2])

    adj_list = torch.stack(index, dim=0)
    # In case adj_list is empty, reshape it into the shape [0, 2] so that indexing works
    adj_list = adj_list.reshape((-1, 2))

    return adj_list, edge_type


def make_dense_adj_list(num_nodes):
    """Compute a dense, unidirectional adjacency list."""
    adj_matrix = torch.ones(num_nodes, num_nodes)
    adj_matrix = torch.tril(adj_matrix, diagonal=-1)  # Unidirectional, no self connections

    adj_list, _ = dense_to_sparse(adj_matrix)
    adj_list.type(torch.int64)
    return adj_list
