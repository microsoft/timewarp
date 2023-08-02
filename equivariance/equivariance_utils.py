import torch
from scipy.spatial.transform import Rotation as R  # type: ignore [import]


def random_rotation_matrix(device=None, dtype=torch.float32):
    """Generate a uniformly distributed random rotation matrix."""
    Q = R.random().as_matrix()
    return torch.tensor(Q, dtype=dtype).to(device)


def apply_rotation(Q, *args):
    """Apply a rotation matrix Q to every input."""
    return tuple((Q @ arg.T).T for arg in args)


def random_translation_vector(device=None, dtype=torch.float32):
    """Generate a normally distributed translation vector."""
    a = torch.randn(1, 3, dtype=dtype).to(device)
    return a


def apply_translation(a, *args):
    """Apply a translation vector to every input."""
    return tuple(arg + a for arg in args)


def random_permutation(num_points: int, device=None) -> torch.Tensor:
    return torch.randperm(num_points).to(device)
