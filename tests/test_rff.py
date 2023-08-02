import pytest
from pathlib import Path

import os
import numpy as np
import torch
import torch.distributions as dist
import mdtraj
from scipy.spatial import distance_matrix


from timewarp.modules.layers.rff_position_encoder import (
    RFFPositionEncoder,
    draw_gaussian_vectors,
    draw_gaussian_vectors_by_scale_distribution,
    apply_rff,
    gamma_shape_rate_from_mean_stddev,
)
from utilities.training_utils import set_seed

TESTDATA_DIR = Path(__file__).resolve().parents[1] / "testdata"


@pytest.fixture
def protein_coordinates() -> np.ndarray:
    pdb = os.path.join(TESTDATA_DIR, "AF-P0C6P0-F1-model_v1-traj-state0.pdb")
    traj = mdtraj.load(pdb)
    positions = traj.xyz[0]
    return positions  # [1618,3]


@pytest.mark.parametrize("rbf_scale", [0.3, 0.7, 1.0, 1.4, 2.0])
def test_rbf_matrix_approximation(
    device: torch.device, protein_coordinates: np.ndarray, rbf_scale: float
):
    """Test accuracy of RFF RBF approximation."""
    set_seed(0)
    nsamples = 256
    gaussian_vectors = draw_gaussian_vectors(3, nsamples, rbf_scale)
    gaussian_vectors = gaussian_vectors.to(device)
    pos = torch.tensor(protein_coordinates, dtype=torch.float32, device=device)

    feat = apply_rff(pos, gaussian_vectors)
    assert feat.shape[0] == pos.shape[0]
    assert feat.shape[1] == nsamples * 2
    kernel_rff = torch.einsum("is,js->ij", feat, feat)
    kernel_exact = np.exp(
        -(distance_matrix(protein_coordinates, protein_coordinates) ** 2.0)
        / (2.0 * rbf_scale**2.0)
    )
    assert torch.max(kernel_rff) <= (1.0 + 1.0e-5)
    error = (kernel_rff.cpu().detach().numpy() - kernel_exact).flatten()
    mean_error = np.mean(error)
    assert mean_error >= -0.05
    assert mean_error <= 0.05


@pytest.mark.parametrize(
    "mean,stddev",
    [
        (1.0, 1.0),
        (1.0, 4.0),
        (2.0, 0.25),
        (2.0, 5.0),
        (5.0, 0.5),
    ],
)
def test_gamma_parameters(
    mean: float,
    stddev: float,
):
    """Test validity of Gamma parametrization by mean and stddev."""
    set_seed(0)
    shape, rate = gamma_shape_rate_from_mean_stddev(mean, stddev)
    gamma = dist.Gamma(shape, rate)
    nsamples = 524288
    samples = gamma.sample((nsamples,)).cpu().detach().numpy()
    assert np.isclose(np.mean(samples), mean, atol=0.05)
    assert np.isclose(np.std(samples), stddev, atol=0.05)


def test_draw_gaussian_vectors_by_scale_distribution():
    set_seed(0)
    ndim = 3
    nsamples = 16
    gaussian_vectors = draw_gaussian_vectors_by_scale_distribution(ndim, nsamples, 1.0, 1.0)
    assert gaussian_vectors.shape[0] == ndim
    assert gaussian_vectors.shape[1] == nsamples


def test_position_encoder(protein_coordinates):
    set_seed(0)
    scale_mean = 0.1
    scale_stddev = 2.0
    encoding_dim = 128
    position_dim = 3

    position_encoder = RFFPositionEncoder(
        position_dim,
        encoding_dim,
        scale_mean,
        scale_stddev,
    )
    coords = torch.tensor(protein_coordinates, dtype=torch.float32)
    coords_batch = torch.stack((coords, coords), dim=0)  # [2,1618,3]
    coords_enc = position_encoder(coords_batch)
    assert coords_enc.shape[0] == coords_batch.shape[0]
    assert coords_enc.shape[1] == coords_batch.shape[1]
    assert coords_enc.shape[2] == encoding_dim
