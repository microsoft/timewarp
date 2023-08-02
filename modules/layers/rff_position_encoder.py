import math
import torch
import torch.nn as nn
import torch.distributions as dist
from dataclasses import dataclass


@dataclass
class RFFPositionEncoderConfig:
    encoding_dim: int  # number of encoding dimensions, must be even
    scale_mean: float  # scale parameter distribution, mean
    scale_stddev: float  # scale parameter distribution, stddev


def draw_gaussian_vectors(ndim: int, nsamples: int, rbf_scale: float):
    size = (ndim, nsamples)
    mean = torch.zeros(size)
    stddev = torch.ones_like(mean) / rbf_scale
    return torch.normal(mean, stddev)


def draw_gaussian_vectors_by_scale_distribution(
    ndim: int, nsamples: int, scale_mean: float, scale_stddev: float
):
    if nsamples == 0:
        return torch.zeros((ndim, nsamples))

    shape, rate = gamma_shape_rate_from_mean_stddev(scale_mean, scale_stddev)
    scale_dist = dist.Gamma(shape, rate)
    gaussian_vectors = []
    for _ in range(nsamples):
        rbf_scale = scale_dist.sample()
        gvec = draw_gaussian_vectors(ndim, 1, rbf_scale)
        gvec = gvec.squeeze(dim=1)
        gaussian_vectors.append(gvec)

    gaussian_vectors = torch.stack(gaussian_vectors, dim=1)
    return gaussian_vectors  # [ndim, nsamples]


def apply_rff(
    input: torch.Tensor,
    gaussian_vectors: torch.Tensor,
):
    """Embeds atom coordinates `input` into Fourier features.

    Arguments
    ---------
    input: (natoms, ndim), ndim typically 3
    gaussian_vectors: (ndim, nsamples)

    Returns
    -------
    feat: (natoms, 2*nsamples) Fourier features, so that
        torch.matmul(feat, feat.transpose()) approximates the RBF kernel.
    """
    nsamples = gaussian_vectors.shape[1]
    ips = torch.matmul(input, gaussian_vectors)
    if nsamples == 0:
        feat = ips
    else:
        feat = math.sqrt(1.0 / nsamples) * torch.cat((torch.cos(ips), torch.sin(ips)), dim=-1)

    return feat


def gamma_shape_rate_from_mean_stddev(mean, stddev):
    """Compute Gamma distribution (shape,rate) parameters from given
    mean and standard deviation.

    Arguments
    ---------
    mean : float, positive, the desired mean.
    stddev : float, positive, the desired standard deviation.

    Returns
    -------
    shape : float, positive, the Gamma shape parameter.
    rate : float, positive, the Gamma rate parameter.
    """
    rate = mean / (stddev**2.0)
    shape = mean * rate
    return shape, rate


class RFFPositionEncoder(nn.Module):
    """A position encoding module suited for 3D atom coordinates.

    The features are randomly created when the module is created and never
    adjusted during training.  The features are constructed following the
    random Fourier feature method but with randomly chosen scales.
    """

    def __init__(
        self,
        position_dim: int,  # number of input dimensions, typically 3
        # encoding_dim can be set to zero to create 0 features.
        encoding_dim: int,  # number of encoding dimensions, must be even.
        scale_mean: float,
        scale_stddev: float,
    ):
        super().__init__()
        self.scale_mean = scale_mean
        self.scale_stddev = scale_stddev

        assert encoding_dim % 2 == 0, "Number of encoding dimensions must be even."

        num_vectors = encoding_dim // 2
        gaussian_vectors = draw_gaussian_vectors_by_scale_distribution(
            position_dim, num_vectors, scale_mean, scale_stddev
        )
        gaussian_vectors = gaussian_vectors.to(torch.float32)
        self.register_buffer("gaussian_vectors", gaussian_vectors, persistent=True)

    def extra_repr(self) -> str:
        position_dim = self.gaussian_vectors.shape[0]
        encoding_dim = self.gaussian_vectors.shape[1]
        return "position_dim={}, encoding_dim={}, scale_mean={}, scale_stddev={}".format(
            position_dim,
            encoding_dim,
            self.scale_mean,
            self.scale_stddev,
        )

    def forward(self, coords: torch.Tensor):
        """Perform a position encoding.

        Arguments
        ---------
        coords: [B,V,position_dim] coordinates.

        Returns
        -------
        coords_enc: [B,V,encoding_dims] encoded coordinates.
        """
        coords_enc = apply_rff(coords, self.gaussian_vectors)
        return coords_enc
