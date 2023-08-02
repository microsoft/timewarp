import abc
from dataclasses import asdict, dataclass

import torch.nn as nn
import torch
from torch import Tensor, BoolTensor
from typing import Sequence, Optional, Tuple

from timewarp.modules.model_wrappers.density_model_base import ConditionalDensityModel
from utilities.cache import Cache
from utilities.logger import TrainingLogger
from timewarp.utils.molecule_utils import get_centre_of_mass


# Type for deltalogp, which may be None to indicate we don't want to compute the likelihood, eg during sampling.
LikType = Optional[Tensor]


class Flow(nn.Module, abc.ABC):
    def __init__(self, atom_embedder: nn.Module):
        super().__init__()
        self.atom_embedder = atom_embedder

    @abc.abstractmethod
    def forward(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords: Tensor,  # [B, num_points, 3]
        z_velocs: Tensor,  # [B, num_points, 3]
        x_features: Tensor,  # [B, num_points, D]
        x_coords: Tensor,  # [B, num_points, 3]
        x_velocs: Tensor,  # [B, num_points, 3]
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        delta_logp: LikType = None,
        reverse: bool = False,
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tuple[Tensor, Tensor, LikType]:
        pass


class ConditionalSequentialFlow(Flow):
    """A generalized nn.Sequential container for conditional normalizing flows on molecules."""

    def __init__(self, layers: Sequence[nn.Module], atom_embedder: nn.Module):
        super().__init__(atom_embedder=atom_embedder)
        self.chain = nn.ModuleList(layers)

    def forward(
        self,
        atom_types: Tensor,  # [B, num_points]
        z_coords: Tensor,  # [B, num_points, 3]
        z_velocs: Tensor,  # [B, num_points, 3]
        x_features: Tensor,  # [B, num_points, D]
        x_coords: Tensor,  # [B, num_points, 3]
        x_velocs: Tensor,  # [B, num_points, 3]
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        delta_logp: LikType = None,
        reverse: bool = False,
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tuple[Tensor, Tensor, LikType]:
        """
        Args:
            delta_logp: Tensor holding the accumulated -ve log-det Jacobian terms.
                If x = f(x') where f(x') where are the previous layers of the flow, then
                this terms is equal to the difference: log p(x) - log p(x')
            reverse: If reverse is True, the module is in "sampling mode" propagating a
                latent sample through towards the observation space.

                If reverse is False, the layer is in "density estimation mode", propagating
                an observed sample through to get the corresponding latent sample and the
                likelihood.
        Return:
            - y: Input x transformed through this flow
            - Updated delta_logp: Subtract the log-det of Jacobian of this transformation for delta_logp,
                which is equivalent to adding log p(y) - log p(x) to delta_logp
        """
        idxs = range(len(self.chain))
        idxs = idxs[::-1] if reverse else idxs

        for i in idxs:
            z_coords, z_velocs, delta_logp = self.chain[i](
                atom_types=atom_types,
                z_coords=z_coords,
                z_velocs=z_velocs,
                x_features=x_features,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                delta_logp=delta_logp,
                reverse=reverse,
                logger=logger,
                cache=cache,
            )

        return z_coords, z_velocs, delta_logp


class ConditionalFlowDensityModel(ConditionalDensityModel):
    def __init__(
        self,
        flow: Flow,
        cache: Optional[Cache] = None,
        use_displacement_as_target: bool = True,
        scale_requires_grad: bool = True,
        ignore_conditional_velocity: bool = False,
    ):
        super().__init__()
        self.flow = flow
        self.coords_prior_log_scale = nn.Parameter(
            torch.tensor(0.0), requires_grad=scale_requires_grad
        )
        self.velocs_prior_log_scale = nn.Parameter(
            torch.tensor(0.0), requires_grad=scale_requires_grad
        )

        self.cache = cache or Cache()

        "If `True`, the conditional velocity `x_velocs` will be zeroed out."
        self.ignore_conditional_velocity = ignore_conditional_velocity
        "If `True`, then the model will internally target the displaced position rather than absolute position."
        self.use_displacement_as_target = use_displacement_as_target

    def log_likelihood(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:  # [batch_size] float tensor
        """Returns the log-likelihood for each element in the batch."""
        if self.ignore_conditional_velocity:
            x_velocs = torch.zeros_like(x_velocs)

        # Predict the change (residual) rather than absolute values
        if self.use_displacement_as_target:
            y_coords_residual = y_coords - x_coords
        else:
            y_coords_residual = y_coords

        y_velocs_residual = y_velocs

        # Canonicalise input position:
        centre_of_mass = get_centre_of_mass(x_coords, masked_elements=masked_elements)
        x_coords = x_coords - centre_of_mass

        z_coords_distribution = torch.distributions.Normal(
            loc=torch.zeros_like(x_coords),
            scale=torch.exp(self.coords_prior_log_scale),
        )
        z_velocs_distribution = torch.distributions.Normal(
            loc=torch.zeros_like(x_velocs),
            scale=torch.exp(self.velocs_prior_log_scale),
        )

        device = x_coords.device
        delta_logp = torch.zeros(x_coords.shape[0], device=device)  # [B]

        # Get atom features
        atom_features = self.flow.atom_embedder(atom_types)  # [B, num_points, D]

        # The output delta_logp = log p(z) - log p(y)
        z_coords, z_velocs, delta_logp = self.flow(
            atom_types=atom_types,  # [B, num_points]
            adj_list=adj_list,  # [num_edges, 2]
            z_coords=y_coords_residual,  # [B, num_points, 3]
            z_velocs=y_velocs_residual,  # [B, num_points, 3]
            x_features=atom_features,  # [B, num_points, D]
            x_coords=x_coords,  # [B, num_points, 3]
            x_velocs=x_velocs,  # [B, num_points, 3]
            edge_batch_idx=edge_batch_idx,  # [num_edges] int64 tensor
            masked_elements=masked_elements,  # [B, num_points] bool tensor
            delta_logp=delta_logp,  # [B]
            reverse=False,
            logger=logger,
            cache=self.cache.empty_like(),
        )

        log_prob_z_coords = z_coords_distribution.log_prob(z_coords)  # [B, num_points, 3]
        log_prob_z_velocs = z_velocs_distribution.log_prob(z_velocs)  # [B, num_points, 3]

        # Masking due to sequences in a batch being of different lengths.
        log_prob_z_coords = ((~masked_elements[:, :, None]) * log_prob_z_coords).sum(
            dim=(-1, -2)
        )  # [B]
        log_prob_z_velocs = ((~masked_elements[:, :, None]) * log_prob_z_velocs).sum(
            dim=(-1, -2)
        )  # [B]

        log_prob_z = log_prob_z_coords + log_prob_z_velocs  # Independent Gaussians
        log_prob_y = log_prob_z - delta_logp  # [B]

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("log_prob_z", log_prob_z.mean())
            logger.log_scalar_async("delta_logp", delta_logp.mean())
            logger.log_scalar_async("log_prob_y", log_prob_y.mean())

            coord_std = torch.exp(self.coords_prior_log_scale)
            veloc_std = torch.exp(self.velocs_prior_log_scale)
            logger.log_scalar_async("coord_std", coord_std.mean())
            logger.log_scalar_async("veloc_std", veloc_std.mean())
        return log_prob_y  # [batch_size]

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Returns conditional samples."""
        y_coords, y_velocs, _ = self.conditional_sample_with_logp(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            num_samples=num_samples,
            logger=logger,
        )

        return y_coords, y_velocs

    def conditional_sample_with_logp(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return conditional samples and log probability of samples."""
        device = x_coords.device
        batch_size = x_coords.shape[0]

        if self.ignore_conditional_velocity:
            x_velocs = torch.zeros_like(x_velocs)

        # Canonicalise input position
        centre_of_mass = get_centre_of_mass(x_coords, masked_elements=masked_elements)
        x_coords = x_coords - centre_of_mass

        # Sample z
        z_coords_distribution = torch.distributions.Normal(
            loc=torch.zeros_like(x_coords),
            scale=torch.exp(self.coords_prior_log_scale),
        )
        z_velocs_distribution = torch.distributions.Normal(
            loc=torch.zeros_like(x_velocs),
            scale=torch.exp(self.velocs_prior_log_scale),
        )

        z_coords = z_coords_distribution.rsample((num_samples,))  # [S, B, V, 3]
        z_velocs = z_velocs_distribution.rsample((num_samples,))  # [S, B, V, 3]
        z_coords = z_coords.reshape(-1, z_coords.shape[-2], z_coords.shape[-1])  # [S * B, V, 3]
        z_velocs = z_velocs.reshape(-1, z_velocs.shape[-2], z_velocs.shape[-1])  # [S * B, V, 3]

        # Get atom features
        atom_features = self.flow.atom_embedder(atom_types)  # [B, num_points, D]
        delta_logp = torch.zeros(x_coords.shape[0], device=device)  # [B]

        # Pass through conditional flow in reverse mode
        y_coords_residual, y_velocs_residual, delta_logp = self.flow(
            atom_types=atom_types.repeat(num_samples, 1),  # [S * B, num_points]
            adj_list=adj_list,  # [num_edges, 2]
            z_coords=z_coords,  # [S * B, num_points, 3]
            z_velocs=z_velocs,  # [S * B, num_points, 3]
            x_features=atom_features.repeat(num_samples, 1, 1),  # [S * B, num_points, D]
            x_coords=x_coords.repeat(num_samples, 1, 1),  # [S * B, num_points, 3]
            x_velocs=x_velocs.repeat(num_samples, 1, 1),  # [S * B, num_points, 3]
            edge_batch_idx=edge_batch_idx,  # [num_edges] int64 tensor TODO: not used
            masked_elements=masked_elements.repeat(
                num_samples, 1
            ),  # [S * B, num_points] bool tensor
            delta_logp=delta_logp.repeat(num_samples),  # [S * B]
            reverse=True,  # In reverse mode for sampling
            logger=logger,
            cache=self.cache.empty_like(),
        )  # [S * B, V, 3], [S * B, V, 3]

        # Un-canonicalise input position
        x_coords = x_coords + centre_of_mass  # [B, V, 3]
        x_coords = x_coords.repeat(num_samples, 1, 1)  # [S * B, V, 3]

        # Model predicts the change (residual) rather than absolute values
        if self.use_displacement_as_target:
            y_coords = x_coords + y_coords_residual  # [S * B, V, 3]
        else:
            y_coords = y_coords_residual  # [S * B, V, 3]

        y_velocs = y_velocs_residual  # [S * B, V, 3]

        y_coords = y_coords.reshape(
            num_samples, batch_size, y_coords.shape[-2], y_coords.shape[-1]
        )  # [S, B, V, 3]
        y_velocs = y_velocs.reshape(
            num_samples, batch_size, y_velocs.shape[-2], y_velocs.shape[-1]
        )  # [S, B, V, 3]

        # Log-density computation.
        log_prob_z_coords = z_coords_distribution.log_prob(z_coords)  # [S * B, num_points, 3]
        log_prob_z_velocs = z_velocs_distribution.log_prob(z_velocs)  # [S * B, num_points, 3]

        # Masking due to sequences in a batch being of different lengths.
        log_prob_z_coords = ((~masked_elements[:, :, None]) * log_prob_z_coords).sum(
            dim=(-1, -2)
        )  # [S * B]
        log_prob_z_velocs = ((~masked_elements[:, :, None]) * log_prob_z_velocs).sum(
            dim=(-1, -2)
        )  # [S * B]

        log_prob_z = log_prob_z_coords + log_prob_z_velocs  # Independent Gaussians
        log_prob_yx = log_prob_z + delta_logp  # [S * B]

        return y_coords, y_velocs, log_prob_yx.reshape(num_samples, batch_size)


@dataclass
class ConditionalFlowDensityConfig:
    scale_requires_grad: bool = True
    ignore_conditional_velocity: bool = False
    use_displacement_as_target: bool = True


def make_conditional_flow_density(config: ConditionalFlowDensityConfig, *args, **kwargs):
    return ConditionalFlowDensityModel(*args, **dict(asdict(config), **kwargs))
