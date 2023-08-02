import sys
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from timewarp.dataloader import ELEMENT_VOCAB
from utilities.logger import TrainingLogger
from timewarp.modules.model_wrappers.density_model_base import (
    ConditionalDensityModel,
    ConditionalDensityModelWithForce,
)
from timewarp.modules.model_wrappers.point_estimate import PointEstimateModel


class InitialStateGaussian(ConditionalDensityModel):
    """Trivial density model that predicts an isotropic Gaussian around the
    initial molecular state.
    """

    def __init__(self):
        super().__init__()
        self.coords_prior_log_scale = nn.Parameter(torch.tensor(0.0))
        self.velocs_prior_log_scale = nn.Parameter(torch.tensor(0.0))

    def log_likelihood(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:

        coord_std = torch.exp(self.coords_prior_log_scale)
        veloc_std = torch.exp(self.velocs_prior_log_scale)

        p_y_coord = torch.distributions.Normal(
            # Mean centered at the initial state
            loc=x_coords,  # [B, V, 3]
            scale=coord_std,
        )
        p_y_veloc = torch.distributions.Normal(
            # Mean centered at the initial state
            loc=x_velocs,  # [B, V, 3]
            scale=veloc_std,
        )

        # Independent Gaussians
        logp_y_coord = p_y_coord.log_prob(y_coords)  # [B, V, 3]
        logp_y_veloc = p_y_veloc.log_prob(y_velocs)  # [B, V, 3]

        # Masking due to sequences in a batch being of different lengths.
        logp_y_coord = ((~masked_elements[:, :, None]) * logp_y_coord).sum(dim=(-1, -2))  # [B]
        logp_y_veloc = ((~masked_elements[:, :, None]) * logp_y_veloc).sum(dim=(-1, -2))  # [B]

        logp_y = logp_y_coord + logp_y_veloc  # [B]

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("coord_std", coord_std.mean())
            logger.log_scalar_async("veloc_std", veloc_std.mean())

        return logp_y  # [B]

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class LearnableLinearGaussian(ConditionalDensityModelWithForce):
    """Predicts a Gaussian centered at a learned linear combination of the
    initial molecular state + velocity vector + force vector.
    """

    def __init__(self):
        super().__init__()
        self.veloc_to_coord_params = nn.Parameter(torch.zeros(len(ELEMENT_VOCAB)))  # [T]
        self.veloc_to_veloc_params = nn.Parameter(torch.zeros(len(ELEMENT_VOCAB)))  # [T]
        self.force_to_veloc_params = nn.Parameter(torch.zeros(len(ELEMENT_VOCAB)))  # [T]

        self.atom_coord_std_params = nn.Parameter(-torch.ones(len(ELEMENT_VOCAB)))  # [T]
        self.atom_veloc_std_params = nn.Parameter(-torch.ones(len(ELEMENT_VOCAB)))  # [T]

    def log_likelihood(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:

        # Std dev can depend on atom type.
        coord_std_params = self.atom_coord_std_params[atom_types]  # [B, V]
        coord_stds = torch.exp(coord_std_params)  # [B, V]
        veloc_std_params = self.atom_veloc_std_params[atom_types]  # [B, V]
        veloc_stds = torch.exp(veloc_std_params)  # [B, V]

        # Coord mean.
        veloc_to_coords = self.veloc_to_coord_params[atom_types][:, :, None]  # [B, V, 1]
        coord_mean = x_coords + veloc_to_coords * x_velocs  # [B, V, 3]

        # Veloc mean.
        force_to_velocs = self.force_to_veloc_params[atom_types][:, :, None]  # [B, V, 1]
        veloc_to_velocs = self.veloc_to_veloc_params[atom_types][:, :, None]  # [B, V, 1]
        veloc_mean = x_velocs + force_to_velocs * x_forces + veloc_to_velocs * x_velocs  # [B, V, 3]

        p_y_coord = torch.distributions.Normal(
            loc=coord_mean, scale=coord_stds[:, :, None].repeat(1, 1, 3)  # [B, V, 3]  # [B, V, 3]
        )
        p_y_veloc = torch.distributions.Normal(
            loc=veloc_mean, scale=veloc_stds[:, :, None].repeat(1, 1, 3)  # [B, V, 3]  # [B, V, 3]
        )

        # Independent Gaussians
        logp_y_coord = p_y_coord.log_prob(y_coords)  # [B, V, 3]
        logp_y_veloc = p_y_veloc.log_prob(y_velocs)  # [B, V, 3]

        # Masking due to sequences in a batch being of different lengths.
        logp_y_coord = ((~masked_elements[:, :, None]) * logp_y_coord).sum(dim=(-1, -2))  # [B]
        logp_y_veloc = ((~masked_elements[:, :, None]) * logp_y_veloc).sum(dim=(-1, -2))  # [B]

        logp_y = logp_y_coord + logp_y_veloc  # [B]

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("coord_std", coord_stds.mean())
            logger.log_scalar_async("veloc_std", veloc_stds.mean())

        return logp_y  # [B]

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class EulerMaruyamaGaussian(ConditionalDensityModelWithForce):
    """Predicts a Gaussian centered at a single Euler-Maruyama step."""

    def __init__(self, step_width_init=1):
        super().__init__()
        # OpenMM uses units of nm, ps, atomic mass units, Kelvin, kJ/mol http://docs.openmm.org/latest/userguide/theory.html#units
        # This value of k_B taken from http://docs.openmm.org/latest/userguide/theory.html#physical-constants
        # There it is reported in J/K. I convert it to kJ/(mol * K)
        self.k_B = 1.380649e-23 * 1e-3 * 6.02214076e23
        # Masses from https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm
        self.mass_vocab = [12.011, 1.00797, 14.0067, 15.9994, 32.06]  # [C, H, N, O, S]

        # These settings taken from default values in simulate_trajectory.py
        self.temperature = 310
        # Conversion of time step from femtosecond to picosecond
        self.delta_t = step_width_init * 0.5 * 1e-3
        self.gamma = 0.3

        # Init learnable parameters
        self.delta_t_factor_param = nn.Parameter(torch.tensor([0.0]))  # [1]
        self.atom_mass_params = nn.Parameter(torch.log(torch.tensor(self.mass_vocab)))  # [T]
        self.atom_coord_std_params = nn.Parameter(-torch.ones(len(ELEMENT_VOCAB)))  # [T]
        self.atom_veloc_std_params = nn.Parameter(-torch.ones(len(ELEMENT_VOCAB)))  # [T]

    def log_likelihood(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:

        p_y_coord, p_y_veloc = self._get_y_dist(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            x_forces=x_forces,
            logger=logger,
        )

        # Independent Gaussians
        logp_y_coord = p_y_coord.log_prob(y_coords)  # [B, V, 3]
        logp_y_veloc = p_y_veloc.log_prob(y_velocs)  # [B, V, 3]

        # Masking due to sequences in a batch being of different lengths.
        logp_y_coord = ((~masked_elements[:, :, None]) * logp_y_coord).sum(dim=(-1, -2))  # [B]
        logp_y_veloc = ((~masked_elements[:, :, None]) * logp_y_veloc).sum(dim=(-1, -2))  # [B]

        logp_y = logp_y_coord + logp_y_veloc  # [B]

        return logp_y  # [B]

    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:

        p_y_coord, p_y_veloc = self._get_y_dist(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            x_forces=x_forces,
            logger=logger,
        )

        # Get samples
        y_coord = p_y_coord.sample((num_samples,))  # [S, B, V, 3]
        y_veloc = p_y_veloc.sample((num_samples,))  # [S, B, V, 3]

        return y_coord, y_veloc

    def _get_y_dist(
        self,
        atom_types: Tensor,
        x_coords: Tensor,
        x_velocs: Tensor,
        x_forces: Tensor,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[torch.distributions.Normal, torch.distributions.Normal]:
        # Coordinate standard deviation can depend on atom type.
        coord_std_params = self.atom_coord_std_params[atom_types]  # [B, V]
        coord_stds = torch.exp(coord_std_params)  # [B, V]

        # Masses depend on atom type.
        masses = torch.exp(self.atom_mass_params[atom_types])  # [B, V]

        # Euler-Maruyama step.
        delta_t_factor = torch.exp(self.delta_t_factor_param)
        coord_mean = x_coords + self.delta_t * delta_t_factor * x_velocs  # [B, V, 3]

        force_term = (x_forces / masses[:, :, None]) * self.delta_t * delta_t_factor  # [B, V, 3]
        friction_term = -self.gamma * x_velocs * self.delta_t * delta_t_factor  # [B, V, 3]
        veloc_mean = x_velocs + force_term + friction_term  # [B, V, 3]
        veloc_stds = torch.sqrt(
            2.0 * self.gamma * self.k_B * self.temperature * self.delta_t * delta_t_factor / masses
        )  # [B, V]

        # Add learnable extra variance for velocities.
        veloc_std_params = self.atom_veloc_std_params[atom_types]  # [B, V]
        veloc_stds = veloc_stds + torch.exp(veloc_std_params)  # [B, V]

        p_y_coord = torch.distributions.Normal(
            loc=coord_mean, scale=coord_stds[:, :, None].repeat(1, 1, 3)  # [B, V, 3]  # [B, V, 3]
        )
        p_y_veloc = torch.distributions.Normal(
            loc=veloc_mean, scale=veloc_stds[:, :, None].repeat(1, 1, 3)  # [B, V, 3]  # [B, V, 3]
        )

        # Compute and monitor various quantities
        if logger is not None:
            logger.log_scalar_async("coord_std", coord_stds.mean())
            logger.log_scalar_async("veloc_std", veloc_stds.mean())

        return p_y_coord, p_y_veloc


class InitialStatePointEstimate(PointEstimateModel):
    """Trivial MSE model that just predicts the initial state."""

    def __init__(self):
        super().__init__()
        # Dummy parameter so that 'optimisation' can occur, which allows tensorboard plotting.
        self.dummy_param = nn.Parameter(torch.zeros(1))  # [1]

    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: Tensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:

        # Bring the dummy parameter into the computation graph
        x_coords = x_coords + 0.0 * self.dummy_param

        # Just predict the initial state for both coords and velocs.
        return x_coords, x_velocs
