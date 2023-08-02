"""Defines the interface for a conditional density model p(y|x)"""
import abc
from typing import Tuple, Optional
from torch import Tensor, BoolTensor

from utilities.logger import TrainingLogger
from timewarp.modules.model_wrappers.base import BaseModelWrapper, BaseModelWrapperWithForce


class ConditionalDensityModel(BaseModelWrapper, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    def forward(
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
    ) -> Tensor:  # [1] float tensor
        """Return the sum of per-atom log-likelihoods for the batch"""
        num_atoms = (~masked_elements).sum(dim=1)  # [batch_size]

        log_likelihood = self.log_likelihood(
            atom_types=atom_types,
            adj_list=adj_list,
            x_coords=x_coords,
            x_velocs=x_velocs,
            y_coords=y_coords,
            y_velocs=y_velocs,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )  # [batch_size]
        log_likelihood_per_atom = log_likelihood / num_atoms  # [batch_size]
        # Loss is average NLL per atom over the batch
        loss = -log_likelihood_per_atom.mean()

        if logger is not None:
            logger.log_scalar_async("nll_loss", loss)

        return loss

    @abc.abstractmethod
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
        """
        Return a sample of y_coords and y_velocs from the conditional distribution:
        p(y_coords, y_velocs| x_coords, x_velocs, atom_types, ...)

        Returns:
            A tuple of positions and velocity tensors sampled from the conditional distribution:
            * y_coords tensor of shape [batch_size, num_points, 3]
            * y_velocs tensor of shape [batch_size, num_points, 3]

            Note that only the entries corresponding to where masked_elements == 0 will be valid.
        """
        pass

    @abc.abstractmethod
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
        """Returns the log-likelihood for each element in the batch"""
        pass


class ConditionalDensityModelWithForce(BaseModelWrapperWithForce, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        y_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        y_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:  # [1] float tensor
        """Return the sum of per-atom log-likelihoods for the batch"""

        num_atoms = (~masked_elements).sum(dim=1)  # [batch_size]

        log_likelihood = self.log_likelihood(
            atom_types=atom_types,
            adj_list=adj_list,
            x_coords=x_coords,
            x_velocs=x_velocs,
            x_forces=x_forces,
            y_coords=y_coords,
            y_velocs=y_velocs,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,
            logger=logger,
        )  # [batch_size]
        log_likelihood_per_atom = log_likelihood / num_atoms  # [batch_size]
        # Loss is average NLL per atom over the batch
        loss = -log_likelihood_per_atom.mean()

        return loss

    @abc.abstractmethod
    def conditional_sample(
        self,
        atom_types: Tensor,  # [batch_size, num_points] int64 tensor
        x_coords: Tensor,  # [batch_size, num_points, 3] float tensor
        x_velocs: Tensor,  # [batch_size, num_points, 3] float tensor
        x_forces: Tensor,  # [batch_size, num_points, 3] float tensor
        adj_list: Tensor,  # [num_edges, 2] int64 tensor
        edge_batch_idx: Tensor,  # [num_edges] int64 tensor
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        num_samples: int,
        logger: Optional[TrainingLogger] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return a sample of y_coords and y_velocs from the conditional distribution:
        p(y_coords, y_velocs| x_coords, x_velocs, atom_types, ...)

        Returns:
            A tuple of positions and velocity tensors sampled from the conditional distribution:
            * y_coords tensor of shape [batch_size, num_points, 3]
            * y_velocs tensor of shape [batch_size, num_points, 3]

            Note that only the entries corresponding to where masked_elements == 0 will be valid.
        """
        pass

    @abc.abstractmethod
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
        masked_elements: BoolTensor,  # [batch_size, num_points] bool tensor
        logger: Optional[TrainingLogger] = None,
    ) -> Tensor:  # [batch_size] float tensor
        """Returns the log-likelihood for each element in the batch"""
        pass
