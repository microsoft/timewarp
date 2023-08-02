from typing import Tuple
import abc

import torch
import torch.nn as nn
from torch import Tensor
from torch import linalg


class ConditionalEquivariantBasis(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def num_rel_basis(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_pointwise_basis(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_state_rel_features(self) -> int:
        """Number of relative features extracted from the molecular state, excluding atom type"""
        pass

    @property
    @abc.abstractmethod
    def num_state_pointwise_features(self) -> int:
        """Number of pointwise features extracted from the molecular state, excluding atom type."""
        pass

    @abc.abstractmethod
    def get_equivariant_features_and_basis(
        self,
        atom_features,  # [B, num_points, D]
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3], z_coords
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass


class ConditionalEquivariantCoordBasis(ConditionalEquivariantBasis):
    """Computes basis vectors constructed out of the *z coordinates*, and also the x coords and velocs"""

    def __init__(self):
        super().__init__()

    @property
    def num_rel_basis(self) -> int:
        return 2  # One relative basis for z coord differences and one for x coord differences

    @property
    def num_pointwise_basis(self) -> int:
        return 1  # One pointwise basis for x velocities.

    @property
    def num_state_rel_features(self) -> int:
        return 2  # One relative feature each from the x and z coord difference norms.

    @property
    def num_state_pointwise_features(self) -> int:
        return 1  # Norm of x velocities.

    def get_equivariant_features_and_basis(
        self,
        atom_features,  # [B, num_points, D]
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3], z_coords
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes:
            - (equivariant) basis vectors from the z *coords*, x coords and x velocs that
            can be used to shift the input in an equivariant way
            - all the pointwise and relative invariant features to be used by the model
            to compute scaling factors for each of the basis vectors
        """
        z_coords = z_untransformed  # Untransformed variables are coordinates, since making basis out of coord vectors.

        # Get a [B, num_points, num_points, 3] matrix of relative_displacements of x
        # where rel_displacements[b, i, j, :] = x_coords[b, i, :] - x_coords[b, j, :]
        x_rel_displacements = x_coords.unsqueeze(-2) - x_coords.unsqueeze(-3)
        x_rel_displacements_norm = linalg.norm(
            x_rel_displacements, ord=2, dim=-1, keepdim=True
        )  # [B, num_points, num_points, 1]
        x_rel_displacement_basis = x_rel_displacements

        # Same for z
        z_rel_displacements = z_coords.unsqueeze(-2) - z_coords.unsqueeze(-3)
        z_rel_displacements_norm = linalg.norm(
            z_rel_displacements, ord=2, dim=-1, keepdim=True
        )  # [B, num_points, num_points, 1]
        z_rel_displacement_basis = z_rel_displacements

        x_velocs_norm = linalg.norm(x_velocs, ord=2, dim=-1, keepdim=True)  # [B, num_points, 1]

        relative_features = torch.cat(
            (z_rel_displacements_norm, x_rel_displacements_norm), dim=-1
        )  # [B, num_points, num_points, num_rel_feat]
        pointwise_features = torch.cat(
            (atom_features, x_velocs_norm), dim=-1
        )  # [B, num_points, num_pointwise_feat]
        relative_basis = torch.stack(
            (z_rel_displacement_basis, x_rel_displacement_basis), dim=-2
        )  # [B, num_points, num_points, num_rel_basis, 3]
        pointwise_basis = x_velocs[:, :, None, :]  # [B, num_points, num_pointwise_basis, 3]

        return relative_features, pointwise_features, relative_basis, pointwise_basis


class ConditionalEquivariantVelocityBasis(ConditionalEquivariantBasis):
    """Computes basis vectors constructed out of the *z velocities*, and also the x coords and velocs"""

    def __init__(self):
        super().__init__()

    @property
    def num_rel_basis(self) -> int:
        return 1  # One relative basis from the x coord difference vectors.

    @property
    def num_pointwise_basis(self) -> int:
        return 2  # One pointwise basis for z velocity and one for x velocity.

    @property
    def num_state_rel_features(self) -> int:
        return 1  # One relative feature from the x coord difference vector norms.

    @property
    def num_state_pointwise_features(self) -> int:
        return 2  # Norm of z and x velocities.

    def get_equivariant_features_and_basis(
        self,
        atom_features,  # [B, num_points, D]
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3], z_velocs
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes:
            - (equivariant) basis vectors from the z *velocs* and x coords and x velocs
            that can be used to shift the input in an equivariant way
            - all the pointwise and relative invariant features to be used by the model
            to compute scaling factors for each of the basis vectors
        """
        z_velocs = z_untransformed  # Untransformed variables are velocities, since making basis out of velocity vectors.

        # Get a [B, num_points, num_points, 3] matrix of relative_displacements of x
        # where rel_displacements[b, i, j, :] = x_coords[b, i, :] - x_coords[b, j, :]
        x_rel_displacements = x_coords.unsqueeze(-2) - x_coords.unsqueeze(-3)
        x_rel_displacements_norm = linalg.norm(
            x_rel_displacements, ord=2, dim=-1, keepdim=True
        )  # [B, num_points, num_points, 1]
        x_rel_displacement_basis = x_rel_displacements  # [B, num_points, num_points, 3]

        x_velocs_norm = linalg.norm(x_velocs, ord=2, dim=-1, keepdim=True)  # [B, num_points, 1]
        z_velocs_norm = linalg.norm(z_velocs, ord=2, dim=-1, keepdim=True)  # [B, num_points, 1]

        relative_features = x_rel_displacements_norm  # [B, num_points, num_points, 1]
        pointwise_features = torch.cat(
            (atom_features, z_velocs_norm, x_velocs_norm), dim=-1
        )  # [B, num_points, num_pointwise_feat]

        relative_basis = x_rel_displacement_basis[
            :, :, :, None, :
        ]  # [B, num_points, num_points, num_rel_basis, 3]
        pointwise_basis = torch.stack(
            (z_velocs, x_velocs), dim=-2
        )  # [B, num_points, num_pointwise_basis, 3]

        return relative_features, pointwise_features, relative_basis, pointwise_basis
