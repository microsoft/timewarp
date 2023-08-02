import abc
from typing import Sequence, Tuple

import torch.nn as nn
from torch import Tensor

from timewarp.modules.layers.equivariant_features_basis import (
    ConditionalEquivariantBasis,
    ConditionalEquivariantCoordBasis,
    ConditionalEquivariantVelocityBasis,
)
from timewarp.modules.layers.feature_processor import FeatureProcessor
from timewarp.modules.layers.mlp import MLP


class ShiftModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        masked_elements,  # [B, num_points]
    ) -> Tensor:  # [B, num_points, 3]
        """
        If inputs are the coordinates and features for N points, this module should
        return a (jointly SO(3) equivariant in x and z) shift vector of shape [N, 3].
        """
        pass


class ScaleModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        masked_elements,  # [B, num_points]
    ) -> Tensor:  # [B, num_points, 1]
        """
        If inputs are the coordinates and features for N points, this module should
        return a (jointly SO(3) *invariant* in x and z) scale vector of shape [N, 1].
        """
        pass


class DenseEquivariantShiftModule(ShiftModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
        basis: ConditionalEquivariantBasis,
    ):
        super().__init__()
        self.basis = basis

        self._input_features_dim = input_features_dim
        self._output_features_dim = output_features_dim
        self.num_pointwise_basis = basis.num_pointwise_basis
        self.num_rel_basis = basis.num_rel_basis
        self.num_state_pointwise_features = basis.num_state_pointwise_features
        self.num_state_rel_features = basis.num_state_rel_features

        # num_state_pointwise_features is the number of pointwise features excluding those
        # that depend on the atom type.
        self._input_pointwise_features_dim = (
            self._input_features_dim + self.num_state_pointwise_features
        )
        # num_state_relative_features is the number of relative features excluding those
        # that depend on the atom type.
        self._input_relative_features_dim = (
            self.num_state_rel_features + 2 * self._input_pointwise_features_dim
        )
        self._processed_pointwise_features_dim = output_features_dim
        self._processed_relative_features_dim = output_features_dim
        # -- Below MLPs output coefficients for pointwise and relative basis vectors --
        # MLPs to get the shifts
        # TODO: Might want to try different MLPs for processing x and z
        self._shift_with_pointwise_mlp = MLP(
            input_dim=self._processed_pointwise_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=self.num_pointwise_basis,
        )
        self._shift_with_relative_mlp = MLP(
            input_dim=self._processed_relative_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=self.num_rel_basis,
        )

    def forward(
        self,
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        masked_elements,  # [B, num_points]
    ) -> Tensor:  # [B, num_points, 3]
        """
        If inputs are the coordinates and features for N points, this module should
        return a (jointly SO(3) equivariant in x and z) shift vector of shape [N, 3].
        """
        # Get the pointwise and relative features and basis vectors.
        (
            relative_features,  # [B, V, V, num_rel_feat]
            pointwise_features,  # [B, V, num_pointwise_feat]
            relative_basis,  # [B, V, V, num_rel_basis, 3]  E.g. num_rel_basis could be 2, one for z and one for x
            pointwise_basis,  # [B, V, num_rel_basis, 3]
        ) = self.basis.get_equivariant_features_and_basis(
            atom_features=x_features,
            adj_list=adj_list,
            z_untransformed=z_untransformed,
            x_coords=x_coords,
            x_velocs=x_velocs,
        )

        # Process the pointwise and relative features in a permutation equivariant way.
        relative_features, pointwise_features = self._process_features(
            relative_features=relative_features,  # [B, V, V, num_rel_feat]
            pointwise_features=pointwise_features,  # [B, V, num_pointwise_feat]
            adj_list=adj_list,
            masked_elements=masked_elements,  # [B, V]
        )

        # Compute the output shift vector based on the basis vectors and processed features.
        shift = self._calc_shift(
            pointwise_features=pointwise_features,  # [B, V, num_pointwise_feat]
            relative_features=relative_features,  # [B, V, V, num_rel_feat]
            pointwise_basis=pointwise_basis,  # [B, V, num_rel_basis, 3]
            relative_basis=relative_basis,  # [B, V, V, num_rel_basis, 3]
            masked_elements=masked_elements,  # [B, V]
        )

        return shift  # [B, V, 3]

    @abc.abstractmethod
    def _process_features(
        self, relative_features, pointwise_features, adj_list, masked_elements
    ) -> Tuple[Tensor, Tensor]:
        """Process the SE(3) *invariant* features in a permutation equivariant way."""
        pass

    def _calc_shift(
        self,
        pointwise_features,  # [B, V, num_pointwise_feat]
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_basis,  # [B, V, num_rel_basis, 3]
        relative_basis,  # [B, V, V, num_rel_basis, 3]
        masked_elements,  # [B, V]
    ):
        """Compute a shift vector for the coupling layer. Let :math:`h_i` denote the pointwise features
        and :math:`h_{ij}` denote the relative features. We also let :math:`e_i` denote the pointwise
        basis and :math:`e_{ij}` denote the relative basis vectors.

        Let :math:`\psi` and :math:`\phi` be pointwise and relative maps respectively that map from
        feature space to scale factors for the basis vectors. Then the shift contribution for atom i
        from each set of basis vectors is computed as:
        ..math::

            shift_i = \psi(h_i) e_i + \sum_j \phi(h_{ij}) e_{ij}.

        There can be more than one set of basis vectors, e.g. one set may be formed out of the
        :math:`x` vectors and another set out of the :math:`y` vectors. The final shift output is
        formed by summing over all sets of basis vectors.
        """
        num_atoms = (~masked_elements).sum(dim=-1)  # [B]

        pointwise_shift = (
            pointwise_basis * self._shift_with_pointwise_mlp(pointwise_features)[..., None]
        )  # [B, V, num_rel_basis, 3]

        relative_shift = (
            relative_basis * self._shift_with_relative_mlp(relative_features)[..., None]
        )  # [B, V, V, num_rel_basis, 3]
        # Mask meaningless entries in relative basis sum.
        unmasked_elements = ~masked_elements[:, None, :, None, None]  # [B, 1, V, 1, 1]
        relative_shift = relative_shift * unmasked_elements  # [B, V, V, num_rel_basis, 3]
        relative_shift = (
            relative_shift.sum(-3) / num_atoms[:, None, None, None]
        )  # [B, V, num_rel_basis, 3]  Sum relative shift vectors atom-wise and normalise.

        all_shifts = pointwise_shift + relative_shift  # [B, V, num_rel_basis, 3]
        shift = (
            all_shifts.sum(dim=-2) / num_atoms[:, None, None]
        )  # [B, V, 3]  Sum over different basis sets, e.g. z and x, and normalise.

        return shift


class DenseEquivariantCoordShiftModule(DenseEquivariantShiftModule):
    """Computes an equivariant shift vector for the z coordinates based on the z velocities.
    Since the coord shift can depend on the velocities, we use the
    ConditionalEquivariantVelocityBasisMixin.
    """

    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
    ):
        super().__init__(
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_layers_dims=hidden_layers_dims,
            basis=ConditionalEquivariantVelocityBasis(),
        )
        self.feature_processor = FeatureProcessor(
            input_pointwise_features_dim=self._input_pointwise_features_dim,
            input_relative_features_dim=self._input_relative_features_dim,
            processed_pointwise_features_dim=self._processed_pointwise_features_dim,
            processed_relative_features_dim=self._processed_relative_features_dim,
            hidden_layers_dims=hidden_layers_dims,
        )

    def _process_features(
        self,
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_features,  # [B, V, num_pointwise_feat]
        adj_list,
        masked_elements,  # [B, V]
    ) -> Tuple[Tensor, Tensor]:
        return self.feature_processor(
            relative_features=relative_features,
            pointwise_features=pointwise_features,
            adj_list=adj_list,
            masked_elements=masked_elements,
        )


class DenseEquivariantVelocShiftModule(DenseEquivariantShiftModule):
    """Computes an equivariant shift vector for the z velocities based on the z coordinates.
    Since the velocity shift can depend on the coordinates, we use the
    ConditionalEquivariantCoordBasisMixin.
    """

    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
    ):
        super().__init__(
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_layers_dims=hidden_layers_dims,
            basis=ConditionalEquivariantCoordBasis(),
        )
        self.feature_processor = FeatureProcessor(
            input_pointwise_features_dim=self._input_pointwise_features_dim,
            input_relative_features_dim=self._input_relative_features_dim,
            processed_pointwise_features_dim=self._processed_pointwise_features_dim,
            processed_relative_features_dim=self._processed_relative_features_dim,
            hidden_layers_dims=hidden_layers_dims,
        )

    def _process_features(
        self,
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_features,  # [B, V, num_pointwise_feat]
        adj_list,
        masked_elements,  # [B, V]
    ) -> Tuple[Tensor, Tensor]:
        return self.feature_processor(
            relative_features=relative_features,
            pointwise_features=pointwise_features,
            adj_list=adj_list,
            masked_elements=masked_elements,
        )


class DenseInvariantScaleModule(ScaleModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
        basis: ConditionalEquivariantBasis,
    ):
        super().__init__()
        self.basis = basis
        self._input_features_dim = input_features_dim
        self._output_features_dim = output_features_dim
        self.num_coord_pointwise_features = basis.num_state_pointwise_features
        self.num_coord_rel_features = basis.num_state_rel_features

        self._input_pointwise_features_dim = (
            self._input_features_dim + self.num_coord_pointwise_features
        )
        self._input_relative_features_dim = (
            self.num_coord_rel_features + 2 * self._input_pointwise_features_dim
        )
        self._processed_pointwise_features_dim = output_features_dim
        self._processed_relative_features_dim = output_features_dim
        # MLPs to get the scale vector (one scale per input point in the sequence)
        # TODO: Might want to try different MLPs for processing x and z
        self._scale_with_pointwise_mlp = MLP(
            input_dim=self._processed_pointwise_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=output_features_dim,
        )
        self._scale_with_relative_mlp = MLP(
            input_dim=self._processed_relative_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=output_features_dim,
        )
        # MLP that maps from latent space (where averaging occurs) to scale space of dim 1.
        self._scale_mlp = MLP(
            input_dim=output_features_dim,
            hidden_layer_dims=hidden_layers_dims,
            out_dim=1,  # Scale is a scalar.
        )

    def forward(
        self,
        adj_list,  # [num_edges, 2]  TODO (change for batching) TODO Currently not used!
        z_untransformed,  # [B, num_points, 3]
        x_features,  # [B, num_points, D]
        x_coords,  # [B, num_points, 3]
        x_velocs,  # [B, num_points, 3]
        masked_elements,  # [B, num_points]
    ) -> Tensor:  # [B, num_points, 3]
        """
        If inputs are the coordinates and features for N points, this module should
        return a (jointly SO(3) equivariant in x and z) shift vector of shape [N, 3].
        """
        # Get the pointwise and relative basis vectors.
        (
            relative_features,  # [B, V, V, num_rel_feat]
            pointwise_features,  # [B, V, num_pointwise_feat]
            _,  # Basis vectors not needed since we want the scaling factors to be *invariant*.
            _,
        ) = self.basis.get_equivariant_features_and_basis(
            atom_features=x_features,
            adj_list=adj_list,
            z_untransformed=z_untransformed,
            x_coords=x_coords,
            x_velocs=x_velocs,
        )

        # Process the pointwise and relative features in a permutation equivariant way.
        relative_features, pointwise_features = self._process_features(
            relative_features=relative_features,  # [B, V, V, num_rel_feat]
            pointwise_features=pointwise_features,  # [B, V, num_pointwise_feat]
            adj_list=adj_list,
            masked_elements=masked_elements,  # [B, V]
        )

        # Compute the output scale factors based on the processed features.
        unnorm_scale = self._calc_scale(
            pointwise_features=pointwise_features,  # [B, V, num_pointwise_feat]
            relative_features=relative_features,  # [B, V, V, num_rel_feat]
            masked_elements=masked_elements,  # [B, V]
        )  # [B, num_points, 1]

        return unnorm_scale

    @abc.abstractmethod
    def _process_features(
        self, relative_features, pointwise_features, adj_list, masked_elements
    ) -> Tuple[Tensor, Tensor]:
        """Process the SE(3) *invariant* features in a permutation equivariant way."""
        pass

    def _calc_scale(self, pointwise_features, relative_features, masked_elements):
        """Compute scale factors for the coupling layer. Let :math:`h_i` denote the pointwise features
        and :math:`h_{ij}` denote the relative features.

        Let :math:`\psi` and :math:`\phi` be pointwise and relative maps respectively that map from
        feature space to feature space. Let :math:`\gamma` be a map from feature space to scalars.
        The scale factor for atom i is:
        ..math::

            scale_i = \gamma( \psi(h_i) + \sum_j \phi(h_{ij}) )
        """
        num_atoms = (~masked_elements).sum(dim=-1)  # [B]

        mlp_relative_features = self._scale_with_relative_mlp(relative_features)  # [B, V, V, 1]
        # Mask out meaningless elements in sum over relative contributions to the scale.
        mlp_relative_features = mlp_relative_features * (
            ~masked_elements[:, None, :, None]
        )  # [B, V, V, 1]

        scale = self._scale_mlp(
            self._scale_with_pointwise_mlp(pointwise_features)  # [B, V, out_features_dim]
            + mlp_relative_features.sum(-2)
            / num_atoms[:, None, None]  # [B, V, V, out_features_dim] -> [B, V, out_features_dim]
        )  # [B, V, 1]

        return scale  # [B, V, 1]


class DenseInvariantCoordScaleModule(DenseInvariantScaleModule):
    """Computes an invariant scale vector for the z coordinates based on the z velocities.
    Since the coord features can depend on the velocities, we use the
    ConditionalEquivariantVelocityBasisMixin, which computes both invariant features and
    equivariant basis vectors, and ignore the basis vectors.
    """

    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
    ):
        super().__init__(
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_layers_dims=hidden_layers_dims,
            basis=ConditionalEquivariantVelocityBasis(),
        )
        self.feature_processor = FeatureProcessor(
            input_pointwise_features_dim=self._input_pointwise_features_dim,
            input_relative_features_dim=self._input_relative_features_dim,
            processed_pointwise_features_dim=self._processed_pointwise_features_dim,
            processed_relative_features_dim=self._processed_relative_features_dim,
            hidden_layers_dims=hidden_layers_dims,
        )

    def _process_features(
        self,
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_features,  # [B, V, num_pointwise_feat]
        adj_list,
        masked_elements,  # [B, V]
    ) -> Tuple[Tensor, Tensor]:
        return self.feature_processor(
            relative_features=relative_features,
            pointwise_features=pointwise_features,
            adj_list=adj_list,
            masked_elements=masked_elements,
        )


class DenseInvariantVelocScaleModule(DenseInvariantScaleModule):
    """Computes an invariant scale vector for the z velocities based on the z coordinates.
    Since the velocity shift can depend on the coordinates, we use the
    ConditionalEquivariantCoordBasisMixin, which computes both invariant features and
    equivariant basis vectors, and ignore the basis vectors.
    """

    def __init__(
        self,
        *,
        input_features_dim: int,
        output_features_dim: int,
        hidden_layers_dims: Sequence[int],
    ):
        super().__init__(
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_layers_dims=hidden_layers_dims,
            basis=ConditionalEquivariantCoordBasis(),
        )
        self.feature_processor = FeatureProcessor(
            input_pointwise_features_dim=self._input_pointwise_features_dim,
            input_relative_features_dim=self._input_relative_features_dim,
            processed_pointwise_features_dim=self._processed_pointwise_features_dim,
            processed_relative_features_dim=self._processed_relative_features_dim,
            hidden_layers_dims=hidden_layers_dims,
        )

    def _process_features(
        self,
        relative_features,  # [B, V, V, num_rel_feat]
        pointwise_features,  # [B, V, num_pointwise_feat]
        adj_list,
        masked_elements,  # [B, V]
    ) -> Tuple[Tensor, Tensor]:
        return self.feature_processor(
            relative_features=relative_features,
            pointwise_features=pointwise_features,
            adj_list=adj_list,
            masked_elements=masked_elements,
        )
