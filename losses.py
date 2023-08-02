import abc
import dataclasses
from functools import singledispatch
from typing import Collection, Optional, OrderedDict, Sequence, Tuple, Union

from multimethod import multimethod

import torch
import torch.nn.functional as F

from utilities.logger import TrainingLogger
from timewarp.dataloader import DenseMolDynBatch
from timewarp.modules.model_wrappers.density_model_base import (
    ConditionalDensityModel,
    ConditionalDensityModelWithForce,
)
from timewarp.utils.openmm import OpenMMProvider
from timewarp.utils.chirality import CiralityChecker

from utilities.model_utils import unflatten_state_dict


def compute_kinetic_energy(
    velocs: torch.Tensor,  # [batch_size, num_atoms, 3]
    masses: torch.Tensor,  # [batch_size, num_atoms]
    kbT: float,
    random_velocs=False,
) -> torch.Tensor:  # [batch_size, ]
    """
    Compute the kinetic energy of a batch of atom velocities.

    Args:
        velocs: Atom velocities. Shape: [batch_size, num_atoms, 3].
        masses: Atom masses. Shape: [batch_size, num_atoms].
        kbT: Boltzmann constant times temperature.
        random_velocs: If `True`, `velocs` are assumed to be realizations of a isotropic Gaussian with zero mean and unit variance.

    Returns:
        kinetic_energy: [batch_size, ]
    """
    if random_velocs:
        return 0.5 * ((velocs**2.0).sum(-1)).sum(-1)
    else:
        return 0.5 * (masses * (velocs**2.0).sum(-1)).sum(-1) / kbT


def _compute_potential_energy_single_protein(
    coord: torch.Tensor,  # [batch_size, num_atoms, 3]
    protein: str,
    mask: torch.Tensor,  # [batch_size, num_atoms]
    openmm_provider: OpenMMProvider,
) -> torch.Tensor:
    batch_size = coord.size(0)
    pot = openmm_provider.get_potential_energy_module(protein)
    return pot(coord[~mask, :].view(batch_size, -1, 3)).squeeze(-1) / openmm_provider.kbT


def compute_potential_energy(
    coords: torch.Tensor,  # [batch_size, num_atoms, 3]
    pdb_names: Sequence[str],  # [batch_size]
    masked_elements: torch.Tensor,  # [batch_size, num_atoms]
    openmm_provider: OpenMMProvider,
    segments: Optional[Sequence[int]] = None,  # [num_segments]
) -> torch.Tensor:
    """
    Compute the potential energy of a batch of atom coordinates.

    Args:
        coords: Atom coordinates. Shape: [batch_size, num_atoms, 3].
        pdb_names: Protein names. Shape: [batch_size, ].
        masked_elements: Mask for atoms that should be ignored. Shape: [batch_size, num_atoms].
        openmm_provider: OpenMMProvider.
        segments: contiguous segments of the batch that contain the same protein. Shape: [num_segments].

    Returns:
        potential_energy: [batch_size, ]
    """
    if segments is not None:
        # Make use of contiguous segments to compute for subset of
        # batch containing the same protein.
        potential_energies = [
            _compute_potential_energy_single_protein(
                coords[segments[i] : segments[i + 1]],
                pdb_names[segments[i]],
                masked_elements[segments[i] : segments[i + 1]],
                openmm_provider=openmm_provider,
            )  # [segment_length, ]
            for i in range(len(segments) - 1)
        ]
    else:
        # Compute each element in the batch independently.
        potential_energies = [
            _compute_potential_energy_single_protein(
                coord, name, mask, openmm_provider=openmm_provider
            )  # [1, ]
            for (name, coord, mask) in zip(pdb_names, coords, masked_elements)
        ]
    return torch.hstack(potential_energies)  # [B, ]


def compute_energy(
    coords: torch.Tensor,  # [batch_size, num_atoms, 3]
    velocs: torch.Tensor,  # [batch_size, num_atoms, 3]
    pdb_names: Sequence[str],  # [batch_size, ]
    masked_elements: torch.Tensor,  # [batch_size, num_atoms]
    openmm_provider: OpenMMProvider,
    random_velocs: bool = False,
    masses: Optional[torch.Tensor] = None,
    segments: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # [batch_size, ]
    """
    Compute the potential and kinetic energy of a batch of coordinates and velocities.

    Args:
        coords: Atom coordinates. Shape: [batch_size, num_atoms, 3].
        velocs: Atom velocities. Shape: [batch_size, num_atoms, 3].
        pdb_names: Protein names. Shape: [batch_size, ].
        masked_elements: Mask for atoms that should be ignored. Shape: [batch_size, num_atoms].
        openmm_provider: OpenMMProvider.
        random_velocs: If `True`, the kinetic energy is computed as if the velocities are
            realizations of a isotropic Gaussian distribution with zero mean and unit variance.
        masses: Atom masses. Shape: [batch_size, num_atoms].
        segments: contiguous segments of the batch that contain the same protein. Shape: [num_segments].

    Returns:
        energy: [batch_size, ]
        (potential_energy, kinetic_energy): [batch_size, ]
    """
    if masses is None:
        # TODO : Instead dispatch on whether or not it's a single `name` or multiple `names`.
        masses_list = [openmm_provider.get_masses(name).to(coords.device) for name in pdb_names]
        max_num_particles = masked_elements.size(-1)

        # NOTE : This assumes a particular ordering of the masking.
        # I.e. if we apply a transform which permutes the atoms, then this will break.
        masses = torch.stack(
            [F.pad(m, (0, max_num_particles - m.size(0)), "constant", 0) for m in masses_list]
        )

    energy_kinetic = compute_kinetic_energy(
        velocs, masses, openmm_provider.kbT, random_velocs=random_velocs
    )

    energy_potential = compute_potential_energy(
        coords, pdb_names, masked_elements, openmm_provider, segments=segments
    )

    energy = energy_kinetic + energy_potential
    return energy, (energy_potential, energy_kinetic)


# TODO : Is there a more suitable place for this?
@multimethod
def batch_to(
    model: ConditionalDensityModel, batch: DenseMolDynBatch, device: Union[str, torch.device]
) -> DenseMolDynBatch:
    """
    Replace attributes of `batch` needed by `model` with value which are now on `device`.
    """
    return dataclasses.replace(
        batch,
        atom_coords=batch.atom_coords.to(device, non_blocking=True),
        atom_velocs=batch.atom_velocs.to(device, non_blocking=True),
        atom_coord_targets=batch.atom_coord_targets.to(device, non_blocking=True),
        atom_veloc_targets=batch.atom_veloc_targets.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        atom_types=batch.atom_types.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
    )


@batch_to.register
def _(
    model: ConditionalDensityModelWithForce, batch: DenseMolDynBatch, device: None
) -> DenseMolDynBatch:
    return batch


@batch_to.register
def _(
    model: ConditionalDensityModelWithForce,
    batch: DenseMolDynBatch,
    device: Union[str, torch.device],
) -> DenseMolDynBatch:
    return dataclasses.replace(
        batch,
        atom_forces=batch.atom_forces.to(device, non_blocking=True),
        atom_force_targets=batch.atom_force_targets.to(device, non_blocking=True),
        atom_coords=batch.atom_coords.to(device, non_blocking=True),
        atom_velocs=batch.atom_velocs.to(device, non_blocking=True),
        atom_coord_targets=batch.atom_coord_targets.to(device, non_blocking=True),
        atom_veloc_targets=batch.atom_veloc_targets.to(device, non_blocking=True),
        masked_elements=batch.masked_elements.to(device, non_blocking=True),
        atom_types=batch.atom_types.to(device, non_blocking=True),
        adj_list=batch.adj_list.to(device, non_blocking=True),
        edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
    )


@batch_to.register
def _(
    model: ConditionalDensityModelWithForce, batch: DenseMolDynBatch, device: None
) -> DenseMolDynBatch:
    return batch


class AbstractLoss(abc.ABC):
    pass


# NOTE : Instead of attaching the loss-computation to the `AbstractLoss` itself,
# we make use of multiple dispatch provided by `multimethod` so that we can further specialize
# implementations on (loss, model)-pairs rather than _just_ model. For example, the `NegativeLogLikelihoodLoss`
# might be differently implemented for `(NegativeLogLikelihoodLoss, ConditionalDensityModel)` and
# `(NegativeLogLikelihoodLoss, ConditionalDensityModelWithForce)`.
# TODO : Implement losses for `ConditionalDensityModelWithForce`.
@multimethod
def get_loss(
    loss: AbstractLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    """Compute `loss` for `model` on `batch`.

    Args:
        loss: Loss to compute.
        model: Model to compute loss for.
        batch: Batch to compute loss on.
        device: Device to move batch to before computing loss.
        logger: Logger to log to.

    Returns:
        Loss value.
    """
    raise TypeError()


class LossWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, loss: Optional[AbstractLoss] = None):
        super().__init__()
        self.module = module
        self.loss = loss

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        assert isinstance(
            self.module, torch.nn.Module
        ), "`module` should be an instance of `torch.nn.Module`"

        nested_state_dict = unflatten_state_dict(state_dict, depth=1)

        if "module" not in nested_state_dict:
            print("`state_dict` seems to be meant for `module`; loading this instead")
            self.module.load_state_dict(state_dict, strict=strict)
        else:
            self.module.load_state_dict(nested_state_dict["module"], strict=strict)

            if "loss" in nested_state_dict:
                if isinstance(self.loss, torch.nn.Module):
                    self.loss.load_state_dict(nested_state_dict["loss"], strict=strict)
                elif strict:
                    # Warn the user.
                    loss_keys = list(nested_state_dict["loss"].keys())
                    print(
                        f"LossWrapper: loss is not a `torch.nn.Module` but `state_dict` contains the following loss parameters {loss_keys}"
                    )

    def forward(self, *args, **kwargs):
        assert self.loss is not None, "`loss` is not given"
        return get_loss(self.loss, self.module, *args, **kwargs)


@singledispatch
def wrap_or_replace_loss(model: torch.nn.Module, loss: AbstractLoss) -> LossWrapper:
    """
    Replace loss if `model` is a `LossWrapper`; otherwise wrap `model` in `loss`.

    Notes:
        If `model` is nested `LossWrapper`, i.e. `wrap_or_replace_loss(model, loss).module` is a `LossWrapper`,
        the returned instance will be \"un-nested\", i.e. `wrap_or_replace(model, loss).module` is NOT a `LossWrapper`.
    """
    return LossWrapper(module=model, loss=loss)


@wrap_or_replace_loss.register
def _(model: LossWrapper, loss: AbstractLoss) -> LossWrapper:
    return wrap_or_replace_loss(model.module, loss)


@singledispatch
def unwrap_loss_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying module if `model` is a `LossWrapper`; otherwise return `model`.
    """
    return model


@unwrap_loss_wrapper.register
def _(model: LossWrapper) -> torch.nn.Module:
    return unwrap_loss_wrapper(model.module)


class NegativeLogLikelihoodLoss(AbstractLoss):
    """
    Negative log-likelihood loss for conditional density models.

    Attributes:
        random_velocs: If `True`, draw random velocities from a Gaussian distribution with zero mean and unit variance.

    Notes:
        This loss is only defined for conditional density models.
    """

    def __init__(self, random_velocs=True):
        super().__init__()
        self.random_velocs = random_velocs


@get_loss.register
def _(
    loss: NegativeLogLikelihoodLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    if device is not None:
        batch = batch_to(model, batch, device)

    if loss.random_velocs:
        x_velocs = torch.randn_like(batch.atom_velocs, device=device).contiguous()
        y_velocs_target = torch.randn_like(batch.atom_veloc_targets, device=device).contiguous()
    else:
        x_velocs = batch.atom_velocs
        y_velocs_target = batch.atom_veloc_targets

    x_coords = batch.atom_coords
    y_coords_target = batch.atom_coord_targets
    atom_types = batch.atom_types
    adj_list = batch.adj_list
    edge_batch_idx = batch.edge_batch_idx
    masked_elements = batch.masked_elements

    return model(
        atom_types=atom_types,
        x_coords=x_coords,
        x_velocs=x_velocs,
        y_coords=y_coords_target,
        y_velocs=y_velocs_target,
        adj_list=adj_list,
        edge_batch_idx=edge_batch_idx,
        masked_elements=masked_elements,
        logger=logger,
    )


class AcceptanceLoss(AbstractLoss):
    """
    Acceptance loss for conditional density models.

    Notes:
        This loss is only defined for conditional density models.
    """

    def __init__(
        self,
        openmm_provider: OpenMMProvider,
        random_velocs: bool = True,
        beta: float = 0.0,
        clamp: bool = False,
        num_samples: int = 1,
        high_energy_threshold: float = -1,
    ):
        """
        Args:
            openmm_provider: OpenMM provider.
            random_velocs: If `True`, draw random velocities from a Gaussian distribution with zero mean and unit variance.
            beta: Scale factor for the entropy term.
            clamp: If `True`, clamp the acceptance probability to [0, 1].
            num_samples: Number of samples to draw from the conditional density model.
        """
        super().__init__()
        self.openmm_provider = openmm_provider
        self.random_velocs = random_velocs
        self.beta = beta
        self.clamp = clamp
        self.num_samples = num_samples
        self.high_energy_threshold = high_energy_threshold
        # If high energies are discarded, check also for chirality changes
        if self.high_energy_threshold != -1:
            self.chirality_checker = CiralityChecker(openmm_provider.pdb_dirs)


@get_loss.register
def _(
    loss: AcceptanceLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    if device is not None:
        batch = batch_to(model, batch, device)

    # TODO : Pass this somewhere.
    random_velocs = loss.random_velocs

    if random_velocs:
        x_velocs = torch.randn_like(batch.atom_velocs, device=device).contiguous()
    else:
        x_velocs = batch.atom_velocs

    x_coords = batch.atom_coords
    atom_types = batch.atom_types
    adj_list = batch.adj_list
    edge_batch_idx = batch.edge_batch_idx
    masked_elements = batch.masked_elements

    num_atoms = (~masked_elements).sum(dim=-1)

    loss_accum = torch.tensor(0.0, device=device)

    # Logging stats.
    energy_accum_x = torch.tensor(0.0, device=device, requires_grad=False)
    energy_potential_accum_x = torch.tensor(0.0, device=device, requires_grad=False)
    energy_kinetic_accum_x = torch.tensor(0.0, device=device, requires_grad=False)

    energy_accum_y = torch.tensor(0.0, device=device, requires_grad=False)
    energy_potential_accum_y = torch.tensor(0.0, device=device, requires_grad=False)
    energy_kinetic_accum_y = torch.tensor(0.0, device=device, requires_grad=False)

    logp_xy_accum = torch.tensor(0.0, device=device, requires_grad=False)
    logp_yx_accum = torch.tensor(0.0, device=device, requires_grad=False)
    neg_log_acceptance_accum = torch.tensor(0.0, device=device, requires_grad=False)

    # TODO : Perform batch computations rather than for-loop.
    for _ in range(loss.num_samples):
        y_coords, y_velocs, logp_xy = model.conditional_sample_with_logp(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,  # type: ignore
            num_samples=1,
            logger=logger,
        )
        y_coords = y_coords.squeeze(0)
        y_velocs = y_velocs.squeeze(0)

        logp_yx = model.log_likelihood(
            atom_types=atom_types,
            x_coords=y_coords,
            x_velocs=y_velocs if random_velocs else -y_velocs,
            y_coords=x_coords,
            y_velocs=x_velocs if random_velocs else -x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,  # type: ignore
            logger=logger,
        )

        masses_list = [
            loss.openmm_provider.get_masses(name).to(x_coords.device) for name in batch.names
        ]
        max_num_particles = atom_types.size(-1)
        # NOTE : This assumes a particular ordering of the masking.
        # I.e. if we apply a transform which permutes the atoms, then this will break.
        masses = torch.stack(
            [F.pad(m, (0, max_num_particles - m.size(0)), "constant", 0) for m in masses_list]
        )

        energy_x, (energy_potential_x, energy_kinetic_x) = compute_energy(
            x_coords,
            x_velocs,
            batch.names,
            masked_elements,
            loss.openmm_provider,
            random_velocs=random_velocs,
            masses=masses,
            segments=batch.segments,
        )
        energy_y, (energy_potential_y, energy_kinetic_y) = compute_energy(
            y_coords,
            y_velocs,
            batch.names,
            masked_elements,
            loss.openmm_provider,
            random_velocs=random_velocs,
            masses=masses,
            segments=batch.segments,
        )

        energy_delta = energy_y - energy_x
        # Clamping to max 0 is equivalent of min(1, acceptance)
        neg_log_acceptance = energy_delta + logp_xy - logp_yx
        neg_conditional_entropy = logp_xy

        if logger is not None:
            energy_accum_x += (energy_x / num_atoms).mean()
            energy_potential_accum_x += (energy_potential_x / num_atoms).mean()
            energy_kinetic_accum_x += (energy_kinetic_x / num_atoms).mean()

            energy_accum_y += (energy_y / num_atoms).mean()
            energy_potential_accum_y += (energy_potential_y / num_atoms).mean()
            energy_kinetic_accum_y += (energy_kinetic_y / num_atoms).mean()

            logp_xy_accum += (logp_xy / num_atoms).mean()
            logp_yx_accum += (logp_yx / num_atoms).mean()
            neg_log_acceptance_accum += (neg_log_acceptance / num_atoms).mean()

        if loss.clamp:
            total_loss = (
                torch.clamp(neg_log_acceptance, max=0) + loss.beta * neg_conditional_entropy
            )
        else:
            total_loss = neg_log_acceptance + loss.beta * neg_conditional_entropy

        # exclude high energy samples
        # if energy difference to conditioning is larger than high_energy_threshold.
        if loss.high_energy_threshold != -1:
            # Check where chirality changes occur
            chirality_changes = loss.chirality_checker.check_changes(
                batch, y_coords, masked_elements
            )
            # add a energy penalty if chirality changes
            energy_delta[chirality_changes] += 100000
            good_energies = energy_delta < loss.high_energy_threshold
            total_loss = total_loss[good_energies.reshape(total_loss.shape)]
            num_atoms = num_atoms[good_energies]
            # return high loss if whole batch has bad energies
            if len(num_atoms) == 0:
                total_loss = torch.tensor(10000.0, device=device)
                num_atoms = torch.tensor(1.0, device=device)

        loss_accum = loss_accum + (total_loss / num_atoms).mean()

    acceptance_loss = loss_accum / loss.num_samples  # type: ignore

    if logger is not None:
        logger.log_scalar_async("energy_x", energy_accum_x / loss.num_samples)
        logger.log_scalar_async("energy_potential_x", energy_potential_accum_x / loss.num_samples)
        logger.log_scalar_async("energy_kinetic_x", energy_kinetic_accum_x / loss.num_samples)

        logger.log_scalar_async("energy_y", energy_accum_y / loss.num_samples)
        logger.log_scalar_async("energy_potential_y", energy_potential_accum_y / loss.num_samples)
        logger.log_scalar_async("energy_kinetic_y", energy_kinetic_accum_y / loss.num_samples)

        logger.log_scalar_async("logp_xy", logp_xy_accum / loss.num_samples)
        logger.log_scalar_async("logp_yx", logp_yx_accum / loss.num_samples)
        logger.log_scalar_async("acceptance_loss", acceptance_loss)
        logger.log_scalar_async("neg_log_acceptance", neg_log_acceptance_accum)

    return acceptance_loss


class EnergyLoss(AbstractLoss):
    """
    Energy loss for conditional density models.

    Notes:
        This loss is currently only applicable to conditional density models.
    """

    def __init__(
        self,
        openmm_provider: OpenMMProvider,
        random_velocs: bool = True,
        num_samples: int = 1,
    ):
        """
        Args:
            openmm_provider: OpenMM provider.
            random_velocs: If `True`, velocities are assumed to be sampled from a isotropic Gaussian with zero mean and unit variance.
            num_samples: Number of samples to use for the energy loss.
        """
        super().__init__()
        self.openmm_provider = openmm_provider
        self.random_velocs = random_velocs
        self.num_samples = num_samples


@get_loss.register
def _(
    loss: EnergyLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    if device is not None:
        batch = batch_to(model, batch, device)

    random_velocs = loss.random_velocs

    if random_velocs:
        x_velocs = torch.randn_like(batch.atom_velocs, device=device).contiguous()
    else:
        x_velocs = batch.atom_velocs

    x_coords = batch.atom_coords
    atom_types = batch.atom_types
    adj_list = batch.adj_list
    edge_batch_idx = batch.edge_batch_idx
    masked_elements = batch.masked_elements

    num_atoms = (~masked_elements).sum(dim=-1)

    loss_accum = torch.tensor(0.0, device=device)

    # Logging stats.
    energy_accum = torch.tensor(0.0, device=device, requires_grad=False)
    energy_potential_accum = torch.tensor(0.0, device=device, requires_grad=False)
    energy_kinetic_accum = torch.tensor(0.0, device=device, requires_grad=False)

    logp_xy_accum = torch.tensor(0.0, device=device, requires_grad=False)

    # TODO : Perform batch computations rather than for-loop.
    for _ in range(loss.num_samples):
        y_coords, y_velocs, logp_xy = model.conditional_sample_with_logp(
            atom_types=atom_types,
            x_coords=x_coords,
            x_velocs=x_velocs,
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,  # type: ignore
            num_samples=1,
            logger=logger,
        )
        y_coords = y_coords.squeeze(0)
        y_velocs = y_velocs.squeeze(0)

        energy, (energy_potential, energy_kinetic) = compute_energy(
            y_coords,
            y_velocs,
            batch.names,
            masked_elements,
            loss.openmm_provider,
            random_velocs=random_velocs,
            segments=batch.segments,
        )

        # Clamping to max 0 is equivalent of min(1, acceptance)
        energy_loss = energy + logp_xy

        if logger is not None:
            energy_accum += (energy / num_atoms).mean()
            energy_potential_accum += (energy_potential / num_atoms).mean()
            energy_kinetic_accum += (energy_kinetic / num_atoms).mean()

            logp_xy_accum += (logp_xy / num_atoms).mean()

        loss_accum = loss_accum + (energy_loss / num_atoms).mean()

    if logger is not None:
        logger.log_scalar_async("energy", energy_accum / loss.num_samples)
        logger.log_scalar_async("energy_potential", energy_potential_accum / loss.num_samples)
        logger.log_scalar_async("energy_kinetic", energy_kinetic_accum / loss.num_samples)

        logger.log_scalar_async("logp_xy", logp_xy_accum / loss.num_samples)

    return loss_accum / loss.num_samples  # type: ignore


class FlippedLoss(AbstractLoss):
    """
    Compute the loss of a model on both the original and flipped batch.

    See `flip_batch` for more details.

    Notes:
        This loss is currently only applicable to conditional density models.
    """

    def __init__(self, loss: AbstractLoss, random_velocs: bool = True):
        """
        Args:
            loss: Loss to compute on both the original and flipped batch.
            random_velocs: If `True`, velocities are assumed to be sampled from a isotropic Gaussian with zero mean and unit variance.
        """
        super().__init__()
        self.loss = loss
        self.random_velocs = random_velocs


def flip_batch(batch: DenseMolDynBatch, random_velocs: bool = False):
    """
    Flips the velocities in the batch.

    Args:
        batch: Batch to flip.
        random_velocs: If `True`, this is effectively a no-op.
    """
    x_coords = batch.atom_coords
    x_velocs = batch.atom_velocs
    y_coords = batch.atom_coord_targets
    y_velocs = batch.atom_veloc_targets

    if not random_velocs:
        x_velocs = -x_velocs
        y_velocs = -y_velocs

    return dataclasses.replace(
        batch,
        atom_coords=y_coords,
        atom_velocs=y_velocs,
        atom_coord_targets=x_coords,
        atom_veloc_targets=x_velocs,
    )


@get_loss.register
def _(
    loss: FlippedLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    return get_loss(
        loss.loss,
        model,
        flip_batch(batch, random_velocs=loss.random_velocs),
        device=device,
        logger=logger,
    )


class ConvexCombinationLoss(AbstractLoss, torch.nn.Module):
    """
    Compute the convex combination of multiple losses.
    """

    def __init__(
        self,
        losses: Collection[AbstractLoss],
        weights: Optional[torch.Tensor] = None,
        pre_softmax_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            losses: Losses to combine.
            weights: Weights to use for each loss.
            pre_softmax_weights: Weights to use for each loss before applying softmax.
                Note that either `weights` or `pre_softmax_weights` must be provided.
        """
        super().__init__()

        self.losses = losses

        # FIXME : Figure out a way to allow persisting the weights while avoiding issues with deepspeed.
        self.register_buffer("_weights", weights, persistent=False)
        self.register_buffer("pre_softmax_weights", pre_softmax_weights, persistent=False)

        assert (
            weights is not None or pre_softmax_weights is not None
        ), "either weights or pre_softmax_weights has to be specified"

    @property
    def weights(self) -> torch.Tensor:
        """
        Weights to use for each loss.
        """
        if self._weights is not None:
            return self._weights  # type: ignore
        elif self.pre_softmax_weights is not None:
            return F.softmax(self.pre_softmax_weights, 0)  # type: ignore
        else:
            raise ValueError()


@get_loss.register
def _(
    loss: ConvexCombinationLoss,
    model: ConditionalDensityModel,
    batch: DenseMolDynBatch,
    device: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
):
    if device is not None:
        batch = batch_to(model, batch, device)

    losses = torch.stack(
        [get_loss(l, model, batch, device=device, logger=logger) for l in loss.losses]
    )
    return (loss.weights * losses).sum()


class AbstractLossSchedule(abc.ABC):
    """Abstract class for loss schedules.

    A loss schedule defines a behavior that can be used to modify the loss in some way based on the current training step.
    """

    @abc.abstractmethod
    def to(self, device: Union[str, torch.device]):
        pass


@multimethod
def loss_schedule_step(
    schedule: AbstractLossSchedule,
    loss: AbstractLoss,
    step_idx: int,
    logger: Optional[TrainingLogger] = None,
):
    """Perform a step of the loss schedule, mutating the `loss` in the process.

    Args:
        schedule: The loss schedule.
        loss: The loss to schedule.
        step_idx: The current step index.
        logger: The logger to use.
    """
    raise TypeError()


# TODO : This doesn't seem like the greatest approach. It's somewhat non-trivial to set this correctly,
# usually requiring manual/visual inspection of the loss schedule. It would be nice to have a more principled way of doing things.
class GeometricLossSchedule(AbstractLossSchedule):
    """Geometrically decreasing loss schedule.

    How this is applied to the loss, depends on the implementation of :func:`loss_schedule`.

    For :class:`ConvexCombinationLoss` this schedule will apply the `factor` to the `pre_softmax_weights` of the loss.
    By choosing different values for `factor`, it is possible to implement schudules with very different limits, e.g.:
    - If `factor < 1` and the `pre_softmax_weights` are positive, the loss weights will converge to uniform.
    - If `factor = [-1, 1]` and the `pre_softmax_weights` are positive, the loss weights will converge to only putting
      weight on the second loss.

    Notes:
        Currently, :func:`loss_schedule_step` is only implemented for :class:`ConvexCombinationLoss`.
        For any other loss, this schedule is a no-op.

    Warning:
        This schedule is not _currently_ compatible with deepspeed.

    Examples:

        Example of schedule where the limit is uniform weights:

        >>> import torch
        >>> from timewarp.losses import NegativeLogLikelihoodLoss, AcceptanceLoss, \
        ...     ConvexCombinationLoss, GeometricLossSchedule, loss_schedule_step
        >>> loss1 = NegativeLogLikelihoodLoss()
        >>> loss2 = AcceptanceLoss(OpenMMProvider("/path/to/pdb/files"))
        >>> loss_computer = ConvexCombinationLoss([loss1, loss2], pre_softmax_weights=torch.tensor([1.0, 10.0]))
        >>> schedule = GeometricLossSchedule(every=1, factor=0.9)
        >>> for step_idx in range(100):
        ...     loss_schedule_step(schedule, loss_computer, step_idx)
        >>> loss_computer.weights
        tensor([0.4999, 0.5001])

        Example of losses where the initial weights are uniform, but the limit is the delta for the first loss:

        >>> import torch
        >>> from timewarp.losses import NegativeLogLikelihoodLoss, AcceptanceLoss, \
        ...     ConvexCombinationLoss, GeometricLossSchedule, loss_schedule_step
        >>> loss1 = NegativeLogLikelihoodLoss()
        >>> loss2 = AcceptanceLoss(OpenMMProvider("/path/to/pdb/files"))
        >>> # Negative `pre_softmax_weights`.
        ... loss_computer = ConvexCombinationLoss([loss1, loss2], pre_softmax_weights=torch.tensor([-1.0, -1.0]))
        >>> # We don't change the weight of the first loss, but keep scaling the second loss.
        ... loss_schedule = GeometricLossSchedule(every=1, factor=torch.tensor([1.0, 2.0]))
        >>> for step_idx in range(100):
        ...     loss_schedule_step(loss_schedule, loss_computer, step_idx)
        >>> loss_computer.weights
        tensor([1., 0.])
    """

    def __init__(
        self,
        every: int,
        factor: Union[float, torch.Tensor],
        maximum: Union[float, torch.Tensor] = float("inf"),
        minimum: Union[float, torch.Tensor] = 0,
    ):
        """
        Args:
            every: How often to apply the schedule.
            factor: The factor to multiply the loss weights with.
            maximum: The maximum allowed value of the loss weights.
            minimum: The minimum allowed value of the loss weights.
        """
        super().__init__()
        self.every = every
        self.factor: torch.Tensor = (
            factor if isinstance(factor, torch.Tensor) else torch.tensor(factor)
        )
        self.maximum: torch.Tensor = (
            maximum if isinstance(maximum, torch.Tensor) else torch.tensor(maximum)
        )
        self.minimum: torch.Tensor = (
            minimum if isinstance(minimum, torch.Tensor) else torch.tensor(minimum)
        )

    def to(self, device: Union[str, torch.device]):
        """Move the schedule to a given device."""
        self.factor = self.factor.to(device)
        self.maximum = self.maximum.to(device)
        self.minimum = self.minimum.to(device)

        return self


@loss_schedule_step.register
def _(
    schedule: GeometricLossSchedule,
    loss: AbstractLoss,
    step_idx: int,
    logger: Optional[TrainingLogger] = None,
):
    # Do nothing by default.
    pass


@loss_schedule_step.register
def _(
    schedule: GeometricLossSchedule,
    loss: ConvexCombinationLoss,
    step_idx: int,
    logger: Optional[TrainingLogger] = None,
):
    assert (
        loss.pre_softmax_weights is not None
    ), "can only adapt `ConvexCombinationLoss` with `pre_softmax_weights` specified"

    if step_idx % schedule.every == 0:
        # When `loss.pre_softmax_weights` are positive and `schedule.factor` is in (0, 1),
        # this will result in slowly moving towards uniform weighting.
        assert isinstance(loss.pre_softmax_weights, torch.Tensor)
        total_factor = schedule.factor ** (step_idx // schedule.every)
        if torch.all(total_factor <= schedule.maximum) and torch.all(
            total_factor >= schedule.minimum
        ):
            loss.pre_softmax_weights *= schedule.factor

        if logger is not None:
            for i in range(len(loss.pre_softmax_weights)):
                logger.log_scalar_async(f"loss_weight_{i}", loss.weights[i])
