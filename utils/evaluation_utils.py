from dataclasses import astuple, dataclass
from functools import singledispatch
import pickle
from git.types import PathLike
import torch
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import os
import sys

from tqdm.auto import tqdm
import openmm
import openmm.app
import mdtraj as md
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timewarp.dataloader import DenseMolDynBatch
from timewarp.modules.model_wrappers.flow import ConditionalFlowDensityModel
from timewarp.equivariance.equivariance_transforms import transform_batch, random_rotation_matrix
from timewarp.utils.openmm.openmm_bridge import OpenmmPotentialEnergyTorch
from timewarp.utils.chirality import check_symmetry_change


ATOM_NAMES = ["C", "H", "N", "O", "S"]

# %%%% Sampling %%%%


def compute_num_proposal_steps(
    current_acceptance_probability: float,
    target_acceptance_per_step: float = 0.9,
    max_num_proposal_steps: int = 100,
):
    """Compute the number of proposal steps needed to achieve acceptance of at least one proposal with the given probability.

    args:
        current_acceptance_probability: acceptance probability from which to compute num steps.
        target_acceptance_per_step: desired probability of at least one acceptance within a proposal batch. Default: 0.9.
        max_num_proposal_steps: maximum number of proposal steps to allow. Default: 100.
    """
    # Make sure that we never touch the boundaries of `(0, 1)`.
    probability_of_rejection = min(max(1 - current_acceptance_probability, 1e-3), 1 - 1e-3)
    # Ensure that we at least take one step.
    return max(
        # Convert to integer.
        # NOTE: Converting _after_ `min` call in case of floating point errors leads to NaN.
        int(
            np.ceil(
                # Don't allow more than `max_num_proposal_steps` proposal steps.
                min(
                    # Replace possible NaN with `inf` so we fall back to `max_num_proposal_steps`.
                    np.nan_to_num(
                        np.log(1 - target_acceptance_per_step) / np.log(probability_of_rejection),
                        nan=np.inf,
                    ),
                    max_num_proposal_steps,
                )
            )
        ),
        1,
    )


@dataclass
class ChainStats:
    """Statistics of a single chain.

    Attributes:
        acceptance_indicator: 1 if the proposal was accepted, 0 otherwise.
        acceptance: acceptance probability of the proposal at a given iteration.
        p_xy: log probability of going from current state `x` to proposed state `y`.
        p_yx: log probability of going from proposed state `y` to current state `x`.
        exponent: negative log acceptance probabilit.
        energies_pot: potential energies of the proposed states.
        energies_kin: kinetic energies of the proposed states.
        energies_pot_delta: difference between potential energies of proposed and current states.
        energies_kin_delta: difference between kinetic energies of proposed and current states.
    """

    acceptance_indicator: np.ndarray
    acceptance: np.ndarray
    p_xy: np.ndarray
    p_yx: np.ndarray
    exponent: np.ndarray
    energies_pot: np.ndarray
    energies_kin: np.ndarray
    energies_pot_delta: np.ndarray
    energies_kin_delta: np.ndarray

    def __len__(self):
        return len(self.acceptance)

    def __getitem__(self, key):
        values = astuple(self)
        return ChainStats(*map(lambda x: x[key], values))

    def thin(self, step):
        """Thin the chain by a given step."""
        values = astuple(self)
        return ChainStats(*map(lambda x: x[0 : x.shape[0] : step], values))

    def save(self, path: PathLike):
        """Save the chain to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: PathLike):
        """Load the chain from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


@singledispatch
def visualize(
    chain_stats: ChainStats,
    include_transition_probs: bool = False,
    include_acceptance_indicator: bool = True,
    include_exponent: bool = True,
    include_current_energy: bool = False,
    delta: bool = False,
    ylims: Optional[Tuple[int, int]] = None,
    figure: Optional[plt.Figure] = None,
    axis: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize the chain statistics.

    Args:
        chain_stats: Chain statistics to visualize.
        include_transition_probs: Whether to include the transition probabilities.
        include_acceptance_indicator: Whether to include the acceptance indicator.
        include_exponent: Whether to include the exponent.
        include_current_energy: Whether to include the current energy.
        delta: Whether to plot the difference between current and proposed energies.
        ylims: Y-axis limits.
    """
    num_steps = len(chain_stats)
    (indices,) = np.where(chain_stats.acceptance_indicator)

    if figure is None and axis is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
    elif figure is not None or axis is not None:
        raise ValueError("Either both or none of `figure` and `axis` must be provided.")

    if delta:
        ax.plot(chain_stats.energies_kin_delta, label="Kinetic energy (Δ)", alpha=0.7)
        ax.plot(chain_stats.energies_pot_delta, label="Potential energy (Δ)", alpha=0.7)
    else:
        ax.plot(chain_stats.energies_kin, label="Kinetic energy", alpha=0.7)
        ax.plot(chain_stats.energies_pot, label="Potential energy", alpha=0.7)

    if include_exponent:
        ax.plot(chain_stats.exponent, label="-log α", alpha=0.8)

    if include_transition_probs:
        ax.plot(chain_stats.p_xy - chain_stats.p_yx, label="log p_xy - log p_yx", alpha=0.7)  # type: ignore

    if include_current_energy:
        e_pot_current = chain_stats.energies_pot.flatten() - chain_stats.energies_pot_delta.flatten()  # type: ignore
        ax.plot(e_pot_current, label="Potential energy (current)", alpha=0.7)

    ax.set_ylim(ylims)
    ax.hlines([0], [0], [num_steps], label="P(accept) = 1", color="g", linestyle="dashed")
    ax.hlines(
        [-np.log(1 / 10)],
        [0],
        [num_steps],
        label="P(accept) = 1/10",
        color="gray",
        linestyle="dashed",
    )
    if include_acceptance_indicator:
        ax.plot(
            indices,
            np.zeros_like(indices),
            "x",
            markersize=12,
            label=f"Accepted (n = {len(indices)})",
            color="r",
        )

    plt.legend()

    return fig, ax


def sample_on_batches(
    batches,
    model,
    device,
    openmm_potential_energy_torch,
    data_augmentation,
    masses,
    random_velocs=False,
):
    """Sample conditioned on samples from the Boltzmann distribution.

    Args:
        batches: Samples from the Boltzmann distribution.
        model: Model to run things on.
        device: device
        openmm_potential_energy_torch: Energy function nof the system.
        data_augmentation: Whether to use data augmentation.
        masses: Masses of the system
        random_velocs: Whether to treat velocities of the conditioning and
            target state as isotropic Gaussians.
    """
    y_coords_model: List[np.ndarray] = []
    y_velocs_model: List[np.ndarray] = []
    traj_coords: List[np.ndarray] = []
    traj_velocs: List[np.ndarray] = []
    traj_coords_conditioning: List[np.ndarray] = []
    traj_velocs_conditioning: List[np.ndarray] = []
    acceptance = []
    p_xys = []
    p_yxs = []
    p_xys_training = []
    p_yxs_training = []
    energies_pot = []
    energies_kin = []

    with torch.no_grad():
        for batch in tqdm(batches):
            if data_augmentation:
                assert isinstance(batch, DenseMolDynBatch)
                batch = transform_batch(batch)

            x_coords = batch.atom_coords.to(device).contiguous()
            y_coord_targets = batch.atom_coord_targets.to(device)

            if random_velocs:
                x_velocs = torch.randn_like(x_coords)
                y_veloc_targets = torch.randn_like(y_coord_targets)
            else:
                x_velocs = batch.atom_velocs.to(device).contiguous()
                y_veloc_targets = batch.atom_veloc_targets.to(device)

            y_coords, y_velocs = model.conditional_sample(
                atom_types=batch.atom_types.to(device, non_blocking=True),
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=batch.adj_list.to(device, non_blocking=True),
                edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
                masked_elements=batch.masked_elements.to(device, non_blocking=True),
                num_samples=1,
            )
            y_coords = y_coords.squeeze(0)
            y_velocs = y_velocs.squeeze(0)
            p_xy = model.log_likelihood(
                atom_types=batch.atom_types.to(device, non_blocking=True),
                x_coords=x_coords,
                x_velocs=x_velocs,
                y_coords=y_coords,
                y_velocs=y_velocs,
                adj_list=batch.adj_list.to(device, non_blocking=True),
                edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
                masked_elements=batch.masked_elements.to(device, non_blocking=True),
            )
            kbT = openmm_potential_energy_torch.kbT
            assert y_coords.shape == x_coords.shape
            e_kin = compute_kinetic_energy(
                y_velocs, masses, random_velocs=random_velocs, kbT=kbT
            ) - compute_kinetic_energy(x_velocs, masses, random_velocs=random_velocs, kbT=kbT)
            e_pot = (
                openmm_potential_energy_torch(y_coords) - openmm_potential_energy_torch(x_coords)
            ) / kbT
            e_pot = e_pot.view(-1)
            assert e_kin.shape == e_pot.shape
            energy = e_pot + e_kin

            p_yx = model.log_likelihood(
                atom_types=batch.atom_types.to(device, non_blocking=True),
                y_coords=x_coords,
                y_velocs=-x_velocs if not random_velocs else x_velocs,
                x_coords=y_coords,
                x_velocs=-y_velocs if not random_velocs else y_velocs,
                adj_list=batch.adj_list.to(device, non_blocking=True),
                edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
                masked_elements=batch.masked_elements.to(device, non_blocking=True),
            )

            assert energy.shape == p_xy.shape
            assert p_yx.shape == p_xy.shape
            exp = energy + p_xy - p_yx
            p_acc = torch.min(torch.tensor(1), torch.exp(-exp))

            # Likelihood training samples
            p_xy_training = model.log_likelihood(
                atom_types=batch.atom_types.to(device, non_blocking=True),
                x_coords=x_coords,
                x_velocs=x_velocs,
                y_coords=y_coord_targets,
                y_velocs=y_veloc_targets,
                adj_list=batch.adj_list.to(device, non_blocking=True),
                edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
                masked_elements=batch.masked_elements.to(device, non_blocking=True),
            )

            p_yx_training = model.log_likelihood(
                atom_types=batch.atom_types.to(device, non_blocking=True),
                x_coords=y_coord_targets,
                x_velocs=-y_veloc_targets if not random_velocs else y_veloc_targets,
                y_coords=x_coords,
                y_velocs=-x_velocs if not random_velocs else x_velocs,
                adj_list=batch.adj_list.to(device, non_blocking=True),
                edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
                masked_elements=batch.masked_elements.to(device, non_blocking=True),
            )

            acceptance.append(p_acc.cpu().detach().numpy())
            p_xys.append(p_xy.cpu().detach().numpy())
            p_yxs.append(p_yx.cpu().detach().numpy())
            p_xys_training.append(p_xy_training.cpu().detach().numpy())
            p_yxs_training.append(p_yx_training.cpu().detach().numpy())
            energies_pot.append(e_pot.cpu().detach().numpy())
            energies_kin.append(e_kin.cpu().detach().numpy())

            y_coords_model.append(y_coords.cpu().numpy())
            y_velocs_model.append(y_velocs.cpu().numpy())
            traj_coords.append(batch.atom_coord_targets.cpu().numpy())
            traj_velocs.append(batch.atom_veloc_targets.cpu().numpy())
            traj_coords_conditioning.append(x_coords.cpu().numpy())
            traj_velocs_conditioning.append(x_velocs.cpu().numpy())

    energies_pot = np.array(energies_pot)
    energies_kin = np.array(energies_kin)
    acceptance = np.array(acceptance)
    ll_reverse = np.array(p_yxs)
    ll_forward = np.array(p_xys)
    ll_reverse_training = np.array(p_yxs_training)
    ll_forward_training = np.array(p_xys_training)
    y_coords_model_arr = np.array(y_coords_model).squeeze(1)
    y_velocs_model_arr = np.array(y_velocs_model).squeeze(1)
    traj_coords_arr = np.array(traj_coords).squeeze(1)
    traj_velocs_arr = np.array(traj_velocs).squeeze(1)
    traj_coords_conditioning_arr = np.array(traj_coords_conditioning).squeeze(1)
    traj_velocs_conditioning_arr = np.array(traj_velocs_conditioning).squeeze(1)
    return (
        y_coords_model_arr,
        y_velocs_model_arr,
        traj_coords_arr,
        traj_velocs_arr,
        traj_coords_conditioning_arr,
        traj_velocs_conditioning_arr,
        ll_reverse,
        ll_forward,
        ll_reverse_training,
        ll_forward_training,
        acceptance,
    )


def sample_on_single_conditional(batch, model, num_samples, sim, step_width, random_velocs, device):
    """Sample conditionally on a single conditioning state with the model and openMM

    Args:
        batch: Initial state of the simulation with all batch information.
        model: Model to run things on.
        num_samples: number of sampling steps.
        sim: openMM simulation environment of the system.
        step_width: Number of MD steps in the openMM simulation.
        random_velocs: Whether to treat velocities of the conditioning state as isotropic Gaussians
            for the model and resample the velocities for every openMM step.
        device: device
    """
    positions = []
    velocities = []
    y_coords_model = []
    y_velocs_model = []
    for _ in tqdm(range(num_samples)):
        sim.context.setPositions(batch.atom_coords.numpy().squeeze(0))
        if random_velocs:
            sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature())
            state = sim.context.getState(getPositions=True, getVelocities=True)
            # x_velocs = state.getVelocities(asNumpy=True)._value
            x_velocs = torch.randn_like(batch.atom_velocs).numpy()
        else:
            x_velocs = batch.atom_velocs.numpy()
            sim.context.setVelocities(x_velocs.squeeze(0))

        y_coords, y_velocs = model.conditional_sample(
            atom_types=batch.atom_types.to(device, non_blocking=True),
            x_coords=batch.atom_coords.to(device, non_blocking=True),
            x_velocs=torch.from_numpy(x_velocs).to(device).float(),
            adj_list=batch.adj_list.to(device, non_blocking=True),
            edge_batch_idx=batch.edge_batch_idx.to(device, non_blocking=True),
            masked_elements=batch.masked_elements.to(device, non_blocking=True),
            num_samples=1,
        )

        sim.step(step_width)
        state = sim.context.getState(getPositions=True, getVelocities=True)
        positions.append(state.getPositions(asNumpy=True)._value)
        velocities.append(state.getVelocities(asNumpy=True)._value)
        y_coords_model.append(y_coords.detach().cpu().numpy())
        y_velocs_model.append(y_velocs.detach().cpu().numpy())
    y_coords_model_conditional = np.array(y_coords_model).squeeze(1).squeeze(1)
    y_velocs_model_conditional = np.array(y_velocs_model).squeeze(1).squeeze(1)
    traj_coords_conditional = np.array(positions)
    traj_velocs_conditional = np.array(velocities)
    traj_coords_conditioning_conditional = np.array(batch.atom_coords.numpy())
    # traj_velocs_conditioning_conditional = np.array(batch.atom_velocs.numpy())

    return (
        y_coords_model_conditional,
        y_velocs_model_conditional,
        traj_coords_conditional,
        traj_velocs_conditional,
        traj_coords_conditioning_conditional,
    )


def compute_kinetic_energy(
    velocs: torch.Tensor,
    masses: torch.Tensor,
    random_velocs: bool = False,
    kbT: Optional[float] = None,
):
    """Compute the kinetic energy of the system.

    Args:
        velocs: Velocities of the system. Shape: [batch_size, num_atoms, 3]
        masses: Masses of the system. Shape: [batch_size, num_atoms]
        random_velocs: Whether the velocities are isotropic Gaussians or not.

    Returns:
        kinetic energy of the system. Shape: [batch_size]
    """
    if random_velocs:
        return 0.5 * ((velocs**2.0).sum(-1)).sum(-1)
    else:
        assert kbT, "Requires kbT to compute energy"
        return 0.5 * (masses * (velocs**2.0).sum(-1)).sum(-1) / kbT


def openmm_step(
    sim: openmm.app.Simulation,
    coords: torch.Tensor,
    velocs: Optional[torch.Tensor] = None,
    num_steps: int = 1,
    integrator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sim.context.setPositions(coords.cpu().numpy().squeeze(0))

    if velocs is not None:
        sim.context.setVelocities(velocs.cpu().numpy().squeeze(0))
    elif integrator is not None:
        sim.context.setVelocitiesToTemperature(integrator.getTemperature())
    else:
        raise ValueError("either `velocs` or `integrator` needs to be specified")

    sim.step(num_steps)
    state = sim.context.getState(getPositions=True, getVelocities=True)
    coords_new = (
        torch.from_numpy(state.getPositions(asNumpy=True)._value).reshape(coords.shape).to(coords)
    )
    velocs_new = (
        torch.from_numpy(state.getVelocities(asNumpy=True)._value).reshape(coords.shape).to(coords)
    )

    return coords_new, velocs_new


# TODO: Implement sampling techniques as classes instead, e.g. `MetropolisHastings`, etc.
def sample_with_model(
    batch: DenseMolDynBatch,
    model: ConditionalFlowDensityModel,
    device: torch.device,
    openmm_potential_energy_torch: OpenmmPotentialEnergyTorch,
    masses: torch.Tensor,
    num_samples: int,
    accept: bool = False,
    random_velocs: bool = False,
    resample_velocs: bool = False,
    initialize_randomly: bool = False,
    num_openmm_steps: int = 0,
    sim: Optional[openmm.app.Simulation] = None,
    openmm_on_proposal: bool = False,
    openmm_on_current: bool = False,
    num_proposal_steps: int = 1,
    adaptive_parallelism: bool = False,
    acceptance_rate_smoothing_factor: float = 0.01,
    rotate: bool = False,
    reference_signs: Optional[torch.tensor] = None,
    chirality_centers: Optional[torch.tensor] = None,
    disable_tqdm: Optional[bool] = False,
):
    """Run a simulation using the model.

    Args:
        batch: Initial state of the simulation with all batch information.
        model: Model to run things on.
        device: device
        openmm_potential_energy_torch: Energy function nof the system.
        masses: Masses of the system
        num_samples: number of sampling steps. This is also equal to the number of energy evaluations.
        accept: Whether to sample from model using Metropolis-Hasting correction
        random_velocs: Whether to treat velocities of the conditioning state as isotropic Gaussians.
        resample_velocs: Whether to resample velocities for conditioning state at every MCMC iteration
        num_openmm_steps: Number of OpenMM steps to take per sample iteration.
            Has no effect if both openmm_on_proposal and openmm_on_current are false.
        sim: openMM simulation environment of the system.
        openmm_on_proposal: Whether to perform OpenMM steps on the proposed state in every MCMC iteration.
        openmm_on_current: Whether to perform OpenMM steps on the accepted state in every MCMC iteration.
        num_proposal_steps: Number of parallel proposal steps to take per sample iteration.
            All states until the first accepted are added the Markov chain.
        adaptive_parallelism: If `True`, `num_proposal_steps` is overridden in such a way to achieve an
            acceptance within each iteration with a desired probability.
        acceptance_rate_smoothing_factor: the smoothing factor used to compute the moving average of
            the acceptance rate, which is in turn used to determine `num_proposal_steps` if `adaptive_parallelism`
            is active.
        rotate: If `True`, the current state will be rotated at every MCMC iteration.
    """
    assert batch.atom_coords.size(0) == 1, "only batch-size of 1 is supported"

    acceptance_indicator = []
    acceptance = []
    p_xys = []
    p_yxs = []
    exponent = []
    energies_pot = []
    energies_kin = []
    energies_pot_delta = []
    energies_kin_delta = []

    x_coords = batch.atom_coords.to(device, non_blocking=True).contiguous()
    if random_velocs:
        x_velocs = torch.randn_like(x_coords)
    else:
        x_velocs = batch.atom_velocs.to(device, non_blocking=True).contiguous()

    edge_batch_idx = batch.edge_batch_idx.to(device, non_blocking=True)
    masked_elements = batch.masked_elements.to(device, non_blocking=True)
    adj_list = batch.adj_list.to(device, non_blocking=True)
    atom_types = batch.atom_types.to(device, non_blocking=True)

    if initialize_randomly:
        # Start from sample rather than data.
        print("Initializaing chain at a random point rather than a data sample.")
        x_coords, x_velocs = model.conditional_sample(
            atom_types=atom_types,
            x_coords=torch.randn_like(x_coords),
            x_velocs=torch.randn_like(x_velocs),
            adj_list=adj_list,
            edge_batch_idx=edge_batch_idx,
            masked_elements=masked_elements,  # type: ignore
            num_samples=1,
        )
        x_coords = x_coords.squeeze(0)
        x_velocs = x_velocs.squeeze(0)

    kbT = openmm_potential_energy_torch.kbT
    velocs_std = (kbT / masses.unsqueeze(0).unsqueeze(-1)).sqrt()

    if openmm_on_current and num_openmm_steps > 0 and sim is not None:
        if random_velocs:
            x_coords, _ = openmm_step(
                sim, x_coords, x_velocs * velocs_std, num_steps=num_openmm_steps
            )
        else:
            x_coords, x_velocs = openmm_step(sim, x_coords, x_velocs, num_steps=num_openmm_steps)

    sampled_coords = [x_coords.detach().cpu().numpy()]
    sampled_velocs = [x_velocs.detach().cpu().numpy()]
    accepted = 0
    if accept:
        print("Sample with the model using Metropolis Hastings")
    else:
        print("Sample with the model by accepting every setp")

    # Small initial acceptance probability so we start out by proposing as many as possible.
    current_acceptance_probability = 1e-3
    # If we're doing adaptive parallelism, the provided `num_proposal_steps` is taken to be the maximum.
    max_num_proposal_steps = num_proposal_steps
    num_proposal_steps = (
        num_proposal_steps
        if not adaptive_parallelism
        else compute_num_proposal_steps(
            current_acceptance_probability, max_num_proposal_steps=max_num_proposal_steps
        )
    )

    with torch.no_grad():
        i = 0
        pbar = tqdm(range(num_samples), disable=disable_tqdm)
        while i < num_samples:
            if random_velocs and resample_velocs:
                # Resample velocities and recompute kinetic energies.
                x_velocs = torch.randn_like(x_velocs)

            if openmm_on_current and num_openmm_steps > 0 and sim is not None:
                if random_velocs:
                    x_coords, _ = openmm_step(
                        sim, x_coords, x_velocs * velocs_std, num_steps=num_openmm_steps
                    )
                else:
                    x_coords, x_velocs = openmm_step(
                        sim, x_coords, x_velocs, num_steps=num_openmm_steps
                    )

            if rotate:
                Q = random_rotation_matrix(device=device, dtype=x_coords.dtype)
                x_coords = (Q @ x_coords.T).T
                x_velocs = (Q @ x_velocs.T).T

            y_coords, y_velocs, p_xy = model.conditional_sample_with_logp(
                atom_types=atom_types,
                x_coords=x_coords,
                x_velocs=x_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements,
                num_samples=num_proposal_steps,
            )
            y_coords = y_coords.squeeze(1)
            y_velocs = y_velocs.squeeze(1)
            x_coords = x_coords.repeat(num_proposal_steps, 1, 1)
            x_velocs = x_velocs.repeat(num_proposal_steps, 1, 1)

            if openmm_on_proposal and sim is not None and num_openmm_steps > 0:
                y_coords, _ = openmm_step(
                    sim, y_coords, y_velocs * velocs_std, num_steps=num_openmm_steps
                )

            e_pot_x = (openmm_potential_energy_torch(x_coords) / kbT).squeeze(-1)
            e_kin_x = compute_kinetic_energy(x_velocs, masses, random_velocs=random_velocs, kbT=kbT)

            assert y_coords.shape == x_coords.shape
            e_kin_y = compute_kinetic_energy(y_velocs, masses, random_velocs=random_velocs, kbT=kbT)
            e_kin = e_kin_y - e_kin_x

            e_pot_y = (openmm_potential_energy_torch(y_coords) / kbT).squeeze(-1)

            # check chirality change
            if chirality_centers is not None and reference_signs is not None:
                chirality_change = check_symmetry_change(
                    y_coords, chirality_centers, reference_signs
                )
                e_pot_y[chirality_change] += 2000

            e_pot = e_pot_y - e_pot_x
            assert e_kin.shape == e_pot.shape
            energy = e_pot + e_kin

            p_yx = model.log_likelihood(
                atom_types=atom_types.repeat(num_proposal_steps, 1),
                y_coords=x_coords,
                y_velocs=-x_velocs if not random_velocs else x_velocs,
                x_coords=y_coords,
                x_velocs=-y_velocs if not random_velocs else y_velocs,
                adj_list=adj_list,
                edge_batch_idx=edge_batch_idx,
                masked_elements=masked_elements.repeat(num_proposal_steps, 1),  # type: ignore
            )

            p_xy = p_xy.reshape(p_yx.shape)

            assert energy.shape == p_xy.shape
            assert p_yx.shape == p_xy.shape
            exp = energy + p_xy - p_yx

            p_acc = torch.min(torch.tensor(1, device=device), torch.exp(-exp))

            if accept:
                accepted_samples = torch.rand(num_proposal_steps, device=device).to(p_acc) < p_acc
                acc_idx = accepted_samples.nonzero(as_tuple=True)[0]
                did_not_accept = len(acc_idx) == 0
                if did_not_accept:
                    first_acc_idx = num_proposal_steps - 1
                else:
                    first_acc_idx = acc_idx[0].cpu().numpy()  # to make the type consistent
                    x_coords[first_acc_idx] = y_coords[first_acc_idx]
                    x_velocs[first_acc_idx] = y_velocs[first_acc_idx]
                    accepted += 1

                # Don't want to keep more samples than `num_samples` around.
                first_acc_idx = min(first_acc_idx, num_samples - i)

                acceptance_indicator.append(accepted_samples[: first_acc_idx + 1].cpu().numpy())

                # Re-compute number of proposals per step.
                current_acceptance_probability = (
                    acceptance_rate_smoothing_factor * (1 - did_not_accept)
                    + (1 - acceptance_rate_smoothing_factor) ** first_acc_idx
                    * current_acceptance_probability
                )
                num_proposal_steps = (
                    num_proposal_steps
                    if not adaptive_parallelism
                    else compute_num_proposal_steps(
                        current_acceptance_probability,
                        max_num_proposal_steps=max_num_proposal_steps,
                    )
                )
            elif (accept is False) & (num_proposal_steps == 1):
                e_kin_x = e_kin_y
                e_pot_x = e_pot_y
                x_coords = y_coords
                x_velocs = y_velocs
                accepted += 1
                first_acc_idx = 0
                acceptance_indicator.append(np.array([True]))
            else:
                raise ValueError("Number of proposals has to be one if everything is accepted!")

            sampled_coords.append(x_coords[: first_acc_idx + 1].detach().cpu().numpy())
            sampled_velocs.append(x_velocs[: first_acc_idx + 1].detach().cpu().numpy())

            x_coords = x_coords[first_acc_idx].unsqueeze(0)
            x_velocs = x_velocs[first_acc_idx].unsqueeze(0)
            pbar.set_description(
                f"Sampling (n = {accepted}, num_proposals_per_step = {num_proposal_steps}, α_mean = {current_acceptance_probability:.3f}, α = {p_acc.mean().detach().cpu().numpy():.5f})"
            )
            i_delta = first_acc_idx + 1
            pbar.update(i_delta)
            i += i_delta

            acceptance.append(p_acc.cpu().detach().numpy()[: first_acc_idx + 1])
            p_xys.append(p_xy.cpu().detach().numpy()[: first_acc_idx + 1])
            p_yxs.append(p_yx.cpu().detach().numpy()[: first_acc_idx + 1])
            exponent.append(exp.cpu().detach().numpy()[: first_acc_idx + 1])
            energies_pot.append(e_pot_y.cpu().detach().numpy()[: first_acc_idx + 1])
            energies_kin.append(e_kin_y.cpu().detach().numpy()[: first_acc_idx + 1])
            energies_pot_delta.append(e_pot.cpu().detach().numpy()[: first_acc_idx + 1])
            energies_kin_delta.append(e_kin.cpu().detach().numpy()[: first_acc_idx + 1])

    sampled_coords = np.concatenate(sampled_coords, axis=0)
    sampled_velocs = np.concatenate(sampled_velocs, axis=0)

    stats = ChainStats(
        acceptance_indicator=np.concatenate(acceptance_indicator, axis=0),
        acceptance=np.concatenate(acceptance, axis=0),
        p_xy=np.concatenate(p_xys, axis=0),
        p_yx=np.concatenate(p_yxs, axis=0),
        exponent=np.concatenate(exponent, axis=0),
        energies_pot=np.concatenate(energies_pot, axis=0),
        energies_kin=np.concatenate(energies_kin, axis=0),
        energies_pot_delta=np.concatenate(energies_pot_delta, axis=0),
        energies_kin_delta=np.concatenate(energies_kin_delta, axis=0),
    )

    return sampled_coords, sampled_velocs, accepted, stats


# %%%% Internal coordinates %%%%
def compute_internal_coordinates(state0pdbpath, adj_list, coords):
    traj = md.load(state0pdbpath)
    traj.xyz = coords
    bonds = md.compute_distances(traj, adj_list)
    # `mdtraj` also returns the indices of the atoms involved in computing each
    # torsion angle. We only care about the acual value, so we just drop the indices,
    # i.e. the first return-value.
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    torsions = phi, psi
    return bonds, torsions


# %%%% Plotting %%%%


def plot_marginal_distribution(
    x_model,
    v_model,
    x_traj,
    v_traj,
    color_model,
    color_traj,
    output_dir,
    protein,
    name,
    step_width,
    atom_types,
    x_bg=None,
    v_bg=None,
):
    fig, axs = plt.subplots(5, 5, figsize=(16, 16), sharey=True, sharex=True)
    for i in range(22):
        if x_bg is not None:
            axs[i // 5, i % 5].hist(
                x_bg[:, i, 0], bins=100, color="blue", alpha=0.5, density=True, label="Boltzmann"
            )
        axs[i // 5, i % 5].hist(
            x_traj[:, i, 0], bins=100, color=color_traj, alpha=0.5, density=True, label="openMM"
        )
        axs[i // 5, i % 5].hist(
            x_model[:, i, 0],
            bins=100,
            color=color_model,
            alpha=0.5,
            density=True,
            label="model",
        )
        axs[i // 5, i % 5].set_title(ATOM_NAMES[atom_types[i]], y=0.7)
    axs[i // 5, i % 5].legend()
    fig.suptitle(f"{name.replace('_', ' ')}: Marginal distribution for Delta x")
    fig.supxlabel("Distance in nm")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_marginal_distribution_delta_x_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()

    # less atoms
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharey=True, sharex=True)
    for i in range(2):
        for j in range(2):
            if x_bg is not None:
                axs[i, j].hist(
                    x_bg[:, i * 5 + j, 0],
                    bins=100,
                    color="blue",
                    alpha=0.5,
                    density=True,
                    label="Boltzmann",
                )
            axs[i, j].hist(
                x_traj[:, i * 5 + j, 0],
                bins=100,
                color=color_traj,
                alpha=0.5,
                density=True,
                label="openMM",
            )
            axs[i, j].hist(
                x_model[:, i * 5 + j, 0],
                bins=100,
                color=color_model,
                alpha=0.5,
                density=True,
                label="model",
            )
            axs[i, j].set_title(ATOM_NAMES[atom_types[i * 5 + j]], y=0.7)
    axs[i, j].legend()
    fig.suptitle(f"{name.replace('_', ' ')}: Marginal distribution for Delta x")
    fig.supxlabel("Distance in nm")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_marginal_distribution_delta_x_smaller_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(5, 5, figsize=(16, 16), sharey=True, sharex=True)
    for i in range(22):
        if v_bg is not None:
            axs[i // 5, i % 5].hist(
                v_bg[:, i, 0], bins=100, color="blue", alpha=0.5, density=True, label="Boltzmann"
            )
        axs[i // 5, i % 5].hist(
            v_traj[:, i, 0],
            bins=100,
            color=color_traj,
            alpha=0.5,
            density=True,
            label="openMM",
        )
        axs[i // 5, i % 5].hist(
            v_model[:, i, 0],
            bins=100,
            color=color_model,
            alpha=0.5,
            density=True,
            label="model",
        )
        axs[i // 5, i % 5].set_title(ATOM_NAMES[atom_types[i]], y=0.7)
    axs[i // 5, i % 5].legend()
    fig.suptitle(f"{name.replace('_', ' ')}: Marginal velocity distribution")
    fig.supxlabel("Velocity in nm/ps")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_marginal_distribution_v_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_bonds(
    bonds_model,
    bonds_traj,
    color_model,
    color_traj,
    output_dir,
    protein,
    name,
    step_width,
    atom_types,
    adj_list,
    bonds_bg=None,
    bonds_intial=None,
):

    fig, axs = plt.subplots(5, 5, figsize=(16, 16), sharey=True, sharex=True)
    for i in range(25):
        if bonds_bg is not None:
            axs[i // 5, i % 5].hist(
                bonds_bg[:, i], bins=100, color="blue", alpha=0.5, density=True, label="Boltzmann"
            )
        axs[i // 5, i % 5].hist(
            bonds_traj[:, i], bins=100, color=color_traj, alpha=0.5, density=True, label="openMM"
        )
        axs[i // 5, i % 5].hist(
            bonds_model[:, i],
            bins=100,
            color=color_model,
            alpha=0.5,
            density=True,
            label="model",
        )
        if bonds_intial is not None:
            axs[i // 5, i % 5].axvline(
                bonds_intial[0, i],
                0,
                0.9,
                color="black",
                linewidth=1,  # label="Conditioning bondlength",
            )
        axs[i // 5, i % 5].set_title(
            f"{ATOM_NAMES[atom_types[adj_list[i, 0]]]}-{ATOM_NAMES[atom_types[adj_list[i, 1]]]}",
            y=0.7,
        )
    axs[i // 5, i % 5].legend()
    fig.suptitle(f"{name.replace('_', ' ')}: Bondlength distributions")
    fig.supxlabel("Bondlength in nm")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_bondlength_distribution_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()

    # less bonds
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharey=True, sharex=True)
    for i in range(2):
        for j in range(2):
            idx = 5 * i + 2 * j
            if bonds_bg is not None:
                axs[i, j].hist(
                    bonds_bg[:, idx],
                    bins=100,
                    color="blue",
                    alpha=0.5,
                    density=True,
                    label="Boltzmann",
                )
            axs[i, j].hist(
                bonds_traj[:, idx],
                bins=100,
                color=color_traj,
                alpha=0.5,
                density=True,
                label="openMM",
            )
            axs[i, j].hist(
                bonds_model[:, idx],
                bins=100,
                color=color_model,
                alpha=0.5,
                density=True,
                label="model",
            )
            if bonds_intial is not None:
                axs[i, j].axvline(
                    bonds_intial[0, idx],
                    0,
                    0.9,
                    color="black",
                    linewidth=1,  # label="Conditioning bondlength",
                )
            axs[i, j].set_title(
                f"{ATOM_NAMES[atom_types[adj_list[idx, 0]]]}-{ATOM_NAMES[atom_types[adj_list[idx, 1]]]}",
                y=0.7,
            )
    axs[i, j].legend()
    fig.suptitle(f"{name.replace('_', ' ')}: Bondlength distributions")
    fig.supxlabel("Bondlength in nm")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_bondlength_distribution_smaller_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_ramachandran(
    torsions, name, title, output_dir, step_width, protein, show_initial_state=False
):
    if torsions[0].shape[-1] == 1:
        plt.figure(figsize=(10, 10))
        plt.title(f"{name.replace('_', ' ')}: Ramachandran plot - {title}")
        plt.hist2d(
            torsions[0].flatten(), torsions[1].flatten(), bins=100, norm=mpl.colors.LogNorm()
        )
        if show_initial_state:
            plt.scatter(torsions[0][0], torsions[1][0], marker="x", color="red", s=50)
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
    elif torsions[0].shape[-1] == 3:
        fig, axs = plt.subplots(1, 3, figsize=(35, 10))
        for i in range(3):

            axs[i].hist2d(torsions[0][:, i], torsions[1][:, i], bins=100, norm=mpl.colors.LogNorm())
            axs[i].scatter(torsions[0][0, i], torsions[1][0, i], marker="x", color="red", s=50)

            axs[i].set_xlim(-np.pi, np.pi)
            axs[i].set_ylim(-np.pi, np.pi)
            axs[i].set_xlabel("Phi")
            axs[i].set_ylabel("Psi")
        fig.suptitle(f"{name.replace('_', ' ')}: Ramachandran plot - {title}")
    else:
        raise NotImplementedError(
            "Ramachandran plot only implemented for one or three angle pairs."
        )
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_ramachandran_{title}_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_transitions(torsions, name, title, output_dir, step_width, protein):
    plt.figure(figsize=(16, 9))
    plt.title(f"{name.replace('_', ' ')}: Psi transitions - {title}")
    plt.plot((torsions[1] - 0.5 + 3) % 6 - 3, linewidth=5)
    plt.xlabel("Sample")
    plt.ylabel("Psi")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_psi_transitions_{title}_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_energy(energies_model, energies_traj, name, title, output_dir, step_width, protein):
    plt.figure(figsize=(16, 9))
    plt.hist(
        energies_model,
        bins=100,
        color="orange",
        alpha=0.5,
        density=True,
        label="model",
        range=(energies_traj.min(), 200),
    )
    plt.hist(
        energies_traj,
        bins=100,
        color="green",
        alpha=0.5,
        density=True,
        label="openMM",
    )
    plt.xlabel("Energy in kJ/mol")
    plt.legend()
    plt.title(f"{name.replace('_', ' ')}: {title} energy distribution")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_{title}_energy_distribution_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
