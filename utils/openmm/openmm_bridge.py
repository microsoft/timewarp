"""Energy and Force computation in OpenMM
"""

from typing import Optional, Tuple
from functools import cached_property
import torch.nn as nn
from torch import Tensor
import numpy as np

# TODO: Make PR to `bgflow` instead of accessing private methods.
from bgflow.distribution.energy.base import Energy, _evaluate_bridge_energy

import multiprocessing as mp
import torch

from bgflow.utils.types import assert_numpy
from bgflow.distribution.energy.openmm import SingleContext, MultiContext
from openmm import unit


def device_to_platform_and_properties(
    device: torch.device, num_threads: int = 1
) -> Tuple[str, dict]:
    """Convert a torch device to a platform and properties dictionary for OpenMM.

    Args:
        device (torch.device): torch device
        num_threads (int, optional): number of threads to use. Defaults to 1.

    Returns:
        Tuple[str, dict]: platform and properties dictionary.
    """
    if device.type == "cpu":
        platform = "CPU"
        properties = dict(Threads=str(num_threads))
    elif device.type == "cuda":
        platform = "CUDA"
        # NOTE : OpenMM expects the value of the properties to be strings.
        properties = dict(CudaDeviceIndex=str(device.index))
    else:
        raise ValueError(f"Unsupported device {device}")

    return platform, properties


class OpenMMEnergy(Energy):
    def __init__(self, bridge, two_event_dims=False):
        event_shape = (bridge.n_atoms, 3) if two_event_dims else (bridge.n_atoms * 3,)
        super().__init__(event_shape)
        self._bridge = bridge

    @property
    def bridge(self):
        return self._bridge

    def _energy(self, batch, no_grads=False):
        return _evaluate_bridge_energy(batch, self._bridge)

    def force(self, batch, temperature=None):
        return self._bridge.evaluate(batch)[1]


class OpenMMBridge:
    """Bridge object to evaluate energies in OpenMM.

    Input positions are in nm, returned energies are dimensionless (units of kT), returned forces are in kT/nm.

    Parameters
    ----------
    openmm_system : simtk.openmm.System
        The OpenMM system object that contains all force objects.
    openmm_integrator : simtk.openmm.Integrator
        A thermostated OpenMM integrator (has to have a method `getTemperature()`.
    platform_name : str, optional
        An OpenMM platform name ('CPU', 'CUDA', 'Reference', or 'OpenCL')
    err_handling : str, optional
        How to handle infinite energies (one of {"warning", "ignore", "exception"}).
    n_workers : int, optional
        The number of processes used to compute energies in batches. This should not exceed the
        most-used batch size or the number of accessible CPU cores. The default is the number
        of logical cpu cores. If a GPU platform is used (CUDA or OpenCL), n_workers is always set to 1
        to sidestep multiprocessing (due to performance issues when using multiprocessing with GPUs).
    n_simulation_steps : int, optional
        If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.
    platform_properties : dict, optional
        Properties defining the OpenMM context used for evaluation. See
        http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html
        for more on how to achieve the desired behavior, e.g. CUDA usage instead of CPU.

    Notes
    ------
    Unlike in `bgflow.distribution.base._Bridge`, we do NOT cache the results
    of the computations.
    Such caching can easily lead to memory issues when keeping multiple instances
    of `OpenMMBridge` around and computing gradients through the energy, since the
    gradient tape will hold a reference to the resulting computation which prohibits
    memory de-allocation.
    """

    _FLOATING_TYPE = np.float64
    _SPATIAL_DIM = 3

    _FLOATING_TYPE = np.float64
    _SPATIAL_DIM = 3

    def __init__(
        self,
        openmm_system,
        openmm_integrator,
        platform_name="CPU",
        err_handling="warning",
        n_workers=mp.cpu_count(),
        n_simulation_steps=0,
        platform_properties=None,
    ):
        from simtk import unit

        _platform_properties = (
            {"Threads": str(max(1, mp.cpu_count() // n_workers))} if platform_name == "CPU" else {}
        )

        if platform_properties is not None:
            _platform_properties.update(platform_properties)

        # Compute all energies in child processes due to a bug in the OpenMM's PME code.
        # This might be problematic if an energy has already been computed in the same program on the parent thread,
        # see https://github.com/openmm/openmm/issues/2602
        self._openmm_system = openmm_system
        self._openmm_integrator = openmm_integrator
        if platform_name in ["CUDA", "OpenCL"] or n_workers == 1:
            self.context_wrapper = SingleContext(
                1,
                openmm_system,
                openmm_integrator,
                platform_name,
                platform_properties=_platform_properties,
            )
        else:
            self.context_wrapper = MultiContext(
                n_workers,
                openmm_system,
                openmm_integrator,
                platform_name,
                platform_properties=_platform_properties,
            )
        self._err_handling = err_handling
        self._n_simulation_steps = n_simulation_steps
        self._unit_reciprocal = 1 / (
            openmm_integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
        ).value_in_unit(unit.kilojoule_per_mole)
        super().__init__()

    @property
    def n_atoms(self):
        return self._openmm_system.getNumParticles()

    @property
    def integrator(self):
        return self._openmm_integrator

    @property
    def n_simulation_steps(self):
        return self._n_simulation_steps

    def _reduce_units(self, x):
        if x is None:
            return None
        return x * self._unit_reciprocal

    def evaluate(
        self,
        batch,
        evaluate_force=True,
        evaluate_energy=True,
        evaluate_positions=False,
        evaluate_path_probability_ratio=False,
    ):
        """
        Compute energies/forces for a batch of positions.
        Parameters:
        -----------
        batch : np.ndarray or torch.Tensor
            A batch of particle positions that has shape (batch_size, num_particles * 3).
        evaluate_force : bool, optional
            Whether to compute forces.
        evaluate_energy : bool, optional
            Whether to compute energies.
        evaluate_positions : bool, optional
            Whether to return positions.
        evaluate_path_probability_ratio : bool, optional
            Whether to compute the log path probability ratio.
            Makes only sense for PathProbabilityIntegrator instances.
        Returns
        -------
        energies : torch.Tensor or None
            The energies in units of kilojoule/mole; its shape  is (len(batch), )
        forces : torch.Tensor or None
            The forces in units of kilojoule/mole/nm; its shape is (len(batch), num_particles*3)
        new_positions : torch.Tensor or None
            The positions in units of nm; its shape is (len(batch), num_particles*3)
        log_path_probability_ratio : torch.Tensor or None
            The logarithmic path probability ratios; its shape  is (len(batch), )
        """

        # make a list of positions
        batch_array = assert_numpy(batch, arr_type=self._FLOATING_TYPE)

        # assert correct number of positions
        assert batch_array.shape[1] == self._openmm_system.getNumParticles() * self._SPATIAL_DIM

        # reshape to (B, N, D)
        batch_array = batch_array.reshape(batch.shape[0], -1, self._SPATIAL_DIM)
        energies, forces, new_positions, log_path_probability_ratio = self.context_wrapper.evaluate(
            batch_array,
            evaluate_energy=evaluate_energy,
            evaluate_force=evaluate_force,
            evaluate_positions=evaluate_positions,
            evaluate_path_probability_ratio=evaluate_path_probability_ratio,
            err_handling=self._err_handling,
            n_simulation_steps=self._n_simulation_steps,
        )

        # divide by kT
        energies = self._reduce_units(energies)
        forces = self._reduce_units(forces)

        # to PyTorch tensors
        energies = torch.tensor(energies).to(batch).reshape(-1, 1) if evaluate_energy else None
        forces = (
            torch.tensor(forces)
            .to(batch)
            .reshape(batch.shape[0], self._openmm_system.getNumParticles() * self._SPATIAL_DIM)
            if evaluate_force
            else None
        )
        new_positions = (
            torch.tensor(new_positions)
            .to(batch)
            .reshape(batch.shape[0], self._openmm_system.getNumParticles() * self._SPATIAL_DIM)
            if evaluate_positions
            else None
        )
        log_path_probability_ratio = (
            torch.tensor(log_path_probability_ratio).to(batch).reshape(-1, 1)
            if evaluate_path_probability_ratio
            else None
        )

        return energies, forces, new_positions, log_path_probability_ratio


class OpenmmPotentialEnergyTorch(nn.Module):
    """
    A torch module to compute the openmm energy of configurations, which allows backpropagation.
    Using the bgflow library
    https://github.com/noegroup/bgflow/tree/ee620a373ee33c1ef9f1183c6655391aaf4bc88a
    """

    def __init__(
        self,
        system,
        integrator,
        platform_name: str,
        platform_properties: Optional[dict] = None,
        bridge_kwargs: dict = {},
    ):
        super().__init__()

        mm_bridge = OpenMMBridge(
            system,
            integrator,
            platform_name=platform_name,
            platform_properties=platform_properties or {},
            **bridge_kwargs,
        )
        # Compute energies in  kJ/mol
        mm_bridge._unit_reciprocal = 1
        self.num_particles = system.getNumParticles()
        self.openmm_energy = OpenMMEnergy(bridge=mm_bridge)

    def forward(self, coords: Tensor) -> Tensor:
        """
        Compute the potential energy of a given configuration coords.
        """
        assert (
            coords.size(-1) == 3
        ), f"last dimension is expected to be of size 3 but it is {coords.size(-1)}"
        assert (
            coords.size(-2) == self.num_particles
        ), f"size {coords.size()} does not align with expected number of particles {self.num_particles}"
        # Input to `openmm_energy` needs to be [B, V * 3].
        coords = coords.view(-1, self.num_particles * 3)
        energy = self.openmm_energy.energy(coords)
        return energy

    def get_integrator(self):
        return self.openmm_energy._bridge.integrator

    @cached_property
    def kbT(self):
        """
        Get the value of the Boltzmann constant times the temperature.
        This is the denominator in the exponent of the Boltzmann distribution.
        """
        return (self.get_integrator().getTemperature() * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
            unit.kilojoule_per_mole
        )
