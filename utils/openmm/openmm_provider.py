import os
import torch
import mdtraj
import openmm as mm

from typing import Dict, Union, Collection, Optional
from functools import cached_property, lru_cache

from openmm import unit

from simulation.md import (
    get_simulation_environment_integrator,
    get_system,
)
from utilities.common import StrPath

from .openmm_bridge import OpenmmPotentialEnergyTorch, device_to_platform_and_properties


class OpenMMProvider:
    """
    Provides convenient way to interact with OpenMM systems for multiple proteins simultaneously.

    Attributes:
        pdb_dirs: collection of directories where PDB files can be located
        parameters: string specifying which the parameters for the integrator,
            e.g. temperature (see `get_simulation_environment_integrator` for more) [default: "T1B-peptides"]
        device: either a string or `torch.device` indicating which device to put the OpenMM simulation on
        cache_size: maximum number of `OpenmmPotentialEnergyTorch` to cache

    Notes:
        There is currently a bug where even if one specifies `device=torch.device("cpu")` but the rest
        of the program runs on a GPU device, we will still see an unreasonable amount of mem. alloc. on the GPU device.
        At the time of writing (2022-05-30) this issue is still unresolved.
    """

    def __init__(
        self,
        pdb_dirs: Union[StrPath, Collection[StrPath]],
        parameters: str = "T1B-peptides",
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        cache_size: int = 8,
    ):
        self.pdb_dirs: Collection[StrPath] = (
            pdb_dirs
            if isinstance(pdb_dirs, Collection) and not isinstance(pdb_dirs, str)
            else [pdb_dirs]
        )

        self.parameters = parameters
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.cache_size = cache_size
        self._potential_energy_cache: Dict[str, OpenmmPotentialEnergyTorch] = dict()

    def clear_cache(self):
        self._potential_energy_cache.clear()

    def clear_cache_to_size(self, size: Optional[int] = None):
        if size is None:
            size = self.cache_size

        if size <= 0:
            self.clear_cache()
            return self

        while len(self._potential_energy_cache) > size:
            # Treated as FIFO, so we drop the _first_ key.
            k_energy = next(iter(self._potential_energy_cache))
            print(f"OpenMMProvider: dropping {k_energy} from cache")
            del self._potential_energy_cache[k_energy]

        return self

    def get_integrator(self):
        return get_simulation_environment_integrator(self.parameters)

    @cached_property
    def kbT(self):
        return (self.get_integrator().getTemperature() * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
            unit.kilojoule_per_mole
        )

    def get_system(self, protein: str):
        state0pdbpath = None
        for pdb_dir in self.pdb_dirs:
            for (dirpath, _, _) in os.walk(str(pdb_dir)):
                # Faster to just check if the target file exists than iterating through all the files
                # in the directory to find a match.
                state0pdbpath_maybe = os.path.join(dirpath, f"{protein}-traj-state0.pdb")
                if os.path.isfile(state0pdbpath_maybe):
                    state0pdbpath = state0pdbpath_maybe
                    break

        if state0pdbpath is None:
            raise ValueError(
                f"could not find PDB file for {protein} in any of the provided paths {self.pdb_dirs}"
            )

        traj = mdtraj.load(state0pdbpath)
        topology = traj.topology.to_openmm()
        positions = traj.openmm_positions(0)
        openmm_model = mm.app.modeller.Modeller(topology, positions)
        system = get_system(openmm_model, self.parameters)
        return system

    def get_potential_energy_module(self, protein: str) -> OpenmmPotentialEnergyTorch:
        if protein in self._potential_energy_cache:
            return self._potential_energy_cache[protein]

        # No need to print all this cache-related information if it's always going to be empty.
        if self.cache_size > 0:
            print(
                f"OpenMMProvider: current size [ {len(self._potential_energy_cache)} / {self.cache_size} ]"
            )
            print(f"OpenMMProvider: contains proteins {list(self._potential_energy_cache.keys())}")
            print(f"OpenMMProvider: missed cache for protein {protein}; creating energy module...")

        # Make sure there's room for one more in the cache.
        self.clear_cache_to_size(self.cache_size - 1)

        system = self.get_system(protein)
        # NOTE : We need an `integrator` instance for every system.
        integrator = get_simulation_environment_integrator(self.parameters)
        # system = simulation.system

        print(
            f"OpenMMProvider: created energy module for {protein} with n={system.getNumParticles()} atoms"
        )

        platform, platform_properties = device_to_platform_and_properties(
            self.device, num_threads=1
        )
        openmm_module = OpenmmPotentialEnergyTorch(
            system,
            integrator,
            platform_name=platform,
            platform_properties=platform_properties,
            bridge_kwargs=dict(n_workers=1),  # Use single-threaded version if we use CPU.
        )

        if self.cache_size > 0:
            self._potential_energy_cache[protein] = openmm_module

        return openmm_module

    @lru_cache(maxsize=2048)
    def get_masses(self, protein: str):
        system = self.get_system(protein)
        num_atoms = system.getNumParticles()
        masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(num_atoms)]
        return torch.tensor(masses, device=self.device)
