from pathlib import Path
import torch
import os
from openmm import unit, Platform
import openmm as mm
import numpy as np
from itertools import islice
import mdtraj
from timewarp.utils.openmm import OpenmmPotentialEnergyTorch
from simulation.md import get_simulation_environment, get_simulation_environment_integrator
from timewarp.datasets import RawMolDynDataset
from timewarp.dataloader import moldyn_dense_collate_fn


def test_opennmm_bridge():
    # compare the energy of openmm with the openmm bridge
    data_dir = (
        Path(os.path.join(os.path.dirname(__file__), "../testdata/output/")).expanduser().resolve()
    )
    protein = "1hgv"
    parameters = "T1-peptides"
    state0pdbpath = os.path.join(data_dir, f"{protein}-traj-state0.pdb")
    simulation = get_simulation_environment(state0pdbpath, parameters)
    integrator = get_simulation_environment_integrator(parameters)

    # load the dataset
    raw_dataset = RawMolDynDataset(data_dir=data_dir, step_width=1000)
    pdb_names = [protein]
    raw_iterator = raw_dataset.make_iterator(pdb_names)
    batches = (moldyn_dense_collate_fn([datapoint]) for datapoint in raw_iterator)
    batches = list(islice(batches, 20))  # Load max 20 timepoints.
    traj_coords = [batch.atom_coord_targets.cpu().numpy() for batch in batches]
    openmm_potential_energy_torch = OpenmmPotentialEnergyTorch(
        simulation.system, integrator, platform_name="CPU"
    )

    # Compute energy and forces with OpenMM
    traj = mdtraj.load(state0pdbpath)
    platform = Platform.getPlatformByName("CPU")
    properties = {"Threads": "1"}
    sim = mm.app.Simulation(
        traj.topology,
        simulation.system,
        integrator,
        platform=platform,
        platformProperties=properties,
    )
    forces = []
    openmm_energies = []
    for positions in traj_coords:
        sim.context.setPositions(positions[0].astype(np.double()))
        state = sim.context.getState(getEnergy=True, getForces=True)
        openmm_energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )

    # Compute energy and and gradient with the OpenMM-bridge
    traj_coords = torch.from_numpy(np.array(traj_coords, dtype=np.double)).squeeze(1)
    with torch.enable_grad():
        traj_coords.requires_grad = True
        openmm_energy_torch = openmm_potential_energy_torch(traj_coords)
        grad = torch.autograd.grad(openmm_energy_torch.sum(), traj_coords)[0]

    # Energies should be equal
    torch.equal(grad, -torch.tensor(forces, dtype=torch.float64))

    # gradient should be equal to the negative force
    torch.equal(
        openmm_energy_torch,
        torch.tensor(openmm_energies, dtype=torch.float64).view(-1, 1),
    )
