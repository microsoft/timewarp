import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.md import (
    get_simulation_environment,
    compute_energy_and_forces,
    compute_energy_and_forces_decomposition,
)
from simulation.md import get_parameters_from_preset
from simulation.md import sample

state0pdbpath = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "implicit-2olx-traj-cpu-state0.pdb"
)
npzpath = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "implicit-2olx-traj-cpu-arrays.npz"
)


def test_get_parameters_from_preset():
    """Test parameter instantiation by name and dict."""
    test_dict = {"par1": 1.0, "par2": 2.0}
    result1 = get_parameters_from_preset(test_dict)
    assert isinstance(result1, dict)

    result2 = get_parameters_from_preset("T1-peptides")
    assert isinstance(result2, dict)
    assert result2["forcefield"] == "amber99-implicit"


def test_openmm_energy_computation_t1peptides():
    """Test energy and force computation against NPZ reference."""
    data = np.load(npzpath)
    ref_energies = data["energies"]
    ref_forces = data["forces"]
    positions = data["positions"]
    velocities = data["velocities"]

    sim = get_simulation_environment(state0pdbpath, "T1-peptides")
    energies, forces = compute_energy_and_forces(sim, positions, velocities)
    np.testing.assert_allclose(ref_energies, energies[:, 0:2], atol=1e-3)
    # Forces can be slightly inaccurate because we store forces in np.float32
    np.testing.assert_allclose(ref_forces, forces, rtol=0.05, atol=1e-2)


def test_openmm_energies_by_force_t1peptides():
    """Test energy and force decomposition self-consistency."""
    data = np.load(npzpath)
    ref_energies = data["energies"]
    ref_forces = data["forces"]
    positions = data["positions"]
    velocities = data["velocities"]

    energies_by_force, forces_by_force = compute_energy_and_forces_decomposition(
        state0pdbpath, "T1-peptides", positions, velocities
    )

    assert len(energies_by_force.keys()) > 1

    # Re-sum the decomposed energies and forces to the total energy and
    # total force, then compare against reference energy and forces, stored
    # in the NPZ file.
    total_energy = sum(energies_by_force.values())
    total_force = sum(forces_by_force.values())

    # Test that the energy decomposition is non-trivial and there are different
    # energies assigned to each force component.  (There may be a few exceptions
    # where the energies are zero, but there should be 3-5 unique energy
    # components.)
    unique_elem_at_0 = len(np.unique([energies[0] for energies in energies_by_force.values()]))
    assert unique_elem_at_0 > 2

    # The computed energies should match the reference energies from the NPZ file
    # exactly.  However, due to platform differences such as use of GPU vs CPU,
    # CUDA version, and summation order, we may observe differences in the values
    # of the order of float32 floating point relative accuracies.  Therefore, we
    # only demand rtol=1.0e-6 for the energies and rtol=0.05 for the forces.
    np.testing.assert_allclose(ref_energies[:, 0], total_energy, rtol=1.0e-6)
    np.testing.assert_allclose(ref_forces, total_force, rtol=0.05, atol=1.0e-3)


def test_openmm_sample_t1peptides():
    """Test sampling via OpenMM integrators."""
    data = np.load(npzpath)
    positions = data["positions"]
    velocities = data["velocities"]

    sim0 = get_simulation_environment(state0pdbpath, "T1-peptides")
    sim1 = get_simulation_environment(state0pdbpath, "T1-peptides")
    sim2 = get_simulation_environment(state0pdbpath, "T1-peptides")
    num_atoms = sim0.system.getNumParticles()

    # Draw two samples with the same seed
    pos, vel = positions[0, :, :], velocities[0, :, :]
    timesteps = [1, 10, 100]
    pos0a, vel0a, for0a = sample(sim0, pos, vel, timesteps, seed=0)
    for arr in [pos0a, vel0a, for0a]:
        assert np.size(arr, 0) == len(timesteps)
        assert np.size(arr, 1) == num_atoms
        assert np.size(arr, 2) == 3

    # FIXME: despite the same random seed being used the resulting
    # data in pos0a and pos0b is different.
    pos0b, vel0b, for0b = sample(sim1, pos, vel, timesteps, seed=0)
    # assert np.allclose(pos0a, pos0b)
    # assert np.allclose(vel0a, vel0b)
    # assert np.allclose(for0a, for0b)

    pos1a, vel1a, for1a = sample(sim2, pos, vel, timesteps, seed=1)
    assert not np.allclose(pos0a, pos1a)
    assert not np.allclose(vel0a, vel1a)
    assert not np.allclose(for0a, for1a)
