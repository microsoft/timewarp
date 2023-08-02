"""Methods for computing force field energies for a given state.
"""
import functools
from packaging import version
import simtk.unit as u


import simtk.openmm  # noqa: F401
import openmm as mm  # type: ignore [import]

# import openmm.unit as u  # type: ignore [import]
import numpy as np


def get_preset_from_dataset(dataset_name):
    """Look up preset based on dataset name.

    Arguments
    ---------
    dataset_name : str
        One of "T1B-peptides", "T1-peptides", "HP-1400", "HP-4000",
        "alanine-dipeptide", or a valid preset name.

    Returns
    -------
    preset_name : str
        The preset name corresponding to the simulation parameters used to
        create the data set.
    """
    # Map newer dataset names to preset's first
    sim_details = {
        "T1B-peptides": "amber14-implicit",
        "T1-peptides": "amber99-implicit-old",
        "HP-1400": "amber99-implicit-old",
        "HP-4000": "amber99-implicit-old",
        "alanine-dipeptide": "amber99-implicit-old",
    }
    if dataset_name in sim_details:
        preset_name = sim_details[dataset_name]
    else:
        preset_name = dataset_name

    return preset_name


def get_parameters_from_preset(preset_or_dataset_name):
    """Set simulation parameters from known preset.
    As of 12/2022 most current simulations use the T1B-peptides preset,
    which uses the amber14-implicit force-field.
    The following data-set were created using this preset:
        - T1B-peptides
        - All 4AA datasets
        - All AD-3 datasets
        - All 2AA datasets

    Arguments
    ---------
    preset_or_dataset_name : str or dict
        One of the dataset names listed in the `get_preset_from_dataset` method.
        or one of the preset names in ["amber99-implicit", "amber14-implicit",
        "amber14-explicit"].
        If it is a dictionary, we directly use the simulation parameters provided.

    Returns
    -------
    parameters : dict
        Parameter dictionary.
    """
    if isinstance(preset_or_dataset_name, dict):
        return preset_or_dataset_name

    preset_name = get_preset_from_dataset(preset_or_dataset_name)

    # Older datasets (AMBER99), created prior to December 2021
    if preset_name == "amber99-implicit-old":
        parameters = {
            "forcefield": "amber99-implicit",
            "temperature": 310.0 * u.kelvin,
            "friction": 0.3 / u.picosecond,
            "timestep": 0.5 * u.femtosecond,
            "integrator": "LangevinIntegrator",
        }
    elif preset_name in ["amber99-implicit", "amber14-implicit", "amber14-explicit"]:
        parameters = {
            "forcefield": preset_name,
            "temperature": 310.0 * u.kelvin,
            "friction": 0.3 / u.picosecond,
            "timestep": 0.5 * u.femtosecond,
            "waterbox_pad": 1.0 * u.nanometers,
            "integrator": "LangevinMiddleIntegrator",
        }
    else:
        raise ValueError("Invalid preset name '%s'" % preset_name)

    return parameters


def get_simulation_environment_integrator(parameters):
    """Obtain integrator from parameters.

    Arguments
    ---------
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    integrator : openmm.Integrator
    """
    parameters = get_parameters_from_preset(parameters)
    temperature = parameters["temperature"]
    friction = parameters["friction"]
    timestep = parameters["timestep"]
    if parameters["integrator"] == "LangevinIntegrator":
        integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    elif parameters["integrator"] == "LangevinMiddleIntegrator":
        assert version.parse(mm.__version__) >= version.parse("7.5")

        # LangevinMiddleIntegrator is equally efficient but second-order accurate.
        # However, the middle integrator is only available in OpenMM version >= 7.5,
        # and we did generate data (T1-peptides, HP-1400, HP-4000, AD-1) using
        # the original LangevinIntegrator.
        integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)

    return integrator


def get_system(model, parameters):
    """Obtain system to generate e.g. a simulation environment.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    system : openmm.system
        System (topology, forcefield).  This
        is required for a simulation object.
    """
    parameters = get_parameters_from_preset(parameters)

    # TODO: use openmmforcefields package to support GAFF2
    # TODO: support CHARMM36 with implicit water

    # amber99-implicit and amber14-implicit
    if parameters["forcefield"].endswith("-implicit"):
        if parameters["forcefield"] == "amber99-implicit":
            forcefield = mm.app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        elif parameters["forcefield"] == "amber14-implicit":
            # (Onufriev, Bashford, Case, "Exploring Protein Native States and
            # Large-Scale Conformational Changes with a modified Generalized
            # Born Model", PROTEINS 2004) using the GB-OBC I parameters
            # (corresponds to `igb=2` in AMBER)
            assert version.parse(mm.__version__) >= version.parse("7.7")
            forcefield = mm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        else:
            raise ValueError("Invalid forcefield parameter '%s'" % parameters["forcefield"])

        model.addExtraParticles(forcefield)

        # Peter Eastman recommends a large cutoff value for implicit solvent
        # models, around 20 Angstrom (= 2nm), see
        # https://github.com/openmm/openmm/issues/3104
        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * u.nanometer,  # == 20 Angstrom
            constraints=None,
        )
    elif parameters["forcefield"] == "amber14-explicit":
        forcefield = mm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        model.addExtraParticles(forcefield)
        model.addSolvent(forcefield, padding=parameters["waterbox_pad"])

        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=mm.app.PME,  # .NoCutoff, .PME for particle mesh Ewald
            constraints=None,  # .HBonds   # constrain H-bonds (fastest vibrations)
        )
    else:
        raise ValueError("Invalid forcefield parameter '%s'" % parameters["forcefield"])

    return system


def get_simulation_environment_from_model(model, parameters):
    """Obtain simulation environment suitable for energy computation.

    Arguments
    ---------
    model : openmm.app.modeller.Modeller
        Fully instantiated OpenMM model.
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    simulation : openmm.Simulation
        Simulation (topology, forcefield and computation parameters).  This
        object can be passed to the compute_forces_and_energy method.
    """
    system = get_system(model, parameters)
    integrator = get_simulation_environment_integrator(parameters)
    simulation = mm.app.Simulation(model.topology, system, integrator)

    return simulation


def get_simulation_environment(state0pdbpath, parameters):
    """Obtain simulation environment suitable for energy computation.

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.
    parameters : dict or str
        Parameter dictionary or preset name.

    Returns
    -------
    simulation : openmm.Simulation
        Simulation (topology, forcefield and computation parameters).  This
        object can be passed to the compute_forces_and_energy method.
    """
    parameters = get_parameters_from_preset(parameters)
    model = get_openmm_model(state0pdbpath)
    return get_simulation_environment_from_model(model, parameters)


def get_openmm_model(state0pdbpath):
    """Create openmm model from pdf file.

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.

    Returns
    -------
    model : openmm.app.modeller.Modeller
        Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.
        This object can also be used to create simulation environments.
    """
    pdb_file = mm.app.pdbfile.PDBFile(state0pdbpath)
    positions = pdb_file.getPositions()
    topology = pdb_file.getTopology()
    model = mm.app.modeller.Modeller(topology, positions)
    return model


@functools.lru_cache
def get_simulation_environment_for_force(state0pdbpath, parameters, force_index):
    """Returns a system with all but one Force objects removed.

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.
    parameters : dict or str
        Parameter dictionary or preset name.
    force_index : int
        Index of the force to retain.
        We must have `0 <= force_index < system.getNumForces()`.

    Returns
    -------
    simulation : openmm.Simulation
    """
    sim = get_simulation_environment(state0pdbpath, parameters)
    assert force_index < sim.system.getNumForces()

    # Remove all forces but the one selected by force_index
    for remove_step in range(force_index):
        sim.system.removeForce(0)  # remove first

    # Desired force is now at index 0
    while sim.system.getNumForces() > 1:
        sim.system.removeForce(1)  # remove second force

    integrator = get_simulation_environment_integrator(parameters)
    return mm.app.Simulation(sim.topology, sim.system, integrator)


def compute_energy_and_forces_decomposition(state0pdbpath, parameters, positions, velocities):
    """Compute potential energy and forces, for each force type.

    Only potential energies are computed using this method.
    For a typical AMBER force field this would yield 5-6 different force types,
    the main ones being:

    * HarmonicBondForce
    * HarmonicAngleForce
    * PeriodicTorsionForce
    * NonbondedForce

    The OpenMM documentation contains a detailed description of these forces:

    * [Standard Forces](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html)
    * [Creating Force Fields documentation](http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html)

    Arguments
    ---------
    state0pdbpath : str
        Pathname for all-atom state0.pdb file created by simulate_trajectory.
    parameters : dict or str
        Parameter dictionary or preset name.
    positions : numpy.array of shape (nframes, natoms, 3)
        XYZ positions of atoms in OpenMM coordinates (nm units).
    velocities : numpy.array of shape (nframes, natoms, 3)
        Velocities of atoms in OpenMM units (nm/ps).

    Returns
    -------
    energies_by_force : dict containing as key the OpenMM force class name and as value a
        numpy array of shape (T,)
        Each element contains the [potential energy for frame]
        energies in kJ/mol.
    forces_by_force : dict containing as key the OpenMM force class name and as value a
        numpy array of shape (T,num_atoms,3) forces in kJ/(mol nm).
    """
    sim_base = get_simulation_environment(state0pdbpath, parameters)
    num_forces = sim_base.system.getNumForces()

    energies_by_force = dict()
    forces_by_force = dict()
    for force_index in range(num_forces):
        force_name = sim_base.system.getForce(force_index).__class__.__name__

        # Create a new simulation environment by scratch; this is necessary
        # because OpenMM internally manages C++ resources through Swig proxies,
        # which cannot be deepcopy'd.
        sim = get_simulation_environment_for_force(state0pdbpath, parameters, force_index)

        # For this specific force, compute energies and forces for each frame
        # given in the positions and velocities array.
        traj_energies = []
        traj_forces = []
        for t in range(np.size(positions, 0)):
            sim.context.setPositions(positions[t, :, :])
            sim.context.setVelocities(velocities[t, :, :])

            state = sim.context.getState(getForces=True, getEnergy=True)
            traj_energies.append(state.getPotentialEnergy().value_in_unit(u.kilojoules_per_mole))
            forces = state.getForces(asNumpy=True)
            forces = forces.value_in_unit(u.kilojoules / (u.mole * u.nanometer))
            forces = forces.astype(np.float32)
            traj_forces.append(forces)

        energies_by_force[force_name] = np.array(traj_energies)
        forces_by_force[force_name] = np.stack(traj_forces)

    return energies_by_force, forces_by_force


def compute_energy_and_forces(simulation, positions, velocities):
    """Compute force field energy and forces.

    Arguments
    ---------
    simulation : openmm.Simulation
        Simulation created from the get_simulation_environment method.
    positions : numpy.array of shape (nframes, natoms, 3)
        XYZ positions of atoms in OpenMM coordinates (nm units).
    velocities : numpy.array of shape (nframes, natoms, 3)
        Velocities of atoms in OpenMM units (nm/ps).

    Returns
    -------
    energies : numpy array of shape (T,3)
        Each row contains [potential, kinetic_integrator, kinetic_velocities]
        energies in kJ/mol.  The kinetic_integrator is the kinetic energy as
        computed by the integrator; see https://github.com/openmm/openmm/blob/ae2fe2fd7db2aae4c5c39fadcee328c6d5bc607d/openmmapi/include/openmm/CustomIntegrator.h#L290
        The kinetic_velocities energy is directly computed from the provided
        velocities.
    forces : numpy array of shape (T,num_atoms,3)
        Forces in kJ/(mol nm).
    """
    all_energies = []
    all_forces = []

    system = simulation.system
    num_atoms = system.getNumParticles()
    masses = [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(num_atoms)]
    masses = np.array(masses)

    for t in range(np.size(positions, 0)):
        simulation.context.setPositions(positions[t, :, :])
        simulation.context.setVelocities(velocities[t, :, :])
        kinetic_energy_from_velocities = 0.5 * np.sum(
            masses * np.sum(velocities[t, :, :] ** 2.0, 1)
        )

        state = simulation.context.getState(getForces=True, getEnergy=True)
        all_energies.append(
            [
                state.getPotentialEnergy().value_in_unit(u.kilojoules_per_mole),
                state.getKineticEnergy().value_in_unit(u.kilojoules_per_mole),
                kinetic_energy_from_velocities,
            ]
        )
        forces = state.getForces(asNumpy=True)
        forces = forces.value_in_unit(u.kilojoules / (u.mole * u.nanometer))
        forces = forces.astype(np.float32)
        all_forces.append(forces)

    energies = np.array(all_energies)
    forces = np.stack(all_forces)

    return energies, forces


def sample(simulation, positions, velocities, timesteps, seed=0):
    """Sample Langevin trajectories from a given starting configuration.

    Arguments
    ---------
    simulation : openmm.Simulation
        Simulation created from the get_simulation_environment method.
    positions : numpy.array of shape (natoms, 3)
        XYZ positions of atoms in OpenMM coordinates (nm units).
    velocities : numpy.array of shape (natoms, 3)
        Velocities of atoms in OpenMM units (nm/ps).
    timesteps : list of integers
        Timesteps at which to sample the trajectory.  For example, a
        list of [1, 10, 100, 1000] will store noutput=4 frames at time
        steps 1, 10, 100, and 1000.
    seed : int
        Random number seed used for sampling.

    Returns
    -------
    sampled_positions : numpy.array of shape (noutput, natoms, 3)
        XYZ positions of atoms in OpenMM coordinates (nm units).
    sampled_velocities : numpy.array of shape (noutput, natoms, 3)
        Velocities of atoms in OpenMM units (nm/ps).
    sampled_forces : numpy array of shape (noutput,num_atoms,3)
        Forces in kJ/(mol nm).
    """
    simulation.integrator.setRandomNumberSeed(seed)

    # Set initial state
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    sampled_positions = []
    sampled_velocities = []
    sampled_forces = []

    current_step = 0
    for step in timesteps:
        assert step > current_step
        delta = step - current_step
        simulation.step(delta)
        current_step = step

        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True, getEnergy=True
        )

        # Positions, velocities, forces
        positions = state.getPositions(asNumpy=True)
        sampled_positions.append(positions.value_in_unit(u.nanometer))
        velocities = state.getVelocities(asNumpy=True)
        sampled_velocities.append(velocities.value_in_unit(u.nanometer / u.picosecond))
        forces = state.getForces(asNumpy=True)
        sampled_forces.append(forces.value_in_unit(u.kilojoules / (u.mole * u.nanometer)))

    sampled_positions = np.stack(sampled_positions)
    sampled_velocities = np.stack(sampled_velocities)
    sampled_forces = np.stack(sampled_forces)

    return sampled_positions, sampled_velocities, sampled_forces
