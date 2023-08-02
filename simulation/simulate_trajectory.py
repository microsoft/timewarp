"""Simulate MD trajectory of protein in water.

Usage:
  simulate_trajectory.py [options] <input.ext> <trajectory.pdb>

Options :
  -h --help             Show this screen.
  --keep-water          Do not remove water molecules from PDBx/mmCIF.
  --preset=<preset>     Use pre-defined simulation settings, listed below.
  --force-field=<ff>    (preset) Force field, "amber99-implicit", "amber14-implicit", or "amber14-explicit". [default: amber14-implicit]
  --waterbox-pad=<pad>  (preset) Waterbox padding width in nm [default: 1.0].
  --temperature=<T>     (preset) System temperature in Kelvin [default: 310].
  --timestep=<ts>       (preset) Integration time step in femtoseconds [default: 1.0].
  --friction=<f>        (preset) Langevin friction in 1.0/ps [default: 0.3].
  --old-integrator      (preset) Use LangevinIntegrator, not LangevinMiddleIntegrator.
  --do-not-minimize     Do not perform energy minimization.
  --min-tol=<mintol>    Energy minimization tolerance in kJ/mol [default: 2.0].
  --burn-in=<burnin>    Number of burn-in steps [default: 2000000].
  --log=<logsteps>      Number steps between stdout report [default: 10000].
  --sampling=<steps>    Number of total integration steps [default: 20000000].
  --spacing=<spacing>   Thinning steps between samples [default: 1000000].
  --spacing_approach=<spacing_approach>  Approach for subsampling [default: logarithmic]
  --no-checkpointing    Do not use checkpointing.
  --cpmins=<cpmins>     Set checkpoint frequency in minutes [default: 5].

The extension of <input.ext> needs to be '.pdb' for processed PDB files that
will be directly simulated.

The following pre-defined simulation settings are available.  All presets use
a temperature of 310K, friction of 0.3/ps, and timestep of 0.5fs.  If you use
the --preset option then the following parameters, marked with "(preset)" are
determined from the preset: --force-field, --waterbox-pad, --explicit-water,
--temperature, --timestep, --friction, --old-integrator.
All other parameters are determined from their respective option values.
The available presets are as follows:

* 'amber99-implicit': AMBER99 with implicit OBC water.
* 'amber14-implicit': AMBER14 with implicit OBC1 water.
* 'amber14-explicit': AMBER14 with explicit TIP3 PFB water, 1nm waterbox.

"""

import os
import sys
from typing import Optional

import openmm as mm
import openmm.unit as u
from docopt import docopt
from openmm.app import PDBFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simulation.md as md
from simulation.npzreporter import NPZReporter, LogarithmicSpacing, UniformWindowedSpacing, Spacing
from simulation.checkpointing import FullCheckpointReporter, load_checkpoint, save_simulation_time


def create_path(filepath):
    """Create directory to the given file."""
    pathname, filename = os.path.split(filepath)
    if pathname:
        os.makedirs(pathname, mode=0o755, exist_ok=True)


if __name__ == "__main__":
    args = docopt(__doc__, version="simulate_trajectory 0.1")
    print(args)

    # Process .cif.gz file and clean
    keep_water = args["--keep-water"]
    cifpath = args["<input.ext>"]
    cifbasename, ext = os.path.splitext(cifpath)
    if ext == ".pdb":
        pdbpath = cifpath
        head, molname = os.path.split(cifbasename)
    else:
        print("Invalid file extension '%s' used for the input file '%s'." % (ext, cifpath))
        sys.exit(1)

    # Load file
    print("Loading '%s' from '%s'" % (molname, pdbpath))

    model = md.get_openmm_model(pdbpath)
    model.addHydrogens()
    if not keep_water:
        model.deleteWater()

    # Save initial simulation topology and positions
    trajfile = args["<trajectory.pdb>"]
    trajfilebase, ext = os.path.splitext(trajfile)
    traj0file = trajfilebase + "-state0.pdb"
    create_path(traj0file)
    trajnpzfile = trajfilebase + "-arrays.npz"

    checkpoint_file_base = trajfilebase + ".chk"
    checkpoint_seconds = 60 * int(args["--cpmins"])
    if args["--no-checkpointing"]:
        checkpoint_file_base = None
    else:
        print("Performing checkpointing every %d seconds." % checkpoint_seconds)

    # Get all atom indices that will be written to NPZ
    protein_atoms = len(model.getPositions())
    print("Pre-processed protein has %d atoms." % protein_atoms)

    # Setup simulation
    if args["--preset"]:
        # setup from preset
        preset_name = args["--preset"]
        print("Using pre-defined preset '%s'." % preset_name)
        parameters = md.get_parameters_from_preset(preset_name)
        print("Using preset '%s'." % preset_name)
    else:
        # setup from user-specified parameters
        parameters = {
            "forcefield": args["--force-field"],
            "temperature": float(args["--temperature"]) * u.kelvin,
            "friction": float(args["--friction"]) / u.picosecond,
            "timestep": float(args["--timestep"]) * u.femtosecond,
            "waterbox_pad": float(args["--waterbox-pad"]) * u.nanometers,
            "integrator": "LangevinMiddleIntegrator",
        }
        if args["--old-integrator"]:
            parameters["integrator"] = "LangevinIntegrator"

        print("Not using preset, but manual configuration:")
        print(parameters)

    simulation = md.get_simulation_environment_from_model(model, parameters)

    all_atoms = len(simulation.context.getState(getPositions=True).getPositions())
    print("After processing, simulation system now has %d atoms." % all_atoms)

    # Write state0 file (potentially added water now).
    PDBFile.writeFile(model.topology, model.positions, open(traj0file, "w"))

    print("OpenMM version: %s" % mm.__version__)
    simulation.context.setPositions(model.positions)

    log_steps = int(args["--log"])
    burnin_steps = int(args["--burn-in"])
    sampling_steps = int(args["--sampling"])
    total_steps = burnin_steps + sampling_steps

    # No checkpoint, start simulation from step 0
    if log_steps > 0:
        # StateDataReporter cannot be pickled, so we always add it afresh.
        simulation.reporters.append(
            mm.app.StateDataReporter(
                sys.stdout,
                log_steps,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                speed=True,
                temperature=True,
                progress=True,
                totalSteps=total_steps,
            )
        )

    # Attempt to continue simulation from checkpoint
    if load_checkpoint(simulation, checkpoint_file_base):
        # Did restore checkpoint, so all reporters and states are ok.
        # Only perform remaining steps
        remaining_steps = sampling_steps - simulation.currentStep
        print(
            "Restored checkpoint at step %d, target is %d, hence %d steps are missing."
            % (simulation.currentStep, sampling_steps, remaining_steps)
        )

        print(
            "Continuing the simulation via completion of the remaining %d steps..."
            % remaining_steps
        )
        simulation.step(remaining_steps)
        # Save also simulation time
        timenpyfile = trajfilebase + "-time.npy"
        save_simulation_time(timenpyfile, simulation.reporters[-1])

        print("Completed remaining SAMPLING steps.")
        del simulation  # needed to force write of NPZ
        sys.exit(0)

    # Energy minimization
    if not args["--do-not-minimize"]:
        tolerance = float(args["--min-tol"])
        print("Performing ENERGY MINIMIZATION to tolerance %2.2f kJ/mol" % tolerance)
        simulation.minimizeEnergy(tolerance=tolerance)
        print("Completed ENERGY MINIMIZATION")

    # Set initial velocities to the right temperature
    temperature = parameters["temperature"]
    print("Initializing VELOCITIES to %s" % temperature)
    simulation.context.setVelocitiesToTemperature(temperature)

    # Burn-in
    if burnin_steps > 0:
        print("Performing BURN-IN SAMPLING for %d steps" % burnin_steps)
        simulation.step(burnin_steps - 1)
        print("Completed BURN-IN SAMPLING")

    print("Re-initializing VELOCITIES to %s" % temperature)
    simulation.context.setVelocitiesToTemperature(temperature)

    # Sampling
    spacing_steps = int(args["--spacing"])
    spacing_factor = 10
    spacing_approach = args["--spacing_approach"]
    spacing: Optional[Spacing] = None
    if spacing_approach == "logarithmic":
        spacing = LogarithmicSpacing(spacing_steps, spacing_factor)

        print(
            "Recording every %d steps with logarithmic policy of factor %d"
            % (spacing_steps, spacing_factor)
        )
    elif spacing_approach == "windowed":
        # +-100 femto-seconds around spacing_steps
        spacing_window = 200
        subsamples = 10
        spacing = UniformWindowedSpacing(
            spacing_steps, spacing_window=spacing_window, subsamples=subsamples
        )
        print(
            f"Recording every {spacing_steps} steps with windowed policy, spacing_window {spacing_window}, subsamples {subsamples}"
        )
    else:
        raise ValueError(
            f"Invalid value for --spacing_approach={spacing_approach}, supported: [logarithmic, windowed]"
        )

    create_path(trajnpzfile)
    simulation.reporters.append(
        NPZReporter(trajnpzfile, spacing, atom_indices=range(protein_atoms))
    )
    print(
        "Performing SAMPLING for %d steps, output '%s' and '%s'"
        % (sampling_steps, trajfile, trajnpzfile)
    )

    # Add checkpointing reporter
    if checkpoint_file_base:
        simulation.reporters.append(
            FullCheckpointReporter(checkpoint_file_base, checkpoint_seconds)
        )

    # Fresh simulation start
    simulation.step(sampling_steps)
    print("Completed SAMPLING")
    if checkpoint_file_base:
        # Save also simulation time
        timenpyfile = trajfilebase + "-time.npy"
        save_simulation_time(timenpyfile, simulation.reporters[-1])

    # Required in order for NPZReporter to be able to properly write out data
    del simulation
