"""
checkpointing.py: Full simulation checkpointing for OpenMM.

The default checkpointing functionality in OpenMM (through `simulation.saveCheckpoint`)
is problematic, as highlighted in [this issue](https://github.com/openmm/openmm/issues/1645).
In particular:

* Only positions and velocities are saved, but additional state such as the Reporter objects present in the simulation are not saved.
* Restoring the simulation state requires re-initializing the Simulation state which is error-prone.

This file provides functionality to take and restore a checkpoint that includes the entire simulation state.
"""

import os
import time
import gzip
import pickle
import numpy as np
import openmm as mm


def safe_replace(base, old, new):
    """Rename base to old, new to base, delete old."""
    base_exists = os.path.exists(base)
    if base_exists:
        os.rename(base, old)

    os.rename(new, base)
    if base_exists:
        os.remove(old)


def state_filenames(checkpoint_file_base):
    """Return (base, old, new) state filenames for given base."""
    state_base = checkpoint_file_base + "-state.chk.gz"
    state_temp_old = state_base + ".old"
    state_temp_new = state_base + ".new"

    return state_base, state_temp_old, state_temp_new


def sim_filenames(checkpoint_file_base):
    """Return (base, old, new) sim filenames for given base."""
    sim_base = checkpoint_file_base + "-sim.pickle.gz"
    sim_temp_old = sim_base + ".old"
    sim_temp_new = sim_base + ".new"

    return sim_base, sim_temp_old, sim_temp_new


def save_checkpoint(simulation, checkpoint_file_base, gzip_compresslevel=3):
    """Create a checkpoint of a simulation.

    Parameters
    ----------
    simulation : Simulation
        The Simulation to generate a checkpoint file for.
    checkpoint_file_base : str
        The base prefix of two files that will be created: "base-state.chk.gz" and "base-sim.pickle.gz".
    gzip_compresslevel : int
        A value between 1 and 9 indicating the [gzip compression strength](https://docs.python.org/3/library/gzip.html#gzip.GzipFile).
    """
    # Save a simulation checkpoint
    state_base, state_temp_old, state_temp_new = state_filenames(checkpoint_file_base)
    with gzip.open(state_temp_new, "wb", compresslevel=gzip_compresslevel) as out:
        out.write(simulation.context.createCheckpoint())

    # Save reporters and currentStep of simulation
    sim_base, sim_temp_old, sim_temp_new = sim_filenames(checkpoint_file_base)
    reporters = simulation.reporters

    # Filter out StateDataReporter as it cannot be pickled
    reporters = [
        rep for rep in reporters if type(rep) is not mm.app.statedatareporter.StateDataReporter
    ]

    sim_state_dict = {
        "currentStep": simulation.currentStep,
        "reporters": reporters,
    }
    with gzip.open(sim_temp_new, "wb", compresslevel=gzip_compresslevel) as out:
        pickle.dump(sim_state_dict, out)

    # All new checkpoint data is written, now perform new -> base rename
    safe_replace(state_base, state_temp_old, state_temp_new)
    safe_replace(sim_base, sim_temp_old, sim_temp_new)


def load_checkpoint(simulation, checkpoint_file_base):
    """Load a checkpoint of a simulation.

    Parameters
    ----------
    simulation : Simulation
        The Simulation to configure from the saved checkpoint file.
        This simulation _must_ already be correct with respect to topology, system, integrator, and platform.
        The load_checkpoint function only restores the positions and velocities of the system, simulation.reporters, and simulation.currentStep to allow
        for continuation of an interrupted simulation run.
    checkpoint_file_base : str
        The base prefix of two files that will be created: "base-state.chk.gz" and "base-sim.pickle.gz".

    Returns
    -------
    result : bool
        True if checkpoint was found and restored, False otherwise.
    """
    if checkpoint_file_base is None:
        return False

    state_base, _, _ = state_filenames(checkpoint_file_base)
    sim_base, _, _ = sim_filenames(checkpoint_file_base)

    if not os.path.exists(state_base) or not os.path.exists(sim_base):
        return False  # Did not restore

    # Restore positions and velocities
    with gzip.open(state_base, "rb") as statefile:
        simulation.loadCheckpoint(statefile)

    # Restore reporters and currentStep
    with gzip.open(sim_base, "rb") as simfile:
        sim_state_dict = pickle.load(simfile)

    simulation.currentStep = sim_state_dict["currentStep"]
    reporters = sim_state_dict["reporters"]
    for reporter in reporters:
        if type(reporter) is FullCheckpointReporter:
            print("Found FullCheckpointReporter among reporters, refreshing it to current time.")
            reporter.refresh()  # Use current time (prevents immediate re-checkpointing)
    simulation.reporters += reporters

    return True


class FullCheckpointReporter(object):
    """FullCheckpointReporter stores checkpoints every n seconds.

    To use, create a FullCheckpointReporter, then add it to the Simulation's list of
    reporters.  The saved checkpoints are meant to be restored with the load_checkpoint method.
    """

    def __init__(self, checkpoint_file_base, checkpoint_seconds):
        """Create a FullCheckpointReporter.

        Parameters
        ----------
        checkpoint_file_base : str
            The base prefix of two files that will be created: "base-state.chk.gz" and "base-sim.pickle.gz".
        checkpoint_seconds : int
            The number of seconds that need to elapse after which a checkpoint is written.
        """
        if checkpoint_seconds <= 0:
            raise ValueError("checkpoint_seconds must be positive.")

        self._checkpoint_file_base = checkpoint_file_base
        self._checkpoint_seconds = checkpoint_seconds
        self._last_report = None

    def refresh(self):
        self._last_report = time.time()

    def describeNextReport(self, simulation):
        if self._last_report is None:
            self.refresh()

        return (1000, False, False, False, False, None)  # PVFE

    def report(self, simulation, state):
        now = time.time()
        delta = int(now - self._last_report)
        if delta < self._checkpoint_seconds:
            return  # Not enough time elapsed yet

        # Enough time did elapse, make a snapshot
        print(
            "Checkpointing: %d seconds elapsed since the last checkpoint, writing checkpoint to '%s*'..."
            % (delta, self._checkpoint_file_base)
        )
        save_checkpoint(simulation, self._checkpoint_file_base)

        # Save also the simulation time
        timenpyfile = self._checkpoint_file_base[:-4] + "-time.npy"
        save_simulation_time(timenpyfile, self)

        # Saving a checkpoint may take quite some time in itself, so store current time after successful saving as reference for when to take the next checkpoint
        self.refresh()


def save_simulation_time(timenpyfile, full_checkpoint_reporter):
    try:
        simulation_time = np.load(timenpyfile)
    except FileNotFoundError:
        simulation_time = 0
    now = time.time()
    delta = int(now - full_checkpoint_reporter._last_report)
    np.save(timenpyfile, simulation_time + delta)
