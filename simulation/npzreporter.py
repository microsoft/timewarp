"""
npzreporter.py: Output simulation trajectory data in NPZ format.

We define a new NPZReporter class that interfaces with OpenMM's Simulation
class to store simulation trajectory data into a compressed NPZ file.  The
simulation data includes positions, velocities, and forces.

Modelled after OpenMM's PDBReporter.
"""
import warnings
from typing import Optional

import numpy as np
import openmm.unit as unit


class Spacing:
    """A policy that determines when to record trajectory data."""

    def stepsUntilNextReport(self, currentStep):
        raise NotImplementedError("Derived classes need to implement stepsUntilNextReport method.")


class RegularSpacing(Spacing):
    """Regular spacing, every `reportInterval` steps."""

    def __init__(self, reportInterval):
        """Create a regular spacing.

        Parameters
        ----------
        reportInterval : int
            The interval (in time steps) at which to write frames.
        """
        super(RegularSpacing, self).__init__()
        self._reportInterval = reportInterval

    def stepsUntilNextReport(self, currentStep):
        """Return the number of steps until the next reporting step."""
        steps = self._reportInterval - currentStep % self._reportInterval
        return steps


class LogarithmicSpacing(Spacing):
    """Logarithmic stepping helper class.

    In order to save more useful trajectory data we do not merely save every
    k'th step but instead save according to the following spacing scheme:
    ```
       [ 10000, 10001, 10010, 10100, 11000,
         20000, 20001, 20010, 20100, 21000, ...]
    ```
    Here `reportInterval=10000` and `spaceFactor=10`.  The above scheme remains
    compact even for large `reportInterval`s but provides richer trajectory
    data for subsequent machine learning experimentation.
    """

    def __init__(self, reportInterval, spaceFactor=10):
        """Create a logarithmic spacing.

        Parameters
        ----------
        reportInterval : int
            The interval (in time steps) at which to write frames.
        spaceFactor : int
            The logarithmic spacing factor by which to increase spacing.
        """
        self._reportInterval = reportInterval
        if spaceFactor <= 1:
            raise ValueError("spaceFactor must be larger than one.")

        self._spaceFactor = spaceFactor

    def stepsUntilNextReport(self, currentStep):
        """Return the number of steps until the next reporting step.

        Note: this method will never return zero.  When it returns a value of 1
        the report should be created after the next iteration.
        """
        currentStep %= self._reportInterval  # ignore reportInterval start

        nextOffset = 1
        while nextOffset <= currentStep:
            nextOffset *= self._spaceFactor

        nextOffset = min(nextOffset, self._reportInterval)
        return nextOffset - currentStep


class UniformWindowedSpacing(Spacing):
    """Uniform stepping helper class.

    In order to save more useful trajectory data we do not merely save every
    k'th step but additionally save the following extra samples (in order):
    ```
       [ Uniform(range(report_interval-spacing_window, report_interval+spacing_window), subsamples),
         Uniform(range(report_interval*2-spacing_window, report_interval*2+spacing_window), subsamples)]
    ```
    The above scheme remains compact even for large `reportInterval`s but allows for time jitter around the report interval.
    """

    def __init__(
        self,
        report_interval,
        spacing_window: int = 100,
        subsamples: int = 10,
        seed: Optional[int] = None,
    ):
        """Create a windowed spacing.

        Notes: Asserts this is called iteratively with incrementing currentSteps calls.

        Parameters
        ----------
        report_interval : int
            The interval (in time steps) at which to write frames.
        spacing_window : int
            the (half)window (in time steps) from which to uniformly sample around the reporting interval.
        subsamples: int
            The number of subsamples (in addition to the sample centered on the report_interval).
        seed: Optional[int]
            If set, makes the subsampling deterministic.
        """
        self.report_interval = report_interval
        self.spacing_window = spacing_window
        self.subsamples = subsamples
        assert self.subsamples < self.spacing_window * 2
        assert self.report_interval >= self.spacing_window * 2
        self.rng = np.random.RandomState(seed)
        # Setup probabilities for subsampling window, exclude sample 0.
        self.p = np.ones(self.spacing_window * 2)
        self.p[self.spacing_window] = 0
        self.p /= self.p.sum()

        # Draw a fresh subsampling window.
        self._draw_timesteps_to_keep()
        self._previous_current_step = -1
        self._in_overflow = False

    def stepsUntilNextReport(self, currentStep):
        """Return the number of steps until the next reporting step.

        Note: this method will never return zero.  When it returns a value of 1
        the report should be created after the next iteration.
        """

        assert currentStep > self._previous_current_step, "non-monotonic currentStep call"
        currentStep = (
            currentStep + self.report_interval // 2
        ) % self.report_interval - self.report_interval // 2
        self._previous_current_step = currentStep
        if currentStep > 0 and self._in_overflow:
            # We've already sampled a new window, but are still querying the end of the previous window.
            # This should only happen in our testing scripts.
            warnings.warn(
                "Querying previous window while new window drawn, this should not happen during simulation."
            )
            return self._relative_timesteps_to_keep[0] - (currentStep - self.report_interval)
        else:
            self._in_overflow = False

        # Find next timestep to report.
        next_i = np.searchsorted(self._relative_timesteps_to_keep, currentStep, side="left")

        if (
            next_i < len(self._relative_timesteps_to_keep)
            and self._relative_timesteps_to_keep[next_i] == currentStep
        ):
            # If we're currently at a step that we're storing, we should report steps till the next report.
            next_i += 1

        if next_i >= len(self._relative_timesteps_to_keep):
            self._draw_timesteps_to_keep()
            self._in_overflow = True
            result = self._relative_timesteps_to_keep[0] - (currentStep - self.report_interval)
        else:
            # Return steps till next subsample.
            result = self._relative_timesteps_to_keep[next_i] - currentStep

        return result

    def _draw_timesteps_to_keep(self):
        # Draw random subsamples in window [-self.spacing_window, self.spacing_window), always add 0.
        self._relative_timesteps_to_keep = (
            self.rng.choice(self.spacing_window * 2, self.subsamples, replace=False, p=self.p)
            - self.spacing_window
        )
        # Add 0
        self._relative_timesteps_to_keep = np.concatenate(
            (self._relative_timesteps_to_keep, np.array([0], dtype=int))
        )
        # Sort ascending.
        self._relative_timesteps_to_keep = np.sort(self._relative_timesteps_to_keep)


class NPZReporter(object):
    """NPZReporter outputs positions, velocities, and forces for each frame.

    To use, create a NPZReporter, then add it to the Simulation's list of
    reporters.

    The created NPZ file will contain the following arrays:
      * 'time': (T,) array, simulation time in picoseconds.
      * 'energies': (T,2) array, each row containing [potential, kinetic]
        energies in kJ/mol.
      * 'positions': (T,num_atoms,3) array, positions in nm.
      * 'velocities': (T,num_atoms,3) array, velocities in nm/ps.
      * 'forces': (T,num_atoms,3) array, forces in kJ/(mol nm).
    """

    def __init__(self, filename, spacing, atom_indices=None):
        """Create a NPZReporter.

        Parameters
        ----------
        filename : string
            The filename to write to, should end with '.npz'.
        spacing : Spacing
            The report spacing at which to write frames.
        atom_indices : Range or List or None
            The list of atoms to record in that order in the NPZ file.
            If None, all atom coordinates are saved.
        """
        self._filename = filename
        self._spacing = spacing
        self._atom_indices = atom_indices
        self._nextModel = 0
        self._positions = []
        self._velocities = []
        self._forces = []
        self._energies = []
        self._time = []
        self._step = []

    def describeNextReport(self, simulation):
        steps = self._spacing.stepsUntilNextReport(simulation.currentStep)
        return (steps, True, True, True, True, None)  # PVFE

    def filter_atoms(self, data):
        if self._atom_indices:
            data = data[self._atom_indices, :]
        return data

    def report(self, simulation, state):
        self._time.append(state.getTime().value_in_unit(unit.picoseconds))
        self._step.append(simulation.currentStep)
        self._energies.append(
            [
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
            ]
        )

        # Positions
        positions = state.getPositions(asNumpy=True)
        positions = positions.value_in_unit(unit.nanometer)
        positions = positions.astype(np.float32)
        positions = self.filter_atoms(positions)
        self._positions.append(positions)

        # Velocities
        velocities = state.getVelocities(asNumpy=True)
        velocities = velocities.value_in_unit(unit.nanometer / unit.picosecond)
        velocities = velocities.astype(np.float32)
        velocities = self.filter_atoms(velocities)
        self._velocities.append(velocities)

        # Forces
        forces = state.getForces(asNumpy=True)
        forces = forces.value_in_unit(unit.kilojoules / (unit.mole * unit.nanometer))
        forces = forces.astype(np.float32)
        forces = self.filter_atoms(forces)
        self._forces.append(forces)

    def __del__(self):
        # Save all trajectory data to the NPZ file
        step = np.array(self._step)
        time = np.array(self._time)
        energies = np.array(self._energies)

        positions = np.stack(self._positions)
        velocities = np.stack(self._velocities)
        forces = np.stack(self._forces)

        np.savez_compressed(
            self._filename,
            step=step,
            time=time,
            energies=energies,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )
