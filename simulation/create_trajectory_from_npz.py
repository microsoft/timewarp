"""Create a PDB trajectory from a state0 and NPZ file.

Usage:
  create_trajectory_from_npz.py <input-state0.pdb> <input-arrays.npz> <trajectory.pdb>

Options :
  -h --help             Show this screen.
"""

import numpy as np
import mdtraj as md
from docopt import docopt


def find_large_step_sequence(steps):
    """Find the largest-increment step in a sequence of steps.

    The NPZ trajectory files store logarithmically or regularly spaced
    indices.  This function recovers the largest regular-increment
    sub-sequence from the ordered sequence of steps.
    """
    step0 = steps[0]
    biggestdiff = 0
    lastdelta = 0
    delta = 0
    for i in range(1, len(steps)):
        step = steps[i]
        diff = step - step0
        lastdelta = delta
        delta = step - steps[i - 1]
        if diff > biggestdiff:
            biggestdiff = diff
        if delta < lastdelta:
            break

    bigstep = i
    return biggestdiff, bigstep


if __name__ == "__main__":
    args = docopt(__doc__, version="create_trajectory_from_npz 0.1")
    print(args)

    traj0 = md.load(args["<input-state0.pdb>"])
    print("Loaded topology:")
    print(traj0)
    data = np.load(args["<input-arrays.npz>"])

    biggestdiff, bigstep = find_large_step_sequence(data["step"])
    indices = list(range(0, len(data["step"]), bigstep))

    print("Found %d frames in a regular subsequence." % len(indices))
    positions = data["positions"][indices, :, :]
    time = data["time"][indices]

    traj = md.Trajectory(
        positions,
        traj0.topology,
        time=time,
        unitcell_lengths=traj0.unitcell_lengths,
        unitcell_angles=traj0.unitcell_angles,
    )
    print("Constructed trajectory:")
    print(traj)
    pdbout = args["<trajectory.pdb>"]
    print("Saving to '%s'..." % pdbout)
    traj.save_pdb(pdbout)
    print("Saved.")
