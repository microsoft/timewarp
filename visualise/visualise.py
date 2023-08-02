"""Visualization functions."""

import subprocess
import os
import sys
import mdtraj as md
import numpy as np
import numpy.typing as npt
import tempfile
from typing import Union

StrPath = Union[str, os.PathLike]


def writepdb(
    state0filepath: StrPath, positions: npt.NDArray[np.float64], outputpdbfilepath: StrPath
) -> None:
    """Write new PDB file from topology and coordinates.

    Parameters
    ----------
    state0filepath : string
        The path to the state0.pdb file specifying the topology.
    positions : np.array
        Positions, shape should be (natoms, 3).
    outputpdbfilepath : string
        File to write PNG rendering to.
    """
    traj0 = md.load(state0filepath)
    traj = md.Trajectory(positions, traj0.topology)
    traj.save_pdb(outputpdbfilepath)


def _visualise(pdbfilepath: StrPath, outputpngfilepath: StrPath) -> None:
    """Create a stick visualization from a pdbfile.

    Parameters
    ----------
    pdbfilepath : string
        The path to the .pdb file specifying the topology.
    outputpngfilepath : string
        File to write PNG rendering to.
    """
    import __main__

    __main__.pymol_argv = ["pymol", "-qc"]  # Quiet and no GUI
    import pymol2

    p = pymol2.PyMOL()
    p.start()

    p.cmd.reinitialize()
    name = "pdb"
    p.cmd.load(pdbfilepath, name)
    p.cmd.disable("all")
    p.cmd.enable(name)
    p.cmd.remove("solvent")

    p.cmd.show("sticks")
    p.cmd.show("spheres")
    p.cmd.set("stick_radius", 0.1, "(all)")
    p.cmd.set("sphere_scale", 0.2, "(all)")
    p.cmd.unset("opaque_background")
    res = 1024
    p.cmd.ray(res, res)
    p.cmd.png(outputpngfilepath)

    p.stop()


def visualise(
    state0filepath: StrPath, positions: npt.NDArray[np.float64], outputpngfilepath: StrPath
) -> None:
    """Create a stick visualization from topology and coordinates.

    Parameters
    ----------
    state0filepath : string
        The path to the state0.pdb file specifying the topology.
    positions : np.array
        Positions, shape should be (natoms, 3).
    outputpngfilepath : string
        File to write PNG rendering to.

    Raises:
        RuntimeError if pymol process exits with an error.
    """
    with tempfile.NamedTemporaryFile(prefix="visframe", suffix=".pdb", delete=False) as tmpfile:
        pdbfilepath = tmpfile.name
        writepdb(state0filepath, positions, pdbfilepath)

    """
    This doesn't work!
    p = Process(target=_visualise, args=(pdbfilepath, outputpngfilepath))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        raise RuntimeError(f"pymol timed out at {timeout} seconds. Input file is kept in {pdbfilepath}.")
    exitcode = p.exitcode
    p.close()
    """
    importdir = os.path.dirname(__file__)
    try:
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, '{importdir}'); from visualise import _visualise;"
                f"_visualise('{pdbfilepath}', '{outputpngfilepath}')",
            ],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Exception while running pymol visualiser. Input file is kept at {pdbfilepath}. Stdout/err was:\n{e.stdout.decode()}"
        ) from e
    else:
        os.unlink(pdbfilepath)


if __name__ == "__main__":
    # Test case of a small protein
    data = np.load("../timewarp/testdata/implicit-2olx-traj-arrays.npz")
    positions = data["positions"][0, :, :]
    visualise(
        "../timewarp/testdata/implicit-2olx-traj-state0.pdb",
        positions,
        "outputs/out.png",
    )
