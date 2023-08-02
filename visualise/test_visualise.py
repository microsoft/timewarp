import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from visualise import visualise

DATA_DIR = Path(__file__).resolve().parents[1] / "timewarp/testdata/smallest_molecule"


def test_visualise() -> None:
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.png"
        data = np.load(DATA_DIR / "2olx-traj-arrays.npz")
        visualise.visualise(
            DATA_DIR / "2olx-traj-state0.pdb", data["positions"][0, :, :], output_path
        )
        assert os.path.isfile(output_path)
