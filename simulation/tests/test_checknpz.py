import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.checknpz import check_npz
from simulation.checknpz import estimate_splitR

npzpath = os.path.join(os.path.dirname(__file__), "..", "testdata", "implicit-2olx-traj-arrays.npz")


def test_splitR_diagnostic():
    """Test discriminative power of splitR diagnostic."""
    # Sample from the same distribution
    splitR = estimate_splitR(np.random.normal(size=(1000,)))
    assert splitR <= 1.1

    # Sample from different distributions
    sample1 = np.random.normal(loc=0.0, scale=1.0, size=(500,))
    sample2 = np.random.normal(loc=1.0, scale=1.0, size=(500,))
    sample = np.concatenate([sample1, sample2])
    splitR = estimate_splitR(sample)
    assert splitR > 1.1


def test_check_npz():
    """End-to-end check of NPZ checking function on reference file."""
    assert check_npz(npzpath)
