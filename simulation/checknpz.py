"""Perform basic consistency checks on MD simulation data stored in NPZ files.

Usage:
  checknpz.py <input.npz>...

For any violation of our checks we will report the maximum violation.
We perform the following checks for energies, positions, velocities, and
forces:

    1. Check that there are no NaN or Inf values;
    2. Check that the extremes are within a multiple of the interquartile
       range around the median.
    3. Check for non-stationarity by comparing observables in the first
       half to observables in the second half.
"""

import sys
import time
import numpy as np
from docopt import docopt


def estimate_splitR(values):
    """Estimate the Gelman/Rubin split-R convergence statistic.

    The split-R diagnostic is described in detail in
    [(Gelman and Rubin, 1992)](https://projecteuclid.org/journals/statistical-science/volume-7/issue-4/Inference-from-Iterative-Simulation-Using-Multiple-Sequences/10.1214/ss/1177011136.pdf).
    We split the chain into two equal sized parts (M=2).

    Arguments
    ---------
    values : np.array, axis 0 being the time order of the Markov chain.
        The observables, in time order, stemming from a Markov chain.

    Returns
    -------
    splitR : np.array with shape equal to values.shape[1::]
        The split-R convergence diagnostic, >=1.0.  A value above 1.2 is
        generally considered as evidence of non-convergence of the chain.
    """
    values1, values2 = np.split(values, 2, axis=0)
    mean1, mean2 = np.mean(values1, axis=0), np.mean(values2, axis=0)
    mean12 = np.mean(np.stack([mean1, mean2]), axis=0)
    N = float(values1.shape[0])
    M = 2.0
    between_chain_var = N * np.var(np.stack([mean1 - mean12, mean2 - mean12]), axis=0)
    within_chain_var = np.mean(np.stack([np.var(values1), np.var(values2)]), axis=0)
    pooled_var = ((N - 1.0) / N) * within_chain_var + ((M + 1.0) / (M * N)) * between_chain_var
    splitR = pooled_var / within_chain_var

    return splitR


def check_value_range(values, label, interquartile_factor=5.0, splitR_ub=1.2):
    """Check the extreme values of observables.

    We perform the following checks on a list of array observables:

    1. Check that there are no NaN or Inf values;
    2. Check that the extremes are within a multiple of the interquartile range
       around the median.
    3. Check for non-stationarity by comparing observables in the first half to
       observables in the second half.

    Arguments
    ---------
    values : np.array, time dimension on first axis
        The energy values to check, ordered in time.
    label : str
        A short string describing the values array.  Used for printing.
    interquartile_factor : float, >0.0
        The multiplier to apply to the check.
    splitR_ub : float, >1.0
        The maximum tolerable convergence diagnostic value to consider the chains
        to have converged.  Following (Gelman and Rubin, 1992) we use 1.2.

    Returns
    -------
    checks_passed : bool
        True if all checks passed.
    """
    assert interquartile_factor > 1.0
    assert splitR_ub > 1.0

    # 1. Check that there are no NaN or Inf values
    passed = True
    if np.any(np.isnan(values)):
        print("  ! [%s] found NaN values in observations" % label)
        passed = False
    if np.any(np.isinf(values)):
        print("  ! [%s] found Inf values in observations" % label)
        passed = False

    # the remainder of checks do not make sense if we have NaN/Inf values
    if not passed:
        return False

    # 2. Check that the extremes are within a multiple of the interquartile range
    # around the median.
    iq = np.quantile(values, [0.25, 0.75], axis=0)
    iq_range = iq[1, ...] - iq[0, ...]
    values_median = np.quantile(values, [0.5], axis=0)
    values_min = np.min(values, axis=0)
    values_max = np.max(values, axis=0)
    values_lb = values_median - interquartile_factor * iq_range
    values_ub = values_median + interquartile_factor * iq_range
    if np.any(values_min < values_lb):
        idx = np.argmax(values_lb - values_min)
        exmin = values_min.flatten()[idx]
        exlb = values_lb.flatten()[idx]
        exmed = values_median.flatten()[idx]
        print(
            "  ! [%s] minimum value %.6e smaller than safety lower bound %.6e (= median %.6e - (%f*iqr))"
            % (label, exmin, exlb, exmed, interquartile_factor)
        )
        passed = False
    if np.any(values_max > values_ub):
        idx = np.argmax(values_max - values_ub)
        exmax = values_max.flatten()[idx]
        exub = values_ub.flatten()[idx]
        exmed = values_median.flatten()[idx]
        print(
            "  ! [%s] maximum value %.6e larger than safety upper bound %.6e (= median %.6e + (%f*iqr))"
            % (label, exmax, exub, exmed, interquartile_factor)
        )
        passed = False

    # 3. Check for non-stationarity by comparing energies in the first half to
    #    energies in the second half.
    # splitR is on Rhat scale (not PSRF scale).  We have Rhat = sqrt(PSRF).
    splitR = np.sqrt(estimate_splitR(values))
    if np.any(splitR > splitR_ub):
        idx = np.argmax(splitR)
        exR = splitR.flatten()[idx]
        print(
            "  ! [%s] splitR half-chain convergence diagnostic of %.4e is higher than allowed threshold %.4e"
            % (label, exR, splitR_ub)
        )
        passed = False

    return passed


def check_npz(npzpathname):
    """Perform consistency checks on NPZ data.

    Arguments
    ---------
    npzpathname : str
        The filepath to the NPZ file to analyze.

    Returns
    -------
    passed : bool
        True if all checks have passed, False otherwise.
    """
    data = np.load(npzpathname)

    # Check potential energies
    potential_energies = data["energies"][:, 0].flatten()
    passed_pot = check_value_range(potential_energies, "E_pot")

    # Check kinetic energies
    kinetic_energies = data["energies"][:, 1].flatten()
    passed_kin = check_value_range(kinetic_energies, "E_kin")

    # Check positions; these can be many values, so we relax IQR and R_ub
    # to accomodate the variability.
    pos = data["positions"]
    passed_pos = check_value_range(pos, "positions", interquartile_factor=10.0, splitR_ub=1.5)

    # Check velocities
    veloc = data["velocities"]
    passed_veloc = check_value_range(veloc, "velocities", interquartile_factor=10.0, splitR_ub=1.5)

    # Check velocities
    forces = data["forces"]
    passed_forces = check_value_range(forces, "forces", interquartile_factor=10.0, splitR_ub=1.5)

    passed = passed_pot and passed_kin and passed_pos and passed_veloc and passed_forces
    return passed


def main():
    args = docopt(__doc__, version="checknpz 0.1")

    total_npz = 0
    passed_npz = 0
    failed_npz = 0
    npzpathnames = args["<input.npz>"]
    for npzpathname in npzpathnames:
        print("Checking '%s'..." % npzpathname)
        start_check_time = time.time()
        passed = check_npz(npzpathname)
        end_check_time = time.time()
        total_npz += 1
        if passed:
            passed_npz += 1
            print("  all checks passed in %fs." % (end_check_time - start_check_time))
        else:
            failed_npz += 1

    if len(npzpathnames) > 1:
        print("")
        print("Total statistics:")
        print("   %d files total" % total_npz)
        print("   %d files passed" % passed_npz)
        print("   %d files failed" % failed_npz)

    if passed_npz == len(npzpathnames):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
