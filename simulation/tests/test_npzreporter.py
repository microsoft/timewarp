"""Test NPZ reporter intervals."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from simulation.npzreporter import RegularSpacing, LogarithmicSpacing, UniformWindowedSpacing


def test_regular_spacing():
    spacing = RegularSpacing(1000)
    assert spacing.stepsUntilNextReport(0) == 1000
    assert spacing.stepsUntilNextReport(1) == 999
    assert spacing.stepsUntilNextReport(999) == 1
    assert spacing.stepsUntilNextReport(1000) == 1000
    assert spacing.stepsUntilNextReport(1001) == 999


def test_logarithmic_spacing():
    spacing = LogarithmicSpacing(1000, spaceFactor=10)
    assert spacing.stepsUntilNextReport(0) == 1
    assert spacing.stepsUntilNextReport(1) == 9
    assert spacing.stepsUntilNextReport(2) == 8
    assert spacing.stepsUntilNextReport(10) == 90
    assert spacing.stepsUntilNextReport(11) == 89
    assert spacing.stepsUntilNextReport(99) == 1
    assert spacing.stepsUntilNextReport(100) == 900
    assert spacing.stepsUntilNextReport(990) == 10
    assert spacing.stepsUntilNextReport(999) == 1
    assert spacing.stepsUntilNextReport(1000) == 1


def test_windowed_spacing():
    # Create full subsampling pattern.
    spacing = UniformWindowedSpacing(10, spacing_window=5, subsamples=9)
    assert spacing.stepsUntilNextReport(0) == 1
    assert spacing.stepsUntilNextReport(1) == 1
    assert spacing.stepsUntilNextReport(9) == 1
    assert spacing.stepsUntilNextReport(110) == 1
    assert spacing.stepsUntilNextReport(119) == 1

    # Create deterministic  pattern [0,1,8,9]
    spacing = UniformWindowedSpacing(10, spacing_window=2, subsamples=3)
    assert spacing.stepsUntilNextReport(0) == 1
    assert spacing.stepsUntilNextReport(1) == 7
    assert spacing.stepsUntilNextReport(6) == 2
    assert spacing.stepsUntilNextReport(7) == 1
    assert spacing.stepsUntilNextReport(8) == 1
    assert spacing.stepsUntilNextReport(9) == 1
    assert spacing.stepsUntilNextReport(10) == 1
    assert spacing.stepsUntilNextReport(11) == 7

    # Create random subsampling pattern.
    spacing = UniformWindowedSpacing(10, spacing_window=5, subsamples=2)
    assert any([spacing.stepsUntilNextReport(i) > 1 for i in range(20)])

    # Assert we do not get the same pattern twice:
    spacing = UniformWindowedSpacing(10, spacing_window=5, subsamples=2, seed=42)
    a = [spacing.stepsUntilNextReport(i) > 1 for i in range(10, 20)]
    b = [spacing.stepsUntilNextReport(i) > 1 for i in range(20, 30)]
    assert a == a
    assert a != b

    report_interval = 1000
    spacing_window = 400
    subsamples = 100
    n_report_intervals = 20
    spacing = UniformWindowedSpacing(
        report_interval=report_interval, spacing_window=spacing_window, subsamples=subsamples
    )
    reported_steps = np.array(
        [spacing.stepsUntilNextReport(i) for i in range(report_interval * n_report_intervals)]
    )
    steps_per_interval = [
        np.count_nonzero(
            reported_steps[
                i * report_interval - spacing_window - 1 : i * report_interval + spacing_window - 1
            ]
            == 1
        )
        for i in range(1, n_report_intervals)
    ]
    assert np.all(np.array(steps_per_interval) == subsamples + 1)
