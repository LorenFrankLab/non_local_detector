"""Small simulated recordings shared by detector-level model tests."""

import numpy as np
import pytest

from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.fixture(scope="session")
def clusterless_sim():
    """One run of two tetrodes on a linear track."""
    return make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.arange(0, 80, 20), n_runs=1, seed=0
    )


@pytest.fixture(scope="session")
def sorted_sim():
    """First 2,000 samples of three simulated place cells.

    Returns ``(time, position, spike_times)``; spike trains are cut to the
    same window.
    """
    _, position, spike_times, time, _, _, _, _ = make_simulated_data(
        seed=0, n_neurons=3
    )
    n = 2_000
    return time[:n], position[:n], [st[st <= time[n - 1]] for st in spike_times]
