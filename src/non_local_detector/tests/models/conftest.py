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


@pytest.fixture
def checkpoint_recording():
    """Builder for a small 2-D detector recording.

    Calling it with ``"sorted"`` or ``"clusterless"`` returns
    ``(model, fit_kwargs, predict_kwargs)``.
    """
    from non_local_detector import (
        Environment,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    def recording(family):
        time = np.linspace(0, 2, 61)
        position = np.column_stack((5 + 4 * np.sin(time), 5 + 4 * np.cos(time)))
        spikes = [np.array([0.1, 0.3, 0.8, 1.2, 1.9])]
        kwargs = {
            "environments": Environment(
                place_bin_size=2, position_range=((0, 10), (0, 10))
            ),
            "infer_track_interior": False,
        }
        if family == "sorted":
            model = NonLocalSortedSpikesDetector(**kwargs)
            args = {"spike_times": spikes}
        else:
            model = NonLocalClusterlessDetector(**kwargs)
            args = {
                "spike_times": spikes,
                "spike_waveform_features": [np.arange(10.0).reshape(5, 2)],
            }
        fit = dict(position_time=time, position=position, **args)
        predict = dict(
            position_time=time,
            position=position,
            time_edges=np.linspace(0, 2, 101),
            **args,
        )
        return model, fit, predict

    return recording
