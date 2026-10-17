"""The No-Spike likelihood's progress bar follows the caller's setting.

Detectors decode in chunks, so a bar that cannot be disabled prints once per
chunk even when every other likelihood was asked to stay quiet.
"""

import numpy as np
import pytest

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.likelihoods import predict_no_spike_log_likelihood

SAMPLING_FREQUENCY = 100.0
BAR_DESC = "No Spike Likelihood"


@pytest.fixture
def run_data():
    """A track sweep with two place cells and their one-feature marks."""
    time = np.arange(0.0, 20.0, 1.0 / SAMPLING_FREQUENCY)
    position = (50.0 + 40.0 * np.sin(2.0 * np.pi * time / 6.0))[:, np.newaxis]
    rng = np.random.default_rng(0)
    spike_times = []
    for center in (30.0, 70.0):
        rate = 10.0 * np.exp(-0.5 * ((position[:, 0] - center) / 8.0) ** 2) + 0.5
        fired = rng.random(time.shape[0]) < rate / SAMPLING_FREQUENCY
        spike_times.append(
            time[fired] + rng.uniform(0.0, 1.0 / SAMPLING_FREQUENCY, int(fired.sum()))
        )
    marks = [rng.normal(100.0, 10.0, (len(times), 1)) for times in spike_times]
    return time, position, spike_times, marks


@pytest.mark.unit
@pytest.mark.parametrize("disable", [True, False])
def test_direct_call_honors_disable_progress_bar(capfd, run_data, disable):
    time, _, spike_times, _ = run_data
    predict_no_spike_log_likelihood(
        spike_times,
        time_edges=time_edges_from_centers(time),
        disable_progress_bar=disable,
    )
    assert (BAR_DESC in capfd.readouterr().err) is not disable


@pytest.mark.integration
@pytest.mark.parametrize("clusterless", [False, True])
def test_detector_setting_disables_no_spike_bar(capfd, run_data, clusterless):
    """``disable_progress_bar`` in the algorithm params silences No-Spike too."""
    time, position, spike_times, marks = run_data
    params = {"disable_progress_bar": True}
    if clusterless:
        detector = NonLocalClusterlessDetector(
            clusterless_algorithm_params={**params, "waveform_std": 10.0}
        ).fit(time, position, spike_times, marks)
        detector.compute_log_likelihood(
            time,
            position,
            spike_times,
            marks,
            time_edges=time_edges_from_centers(time),
        )
    else:
        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm_params=params
        ).fit(time, position, spike_times)
        detector.compute_log_likelihood(
            time,
            position,
            spike_times,
            time_edges=time_edges_from_centers(time),
        )
    assert BAR_DESC not in capfd.readouterr().err
