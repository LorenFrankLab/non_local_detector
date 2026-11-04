"""Small simulated recordings and detectors for the saved-model format tests."""

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector import time_edges_from_centers
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

SORTED_ALGORITHMS = sorted(_SORTED_SPIKES_ALGORITHMS)
CLUSTERLESS_ALGORITHMS = sorted(_CLUSTERLESS_ALGORITHMS)
SORTED_CLASSES = (
    nld.NonLocalSortedSpikesDetector,
    nld.ContFragSortedSpikesClassifier,
    nld.NoSpikeContFragSortedSpikesClassifier,
    nld.MultiEnvironmentSortedSpikesClassifier,
    nld.SortedSpikesDecoder,
)
CLUSTERLESS_CLASSES = (
    nld.NonLocalClusterlessDetector,
    nld.ContFragClusterlessClassifier,
    nld.NoSpikeContFragClusterlessClassifier,
    nld.MultiEnvironmentClusterlessClassifier,
    nld.ClusterlessDecoder,
)
N_DECODE = 300


@pytest.fixture(scope="session")
def sorted_recording():
    """Three place cells on a linear track: (time, position, spike_times)."""
    _, position, spike_times, time, _, _, _, _ = make_simulated_data(
        seed=0, n_neurons=3
    )
    n = 2_000
    return time[:n], position[:n], [st[st <= time[n - 1]] for st in spike_times]


@pytest.fixture(scope="session")
def clusterless_recording():
    """Two tetrodes on a linear track: (time, position, spike_times, features)."""
    sim = make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.arange(0, 80, 20), n_runs=1, seed=0
    )
    return sim.position_time, sim.position, sim.spike_times, sim.spike_waveform_features


def algorithm_params(algorithm):
    """Fit parameters that keep the small fits quick and quiet."""
    params = {"disable_progress_bar": True}
    if algorithm == "sorted_spikes_mrf":
        params["penalty"] = 1.0
    if algorithm == "clusterless_gmm":
        params.update(gmm_components_occupancy=4, gmm_components_joint=4)
    return params


def make_detector(cls, algorithm, **kwargs):
    """An unfitted ``cls`` with ``algorithm`` (sorted or clusterless)."""
    key = "clusterless" if algorithm.startswith("clusterless") else "sorted_spikes"
    kwargs[f"{key}_algorithm"] = algorithm
    kwargs[f"{key}_algorithm_params"] = algorithm_params(algorithm)
    return cls(**kwargs)


def recording_args(detector, sorted_recording, clusterless_recording):
    """``(time, position, spike_args, fit_kwargs)`` for fitting ``detector``."""
    if hasattr(detector, "clusterless_algorithm"):
        time, position, spike_times, features = clusterless_recording
        spikes = {"spike_times": spike_times, "spike_waveform_features": features}
    else:
        time, position, spike_times = sorted_recording
        spikes = {"spike_times": spike_times}
    fit_kwargs = {}
    names = [env.environment_name for env in detector.environments]
    if len(names) > 1:
        half = len(time) // 2
        fit_kwargs["environment_labels"] = np.where(
            np.arange(len(time)) < half, names[0], names[1]
        )
    return time, position, spikes, fit_kwargs


def decode(detector, time, position, spikes, **kwargs):
    """The acausal posterior and state probabilities of the first bins."""
    results = detector.predict(
        time_edges=time_edges_from_centers(time[:N_DECODE]),
        position_time=time,
        position=position,
        **spikes,
        **kwargs,
    )
    return (
        results.acausal_posterior.values,
        results.acausal_state_probabilities.values,
    )
