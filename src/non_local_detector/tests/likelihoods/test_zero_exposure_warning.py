"""Every encoding fit warns when it has zero training exposure.

Exposure is what a position sample contributes to occupancy: its weight, zero
for a NaN position, times the time its interpolation basis spends inside the
encoding support. With none, rates are unidentified and every backend returns
floor rates (sorted spikes) or zero-rate electrodes (clusterless). That is
usually a mask or label mistake, so each backend must say so rather than
return a floor model silently.
"""

import warnings

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.likelihoods.common import RATE_EPS_HZ

pytestmark = pytest.mark.unit

MESSAGE = "zero training exposure"
N_SAMPLES = 201
ALGORITHMS = [*_SORTED_SPIKES_ALGORITHMS, *_CLUSTERLESS_ALGORITHMS]


@pytest.fixture(scope="module")
def run():
    """Back-and-forth motion on [0, 40] cm sampled at 20 Hz, two units."""
    time = np.arange(N_SAMPLES) / 20.0
    position = (20.0 + 18.0 * np.sin(2.0 * np.pi * time / 5.0))[:, None]
    environment = Environment(
        place_bin_size=2.0, position_range=((0.0, 40.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    spike_times = [time[5:-5:7] + 0.01, time[3:-5:11] + 0.02]
    rng = np.random.default_rng(0)
    marks = [rng.normal(size=(len(times), 2)) for times in spike_times]
    return time, position, environment, spike_times, marks


def _fit(algorithm, run, weights, position=None, **support_kwargs):
    time, run_position, environment, spike_times, marks = run
    position = run_position if position is None else position
    kwargs = {"disable_progress_bar": True, **support_kwargs}
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        fit = _SORTED_SPIKES_ALGORITHMS[algorithm][0]
        if algorithm == "sorted_spikes_glm":
            kwargs.update(
                place_bin_edges=environment.place_bin_edges_,
                edges=environment.edges_,
                is_track_interior=environment.is_track_interior_,
                is_track_boundary=environment.is_track_boundary_,
            )
        return fit(time, position, spike_times, environment, weights=weights, **kwargs)
    fit = _CLUSTERLESS_ALGORITHMS[algorithm][0]
    if algorithm == "clusterless_gmm":
        kwargs.update(gmm_components_occupancy=2, gmm_components_joint=2)
    return fit(
        time, position, spike_times, marks, environment, weights=weights, **kwargs
    )


def _fit_recording_warnings(algorithm, run, weights, **fit_kwargs):
    """Fit, returning the model and its zero-exposure warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _fit(algorithm, run, weights, **fit_kwargs)
    return model, [w for w in caught if MESSAGE in str(w.message)]


def _assert_floor_model(algorithm, model):
    """The consequence the warning states: floor rates or zero-rate electrodes."""
    if algorithm in _SORTED_SPIKES_ALGORITHMS:
        interior = np.asarray(model["is_track_interior"]).ravel()
        place_fields = np.asarray(model["place_fields"])[:, interior]
        np.testing.assert_allclose(place_fields, RATE_EPS_HZ, rtol=1e-5)
    else:
        np.testing.assert_array_equal(np.asarray(model["mean_rates"]), 0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_all_zero_weights_warn(algorithm, run):
    model, caught = _fit_recording_warnings(algorithm, run, np.zeros(N_SAMPLES))

    assert len(caught) == 1
    _assert_floor_model(algorithm, model)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_weights_only_on_nan_positions_warn(algorithm, run):
    """Positive weights on NaN-position samples carry no exposure."""
    position = run[1].copy()
    position[100:120] = np.nan
    weights = np.zeros(N_SAMPLES)
    weights[100:120] = 1.0

    model, caught = _fit_recording_warnings(algorithm, run, weights, position=position)

    assert len(caught) == 1
    _assert_floor_model(algorithm, model)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_weights_only_outside_encoding_time_range_warn(algorithm, run):
    """Positive weights on samples whose basis lies outside the support."""
    time = run[0]
    weights = (time > 8.0).astype(float)

    model, caught = _fit_recording_warnings(
        algorithm, run, weights, encoding_time_range=(0.0, 5.0)
    )

    assert len(caught) == 1
    _assert_floor_model(algorithm, model)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_positive_exposure_does_not_warn(algorithm, run):
    _, caught = _fit_recording_warnings(algorithm, run, np.ones(N_SAMPLES))

    assert not caught
