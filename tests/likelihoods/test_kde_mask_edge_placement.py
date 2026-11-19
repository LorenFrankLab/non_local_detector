"""KDE encoding fits place spikes by the occupancy's rule at a weight transition.

Occupancy puts each weighted tracking sample's exposure at that sample's
position. Where the weights of a spike's two bracketing samples differ, the
spike's share of each sample's weight sits at that sample, so spikes and
occupancy follow one rule and the intensity past a mask edge stays near the
edge rate instead of growing with distance. Between samples of equal weight the
spike keeps its interpolated position and weight.
"""

import numpy as np
import pytest

from non_local_detector import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)

pytestmark = pytest.mark.unit

SIGMA = float(np.sqrt(12.5))


def _fit(algorithm, position_time, position, spike_times, environment, weights):
    """Fit one electrode / neuron and return its stored entries and intensity.

    Returns
    -------
    positions : np.ndarray, shape (n_entries,)
        Stored encoding positions, in the stored dtype.
    entry_weights : np.ndarray, shape (n_entries,)
        Stored encoding weights, in the stored dtype.
    features : np.ndarray, shape (n_entries,) or None
        First mark column of each entry (clusterless only).
    intensity : np.ndarray, shape (n_interior_bins,)
        Fitted ground-process intensity / place field in Hz.
    """
    if algorithm == "clusterless_kde":
        # The mark of each spike is its index, so entries map back to spikes.
        features = np.stack(
            [np.arange(spike_times.size, dtype=float), np.zeros(spike_times.size)],
            axis=1,
        )
        model = fit_clusterless_kde_encoding_model(
            position_time,
            position,
            [spike_times],
            [features],
            environment,
            weights,
            position_std=SIGMA,
            disable_progress_bar=True,
        )
        return (
            np.asarray(model["encoding_positions"][0])[:, 0],
            np.asarray(model["encoding_weights"][0]),
            np.asarray(model["encoding_spike_waveform_features"][0], float)[:, 0],
            np.asarray(model["summed_ground_process_intensity"], float),
        )
    model = fit_sorted_spikes_kde_encoding_model(
        position_time,
        position,
        [spike_times],
        environment,
        weights,
        position_std=SIGMA,
        disable_progress_bar=True,
    )
    (marginal,) = model["marginal_models"]
    interior = np.asarray(model["is_track_interior"])
    return (
        np.asarray(marginal.samples_)[:, 0],
        np.asarray(marginal.weights_),
        None,
        np.asarray(model["place_fields"], float)[0, interior],
    )


ALGORITHMS = ["clusterless_kde", "sorted_spikes_kde"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_spike_shares_sit_on_their_samples_at_weight_transitions(algorithm):
    """Each transition spike's shares sit on its bracketing samples; others interpolate."""
    position_time = np.arange(6.0)
    position = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])[:, None]
    weights = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 1.0])
    # Equal weights (interpolated), 1 -> 0.5 (two shares), 0.5 -> 0 (left
    # share), 0 -> 0 (interpolated, zero weight), 0 -> 1 (right share).
    spike_times = np.array([0.5, 1.25, 2.5, 3.5, 4.75])
    environment = Environment(
        place_bin_size=1.0, position_range=[(0.0, 10.0)]
    ).fit_place_grid(np.array([[0.0], [10.0]]), infer_track_interior=False)

    positions, entry_weights, features, _ = _fit(
        algorithm, position_time, position, spike_times, environment, weights
    )

    np.testing.assert_allclose(positions, [1.0, 2.0, 4.0, 4.0, 7.0, 10.0])
    np.testing.assert_allclose(
        entry_weights, [1.0, 0.75, 0.25 * 0.5, 0.5 * 0.5, 0.0, 0.75], atol=1e-7
    )
    if features is not None:
        np.testing.assert_array_equal(features, [0, 1, 1, 2, 3, 4])


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("weights", ["omitted", "all equal"])
def test_spikes_interpolate_without_weight_transitions(algorithm, weights):
    """With no weight transition each spike keeps its interpolated position and weight."""
    position_time = np.arange(0.0, 10.0, 1 / 30)
    position = (50 + 40 * np.sin(position_time))[:, None]
    spike_times = np.arange(0.0123, 9.95, 0.037)
    sample_weights = None if weights == "omitted" else np.full(position_time.size, 0.5)
    environment = Environment(
        place_bin_size=2.0, position_range=[(0.0, 100.0)]
    ).fit_place_grid(np.array([[0.0], [100.0]]), infer_track_interior=False)

    positions, entry_weights, features, _ = _fit(
        algorithm, position_time, position, spike_times, environment, sample_weights
    )

    expected_weight = 1.0 if weights == "omitted" else 0.5
    np.testing.assert_array_equal(
        positions,
        np.interp(spike_times, position_time, position[:, 0]).astype(positions.dtype),
    )
    np.testing.assert_array_equal(entry_weights, expected_weight)
    if features is not None:
        np.testing.assert_array_equal(features, np.arange(spike_times.size))


def _float64_intensity(position_time, position, spike_times, weights, centers):
    """Weighted KDE intensity in float64 with the transition placement rule.

    Uniform samples, each carrying one sampling interval of exposure (the
    endpoint half-cells make the first and last whole).
    """
    dt = position_time[1] - position_time[0]
    x = position[:, 0]

    def kernel(points, samples):
        return np.exp(-0.5 * ((points[:, None] - samples[None, :]) / SIGMA) ** 2)

    exposure = weights * dt
    occupancy = kernel(centers, x) @ exposure

    left = np.clip(
        np.searchsorted(position_time, spike_times, side="right") - 1,
        0,
        position_time.size - 1,
    )
    right = np.minimum(left + 1, position_time.size - 1)
    fraction = np.where(right > left, (spike_times - position_time[left]) / dt, 0.0)
    same = weights[left] == weights[right]
    entry_positions = np.concatenate(
        [
            np.interp(spike_times[same], position_time, x),
            x[left[~same]],
            x[right[~same]],
        ]
    )
    entry_weights = np.concatenate(
        [
            weights[left[same]],
            (1 - fraction[~same]) * weights[left[~same]],
            fraction[~same] * weights[right[~same]],
        ]
    )
    spike_density = kernel(centers, entry_positions) @ entry_weights
    return spike_density / occupancy


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_intensity_past_a_mask_edge_matches_float64_rule(algorithm):
    """A spatial mask crossed at video rate: intensity past the edge stays bounded.

    One spike train at a position-independent 200 Hz on a 100 cm/s run tracked
    at 30 Hz (3.3 cm steps) with weights ``position < 100``. Interpolating the
    spikes past the edge made the intensity grow like
    ``exp(step * distance / position_std**2)`` (about 7,000x the true rate at
    151 cm).
    """
    rate = 200.0
    position_time = np.arange(round(12.0 * 30.0)) / 30.0
    phase = np.mod(100.0 * position_time, 400.0)
    position = np.where(phase <= 200.0, phase, 400.0 - phase)[:, None]
    spike_times = np.arange(0.00123, position_time[-1], 1 / rate)
    weights = (position[:, 0] < 100.0).astype(float)
    environment = Environment(
        place_bin_size=2.0, position_range=[(0.0, 200.0)]
    ).fit_place_grid(np.array([[0.0], [200.0]]), infer_track_interior=False)
    centers = np.asarray(environment.place_bin_centers_)[:, 0]

    _, _, _, intensity = _fit(
        algorithm, position_time, position, spike_times, environment, weights
    )

    expected = _float64_intensity(
        position_time, position, spike_times, weights, centers
    )
    np.testing.assert_allclose(intensity, expected, rtol=1e-4)
    last_inside = np.flatnonzero(centers < 100.0)[-1]
    assert intensity[centers > 100.0].max() <= 1.1 * intensity[last_inside]
