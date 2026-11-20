"""KDE encoding fits place a spike between samples of unequal weight by their shares.

Occupancy puts each weighted tracking sample's exposure at that sample's
position. Where the weights of a spike's two bracketing samples differ, the
spike's weight is split between them as its time is, ``(1 - f) * w_left`` and
``f * w_right``, and the spike is one entry at the mean of the two sample
positions weighted by those shares. A spike next to an unweighted sample so
sits on its weighted sample, and the intensity past a mask edge stays near the
edge rate instead of growing with distance. Between samples of equal weight the
spike keeps its interpolated position and weight.
"""

import numpy as np
import pytest
from track_linearization import make_track_graph

from non_local_detector import Environment
from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.likelihoods.common import place_encoding_spikes
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
def test_transition_spikes_sit_at_the_share_weighted_mean_of_their_samples(
    algorithm,
):
    """One entry per spike: at a transition, at its samples' share-weighted mean."""
    position_time = np.arange(6.0)
    position = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])[:, None]
    weights = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 1.0])
    # Equal weights (interpolated), 1 -> 0.5 (between its samples), 0.5 -> 0
    # (on the left sample), 0 -> 0 (interpolated, zero weight), 0 -> 1 (on the
    # right sample).
    spike_times = np.array([0.5, 1.25, 2.5, 3.5, 4.75])
    environment = Environment(
        place_bin_size=1.0, position_range=[(0.0, 10.0)]
    ).fit_place_grid(np.array([[0.0], [10.0]]), infer_track_interior=False)

    positions, entry_weights, features, _ = _fit(
        algorithm, position_time, position, spike_times, environment, weights
    )

    # The 1 -> 0.5 spike at fraction 0.25 between the samples at 2 and 4 cm.
    left_share, right_share = (1 - 0.25) * 1.0, 0.25 * 0.5
    mean_position = (left_share * 2.0 + right_share * 4.0) / (left_share + right_share)
    np.testing.assert_allclose(positions, [1.0, mean_position, 4.0, 7.0, 10.0])
    np.testing.assert_allclose(
        entry_weights,
        [1.0, left_share + right_share, (1 - 0.5) * 0.5, 0.0, 0.75 * 1.0],
        atol=1e-7,
    )
    if features is not None:
        np.testing.assert_array_equal(features, [0, 1, 2, 3, 4])


def _share_weighted_placement(position_time, position, spike_times, weights):
    """Float64 reference of the placement rule in raw position coordinates.

    Returns
    -------
    positions : np.ndarray, shape (n_spikes, n_position_dims)
    spike_weights : np.ndarray, shape (n_spikes,)
    """
    left = np.searchsorted(position_time, spike_times, side="right") - 1
    right = left + 1
    fraction = (spike_times - position_time[left]) / (
        position_time[right] - position_time[left]
    )
    left_share = (1 - fraction) * weights[left]
    right_share = fraction * weights[right]
    total = left_share + right_share
    same = weights[left] == weights[right]
    # The total is 0 only between two unweighted samples, where spikes interpolate.
    share_fraction = np.divide(
        right_share, total, out=np.zeros_like(total), where=total > 0
    )
    toward_right = np.where(same, fraction, share_fraction)
    positions = position[left] + toward_right[:, None] * (
        position[right] - position[left]
    )
    spike_weights = np.where(same, weights[left], total)
    return positions, spike_weights


def _l_track_linear(points):
    """Linear position on the track (0, 0)-(100, 0)-(100, 100), edge spacing 0."""
    return np.where(points[:, 1] == 0.0, points[:, 0], 100.0 + points[:, 1])


@pytest.mark.parametrize("geometry", ["2-D", "track graph"])
def test_placement_in_decoding_coordinates(geometry):
    """2-D placement keeps both columns; on a track graph it is linearized."""
    position_time = np.arange(7.0)
    # One spike in each of the 1 -> 0.5, 0.5 -> 0, 0 -> 0, 0 -> 1 and 1 -> 1
    # intervals; none in the 0 -> 0 interval that turns the track's corner.
    weights = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0])
    spike_times = np.array([0.25, 1.5, 2.5, 4.6, 5.5])
    if geometry == "2-D":
        position = np.array(
            [[0, 0], [2, 1], [4, 3], [6, 2], [8, 5], [10, 4], [12, 7]], dtype=float
        )
        environment = Environment(
            place_bin_size=1.0, position_range=[(0.0, 12.0), (0.0, 7.0)]
        ).fit_place_grid(
            np.array([[0.0, 0.0], [12.0, 7.0]]), infer_track_interior=False
        )
    else:
        position = np.array(
            [[10, 0], [30, 0], [50, 0], [70, 0], [100, 20], [100, 40], [100, 70]],
            dtype=float,
        )
        edges = [(0, 1), (1, 2)]
        environment = Environment(
            track_graph=make_track_graph(
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], edges
            ),
            edge_order=edges,
            edge_spacing=0.0,
            place_bin_size=2.0,
        ).fit_place_grid()
    support, weights, _, position = prepare_encoding_support(
        position_time, position, weights
    )

    positions, spike_weights = place_encoding_spikes(
        position_time, position, spike_times, weights, environment, support
    )

    expected_positions, expected_weights = _share_weighted_placement(
        position_time, position, spike_times, weights
    )
    if geometry == "track graph":
        expected_positions = _l_track_linear(expected_positions)[:, None]
    assert positions.shape == expected_positions.shape
    np.testing.assert_allclose(positions, expected_positions, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(spike_weights, expected_weights, rtol=1e-12)


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
    endpoint half-cells make the first and last whole); every spike lies
    strictly between the first and last sample.
    """
    dt = position_time[1] - position_time[0]

    def kernel(points, samples):
        return np.exp(-0.5 * ((points[:, None] - samples[None, :]) / SIGMA) ** 2)

    exposure = weights * dt
    occupancy = kernel(centers, position[:, 0]) @ exposure
    entry_positions, entry_weights = _share_weighted_placement(
        position_time, position, spike_times, weights
    )
    spike_density = kernel(centers, entry_positions[:, 0]) @ entry_weights
    return spike_density / occupancy


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("floor", [0.0, 1e-3, 1e-6, 1e-12])
def test_intensity_past_a_mask_edge_matches_float64_rule(algorithm, floor):
    """A spatial mask crossed at video rate: intensity past the edge stays bounded.

    One spike train at a position-independent 200 Hz on a 100 cm/s run tracked
    at 30 Hz (3.3 cm steps) with weight 1 below 100 cm and ``floor`` past it.
    Interpolating the spikes past the edge made the intensity grow like
    ``exp(step * distance / position_std**2)`` (about 7,000x the true rate at
    151 cm with a zero floor).
    """
    rate = 200.0
    position_time = np.arange(round(12.0 * 30.0)) / 30.0
    phase = np.mod(100.0 * position_time, 400.0)
    position = np.where(phase <= 200.0, phase, 400.0 - phase)[:, None]
    spike_times = np.arange(0.00123, position_time[-1], 1 / rate)
    weights = np.where(position[:, 0] < 100.0, 1.0, floor)
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
