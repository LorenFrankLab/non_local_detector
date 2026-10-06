"""Direct likelihoods take decode bin edges and return one row per bin.

``time_edges`` of shape ``(n_bins + 1,)`` define the decode bins: bin ``i`` is
``[time_edges[i], time_edges[i + 1])`` and the final bin also holds
``time_edges[-1]``. Every bin, including the last, can own a spike; spikes
outside ``[time_edges[0], time_edges[-1]]`` are not counted. Positions for the
local paths are evaluated at the bin centers.
"""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.likelihoods.common import (
    get_spikecount_per_time_bin,
    select_spikes_in_rows,
)
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    fit_registered_backends,
)

EDGES = np.arange(6.0)  # five bins


# ------------------------------------------------------------ event assignment
@pytest.mark.unit
def test_every_bin_including_the_last_can_own_a_spike():
    counts = get_spikecount_per_time_bin(
        np.array([0.5, 1.5, 2.5, 3.5, 4.5]), time_edges=EDGES
    )
    np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1])


@pytest.mark.unit
def test_boundaries_are_left_closed_and_the_final_edge_is_included():
    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(EDGES.copy(), time_edges=EDGES), [1, 1, 1, 1, 2]
    )


@pytest.mark.unit
def test_events_outside_the_edges_are_not_counted():
    spikes = np.array([-1e-9, -0.5, 5.0 + 1e-9, 6.0])
    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(spikes, time_edges=EDGES), [0, 0, 0, 0, 0]
    )


@pytest.mark.unit
@pytest.mark.parametrize("ascending", [True, False])
def test_any_row_partition_reproduces_the_full_counts(ascending):
    rng = np.random.default_rng(0)
    edges = np.sort(rng.uniform(0, 10, 12))
    spikes = np.concatenate([edges, rng.uniform(edges[0] - 1, edges[-1] + 1, 50)])
    spikes = np.sort(spikes) if ascending else rng.permutation(spikes)
    full = get_spikecount_per_time_bin(spikes, time_edges=edges)
    assert full.shape == (11,)
    assert full.sum() == np.sum((spikes >= edges[0]) & (spikes <= edges[-1]))
    for cuts in ([0, 4, 11], [0, 1, 10, 11], [0, 5, 6, 11]):
        parts = [
            get_spikecount_per_time_bin(spikes, time_edges=edges, row_slice=slice(a, b))
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]
        np.testing.assert_array_equal(np.concatenate(parts), full)
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            selection = select_spikes_in_rows(spikes, a, b, time_edges=edges)
            assert selection.n_rows == b - a
            assert np.all((selection.bin_ind >= 0) & (selection.bin_ind < b - a))


# ------------------------------------------------------------ all backends
@pytest.fixture(scope="module")
def encoding_data():
    rng = np.random.default_rng(3)
    position_time = np.arange(0.0, 20.0, 0.01)
    position = (50.0 + 45.0 * np.sin(2 * np.pi * position_time / 5.0))[:, None]
    environment = Environment(
        environment_name="line", place_bin_size=10.0, position_range=((0.0, 100.0),)
    ).fit_place_grid(position=position, infer_track_interior=False)
    # Unit 0 fires near 25 cm, unit 1 near 75 cm, so rates differ by position.
    spike_times, features = [], []
    for center in (25.0, 75.0):
        near = np.abs(position[:, 0] - center) < 10.0
        times = np.sort(rng.choice(position_time[near], 300, replace=False))
        spike_times.append(times + 0.001)
        features.append(rng.standard_normal((300, 2)) * 3.0 + center / 5.0)
    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": spike_times,
        "encoding_features": features,
    }


@pytest.fixture(scope="module")
def fitted(encoding_data):
    return {
        name: (predict, model, is_clusterless)
        for name, predict, model, is_clusterless in fit_registered_backends(
            encoding_data
        )
    }


def _predict(
    fitted, name, time_edges, spike_times, *, is_local, position_time, position
):
    predict, model, is_clusterless = fitted[name]
    rng = np.random.default_rng(0)
    args = [position_time, position, spike_times]
    if is_clusterless:
        args.append([rng.standard_normal((len(t), 2)) + 5.0 for t in spike_times])
    return np.asarray(predict(*args, time_edges=time_edges, **model, is_local=is_local))


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("is_local", [False, True])
def test_backend_returns_one_row_per_bin_and_owns_the_final_edge(
    fitted, encoding_data, name, is_local
):
    edges = np.linspace(2.0, 3.0, 11)
    position_args = {
        "position_time": encoding_data["position_time"],
        "position": encoding_data["position"],
    }
    empty = _predict(
        fitted,
        name,
        edges,
        [np.array([]), np.array([])],
        is_local=is_local,
        **position_args,
    )
    assert empty.shape[0] == 10
    at_final_edge = _predict(
        fitted,
        name,
        edges,
        [np.array([3.0]), np.array([])],
        is_local=is_local,
        **position_args,
    )
    changed = np.any(at_final_edge != empty, axis=1)
    np.testing.assert_array_equal(changed, [False] * 9 + [True])
    outside = _predict(
        fitted,
        name,
        edges,
        [np.array([1.999, 3.001]), np.array([])],
        is_local=is_local,
        **position_args,
    )
    np.testing.assert_array_equal(outside, empty)


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize(
    "edges",
    [
        np.array([2.0, 2.1, 2.1, 2.3]),
        np.array([2.0, 2.2, 2.1, 2.3]),
        np.array([2.0, np.nan, 2.3]),
        np.array([2.0]),
        np.array([[2.0, 2.1], [2.2, 2.3]]),
    ],
    ids=["repeated", "decreasing", "nan", "single-edge", "2d"],
)
def test_backend_rejects_invalid_edges(fitted, encoding_data, name, edges):
    with pytest.raises((DataError, ValidationError)):
        _predict(
            fitted,
            name,
            edges,
            [np.array([2.05]), np.array([])],
            is_local=False,
            position_time=encoding_data["position_time"],
            position=encoding_data["position"],
        )


@pytest.mark.unit
def test_no_spike_returns_one_row_per_bin_and_owns_the_final_edge():
    edges = np.linspace(0.0, 1.0, 11)
    empty = np.asarray(
        predict_no_spike_log_likelihood([np.array([])], time_edges=edges)
    )
    final = np.asarray(
        predict_no_spike_log_likelihood([np.array([1.0])], time_edges=edges)
    )
    assert empty.shape[0] == 10
    np.testing.assert_array_equal(
        np.any(final != empty, axis=tuple(range(1, final.ndim))), [False] * 9 + [True]
    )
    with pytest.raises(DataError):
        predict_no_spike_log_likelihood(
            [np.array([])], time_edges=np.array([0.0, 1.0, 0.5])
        )


@pytest.mark.integration
@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize(
    ("edges", "expected_position"),
    [
        # Left edge before the jump, center after it.
        (np.array([4.86, 4.96, 5.06, 5.16]), [25.0, 75.0, 75.0]),
        # Center before the jump, right edge after it.
        (np.array([4.81, 4.91, 5.01, 5.11]), [25.0, 25.0, 75.0]),
    ],
    ids=["left-edge-before-jump", "right-edge-after-jump"],
)
def test_local_position_is_evaluated_at_bin_centers(
    fitted, encoding_data, name, edges, expected_position
):
    """With no decoding spikes the local row equals the non-local row at the
    animal's bin; the animal jumps from 25 to 75 cm at t = 5 s."""
    position_time = np.arange(0.0, 10.0, 0.001)
    position = np.where(position_time < 5.0, 25.0, 75.0)[:, None]
    no_spikes = [np.array([]), np.array([])]
    local = _predict(
        fitted,
        name,
        edges,
        no_spikes,
        is_local=True,
        position_time=position_time,
        position=position,
    )[:, 0]
    non_local = _predict(
        fitted,
        name,
        edges,
        no_spikes,
        is_local=False,
        position_time=position_time,
        position=position,
    )
    interior_centers = np.asarray(encoding_data["environment"].place_bin_centers_)[:, 0]
    low = int(np.argmin(np.abs(interior_centers - 25.0)))
    high = int(np.argmin(np.abs(interior_centers - 75.0)))
    assert not np.isclose(non_local[0, low], non_local[0, high], rtol=1e-3)
    expected = [
        non_local[i, low if p == 25.0 else high]
        for i, p in enumerate(expected_position)
    ]
    np.testing.assert_allclose(local, expected, rtol=1e-3)
