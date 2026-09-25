"""Decode time edges: validation, the detector uniformity guard, and grid helpers.

Decode bins are given by ``n_bins + 1`` edges. Every public boundary validates
them once (one-dimensional, at least two finite, strictly increasing values
whose spacing the timestamp precision can resolve). Detector entry points also
require uniform bins, because the HMM applies one transition per row; the
tolerance is a few units in the last place of the largest edge, so uniform
Unix-epoch grids pass and real irregularity is still detected.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.time_edges import (
    time_edges_from_centers,
    uniform_time_bin_width,
    validate_time_edges,
)

ORIGINS = [0.0, 1234.5678, 1.7e9, 1.75e9 + 3600.0]
WIDTHS = [0.001, 0.002, 1 / 1500]


def _grids(t0, dt, n):
    """Uniform grids as users build them."""
    i = np.arange(n + 1)
    return {
        "t0+i*dt": t0 + i * dt,
        "arange": np.arange(n + 1) * dt + t0,
        "linspace": np.linspace(t0, t0 + n * dt, n + 1),
        "cumsum": t0 + np.concatenate([[0.0], np.cumsum(np.full(n, dt))]),
        "centers": time_edges_from_centers(t0 + (i[:-1] + 0.5) * dt)
        if n > 1
        else t0 + i * dt,
    }


# --------------------------------------------------------------- validation
@pytest.mark.unit
@pytest.mark.parametrize(
    "edges",
    [np.array([]), np.array([1.0]), np.zeros((3, 2)), np.array(["a", "b"])],
    ids=["empty", "single-edge", "2d", "strings"],
)
def test_malformed_edges_are_rejected(edges):
    with pytest.raises(ValidationError):
        validate_time_edges(edges)


@pytest.mark.unit
@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, 1.0, np.inf]),
        np.array([0.0, 1.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 1.0, 3.0]),
    ],
    ids=["nan", "inf", "repeated", "decreasing"],
)
def test_invalid_edge_values_are_rejected(edges):
    with pytest.raises(DataError):
        validate_time_edges(edges)


@pytest.mark.unit
def test_single_bin_and_integer_edges_are_valid():
    np.testing.assert_array_equal(validate_time_edges(np.array([0, 1])), [0.0, 1.0])
    assert validate_time_edges(np.array([0, 1])).dtype == np.float64
    edges = np.array([1.7e9, 1.7e9 + 0.002])
    np.testing.assert_array_equal(validate_time_edges(edges), edges)


@pytest.mark.unit
def test_array_likes_are_accepted():
    """Sequences and pandas indexes (e.g. a position index) are converted."""
    pd = pytest.importorskip("pandas")
    np.testing.assert_array_equal(validate_time_edges([0.0, 0.5, 1.0]), [0, 0.5, 1])
    np.testing.assert_array_equal(
        validate_time_edges(pd.Index([0.0, 0.5, 1.0])), [0, 0.5, 1]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t0", "dt"),
    [(1.7e9, 0.002), (0.0, 0.002), (0.0, 0.02)],
    ids=["float32-epoch", "float32-2ms-1h", "float32-20ms-1h"],
)
def test_edges_that_cannot_resolve_their_bins_fail_with_a_precision_error(t0, dt):
    """float32 cannot represent the requested spacing; the error names the
    precision problem instead of reporting repeated or irregular bins."""
    edges = (t0 + np.arange(int(3600 / dt) + 1) * dt).astype(np.float32)
    with pytest.raises(DataError, match="precision"):
        validate_time_edges(edges)


# --------------------------------------------------------------- uniformity
@pytest.mark.unit
@pytest.mark.parametrize("t0", ORIGINS)
@pytest.mark.parametrize("dt", WIDTHS)
@pytest.mark.parametrize("n", [1, 10, 1000])
def test_uniform_grids_are_accepted_at_any_origin(t0, dt, n):
    for name, edges in _grids(t0, dt, n).items():
        width = uniform_time_bin_width(edges)
        assert width == pytest.approx(dt, rel=1e-3), name


@pytest.mark.unit
def test_long_epoch_grid_is_accepted():
    """1.8 million 2 ms bins at a Unix-epoch origin."""
    edges = 1.7e9 + np.arange(1_800_001) * 0.002
    assert uniform_time_bin_width(edges) == pytest.approx(0.002, rel=1e-6)


@pytest.mark.unit
def test_irregular_grids_are_rejected_with_the_transition_reason():
    ten_then_fifty = np.concatenate(
        [np.arange(0, 1, 0.01), 1 + np.arange(0, 1.0001, 0.05)]
    )
    with pytest.raises(DataError, match="transition"):
        uniform_time_bin_width(ten_then_fifty)


@pytest.mark.unit
def test_small_displacement_is_detected_at_epoch_origin():
    """One edge moved by 1e-3 of a 2 ms bin is resolvable in float64 at 1.7e9."""
    edges = 1.7e9 + np.arange(100_001) * 0.002
    edges[50_000] += 1e-3 * 0.002
    with pytest.raises(DataError):
        uniform_time_bin_width(edges)


# --------------------------------------------------------- grid construction
@pytest.mark.unit
@pytest.mark.parametrize("t0", [0.0, 1.7e9])
def test_edges_from_centers_are_centered_on_the_samples(t0):
    centers = t0 + np.arange(1000) * 0.002
    edges = time_edges_from_centers(centers)
    assert edges.shape == (1001,)
    np.testing.assert_allclose(
        edges[:-1] + 0.5 * np.diff(edges), centers, rtol=0, atol=4 * np.spacing(t0 + 2)
    )
    assert uniform_time_bin_width(edges) == pytest.approx(0.002, rel=1e-6)


@pytest.mark.unit
def test_edges_from_centers_rejects_ambiguous_input():
    with pytest.raises(ValidationError, match="duration"):
        time_edges_from_centers(np.array([1.0]))
    with pytest.raises(DataError):
        time_edges_from_centers(np.array([0.0, 1.0, 3.0]))


@pytest.mark.unit
@pytest.mark.parametrize("t0", [0.0, 1.7e9])
def test_calculate_time_edges_is_origin_independent(t0):
    """0.7 s at 500 Hz is 350 bins at any origin (the old helper gave 351 at
    1.7e9) and the grid passes the detector guard."""
    detector = NonLocalSortedSpikesDetector(sampling_frequency=500)
    edges = detector.calculate_time_edges(np.array([t0, t0 + 0.7]))
    assert edges.shape == (351,)
    assert edges[0] == t0
    assert uniform_time_bin_width(edges) == pytest.approx(0.002, rel=1e-6)


@pytest.mark.unit
def test_calculate_time_edges_rejects_partial_bins_unless_trimmed():
    detector = NonLocalSortedSpikesDetector(sampling_frequency=500)
    with pytest.raises(ValidationError, match="whole number"):
        detector.calculate_time_edges(np.array([0.0, 0.3005]))
    edges = detector.calculate_time_edges(np.array([0.0, 0.3005]), trim=True)
    assert edges.shape == (151,)
    assert edges[-1] <= 0.3005
    with pytest.raises(ValidationError):
        detector.calculate_time_edges(np.array([0.0, 0.001]), trim=True)
