"""Detector entry points decode on explicit bin edges.

``predict``, ``most_likely_sequence`` and ``estimate_parameters`` take
keyword-only ``time_edges`` of shape ``(n_bins + 1,)`` and return one row per
bin at the bin centers. The edges must be valid and uniform (the HMM applies
one transition per bin) and are checked before any fitting or stored state
changes. Bins affected by a NaN position sample are decoded as missing, EM
encoding weights are interpolated from the bin centers onto the position
samples, and the bin width learned by ``estimate_parameters`` is enforced when
the fitted transitions are reused.
"""

import numpy as np
import pytest

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.tests.models.test_failed_calls_preserve_model import (
    _assert_unchanged,
    _snapshot,
)

FAMILIES = ["sorted", "clusterless"]


@pytest.fixture
def family(request, sorted_sim, clusterless_sim):
    """``(detector class, position_time, fit args, spike args)``."""
    if request.param == "sorted":
        time, position, spike_times = sorted_sim
        return (
            NonLocalSortedSpikesDetector,
            time,
            {"position_time": time, "position": position, "spike_times": spike_times},
            {"spike_times": spike_times},
        )
    sim = clusterless_sim
    spikes = {
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
    }
    return (
        NonLocalClusterlessDetector,
        sim.position_time,
        {"position_time": sim.position_time, "position": sim.position, **spikes},
        spikes,
    )


def _fitted(family):
    detector_cls, _, fit_args, _ = family
    detector = detector_cls()
    detector.fit(**fit_args)
    return detector


def _decode_args(family, time_edges, **kwargs):
    _, _, fit_args, spike_args = family
    return {
        **spike_args,
        "time_edges": time_edges,
        "position": fit_args["position"],
        "position_time": fit_args["position_time"],
        **kwargs,
    }


def _coarse_edges(position_time, factor=4, n_bins=200):
    """Uniform bins ``factor`` samples wide, offset from the position samples."""
    dt = np.median(np.diff(position_time))
    return position_time[0] + 0.3 * dt + np.arange(n_bins + 1) * factor * dt


# ----------------------------------------------------------------- rows
@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_predict_returns_one_row_per_bin_at_the_bin_centers(family):
    detector = _fitted(family)
    edges = _coarse_edges(family[1])
    results = detector.predict(**_decode_args(family, edges))
    centers = edges[:-1] + 0.5 * np.diff(edges)
    assert results.acausal_posterior.shape[0] == 200
    np.testing.assert_allclose(results.time.values, centers, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        results.acausal_state_probabilities.sum("states"), 1.0, atol=1e-5
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_most_likely_sequence_returns_one_row_per_bin(family):
    detector = _fitted(family)
    edges = _coarse_edges(family[1])
    sequence = detector.most_likely_sequence(**_decode_args(family, edges))
    assert sequence.shape[0] == 200
    np.testing.assert_allclose(
        sequence.index.values, edges[:-1] + 0.5 * np.diff(edges), atol=1e-12
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_sample_centered_edges_keep_one_row_per_position_sample(family):
    """``time_edges_from_centers`` reproduces the one-row-per-sample grid."""
    detector = _fitted(family)
    position_time = family[1][:300]
    results = detector.predict(
        **_decode_args(family, time_edges_from_centers(position_time))
    )
    np.testing.assert_allclose(results.time.values, position_time, atol=1e-12)


# ----------------------------------------------------------------- API
@pytest.mark.unit
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_old_time_argument_is_rejected(family):
    detector = _fitted(family)
    args = _decode_args(family, _coarse_edges(family[1]))
    edges = args.pop("time_edges")
    with pytest.raises(TypeError):
        detector.predict(**args, time=edges)
    spike_args = family[3]
    with pytest.raises(TypeError):
        detector.predict(*spike_args.values(), edges)


# ----------------------------------------------------------------- validation
def _bad_grids(position_time):
    edges = _coarse_edges(position_time)
    repeated = edges.copy()
    repeated[10] = repeated[9]
    nan = edges.copy()
    nan[5] = np.nan
    nonuniform = np.concatenate(
        [edges[:100], edges[99] + 5 * np.cumsum(np.diff(edges[:101]))]
    )
    return {
        "repeated": repeated,
        "decreasing": edges[::-1].copy(),
        "nan": nan,
        "nonuniform": nonuniform,
        "single-edge": edges[:1],
    }


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
@pytest.mark.parametrize(
    "grid", ["repeated", "decreasing", "nan", "nonuniform", "single-edge"]
)
def test_invalid_edges_fail_before_state_changes(family, grid):
    _, position_time, fit_args, _ = family
    bad = _bad_grids(position_time)[grid]
    detector = _fitted(family)
    snapshot = _snapshot(detector)
    with pytest.raises((DataError, ValidationError)):
        detector.predict(**_decode_args(family, bad))
    _assert_unchanged(detector, snapshot)
    with pytest.raises((DataError, ValidationError)):
        detector.most_likely_sequence(**_decode_args(family, bad))
    _assert_unchanged(detector, snapshot)
    with pytest.raises((DataError, ValidationError)):
        detector.estimate_parameters(**fit_args, time_edges=bad, max_iter=1)
    _assert_unchanged(detector, snapshot)


@pytest.mark.unit
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_is_missing_must_have_one_entry_per_bin(family):
    detector = _fitted(family)
    edges = _coarse_edges(family[1])
    with pytest.raises(ValidationError, match="is_missing"):
        detector.predict(**_decode_args(family, edges, is_missing=np.zeros(201, bool)))


# ----------------------------------------------------------------- missing
def _expected_missing(edges, position_time, is_nan):
    """Bins owning any time at which interpolation reads a NaN sample.

    Linear interpolation (``scipy.interpolate.interpn``, extrapolating from
    the end intervals) reads sample ``k`` at times in ``[t[k - 1], t[k + 1])``,
    reaching to infinity when ``k`` is in an end interval. Bin ``i`` owns
    ``[edges[i], edges[i + 1])``, and the final bin also owns ``edges[-1]``.
    """
    n = position_time.size
    missing = np.zeros(edges.size - 1, bool)
    for k in np.flatnonzero(is_nan):
        lo = position_time[k - 1] if k > 1 else -np.inf
        hi = position_time[k + 1] if k < n - 2 else np.inf
        for i in range(edges.size - 1):
            closed_end = (
                edges[i + 1] >= lo if i == edges.size - 2 else edges[i + 1] > lo
            )
            if edges[i] < hi and closed_end:
                missing[i] = True
    return missing


@pytest.mark.unit
def test_interpolation_reads_a_nan_sample_on_a_half_open_span():
    """Pins the convention the missing-bin oracle relies on: at the lower
    neighbour's time the NaN sample still enters with weight zero, at the
    upper neighbour's time it no longer does."""
    import scipy.interpolate

    position_time = np.arange(10.0)
    position = np.arange(10.0)[:, None]
    position[5] = np.nan
    at = np.array([3.9, 4.0, 4.5, 5.9, 6.0])
    read = scipy.interpolate.interpn(
        (position_time,), position, at, bounds_error=False, fill_value=None
    )[:, 0]
    np.testing.assert_array_equal(np.isfinite(read), [True, False, False, False, True])


@pytest.mark.unit
def test_missing_bins_cover_every_time_in_the_bin():
    """Local likelihoods interpolate position at spike times, not only at bin
    centers, so a bin is missing when any time it owns reads a NaN sample. On a
    sample-centered grid with one NaN sample ``k`` that is bins ``k - 1`` to
    ``k + 1``: bin ``k + 1`` has a finite center but owns times below
    ``t[k + 1]`` that read NaN."""
    from non_local_detector.models.base import _missing_bins
    from non_local_detector.time_edges import time_edges_from_centers

    position_time = np.arange(10.0)
    position = np.arange(10.0)[:, None]
    position[5] = np.nan
    edges = time_edges_from_centers(position_time)
    missing = _missing_bins(edges, None, position_time, position)
    np.testing.assert_array_equal(np.flatnonzero(missing), [4, 5, 6])
    np.testing.assert_array_equal(
        missing, _expected_missing(edges, position_time, ~np.isfinite(position[:, 0]))
    )


@pytest.mark.unit
def test_missing_bins_include_a_nan_read_at_the_closed_final_edge():
    """The final bin also owns ``edges[-1]``; a NaN read exactly there marks it."""
    from non_local_detector.models.base import _missing_bins

    position_time = np.arange(10.0)
    position = np.arange(10.0)[:, None]
    position[6] = np.nan  # read on [5, 7)
    edges = np.array([1.0, 3.0, 5.0])
    np.testing.assert_array_equal(
        _missing_bins(edges, None, position_time, position), [False, True]
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", ["clusterless"], indirect=True)
def test_spike_with_nan_position_in_a_finite_center_bin_is_missing(family):
    """A decoding spike at ``t[k] + 0.75 dt`` with ``position[k]`` NaN reads a
    NaN position although its bin's center ``t[k + 1]`` reads a finite one.
    The bin must be missing, or the local likelihood turns the posterior NaN."""
    detector = _fitted(family)
    position_time = family[1][:300]
    position = family[2]["position"].astype(float)[:300].copy()
    position = position if position.ndim > 1 else position[:, None]
    k = 50
    position[k] = np.nan
    dt = position_time[1] - position_time[0]
    spike_times = [s.copy() for s in family[3]["spike_times"]]
    features = [f.copy() for f in family[3]["spike_waveform_features"]]
    spike_times[0] = np.sort(np.append(spike_times[0], position_time[k] + 0.75 * dt))
    features[0] = np.vstack([features[0], features[0][:1]])
    edges = time_edges_from_centers(position_time)
    results = detector.predict(
        **_decode_args(
            family,
            edges,
            position=position,
            position_time=position_time,
            spike_times=spike_times,
            spike_waveform_features=features,
        ),
        return_outputs="log_likelihood",
    )
    is_zero_row = np.all(results.log_likelihood.values == 0.0, axis=1)
    np.testing.assert_array_equal(np.flatnonzero(is_zero_row), [k - 1, k, k + 1])
    assert np.all(np.isfinite(results.acausal_posterior.values))


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_bins_touched_by_nan_position_are_missing(family):
    detector = _fitted(family)
    position_time = family[1]
    position = family[2]["position"].astype(float).copy()
    position = position if position.ndim > 1 else position[:, None]
    is_nan = np.zeros(position_time.size, bool)
    is_nan[[40, 41, 42, 301, 555]] = True
    position[is_nan] = np.nan
    edges = _coarse_edges(position_time)
    results = detector.predict(
        **_decode_args(family, edges, position=position),
        return_outputs="log_likelihood",
    )
    expected = _expected_missing(edges, position_time, is_nan)
    assert 0 < expected.sum() < expected.size
    is_zero_row = np.all(results.log_likelihood.values == 0.0, axis=1)
    np.testing.assert_array_equal(is_zero_row, expected)
    assert np.all(np.isfinite(results.acausal_posterior.values))


# ----------------------------------------------------------------- EM
@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_em_encoding_weights_are_interpolated_from_bin_centers(family, monkeypatch):
    """Equal bin and sample counts do not mean equal coordinates: edges offset
    by half a sample put the bin centers one sample later, and the weights are
    still interpolated (samples before the first edge get none)."""
    detector_cls, position_time, fit_args, _ = family
    dt = np.median(np.diff(position_time))
    edges = position_time[0] + 0.5 * dt + np.arange(position_time.size + 1) * dt
    detector = detector_cls()
    recorded = {"probabilities": [], "weights": []}
    original_predict = detector._predict
    original_fit_encoding = detector.fit_encoding_model

    def spy_predict(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        recorded["probabilities"].append(np.asarray(result[1]))
        return result

    def spy_fit_encoding(*args, weights=None, **kwargs):
        if weights is not None:
            recorded["weights"].append(np.asarray(weights))
        return original_fit_encoding(*args, weights=weights, **kwargs)

    monkeypatch.setattr(detector, "_predict", spy_predict)
    monkeypatch.setattr(detector, "fit_encoding_model", spy_fit_encoding)
    detector.estimate_parameters(
        **fit_args,
        time_edges=edges,
        max_iter=1,
        min_encoding_local_mass=0.0,
        min_encoding_local_ess=0.0,
    )
    local = detector.state_names.index("Local")
    centers = edges[:-1] + 0.5 * np.diff(edges)
    inside = (position_time >= edges[0]) & (position_time <= edges[-1])
    expected = np.where(
        inside,
        np.interp(position_time, centers, recorded["probabilities"][0][:, local]),
        0.0,
    )
    assert not inside[0] and inside[1:].all()
    np.testing.assert_allclose(recorded["weights"][0], expected, rtol=1e-6)


# ----------------------------------------------------------------- transitions
@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_learned_transitions_require_the_same_bin_width(family):
    detector_cls, position_time, fit_args, _ = family
    detector = detector_cls()
    edges = _coarse_edges(position_time, factor=2)
    detector.estimate_parameters(**fit_args, time_edges=edges, max_iter=1)
    width = float(np.mean(np.diff(edges)))
    assert detector.transition_time_bin_width_ == pytest.approx(width, rel=1e-9)

    detector.predict(**_decode_args(family, edges))
    wider = _coarse_edges(position_time, factor=4)
    with pytest.raises(ValidationError, match="bin width"):
        detector.predict(**_decode_args(family, wider))
    with pytest.raises(ValidationError, match="bin width"):
        detector.most_likely_sequence(**_decode_args(family, wider))

    # A plain fit replaces the learned transitions and the recorded width.
    detector.fit(**fit_args)
    assert detector.transition_time_bin_width_ is None
    detector.predict(**_decode_args(family, wider))


# ----------------------------------------------------------------- callbacks
@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
def test_unmarked_callback_requires_row_slices_for_chunking(family, monkeypatch):
    """Full-grid overrides work; chunking needs explicit global row ownership."""
    detector = _fitted(family)
    position_time = family[1]
    edges = _coarse_edges(position_time)
    chunk_starts = [67, 134]  # np.array_split(range(200), 3)
    spike_times = [
        np.sort(np.concatenate([s, edges[chunk_starts]]))
        for s in family[3]["spike_times"]
    ]
    args = _decode_args(family, edges, spike_times=spike_times)
    reference = detector.predict(**args, return_outputs="log_likelihood")

    original = detector.compute_log_likelihood
    received = []

    def unmarked(time_edges, *args, is_missing=None):
        received.append(len(time_edges))
        return original(*args, time_edges=time_edges, is_missing=is_missing)

    monkeypatch.setattr(detector, "compute_log_likelihood", unmarked)
    full = detector.predict(**args, n_chunks=1, return_outputs="log_likelihood")
    np.testing.assert_allclose(
        full.log_likelihood.values, reference.log_likelihood.values, rtol=1e-6
    )
    with pytest.raises(ValidationError, match="row_slice_aware"):
        detector.predict(**args, n_chunks=3, cache_likelihood=False)
    assert received == [len(edges)]


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_chunked_predict_matches_unchunked(family):
    detector = _fitted(family)
    args = _decode_args(
        family, _coarse_edges(family[1]), return_outputs="log_likelihood"
    )
    reference = detector.predict(**args)
    chunked = detector.predict(**args, n_chunks=4, cache_likelihood=False)
    np.testing.assert_allclose(
        chunked.log_likelihood.values,
        reference.log_likelihood.values,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.unit
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_callback_returning_the_wrong_row_count_is_rejected(
    family, monkeypatch, n_chunks
):
    """A callback written for one row per edge would be silently misaligned."""
    detector = _fitted(family)
    edges = _coarse_edges(family[1])
    original = detector.compute_log_likelihood

    def one_row_per_edge(time_edges, *args, is_missing=None):
        rows = original(*args, time_edges=time_edges)
        return np.concatenate([rows, rows[-1:]])

    monkeypatch.setattr(detector, "compute_log_likelihood", one_row_per_edge)
    with pytest.raises(ValidationError, match="row"):
        detector.predict(**_decode_args(family, edges), n_chunks=n_chunks)


# ----------------------------------------------------------------- position
@pytest.mark.unit
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_position_must_match_position_time(family):
    detector = _fitted(family)
    edges = _coarse_edges(family[1])
    position = family[2]["position"].astype(float).copy()
    position = position if position.ndim > 1 else position[:, None]
    position[5] = np.nan
    with pytest.raises(ValidationError, match="position_time"):
        detector.predict(**_decode_args(family, edges, position=position[:-10]))
    backwards = family[1][::-1].copy()
    with pytest.raises((ValidationError, DataError), match="position_time"):
        detector.predict(
            **_decode_args(family, edges, position=position, position_time=backwards)
        )


@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
def test_infinite_position_is_missing_like_nan(family):
    detector = _fitted(family)
    position_time = family[1]
    edges = _coarse_edges(position_time)
    position = family[2]["position"].astype(float).copy()
    position = position if position.ndim > 1 else position[:, None]
    position[100] = np.inf
    results = detector.predict(
        **_decode_args(family, edges, position=position),
        return_outputs="log_likelihood",
    )
    is_inf = np.zeros(position_time.size, bool)
    is_inf[100] = True
    np.testing.assert_array_equal(
        np.all(results.log_likelihood.values == 0.0, axis=1),
        _expected_missing(edges, position_time, is_inf),
    )


# ----------------------------------------------------------------- EM support
@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
def test_em_weights_are_zero_outside_the_decoded_edges(family, monkeypatch):
    """Position samples the decode grid does not cover carry no posterior."""
    detector_cls, position_time, fit_args, _ = family
    edges = _coarse_edges(position_time, factor=4, n_bins=300)
    detector = detector_cls()
    weights = []
    original = detector.fit_encoding_model

    def spy(*args, **kwargs):
        if kwargs.get("weights") is not None:
            weights.append(np.asarray(kwargs["weights"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "fit_encoding_model", spy)
    detector.estimate_parameters(
        **fit_args,
        time_edges=edges,
        max_iter=1,
        min_encoding_local_mass=0.0,
        min_encoding_local_ess=0.0,
    )
    outside = (position_time < edges[0]) | (position_time > edges[-1])
    assert outside.any() and (~outside).any()
    np.testing.assert_array_equal(weights[0][outside], 0.0)
    assert np.all(weights[0][~outside] > 0.0)


# ----------------------------------------------------------------- width
@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
def test_width_is_recorded_only_when_transitions_are_learned(family):
    detector_cls, position_time, fit_args, _ = family
    detector = detector_cls()
    edges = _coarse_edges(position_time, factor=2)
    detector.estimate_parameters(
        **fit_args, time_edges=edges, max_iter=1, estimate_discrete_transition=False
    )
    assert detector.transition_time_bin_width_ is None
    detector.predict(**_decode_args(family, _coarse_edges(position_time, factor=4)))


@pytest.mark.unit
def test_equal_widths_at_different_origins_are_accepted():
    """The recorded width carries its own grid's rounding; a 2 ms grid learned
    at a Unix-epoch origin still matches a 2 ms grid at origin 0."""
    from non_local_detector.models.base import _decode_time_edges
    from non_local_detector.time_edges import calculate_time_edges

    learned = calculate_time_edges([1.7e9, 1.7e9 + 100 * 0.002], 500.0)
    width = (learned[-1] - learned[0]) / 100
    tolerance = max(4 * np.spacing(np.max(np.abs(learned))), 1e-3 * width)
    for origin in (0.0, 100.0, 1.7e9):
        _decode_time_edges(origin + np.arange(51) * 0.002, width, tolerance)
    with pytest.raises(ValidationError, match="bin width"):
        _decode_time_edges(np.arange(51) * 0.004, width, tolerance)
    with pytest.raises(ValidationError, match="bin width"):
        _decode_time_edges(np.arange(51) * 0.002 * 1.002, width, tolerance)


@pytest.mark.unit
@pytest.mark.parametrize("nan_index", [0, 1, 8, 9])
def test_nan_near_the_ends_marks_extrapolated_bins_missing(nan_index):
    """Positions outside the samples are extrapolated from the end intervals,
    so a NaN in either end interval affects every bin beyond that end."""
    from non_local_detector.models.base import _missing_bins

    position_time = np.arange(10.0)
    position = np.arange(10.0)[:, None]
    position[nan_index] = np.nan
    edges = np.arange(-3.0, 13.5, 0.5)
    centers = edges[:-1] + 0.25
    missing = _missing_bins(edges, None, position_time, position)
    import scipy.interpolate

    interpolated = scipy.interpolate.interpn(
        (position_time,), position, centers, bounds_error=False, fill_value=None
    )
    # Every bin whose center reads a NaN is missing.
    assert np.all(missing[~np.isfinite(interpolated[:, 0])])
    assert not np.all(missing)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_missing_bins_match_the_interpolation_oracle(seed):
    """Random masks, edges on sample times, and NaN at either end."""
    from non_local_detector.models.base import _missing_bins

    rng = np.random.default_rng(seed)
    position_time = np.sort(rng.uniform(0, 10, 30))
    position = rng.uniform(0, 100, (30, 1))
    is_nan = rng.random(30) < 0.15
    is_nan[rng.integers(0, 2) * 29] = True
    position[is_nan] = np.nan
    edges = np.sort(
        np.concatenate([rng.uniform(-2, 12, 20), rng.choice(position_time, 5)])
    )
    edges = np.unique(edges)
    user_mask = rng.random(edges.size - 1) < 0.1

    expected = _expected_missing(edges, position_time, is_nan)
    np.testing.assert_array_equal(
        _missing_bins(edges, user_mask, position_time, position), expected | user_mask
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_most_likely_sequence_marks_nan_position_bins_missing(family, monkeypatch):
    from non_local_detector.models import base

    detector = _fitted(family)
    position_time = family[1]
    position = family[2]["position"].astype(float).copy()
    position = position if position.ndim > 1 else position[:, None]
    is_nan = np.zeros(position_time.size, bool)
    is_nan[[40, 41, 42, 301, 555]] = True
    position[is_nan] = np.nan
    edges = _coarse_edges(position_time)
    captured = {}
    original = base.most_likely_sequence

    def spy(*args, is_missing=None, **kwargs):
        captured["is_missing"] = is_missing
        return original(*args, is_missing=is_missing, **kwargs)

    monkeypatch.setattr(base, "most_likely_sequence", spy)
    detector.most_likely_sequence(**_decode_args(family, edges, position=position))
    np.testing.assert_array_equal(
        captured["is_missing"], _expected_missing(edges, position_time, is_nan)
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted"], indirect=True)
def test_estimate_parameters_decodes_user_missing_bins_as_missing(family):
    detector_cls, position_time, fit_args, _ = family
    edges = _coarse_edges(position_time)
    is_missing = np.zeros(200, bool)
    is_missing[[0, 57, 58, 199]] = True
    results = detector_cls().estimate_parameters(
        **fit_args,
        time_edges=edges,
        is_missing=is_missing,
        max_iter=1,
        return_outputs="log_likelihood",
    )
    np.testing.assert_array_equal(
        np.all(results.log_likelihood.values == 0.0, axis=1), is_missing
    )
    with pytest.raises(ValidationError, match="is_missing"):
        detector_cls().estimate_parameters(
            **fit_args, time_edges=edges, is_missing=np.zeros(201, bool), max_iter=1
        )


@pytest.mark.integration
@pytest.mark.parametrize("family", FAMILIES, indirect=True)
def test_unix_epoch_origin_matches_origin_zero(family):
    """Shifting every timestamp to a Unix-epoch origin keeps the decode."""
    detector_cls, position_time, fit_args, spike_args = family
    shift = 1.7e9
    edges = _coarse_edges(position_time)
    results = []
    for offset in (0.0, shift):
        detector = detector_cls()
        shifted_fit = {
            **fit_args,
            "position_time": fit_args["position_time"] + offset,
            "spike_times": [s + offset for s in fit_args["spike_times"]],
        }
        detector.fit(**shifted_fit)
        decode = {k: v for k, v in shifted_fit.items() if k != "position"}
        results.append(
            detector.predict(
                **{k: decode[k] for k in spike_args},
                time_edges=edges + offset,
                position=fit_args["position"],
                position_time=decode["position_time"],
                return_outputs="log_likelihood",
            )
        )
    np.testing.assert_allclose(
        results[1].acausal_posterior.values,
        results[0].acausal_posterior.values,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        results[1].time.values - shift, results[0].time.values, atol=1e-6
    )


@pytest.mark.unit
def test_every_detector_entry_point_takes_keyword_only_time_edges():
    import inspect

    from non_local_detector import (
        ClusterlessDecoder,
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
        SortedSpikesDecoder,
    )

    for cls in (
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
        ClusterlessDecoder,
        SortedSpikesDecoder,
    ):
        for method in ("predict", "estimate_parameters", "most_likely_sequence"):
            parameters = inspect.signature(getattr(cls, method)).parameters
            assert "time" not in parameters, (cls, method)
            assert parameters["time_edges"].kind is inspect.Parameter.KEYWORD_ONLY, (
                cls,
                method,
            )


@pytest.mark.unit
@pytest.mark.parametrize("edge_dtype", [np.float32, np.float64])
def test_row_aware_chunks_keep_higher_precision_spikes_in_their_bin(edge_dtype):
    """Excluding a shared chunk edge must not open a gap wider than the spike
    times' precision: float32 edges with float64 spikes just below an edge."""
    from non_local_detector.core import accepts_row_slice, row_slice_aware
    from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
    from non_local_detector.models.base import _prepare_likelihood_callback

    edges = np.array([0.0, 1.0, 2.0], dtype=edge_dtype)
    spikes = np.array([0.99999999, 1.0])

    @row_slice_aware
    def likelihood(time_edges, spike_times, is_missing=None, row_slice=None):
        return get_spikecount_per_time_bin(
            spike_times, time_edges=time_edges, row_slice=row_slice
        )[:, None]

    callback = _prepare_likelihood_callback(likelihood, edges, has_no_spike=False)
    # The adapter takes the chunk's global row range from core, like a marked
    # callback, so it never has to recover it from the row coordinates.
    assert accepts_row_slice(callback)
    centers = edges[:-1] + 0.5 * np.diff(edges)
    chunked = np.concatenate(
        [
            callback(centers, spikes, is_missing=None, row_slice=slice(i, i + 1))
            for i in range(2)
        ]
    )
    np.testing.assert_array_equal(
        chunked[:, 0], get_spikecount_per_time_bin(spikes, time_edges=edges)
    )
    np.testing.assert_array_equal(chunked[:, 0], [1, 1])
