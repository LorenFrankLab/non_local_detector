"""Grid preparation is per prediction; chunk workspace is per requested row."""

import numpy as np
import pytest

from non_local_detector.core import row_slice_aware
from non_local_detector.exceptions import DataError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _SORTED_SPIKES_ALGORITHMS,
)
from non_local_detector.likelihoods.common import _SpikeTimeOrder
from non_local_detector.tests.models.test_rate_units_and_persistence import (
    ALGORITHMS,
    _decoder,
)


@pytest.fixture(scope="module", params=ALGORITHMS)
def backend(request):
    name = request.param
    time = np.arange(30) / 30
    position = np.zeros((30, 1))
    spikes = [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]
    args = (spikes,)
    if name.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
    detector = _decoder(name, calibration=True).fit(time, position, *args)
    registry = (
        _CLUSTERLESS_ALGORITHMS
        if name.startswith("clusterless")
        else _SORTED_SPIKES_ALGORITHMS
    )
    predict = registry[name][1]
    encoding = next(iter(detector.encoding_model_.values()))
    return detector, time, position, args, predict, encoding


@pytest.mark.integration
@pytest.mark.parametrize("family", ["sorted_spikes_kde", "clusterless_kde"])
@pytest.mark.parametrize("learned_clock", [False, True])
@pytest.mark.parametrize("nan_gap", [False, True])
def test_detector_chunk_callbacks_do_not_scan_the_complete_grid(
    family, learned_clock, nan_gap, monkeypatch
):
    """Exercise actual callback preparation without allocating HMM posteriors."""
    time = np.arange(30) / 30
    position = np.zeros((30, 1))
    args = ([np.array([0.1, 0.3, 0.5, 0.7, 0.9])],)
    if family.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
    detector = _decoder(family, calibration=True).fit(time, position, *args)
    if nan_gap:
        position = position.copy()
        position[6:8] = np.nan
    edges = np.arange(180_001) * 0.002
    if learned_clock:
        detector.transition_time_bin_width_ = 0.002
        detector._transition_time_bin_width_tolerance = 2e-6
    original = detector.compute_log_likelihood
    diff = np.diff
    full_scans = []

    @row_slice_aware
    def measured(
        *args,
        time_edges,
        row_slice=None,
        is_missing=None,
        _time_grid=None,
        _position_context=None,
        _spike_time_order=None,
    ):
        def tracked_diff(values, *args, **kwargs):
            if np.shape(values) == edges.shape:
                full_scans.append(np.size(values))
            return diff(values, *args, **kwargs)

        keywords = {
            "time_edges": time_edges,
            "row_slice": row_slice,
            "is_missing": is_missing,
            "_spike_time_order": _spike_time_order,
        }
        if _time_grid is not None:
            keywords["_time_grid"] = _time_grid
        if _position_context is not None:
            keywords["_position_context"] = _position_context
        with monkeypatch.context() as patch:
            from non_local_detector import time_edges as grid_module
            from non_local_detector.models import base

            def unexpected_precision_scan(*args, **kwargs):
                pytest.fail(
                    "Prepared chunk rescanned complete-grid timestamp precision"
                )

            def unexpected_position_preparation(*args, **kwargs):
                pytest.fail("Prepared chunk rebuilt complete-grid tracking missingness")

            patch.setattr(np, "diff", tracked_diff)
            patch.setattr(grid_module, "_spacing_tolerance", unexpected_precision_scan)
            patch.setattr(base, "_missing_bins", unexpected_position_preparation)
            result = original(*args, **keywords)
        assert not full_scans, "Every requested chunk scanned the full recording grid"
        return result

    def request_chunks(*, time, log_likelihood_func, log_likelihood_args, **kwargs):
        for rows in [slice(100, 120), slice(200, 220), slice(300, 320)]:
            result = log_likelihood_func(
                time[rows], *log_likelihood_args, row_slice=rows, is_missing=None
            )
            assert result.shape[0] == 20
        return ()

    from non_local_detector.models import base

    monkeypatch.setattr(detector, "compute_log_likelihood", measured)
    monkeypatch.setattr(base, "chunked_filter_smoother", request_chunks)
    detector._predict(edges, log_likelihood_args=(time, position, *args), n_chunks=3)


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True])
def test_prepared_backend_requests_allocate_only_requested_grid_rows(
    backend, is_local, monkeypatch
):
    from non_local_detector import time_edges as grid_module

    _, time, position, args, predict, encoding = backend
    edges = np.arange(180_001) * 0.002
    context = grid_module._DecodeTimeGrid(edges)
    diff = np.diff

    def unexpected_validation(*args, **kwargs):
        pytest.fail("Prepared chunk must not validate the full grid again")

    def bounded_diff(values, *args, **kwargs):
        assert np.shape(values) != edges.shape, (
            "Duration slicing allocated the full grid"
        )
        return diff(values, *args, **kwargs)

    monkeypatch.setattr(grid_module, "validate_time_edges", unexpected_validation)
    monkeypatch.setattr(np, "diff", bounded_diff)
    result = predict(
        time,
        position,
        *args,
        time_edges=edges,
        row_slice=slice(100, 120),
        is_local=is_local,
        _time_grid=context,
        _spike_time_order=_SpikeTimeOrder(),
        **encoding,
    )
    assert result.shape[0] == 20
    assert np.all(np.isfinite(result))


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True])
def test_prepared_variable_duration_chunks_preserve_standalone_likelihoods(
    backend, is_local
):
    from non_local_detector.time_edges import _DecodeTimeGrid

    _, time, position, _, predict, encoding = backend
    edges = 1.7e9 + np.r_[0, np.cumsum(np.resize([0.002, 0.004], 40))]
    position_time = 1.7e9 + time
    spikes = [edges[[0, 5, 20, 39, 40]]]
    args = (spikes,)
    if predict in [pair[1] for pair in _CLUSTERLESS_ALGORITHMS.values()]:
        args = (*args, [np.zeros((5, 1))])
    options = dict(time_edges=edges, is_local=is_local, **encoding)
    full = np.asarray(predict(position_time, position, *args, **options))
    context = _DecodeTimeGrid(edges)
    prepared = np.asarray(
        predict(position_time, position, *args, _time_grid=context, **options)
    )
    np.testing.assert_array_equal(prepared, full)
    chunks = np.concatenate(
        [
            np.asarray(
                predict(
                    position_time,
                    position,
                    *args,
                    _time_grid=context,
                    row_slice=rows,
                    **options,
                )
            )
            for rows in [slice(0, 1), slice(1, 20), slice(20, 40)]
        ]
    )
    np.testing.assert_allclose(chunks, full, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_standalone_likelihood_revalidates_edges_on_every_call(backend):
    _, time, position, args, predict, encoding = backend
    edges = np.arange(21) * 0.002
    predict(time, position, *args, time_edges=edges, **encoding)
    edges[7] = edges[6]
    with pytest.raises(DataError, match="strictly increasing"):
        predict(time, position, *args, time_edges=edges, **encoding)


@pytest.mark.unit
def test_standalone_likelihood_rechecks_mutated_spike_order(backend):
    _, time, position, _, predict, encoding = backend
    edges = np.arange(21) * 0.002
    spikes = [edges[[0, 5, 10, 20]].copy()]
    args = (spikes,)
    if predict in [pair[1] for pair in _CLUSTERLESS_ALGORITHMS.values()]:
        args = (*args, [np.zeros((4, 1))])
    options = dict(time_edges=edges, row_slice=slice(3, 20), **encoding)
    expected = np.asarray(predict(time, position, *args, **options))
    spikes[0][:] = spikes[0][::-1]
    actual = np.asarray(predict(time, position, *args, **options))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("prepared_durations", [False, True])
def test_prepared_no_spike_requests_do_not_scan_the_recording(
    monkeypatch, prepared_durations
):
    from non_local_detector.likelihoods.no_spike import (
        predict_no_spike_log_likelihood,
    )
    from non_local_detector.time_edges import _DecodeTimeGrid

    edges = 1.7e9 + np.arange(180_001) * 0.002
    context = _DecodeTimeGrid(edges)
    durations = np.diff(edges) if prepared_durations else None
    expected = -5.0 * np.diff(edges[100:121])
    diff = np.diff

    def bounded_diff(values, *args, **kwargs):
        assert np.shape(values) != edges.shape
        return diff(values, *args, **kwargs)

    monkeypatch.setattr(np, "diff", bounded_diff)
    result = predict_no_spike_log_likelihood(
        [[]],
        no_spike_rate=5.0,
        time_edges=edges,
        row_slice=slice(100, 120),
        _time_grid=context,
        _time_bin_sizes=durations,
    )
    np.testing.assert_array_equal(np.asarray(result)[:, 0], expected.astype(np.float32))


@pytest.mark.unit
@pytest.mark.parametrize("context_type", ["other_edges", "unsupported"])
def test_mismatched_context_cannot_skip_fresh_grid_validation(context_type):
    from non_local_detector.time_edges import _DecodeTimeGrid, _resolve_time_grid

    original = np.arange(21) * 0.002
    context = _DecodeTimeGrid(original) if context_type == "other_edges" else object()
    edges = original.copy()
    edges[7] = edges[6]
    with pytest.raises(DataError, match="strictly increasing"):
        _resolve_time_grid(edges, context)
