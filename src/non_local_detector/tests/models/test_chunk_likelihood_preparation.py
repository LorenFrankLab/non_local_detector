"""Recording-wide work must be prepared once, not repeated in every chunk."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.core import row_slice_aware
from non_local_detector.likelihoods.common import _SpikeTimeOrder, select_spikes_in_rows


@pytest.mark.unit
@pytest.mark.parametrize("array_type", [np.asarray, jnp.asarray, list])
def test_prepared_spike_times_convert_original_input_only_once(monkeypatch, array_type):
    """Reuse the host times for NumPy, JAX and list inputs across row requests."""
    time = np.arange(10.0)
    spikes = array_type([0.5, 1.5, 3.5, 6.5, 8.5])
    original_asarray = np.asarray
    conversions = []

    def tracked_asarray(values, *args, **kwargs):
        if values is spikes:
            conversions.append(values)
        return original_asarray(values, *args, **kwargs)

    order = _SpikeTimeOrder()
    monkeypatch.setattr(np, "asarray", tracked_asarray)
    for row_start, row_stop in [(0, 3), (3, 7), (7, 10)]:
        selection = select_spikes_in_rows(
            spikes, time, row_start, row_stop, _spike_time_order=order
        )
        indexer, rows = selection.indexer, selection.bin_ind
        selected = original_asarray(spikes)[indexer]
        np.testing.assert_array_equal(
            rows, np.digitize(selected, time[1:-1]) - row_start
        )
    assert len(conversions) == 1


@pytest.mark.unit
@pytest.mark.parametrize("rows", [(3, 3), (9, 10)])
@pytest.mark.parametrize("prepared", [False, True])
def test_non_owning_rows_do_not_read_spike_times(monkeypatch, rows, prepared):
    """Empty requests and the terminal empty row need no spike transfer/check."""
    time = np.arange(10.0)
    spikes = np.arange(10_000.0)
    original_asarray = np.asarray

    def tracked_asarray(values, *args, **kwargs):
        if values is spikes:
            pytest.fail("Rows that cannot own spikes should not read spike times")
        return original_asarray(values, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", tracked_asarray)
    selection = select_spikes_in_rows(
        spikes, time, *rows, _spike_time_order=_SpikeTimeOrder() if prepared else None
    )
    indexer, bin_ind = selection.indexer, selection.bin_ind
    assert spikes[indexer].size == 0
    assert bin_ind.size == 0


@pytest.mark.unit
@pytest.mark.parametrize("row_start,row_stop", [(0, 20), (401, 421), (980, 1001)])
@pytest.mark.parametrize("ascending", [True, False])
def test_binning_scans_only_requested_boundaries(
    monkeypatch, row_start, row_stop, ascending
):
    """Digitize checks monotonicity linearly; its bins must be chunk-sized.

    The independent full-timeline reference also pins global ownership for
    irregular times, boundary spikes, the inclusive endpoint and unsorted input.
    """
    time = np.cumsum(np.resize([0.002, 0.003, 0.004], 1001))
    spikes = np.sort(np.r_[time, (time[:-1] + time[1:]) / 2])
    if not ascending:
        spikes = spikes[::-1]
    global_rows = np.digitize(spikes, time[1:-1])
    keep = (global_rows >= row_start) & (global_rows < row_stop)
    digitize = np.digitize
    scanned_sizes = []

    def tracked_digitize(values, bins, **kwargs):
        scanned_sizes.append(len(bins))
        return digitize(values, bins, **kwargs)

    monkeypatch.setattr(np, "digitize", tracked_digitize)
    selection = select_spikes_in_rows(spikes, time, row_start, row_stop)
    indexer, local_rows = selection.indexer, selection.bin_ind

    np.testing.assert_array_equal(spikes[indexer], spikes[keep])
    np.testing.assert_array_equal(local_rows, global_rows[keep] - row_start)
    assert sum(scanned_sizes) <= row_stop - row_start


@pytest.fixture(scope="module", params=["sorted", "clusterless"])
def fitted_detector(request):
    """Small real encoding models, including the default No-Spike state."""
    position_time = np.linspace(0, 10, 201)
    position = (10 + 8 * np.sin(position_time))[:, None]
    spikes = [np.arange(0.025, 10, 0.2)]
    env = Environment(place_bin_size=5.0)
    args = (position_time, position, spikes)
    if request.param == "sorted":
        detector = NonLocalSortedSpikesDetector(environments=env)
    else:
        detector = NonLocalClusterlessDetector(environments=env)
        features = [np.random.default_rng(42).normal(size=(len(spikes[0]), 2))]
        args = (*args, features)
    detector.fit(*args)
    return detector, args


@pytest.mark.integration
@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_no_spike_duration_prepared_once_per_prediction(
    fitted_detector, monkeypatch, covariate, n_chunks
):
    """Use the full median once; refresh it even if the same time array mutates."""
    detector, args = fitted_detector
    time = np.r_[0.0, np.cumsum(np.resize([0.002, 0.003, 0.004], 30))]
    transitions = detector.discrete_state_transitions_
    if covariate:
        transitions = np.broadcast_to(transitions, (len(time), *transitions.shape))
    median = np.median
    durations = []

    def tracked_median(values, *args, **kwargs):
        result = median(values, *args, **kwargs)
        durations.append(result)
        return result

    for _ in range(2):
        expected = np.asarray(detector.compute_log_likelihood(time, *args))
        durations.clear()

        with monkeypatch.context() as patch:
            patch.setattr(np, "median", tracked_median)
            result = detector._predict(
                time,
                log_likelihood_args=args,
                cache_likelihood=False,
                n_chunks=n_chunks,
                discrete_state_transitions=transitions,
                accumulate_log_likelihoods=True,
            )

        np.testing.assert_allclose(result[5], expected, rtol=1e-5, atol=1e-6)
        assert durations == [median(np.diff(time))]
        time *= 2


@pytest.mark.integration
def test_cached_likelihood_skips_duration_preparation(fitted_detector, monkeypatch):
    """An EM step reusing likelihoods should do no timeline preparation."""
    detector, args = fitted_detector
    time = np.arange(31) * 0.002
    expected = np.asarray(detector.compute_log_likelihood(time, *args))

    def unexpected_median(*args, **kwargs):
        pytest.fail("No-Spike duration is unused when likelihoods are supplied")

    monkeypatch.setattr(np, "median", unexpected_median)
    monkeypatch.setattr(_SpikeTimeOrder, "get", unexpected_median)
    result = detector._predict(time, log_likelihoods=expected, n_chunks=3)
    np.testing.assert_array_equal(result[5], expected)


@pytest.mark.integration
def test_custom_likelihood_signature_remains_supported(fitted_detector, monkeypatch):
    """An override with the existing row-slice API needs no new keyword."""
    detector, args = fitted_detector
    original = detector.compute_log_likelihood
    time = np.arange(31) * 0.002
    expected = np.asarray(original(time, *args))

    @row_slice_aware
    def custom_likelihood(time, *args, is_missing=None, row_slice=None):
        return original(time, *args, is_missing=is_missing, row_slice=row_slice)

    monkeypatch.setattr(detector, "compute_log_likelihood", custom_likelihood)
    result = detector._predict(
        time,
        log_likelihood_args=args,
        cache_likelihood=False,
        n_chunks=3,
        accumulate_log_likelihoods=True,
    )
    np.testing.assert_allclose(result[5], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_direct_likelihood_uses_fresh_order_preparation(fitted_detector, monkeypatch):
    """Standalone likelihood calls share state work but never retain ordering."""
    detector, fitted_args = fitted_detector
    spikes = fitted_args[2][0].copy()
    args = (*fitted_args[:2], [spikes], *fitted_args[3:])
    time = np.arange(31) * 0.002
    original_all = np.all
    checks = []

    def tracked_all(values, *args, **kwargs):
        result = original_all(values, *args, **kwargs)
        if getattr(values, "shape", None) == (len(spikes) - 1,):
            checks.append(bool(result))
        return result

    monkeypatch.setattr(np, "all", tracked_all)
    detector.compute_log_likelihood(time, *args)
    spikes[:] = spikes[::-1]
    detector.compute_log_likelihood(time, *args)
    assert checks == [True, False]


@pytest.mark.integration
@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_spike_order_checked_once_per_prediction(
    fitted_detector, monkeypatch, covariate, n_chunks
):
    """Check once across states/chunks; changed arrays get a fresh check next run.

    Track the recording-length boolean reduction, rather than a cache helper,
    so the test fails if any backend still scans all spikes on every chunk.
    The array is shuffled in place between predictions along with its features.
    """
    detector, fitted_args = fitted_detector
    time = np.arange(31) * 0.002
    spikes = np.linspace(0.0, 0.059, 503)
    args = (*fitted_args[:2], [spikes])
    if len(fitted_args) == 4:
        features = np.random.default_rng(9).normal(size=(len(spikes), 2))
        args = (*args, [features])
    transitions = detector.discrete_state_transitions_
    if covariate:
        transitions = np.broadcast_to(transitions, (len(time), *transitions.shape))
    original_all = np.all
    checks = []

    def tracked_all(values, *args, **kwargs):
        result = original_all(values, *args, **kwargs)
        if getattr(values, "shape", None) == (len(spikes) - 1,):
            checks.append(bool(result))
        return result

    for ascending in (True, False):
        expected = np.asarray(detector.compute_log_likelihood(time, *args))
        checks.clear()
        with monkeypatch.context() as patch:
            patch.setattr(np, "all", tracked_all)
            result = detector._predict(
                time,
                log_likelihood_args=args,
                cache_likelihood=False,
                n_chunks=n_chunks,
                discrete_state_transitions=transitions,
                accumulate_log_likelihoods=True,
            )
        assert checks == [ascending]
        np.testing.assert_allclose(result[5], expected, rtol=1e-5, atol=1e-6)
        order = np.random.default_rng(10).permutation(len(spikes))
        spikes[:] = spikes[order]
        if len(args) == 4:
            args[3][0][:] = args[3][0][order]
