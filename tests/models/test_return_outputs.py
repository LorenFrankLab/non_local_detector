"""Tests for the return_outputs parameter in model predict() methods.

Tests the string-based interface for controlling which outputs are returned
from predict() methods, and that the removed boolean output flags are rejected.
"""

import gc

import jax
import numpy as np
import pytest

from non_local_detector import (
    NonLocalSortedSpikesDetector,
    core,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from tests.models._helpers import SORTED_SIM_RATE


@pytest.fixture(scope="module")
def simple_fitted_detector(sorted_sim):
    """A detector fit and decoded on the shared 2,000-sample recording.

    Module-scoped: tests only call ``detector.predict(...)`` with different
    ``return_outputs`` flags and do not mutate the detector. Which outputs are
    returned, and their shapes, do not depend on the recording length.
    """
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={
            "position_std": 6.0,
            "block_size": (2**12),
        },
        sampling_frequency=SORTED_SIM_RATE,
    ).fit(time, position, spike_times)

    return detector, spike_times, time, position


@pytest.mark.integration
class TestReturnOutputsParameter:
    """Test the return_outputs parameter with various inputs.

    These tests run a full HMM predict against a fitted detector shared via
    a module-scoped fixture.
    """

    def test_default_returns_smoother_only(self, simple_fitted_detector):
        """Test that default (None) returns only smoother and marginal likelihood."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
        )

        # Should have smoother outputs
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results
        assert "marginal_log_likelihoods" in results.attrs

        # Should NOT have filter, predictive, or log_likelihood
        assert "causal_posterior" not in results
        assert "causal_state_probabilities" not in results
        assert "predictive_state_probabilities" not in results
        assert "log_likelihood" not in results

    def test_return_filter_string(self, simple_fitted_detector):
        """Test return_outputs='filter' returns filter distribution."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs="filter",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results

        # Should have filter
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results

        # Should NOT have predictive or log_likelihood
        assert "predictive_state_probabilities" not in results
        assert "log_likelihood" not in results

        # Verify shapes
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        n_states = results.acausal_state_probabilities.shape[1]

        assert results.causal_posterior.shape == (n_time, n_state_bins)
        assert results.causal_state_probabilities.shape == (n_time, n_states)

    def test_return_predictive_string(self, simple_fitted_detector):
        """Test return_outputs='predictive' returns the predictive state probabilities.

        The string means the same as the list ['predictive']: the
        (n_time, n_state_bins) predictive posterior has its own name.
        """
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs="predictive",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        assert "predictive_state_probabilities" in results
        assert "predictive_posterior" not in results

        # Should NOT have filter or log_likelihood
        assert "causal_posterior" not in results
        assert "log_likelihood" not in results

        n_time = len(time)
        n_states = results.acausal_state_probabilities.shape[1]
        assert results.predictive_state_probabilities.shape == (n_time, n_states)
        predictive_sums = results.predictive_state_probabilities.sum(dim="states")
        assert np.allclose(predictive_sums.values, 1.0, atol=1e-10)

    def test_return_both_predictive_outputs(self, simple_fitted_detector):
        """Both predictive outputs are requested by name."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs=["predictive", "predictive_posterior"],
        )

        n_state_bins = results.acausal_posterior.shape[1]
        assert results.predictive_posterior.shape == (len(time), n_state_bins)
        predictive_posterior_sums = results.predictive_posterior.sum(dim="state_bins")
        assert np.allclose(predictive_posterior_sums.values, 1.0, atol=1e-10)
        assert "predictive_state_probabilities" in results

    def test_return_predictive_posterior_string(self, simple_fitted_detector):
        """Test return_outputs='predictive_posterior' returns full predictive posterior."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs="predictive_posterior",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have predictive_posterior but NOT predictive_state_probabilities
        assert "predictive_posterior" in results
        assert "predictive_state_probabilities" not in results

        # Should NOT have filter or log_likelihood
        assert "causal_posterior" not in results
        assert "log_likelihood" not in results

        # Verify shape of full version (state bins)
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        assert results.predictive_posterior.shape == (n_time, n_state_bins)

        # Verify probabilities sum to 1 (full version)
        predictive_posterior_sums = results.predictive_posterior.sum(dim="state_bins")
        assert np.allclose(predictive_posterior_sums.values, 1.0, atol=1e-10)

    def test_return_log_likelihood_string(self, simple_fitted_detector):
        """Test return_outputs='log_likelihood' returns log likelihoods."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs="log_likelihood",
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have log_likelihood
        assert "log_likelihood" in results

        # Should NOT have filter or predictive
        assert "causal_posterior" not in results
        assert "predictive_state_probabilities" not in results

        # Verify shape
        n_time = len(time)
        n_state_bins = results.acausal_posterior.shape[1]
        assert results.log_likelihood.shape == (n_time, n_state_bins)

    def test_return_all_string(self, simple_fitted_detector):
        """Test return_outputs='all' returns all outputs."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs="all",
        )

        # Should have everything
        assert "acausal_posterior" in results
        assert "acausal_state_probabilities" in results
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results
        assert "predictive_state_probabilities" in results
        assert "predictive_posterior" in results
        assert "log_likelihood" in results
        assert "marginal_log_likelihoods" in results.attrs

    @pytest.mark.parametrize(
        "outputs",
        [
            pytest.param(["filter", "log_likelihood"], id="list"),
            pytest.param({"filter", "log_likelihood"}, id="set"),
        ],
    )
    def test_return_multiple_outputs_iterable(self, simple_fitted_detector, outputs):
        """Test return_outputs accepts any iterable of strings (list, set, ...)."""
        detector, spike_times, time, position = simple_fitted_detector

        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            return_outputs=outputs,
        )

        # Should have smoother (always)
        assert "acausal_posterior" in results

        # Should have filter and log_likelihood
        assert "causal_posterior" in results
        assert "causal_state_probabilities" in results
        assert "log_likelihood" in results

        # Should NOT have predictive outputs
        assert "predictive_state_probabilities" not in results
        assert "predictive_posterior" not in results

    @pytest.mark.parametrize(
        ("return_outputs", "exc", "match"),
        [
            ("invalid_option", ValueError, "Invalid return_outputs"),
            ({"filter", "invalid_option"}, ValueError, "Invalid return_outputs"),
            (123, ValidationError, "return_outputs must be"),
        ],
        ids=["invalid-string", "invalid-set-element", "invalid-type"],
    )
    def test_invalid_return_outputs_raises_error(
        self, simple_fitted_detector, return_outputs, exc, match
    ):
        """Test that an invalid return_outputs string, element, or type raises."""
        detector, spike_times, time, position = simple_fitted_detector

        with pytest.raises(exc, match=match):
            detector.predict(
                spike_times=spike_times,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=time,
                return_outputs=return_outputs,
            )


def _live_state_bin_rows(n_state_bins: int, row_counts: set[int]) -> int:
    """Rows of the live device arrays shaped like (part of) a posterior.

    Counts every live 2-D array whose second axis is the state bins and whose
    first axis is the whole timeline or one chunk of it.
    """
    return sum(
        array.shape[0]
        for array in jax.live_arrays()
        if not array.is_deleted()
        and array.ndim == 2
        and array.shape[1] == n_state_bins
        and array.shape[0] in row_counts
    )


def _rows_held_per_chunk(detector, spike_times, time, position, monkeypatch, **kw):
    """Live posterior rows during each forward and backward chunk of ``predict``.

    Read wherever a chunk's state probabilities are taken, after the filter or
    the smoother of that chunk has run. The last reading is the backward
    pass's final chunk, just before the smoothed chunks are concatenated.
    Rows live before the call are subtracted.
    """
    n_state_bins = len(detector.state_space_.interior_state_ind)
    n_time = len(time)
    row_counts = {n_time} | {
        len(chunk) for chunk in np.array_split(np.arange(n_time), kw["n_chunks"])
    }
    gc.collect()
    baseline = _live_state_bin_rows(n_state_bins, row_counts)
    readings = []
    state_marginals = core._state_marginals

    def reading_state_marginals(posterior, aggregation):
        readings.append(_live_state_bin_rows(n_state_bins, row_counts) - baseline)
        return state_marginals(posterior, aggregation)

    with monkeypatch.context() as patch:
        patch.setattr(core, "_state_marginals", reading_state_marginals)
        results = detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            position=position,
            position_time=time,
            **kw,
        )
    return readings, results


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_unrequested_posteriors_are_not_held(
    simple_fitted_detector, monkeypatch, n_chunks
):
    """The default outputs hold neither the filtered nor the predictive posterior.

    The predictive posterior is not kept past its chunk, and the filtered one
    is released by the time the smoothed posterior is concatenated, so both
    the peak and the smoothed concatenation hold fewer posterior rows than a
    decode that returns the two. The returned arrays are the same.
    """
    detector, spike_times, time, position = simple_fitted_detector
    n_time = len(time)
    default_rows, default = _rows_held_per_chunk(
        detector, spike_times, time, position, monkeypatch, n_chunks=n_chunks
    )
    # Not "all": it also caches the log likelihoods on the device.
    requested_rows, requested = _rows_held_per_chunk(
        detector,
        spike_times,
        time,
        position,
        monkeypatch,
        n_chunks=n_chunks,
        return_outputs=["filter", "predictive_posterior"],
    )
    # Filtered and predictive per forward chunk, smoothed per backward chunk.
    assert len(default_rows) == len(requested_rows) == 3 * n_chunks
    # Without them, the smoothed concatenation holds two posteriors fewer...
    assert default_rows[-1] <= requested_rows[-1] - 2 * n_time
    # ...and the peak at least one fewer: the filter itself returns its
    # chunk's predictive rows alongside the filtered ones.
    assert max(default_rows) <= max(requested_rows) - n_time
    for name in ("acausal_posterior", "acausal_state_probabilities"):
        np.testing.assert_array_equal(default[name].values, requested[name].values)
    np.testing.assert_array_equal(
        default.attrs["marginal_log_likelihoods"],
        requested.attrs["marginal_log_likelihoods"],
    )
