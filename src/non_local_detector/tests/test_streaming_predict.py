"""Chunked-parity tests for detector.predict(n_chunks=K).

Verifies that the streaming `n_chunks > 1` path produces the same
posteriors as the unchunked `n_chunks = 1` baseline, end-to-end through
the full detector.fit() + predict() pipeline.

The core-level `chunked_filter_smoother` is already tested in
`tests/core/test_chunked_parity.py` against a mock log-likelihood
function.  These tests cover the higher-level integration: the
detector's likelihood function is called with a time slice per chunk,
and the accumulated posterior must agree with the single-shot version.

Tolerance: 1e-4 (absolute + relative).  Matches existing chunked-parity
tests in ``tests/core/test_chunked_parity.py``.  Chunking triggers
floating-point reassociation at chunk boundaries; differences below
this threshold are benign.
"""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector import (
    ContFragClusterlessClassifier,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

_TOL = dict(atol=1e-4, rtol=1e-4)


@pytest.fixture(scope="module")
def sorted_spikes_fitted():
    """Tiny fitted sorted-spikes detector — reused across chunked-parity tests."""
    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=10)
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": 4096},
    ).fit(time, position, spike_times, is_training=~is_event)
    return detector, dict(
        time=time, position=position, spike_times=spike_times, position_time=time
    )


@pytest.fixture(scope="module")
def clusterless_fitted():
    """Tiny fitted clusterless detector — reused across chunked-parity tests."""
    sim = make_simulated_run_data(n_tetrodes=4)
    # Use time-bin centres as the decoding ``time`` (edges has n_time_bins+1 entries).
    time = 0.5 * (sim.edges[:-1] + sim.edges[1:])
    detector = ContFragClusterlessClassifier(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 1000,
        },
    ).fit(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
    )
    return detector, dict(
        time=time,
        position=sim.position,
        position_time=sim.position_time,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
    )


class TestStreamingPredictParity:
    """``detector.predict(n_chunks=K)`` must match ``n_chunks=1`` on real detector flow.

    This is the gate the streaming-likelihood plan must keep green throughout.
    Writing this test FIRST (TDD / RED step) verifies the plan's claim that
    the streaming plumbing already works end-to-end before we layer caching
    on top.
    """

    @pytest.mark.integration
    def test_sorted_spikes_chunked_matches_unchunked(self, sorted_spikes_fitted):
        detector, predict_kwargs = sorted_spikes_fitted
        r1 = detector.predict(**predict_kwargs, n_chunks=1)
        r5 = detector.predict(**predict_kwargs, n_chunks=5)
        np.testing.assert_allclose(
            r5.acausal_posterior.values,
            r1.acausal_posterior.values,
            **_TOL,
            err_msg="sorted-spikes acausal_posterior drifts between n_chunks=1 and n_chunks=5",
        )
        np.testing.assert_allclose(
            r5.acausal_state_probabilities.values,
            r1.acausal_state_probabilities.values,
            **_TOL,
            err_msg="sorted-spikes acausal_state_probabilities drifts",
        )

    @pytest.mark.integration
    def test_clusterless_chunked_matches_unchunked(self, clusterless_fitted):
        detector, predict_kwargs = clusterless_fitted
        r1 = detector.predict(**predict_kwargs, n_chunks=1)
        r5 = detector.predict(**predict_kwargs, n_chunks=5)
        np.testing.assert_allclose(
            r5.acausal_posterior.values,
            r1.acausal_posterior.values,
            **_TOL,
            err_msg="clusterless acausal_posterior drifts between n_chunks=1 and n_chunks=5",
        )
        np.testing.assert_allclose(
            r5.acausal_state_probabilities.values,
            r1.acausal_state_probabilities.values,
            **_TOL,
            err_msg="clusterless acausal_state_probabilities drifts",
        )
