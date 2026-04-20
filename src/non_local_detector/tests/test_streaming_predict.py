"""Streaming-predict tests: chunked-parity + `_resolve_n_chunks` heuristic.

Two test groups:

1. **Chunked-parity integration tests**: verify
   ``detector.predict(n_chunks=K)`` produces the same posteriors as
   ``n_chunks=1`` end-to-end through fit() + predict(). These guard
   the invariant that any changes to the streaming path (caching,
   auto-chunking, etc.) stay behaviour-preserving.

2. **``_resolve_n_chunks`` unit tests**: verify the memory-aware
   heuristic passes explicit ints through unchanged, and for
   ``"auto"`` picks ``n_chunks`` so each chunk's likelihood slab fits
   in the budget.  Uses an explicit ``memory_budget_bytes`` kwarg so
   the tests run on CPU without needing a GPU.

Tolerance on the integration tests: 1e-4 (abs + rel). Matches existing
chunked-parity tests in ``tests/core/test_chunked_parity.py``. Chunking
causes benign floating-point reassociation at chunk boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from non_local_detector import (
    ContFragClusterlessClassifier,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from non_local_detector.streaming import _resolve_n_chunks

_TOL = {"atol": 1e-4, "rtol": 1e-4}


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
    return detector, {
        "time": time,
        "position": position,
        "spike_times": spike_times,
        "position_time": time,
    }


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
    return detector, {
        "time": time,
        "position": sim.position,
        "position_time": sim.position_time,
        "spike_times": sim.spike_times,
        "spike_waveform_features": sim.spike_waveform_features,
    }


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

    @pytest.mark.integration
    def test_sorted_spikes_auto_matches_unchunked(self, sorted_spikes_fitted):
        """``n_chunks='auto'`` must produce identical output to ``n_chunks=1``
        on a small workload where the heuristic resolves to 1 (on CPU the
        device-memory query returns None → fallback to 1)."""
        detector, predict_kwargs = sorted_spikes_fitted
        r1 = detector.predict(**predict_kwargs, n_chunks=1)
        ra = detector.predict(**predict_kwargs, n_chunks="auto")
        np.testing.assert_allclose(
            ra.acausal_posterior.values,
            r1.acausal_posterior.values,
            **_TOL,
            err_msg="sorted-spikes n_chunks='auto' drifts from n_chunks=1",
        )

    @pytest.mark.integration
    def test_clusterless_auto_matches_unchunked(self, clusterless_fitted):
        detector, predict_kwargs = clusterless_fitted
        r1 = detector.predict(**predict_kwargs, n_chunks=1)
        ra = detector.predict(**predict_kwargs, n_chunks="auto")
        np.testing.assert_allclose(
            ra.acausal_posterior.values,
            r1.acausal_posterior.values,
            **_TOL,
            err_msg="clusterless n_chunks='auto' drifts from n_chunks=1",
        )


# ---------------------------------------------------------------------------
# _resolve_n_chunks heuristic (Task 1) — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveNChunksPassthrough:
    """Explicit int passthrough: the heuristic must not override user choice."""

    def test_passthrough_one(self):
        assert _resolve_n_chunks(1, n_time=1000, n_state_bins=200) == 1

    def test_passthrough_many(self):
        assert _resolve_n_chunks(7, n_time=1000, n_state_bins=200) == 7

    def test_passthrough_ignores_budget(self):
        """Explicit int should return unchanged even if it wouldn't fit the budget."""
        # 1000 × 200 × 4 = 800 kB/chunk at n_chunks=1; budget says only 100 B
        # — should still return 1 because the user asked for 1.
        assert _resolve_n_chunks(
            1, n_time=1000, n_state_bins=200, memory_budget_bytes=100
        ) == 1


@pytest.mark.unit
class TestResolveNChunksAuto:
    """``'auto'``: pick smallest n_chunks so per-chunk likelihood fits budget."""

    def test_auto_small_workload_fits_in_one_chunk(self):
        """Fits: expect 1 chunk."""
        # 1000 time bins × 200 state bins × 4 B = 800 kB; budget 10 MB → fits.
        n = _resolve_n_chunks(
            "auto",
            n_time=1_000,
            n_state_bins=200,
            memory_budget_bytes=10_000_000,
        )
        assert n == 1

    def test_auto_chronic_scale_picks_many_chunks(self):
        """Chronic recording × 8 GB budget picks multiple chunks."""
        # 1.8 M time × 13 680 state bins × 4 B = 98 GB; 40% of 8 GB = 3.2 GB.
        # chunk_size = 3.2 GB / (13680*4) = ~58 500 time bins → ~30 chunks.
        n = _resolve_n_chunks(
            "auto",
            n_time=1_800_000,
            n_state_bins=13_680,
            memory_budget_bytes=8 * 2**30,
        )
        assert n >= 12, f"Chronic-scale workload should pick many chunks, got {n}"

    def test_auto_budget_boundary_picks_correct_count(self):
        """Budget exactly = chunk_size × per-time-bytes → chunk_size chunks of right shape."""
        # n_time=1000, n_state_bins=200, dtype=4.  Budget allows 100 time bins
        # per chunk (100 × 200 × 4 = 80 kB × safety_fraction 0.40 = 32 kB)
        # — so effective per-chunk budget = 0.40 × 200_000 = 80_000 bytes →
        # chunk_size = 80_000 / (200 × 4) = 100 → n_chunks = 10.
        n = _resolve_n_chunks(
            "auto",
            n_time=1_000,
            n_state_bins=200,
            memory_budget_bytes=200_000,  # × 0.40 = 80 kB usable
        )
        assert n == 10, f"Expected exactly 10 chunks at budget=200 kB, got {n}"

    def test_auto_tiny_budget_caps_at_n_time(self):
        """Budget smaller than a single time bin → chunk_size=1, n_chunks=n_time."""
        n = _resolve_n_chunks(
            "auto",
            n_time=50,
            n_state_bins=10,
            memory_budget_bytes=4,  # one time bin fits (4 B × 0.40 = 1.6 B, ceil to 1)
        )
        # Should cap at n_time (one time bin per chunk) rather than blow up.
        assert 1 <= n <= 50

    def test_auto_respects_safety_fraction(self):
        """Higher safety_fraction (more headroom) → same budget → more chunks."""
        base = _resolve_n_chunks(
            "auto",
            n_time=1_000_000,
            n_state_bins=1_000,
            memory_budget_bytes=1 * 2**30,
            safety_fraction=0.40,
        )
        conservative = _resolve_n_chunks(
            "auto",
            n_time=1_000_000,
            n_state_bins=1_000,
            memory_budget_bytes=1 * 2**30,
            safety_fraction=0.10,  # only 10% of budget usable → more chunks
        )
        assert conservative > base


@pytest.mark.unit
class TestResolveNChunksValidation:
    """Input validation.  Rejects bad values at the boundary."""

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks(0, n_time=100, n_state_bins=10)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks(-3, n_time=100, n_state_bins=10)

    def test_rejects_bool(self):
        """``bool`` is an ``int`` subclass in Python — reject True/False explicitly."""
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks(True, n_time=100, n_state_bins=10)

    def test_rejects_non_int_non_auto_string(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks("banana", n_time=100, n_state_bins=10)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks(3.5, n_time=100, n_state_bins=10)  # type: ignore[arg-type]

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="must be"):
            _resolve_n_chunks(None, n_time=100, n_state_bins=10)  # type: ignore[arg-type]

    def test_rejects_nonpositive_budget(self):
        with pytest.raises(ValueError, match="memory_budget_bytes"):
            _resolve_n_chunks(
                "auto",
                n_time=100,
                n_state_bins=10,
                memory_budget_bytes=0,
            )

    def test_rejects_nonpositive_n_time(self):
        with pytest.raises(ValueError, match="n_time"):
            _resolve_n_chunks("auto", n_time=0, n_state_bins=10)

    def test_rejects_nonpositive_n_state_bins(self):
        with pytest.raises(ValueError, match="n_state_bins"):
            _resolve_n_chunks("auto", n_time=100, n_state_bins=0)


@pytest.mark.unit
class TestResolveNChunksMathInvariants:
    """Properties that should hold for any resolved ``"auto"`` result."""

    @pytest.mark.parametrize(
        "n_time,n_state_bins,budget",
        [
            (1_000, 200, 10_000_000),
            (1_800_000, 13_680, 8 * 2**30),
            (500_000, 1_000, 100_000_000),
            (10_000, 100, 1_000_000),
        ],
    )
    def test_per_chunk_bytes_under_budget(self, n_time, n_state_bins, budget):
        """Resolved n_chunks must produce chunks that fit the effective budget."""
        n = _resolve_n_chunks(
            "auto",
            n_time=n_time,
            n_state_bins=n_state_bins,
            memory_budget_bytes=budget,
        )
        chunk_size = math.ceil(n_time / n)
        per_chunk_bytes = chunk_size * n_state_bins * 4  # fp32
        # safety_fraction default 0.40
        assert per_chunk_bytes <= budget * 0.40 * 1.01, (  # 1% slack for ceil
            f"chunk={chunk_size}, per_chunk={per_chunk_bytes} bytes, "
            f"budget*safety={budget * 0.40} bytes"
        )

    def test_auto_returns_positive_int(self):
        """Always returns int >= 1, never zero / float / negative."""
        n = _resolve_n_chunks(
            "auto",
            n_time=1_000,
            n_state_bins=200,
            memory_budget_bytes=10_000_000,
        )
        assert isinstance(n, int) and n >= 1
