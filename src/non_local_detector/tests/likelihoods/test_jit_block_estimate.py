"""Prerequisite equivalence tests and JIT-fused block_estimate tests for clusterless KDE log likelihood.

Currently contains only the Task 0 prerequisite equivalence tests
(searchsorted vs digitize). Tasks 1+ will extend this file with
``TestJitBlockEstimateAccuracy`` and ``TestJitBlockEstimateJaxpr``.
"""

import jax.numpy as jnp
import numpy as np


class TestSearchsortedDigitizeEquivalence:
    """Verify jnp.searchsorted matches np.digitize for spike binning.

    Note: ``jnp.searchsorted`` returns int32 while ``np.digitize`` returns
    int64. ``np.testing.assert_array_equal`` compares values, not dtypes, so
    these tests verify value equivalence only. Both are valid index dtypes
    for ``jax.ops.segment_sum`` — Task 3 (electrode scan) must use the int32
    result directly rather than assuming int64.
    """

    def test_random_spike_times(self):
        """Equivalence on 1000 random spike times."""
        rng = np.random.default_rng(42)
        time_edges = np.linspace(0, 10, 501)  # 500 time bins
        spike_times = np.sort(rng.uniform(-0.1, 10.1, 1000))  # some out of bounds

        reference = np.digitize(spike_times, time_edges[1:-1])
        # np.digitize(x, bins) with default args is equivalent to
        # searchsorted(bins, x, side='right')
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side="right")

        np.testing.assert_array_equal(np.asarray(result), reference)

    def test_spikes_on_edges(self):
        """Spikes exactly on bin edges are handled correctly."""
        time_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        spike_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.5])

        reference = np.digitize(spike_times, time_edges[1:-1])
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side="right")

        np.testing.assert_array_equal(np.asarray(result), reference)

    def test_empty_spikes(self):
        """Empty spike array returns empty result."""
        time_edges = np.linspace(0, 10, 101)
        spike_times = np.array([])

        reference = np.digitize(spike_times, time_edges[1:-1])
        result = jnp.searchsorted(time_edges[1:-1], spike_times, side="right")

        np.testing.assert_array_equal(np.asarray(result), reference)
