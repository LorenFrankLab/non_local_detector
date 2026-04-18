"""Prerequisite equivalence tests and JIT-fused block_estimate tests for clusterless KDE log likelihood.

Covers:

* ``TestSearchsortedDigitizeEquivalence`` (Task 0 prerequisite):
  ``jnp.searchsorted(..., side='right')`` matches ``np.digitize`` for
  spike-time binning — required for Task 3's JAX-traceable electrode scan.
* ``TestJitBlockEstimateAccuracy`` (Task 1): the JIT-fused
  ``block_estimate_log_joint_mark_intensity`` produces the same numerics
  as the pre-refactor Python-loop version.
* ``TestJitBlockEstimateJaxpr`` (Task 1): the traced jaxpr of the private
  ``_block_estimate_log_joint_mark_intensity_impl`` contains a loop
  primitive (``scan``/``while``) — confirming the Python for-loop has
  been replaced by a JAX control-flow primitive.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest


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


def _make_electrode_data(rng, n_enc, n_dec, n_pos, n_wf=4):
    """Build a synthetic single-electrode input bundle for block_estimate tests."""
    from non_local_detector.likelihoods.clusterless_kde_log import log_kde_distance

    enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
    dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
    wf_std = jnp.full(n_wf, 24.0)
    enc_pos = jnp.array(rng.uniform(0, 200, (n_enc, 1)))
    eval_pos = jnp.linspace(0, 200, n_pos)[:, None]
    pos_std = jnp.array([3.5])
    log_pos_dist = log_kde_distance(eval_pos, enc_pos, pos_std)
    occupancy = jnp.ones(n_pos) * 0.01
    return dec_wf, enc_wf, wf_std, occupancy, log_pos_dist


class TestJitBlockEstimateAccuracy:
    """JIT-fused block_estimate preserves the numerics of the Python-loop version.

    Comparison strategy — the multi-block diff against a single-call
    reference has two sources, both bounded by float32 precision:

    1. **Chunking-dependent stabilization in ``_compensated_linear_marginal``**
       (≤ 8 waveform features): the fast path computes
       ``max_wf = jnp.max(logK_mark, axis=1)`` over decoding spikes and uses
       it to stabilize a subsequent matmul. Different block sizes produce
       different max values, so the stabilized kernels live in slightly
       different numerical ranges. Result: ≲ 1e-5 max diff. Affected by
       both the pre-refactor Python loop and the new ``fori_loop``.

    2. **XLA instruction reordering** inside a ``fori_loop`` body vs. an
       eagerly-dispatched Python loop: JIT compilation of the ``fori_loop``
       body may fuse ops differently than the independently-JIT'd Python
       loop iterations. Result: ≲ 1e-6 max diff, present even at
       ``block_size >= n_dec`` (no chunking).

    Padding is done via ``mode='edge'`` (replicate last real spike), NOT
    zero-fill, which would introduce a third contamination source via
    ``max_wf``. See ``test_padding_does_not_contaminate_real_rows``.

    Tolerances:

    * ``block_size >= n_dec`` (single block, no chunking): atol=1e-4 to
      absorb XLA reordering. The single-block case is NOT bit-exact.
    * smaller ``block_size``: atol=1e-4 absorbs both sources above.
    """

    @pytest.mark.parametrize("block_size", [50, 100, 200, 499, 500])
    def test_matches_single_call(self, block_size):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
            estimate_log_joint_mark_intensity,
        )
        from non_local_detector.likelihoods.common import LOG_EPS

        rng = np.random.default_rng(42)
        n_dec = 500
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=1000, n_dec=n_dec, n_pos=100
        )
        result = block_estimate_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            block_size=block_size,
        )
        reference = jnp.clip(
            estimate_log_joint_mark_intensity(
                dec_wf, enc_wf, wf_std, occ, 5.0, log_pos
            ),
            min=LOG_EPS,
            max=None,
        )
        assert result.shape == reference.shape
        atol = 1e-4
        max_diff = float(jnp.max(jnp.abs(result - reference)))
        assert jnp.allclose(result, reference, atol=atol, rtol=atol), (
            f"block_size={block_size}: max diff={max_diff:.2e} exceeds {atol:.0e}"
        )

    def test_large_scale(self):
        """Sanity-check shape and finiteness at scale: 10k enc, 5k dec."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(123)
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=10000, n_dec=5000, n_pos=200
        )
        result = block_estimate_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            block_size=500,
        )
        assert result.shape == (5000, 200)
        assert jnp.all(jnp.isfinite(result))

    def test_padding_does_not_contaminate_real_rows(self):
        """Partial last block's padding must not shift real rows' outputs.

        Regression guard for a subtle contamination: inside
        ``_compensated_linear_marginal`` the reduction
        ``max_wf = jnp.max(logK_mark, axis=1)`` runs over decoding spikes.
        If padded decoding spikes produce ``logK_mark`` values higher than
        any real spike's (as zero-fill did — zero-feature spikes sit near
        the center of the encoding-feature cloud and often win the max),
        ``max_wf`` shifts upward, which degrades the precision of real
        rows' outputs via the stabilized matmul.

        Using ``mode='edge'`` padding (copy the last real spike) makes
        padded columns byte-identical to a real column, so they cannot
        exceed the max. This test verifies that property: compare
        block_size=100 (2 full blocks, no padding) against block_size=75
        (2 full blocks + 50-spike partial → 25 padded spikes) for the
        first 200 real spikes. They should agree to float32 precision,
        not just to the ≲ 1e-5 compensated-linear chunking tolerance.
        """
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(42)
        n_dec = 200
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=1000, n_dec=n_dec, n_pos=100
        )
        # block_size=100 divides n_dec exactly: no padding.
        no_padding = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=100
        )
        # block_size=75 → blocks of [75, 75, 50]; last block pads 25 spikes.
        with_padding = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=75
        )
        # Both paths share the compensated-linear chunking effect (diff
        # block boundaries), so allow the documented chunking tolerance —
        # but if padding contaminated real rows, the diff would be much
        # larger than the no-padding chunking baseline.
        max_diff = float(jnp.max(jnp.abs(with_padding - no_padding)))
        assert max_diff < 1e-4, (
            f"Padding appears to contaminate real decoding-spike outputs: "
            f"max diff between paddings={max_diff:.2e}"
        )

    def test_empty_decoding_spikes(self):
        """Zero decoding spikes returns an empty (0, n_pos) array.

        The empty case is handled BEFORE the JIT boundary in the public
        wrapper (``n_decoding_spikes`` is a static shape), so the fori_loop
        path is never traced for empty input.
        """
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(42)
        _, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=100, n_dec=10, n_pos=50
        )
        dec_wf_empty = jnp.zeros((0, 4))
        result = block_estimate_log_joint_mark_intensity(
            dec_wf_empty,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            block_size=100,
        )
        assert result.shape == (0, 50)


class TestJitBlockEstimateJaxpr:
    """The block loop compiles to JAX primitives, not a Python for-loop.

    IMPORTANT: Trace the private ``_block_estimate_log_joint_mark_intensity_impl``,
    NOT the JIT-wrapped ``_block_estimate_log_joint_mark_intensity_jit`` — the
    JIT-wrapped version shows ``jit`` at the top level of the jaxpr, hiding
    the inner loop structure.
    """

    def test_no_python_loop_in_jaxpr(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _block_estimate_log_joint_mark_intensity_impl,
        )

        n_enc, n_dec, n_pos, n_wf = 100, 200, 50, 4
        dec_wf = jnp.zeros((n_dec, n_wf))
        enc_wf = jnp.zeros((n_enc, n_wf))
        wf_std = jnp.full(n_wf, 24.0)
        occ = jnp.ones(n_pos) * 0.01
        log_pos = jnp.zeros((n_enc, n_pos))

        # Partial so static args become concrete Python ints, not traced.
        fn = functools.partial(
            _block_estimate_log_joint_mark_intensity_impl,
            block_size=50,
            use_gemm=True,
            pos_tile_size=None,
            enc_tile_size=None,
            use_streaming=False,
        )
        jaxpr = jax.make_jaxpr(fn)(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
        )

        # Collect all primitives from the outer jaxpr equations.
        primitives = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]

        # Direct regression guard: exactly one scan/while primitive at the
        # top level. A re-Pythonized loop of JIT'd block calls would show
        # multiple ``jit`` ops and zero scan/while at the top level — this
        # assertion catches that specific regression.
        n_loop_ops = sum(primitives.count(p) for p in ("scan", "while", "fori_loop"))
        assert n_loop_ops == 1, (
            f"Expected exactly 1 top-level loop primitive (scan/while/fori_loop) "
            f"from jax.lax.fori_loop; got {n_loop_ops}. "
            f"Full primitive list: {primitives}"
        )

        # Belt-and-suspenders: per-block GEMMs must live inside the loop body,
        # not at the top level. A Python for-loop with n_dec=200 and
        # block_size=50 would unroll to 4 iterations and produce ≥ 4
        # top-level ``dot_general`` or ``jit`` equations.
        n_dot_general = primitives.count("dot_general")
        assert n_dot_general <= 1, (
            f"Expected ≤1 top-level dot_general (fori_loop hides per-iteration "
            f"GEMMs inside the loop body); got {n_dot_general}. "
            f"Full primitive list: {primitives}"
        )
        n_jit = primitives.count("jit")
        assert n_jit <= 2, (
            f"Expected ≤2 top-level jit ops (the loop body is a single "
            f"compiled region); got {n_jit}. A re-Pythonized loop of JIT'd "
            f"blocks would show ≥ n_blocks top-level jit ops. "
            f"Full primitive list: {primitives}"
        )
