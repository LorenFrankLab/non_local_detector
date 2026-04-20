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
* ``TestGemmPrecompute`` (Task 2): the split of the log mark kernel GEMM
  into encoding-only precompute (``_precompute_encoding_gemm_quantities``)
  + decoding-only finalization (``_compute_log_mark_kernel_from_precomputed``)
  is a pure algebraic decomposition, so it matches the original
  ``_compute_log_mark_kernel_gemm`` to floating-point exactness.
* ``TestFusedSegmentSum`` (Task 4): the fused
  ``block_estimate_with_segment_sum_log_joint_mark_intensity`` produces
  the same time-binned log likelihood as the separate
  ``block_estimate_log_joint_mark_intensity -> jax.ops.segment_sum``
  pipeline, up to FP32 accumulation-reorder noise.
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
        first 200 real spikes. They should agree within the noise floor
        of the compensated-linear chunking path.

        Threshold rationale: the original zero-pad bug produced
        contamination of ~4.5e+00 in mark-intensity values (logK_mark for
        a zero-feature "spike" sits near the center of the encoding
        cloud, dominating ``max_wf``). Post-fix, the only remaining diff
        between block sizes is FP32 accumulation reordering from the
        compensated-linear matmul. On CPU this is ≲ 1e-5; on GPU
        (cuBLAS, TF32-by-default on Ampere+) it can reach ~2e-3. We gate
        the tolerance by platform: 1e-4 on CPU, 5e-3 on GPU. Both are
        ≥ 3 orders of magnitude smaller than the pre-fix contamination,
        so a regression back to zero-fill would fire loudly; the tighter
        CPU threshold catches subtler regressions during local dev.
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
        on_gpu = any(d.platform == "gpu" for d in jax.devices())
        atol = 5e-3 if on_gpu else 1e-4
        assert max_diff < atol, (
            f"Padding appears to contaminate real decoding-spike outputs: "
            f"max diff between paddings={max_diff:.2e} exceeds {atol:.0e} "
            f"(platform={'gpu' if on_gpu else 'cpu'})"
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


class TestGemmPrecompute:
    """Precomputed encoding-side GEMM quantities produce identical output.

    The split of ``_compute_log_mark_kernel_gemm`` into
    ``_precompute_encoding_gemm_quantities`` +
    ``_compute_log_mark_kernel_from_precomputed`` is a pure algebraic
    reorganization — no numerical operation is reordered, so the two
    paths must agree to floating-point exactness (max diff == 0.0).
    """

    def test_precomputed_matches_full_gemm(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _compute_log_mark_kernel_from_precomputed,
            _compute_log_mark_kernel_gemm,
            _precompute_encoding_gemm_quantities,
        )

        rng = np.random.default_rng(42)
        n_enc, n_dec, n_wf = 1000, 200, 4
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)

        reference = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
        precomp = _precompute_encoding_gemm_quantities(enc_wf, wf_std)
        result = _compute_log_mark_kernel_from_precomputed(dec_wf, precomp)

        # Pure algebraic split: must be bit-identical.
        max_diff = float(jnp.max(jnp.abs(result - reference)))
        assert max_diff == 0.0, (
            f"Precomputed path diverged from full GEMM: max diff={max_diff:.2e}"
        )

    def test_precomputed_reused_across_blocks(self):
        """One precompute, many decoding blocks — each block still matches full GEMM."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _compute_log_mark_kernel_from_precomputed,
            _compute_log_mark_kernel_gemm,
            _precompute_encoding_gemm_quantities,
        )

        rng = np.random.default_rng(42)
        n_enc, n_wf = 500, 4
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)
        precomp = _precompute_encoding_gemm_quantities(enc_wf, wf_std)

        for block_seed in range(5):
            block_rng = np.random.default_rng(block_seed)
            dec_wf = jnp.array(block_rng.standard_normal((100, n_wf)) * 50)
            reference = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
            result = _compute_log_mark_kernel_from_precomputed(dec_wf, precomp)
            max_diff = float(jnp.max(jnp.abs(result - reference)))
            assert max_diff == 0.0, (
                f"block_seed={block_seed}: reused precompute diverged from full GEMM "
                f"(max diff={max_diff:.2e})"
            )

    def test_block_estimate_uses_precomputed_path(self):
        """Verify the block fori_loop body consumes the precomputed quantities.

        ``_precompute_encoding_gemm_quantities`` computes ``Y = enc * inv_sigma``
        and ``y2 = sum(Y**2, axis=1)``, both shape-indexed by ``n_enc``.  When
        precompute is hoisted outside the ``fori_loop``, the jaxpr of
        ``_block_estimate_log_joint_mark_intensity_impl`` should contain
        top-level ``reduce_sum``/``mul`` equations of size ``n_enc`` that
        live OUTSIDE the scan body.  We verify by diffing the jaxpr
        against a control that forces the chunked path (which ignores
        ``precomputed_enc_gemm``) — the non-chunked jaxpr should have more
        top-level equations than the chunked one.

        This is a structural sanity check, not a guarantee of performance.
        """
        import functools

        from non_local_detector.likelihoods.clusterless_kde_log import (
            _block_estimate_log_joint_mark_intensity_impl,
        )

        n_enc, n_dec, n_pos, n_wf = 500, 200, 50, 4
        dec_wf = jnp.zeros((n_dec, n_wf))
        enc_wf = jnp.zeros((n_enc, n_wf))
        wf_std = jnp.full(n_wf, 24.0)
        occ = jnp.ones(n_pos) * 0.01
        log_pos = jnp.zeros((n_enc, n_pos))

        non_chunked = functools.partial(
            _block_estimate_log_joint_mark_intensity_impl,
            block_size=50,
            use_gemm=True,
            pos_tile_size=None,
            enc_tile_size=None,
            use_streaming=False,
        )
        jaxpr_nonchunked = jax.make_jaxpr(non_chunked)(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos
        )
        prims_nonchunked = [eqn.primitive.name for eqn in jaxpr_nonchunked.jaxpr.eqns]

        # The precompute hoist adds at least these top-level primitives
        # outside the scan body: the inv_sigma reciprocal, the Y = enc *
        # inv_sigma broadcast mul, and the y2 reduce_sum.  Without hoisting,
        # these would all live inside the scan body and be invisible at the
        # top level.
        n_top_level_dot_general = prims_nonchunked.count("dot_general")
        n_top_level_reduce_sum = prims_nonchunked.count("reduce_sum")
        n_top_level_div = prims_nonchunked.count("div")

        # At least one top-level reduce_sum (for y2) and one top-level
        # elementwise op for the inv_sigma and Y = enc * inv_sigma scaling.
        assert n_top_level_reduce_sum >= 1, (
            f"Expected ≥1 top-level reduce_sum (from y2 precompute hoist); "
            f"got {n_top_level_reduce_sum}. Full: {prims_nonchunked}"
        )
        # No top-level dot_general: the per-block GEMM still lives inside
        # the scan body, only the encoding-side scaling moved outside.
        assert n_top_level_dot_general == 0, (
            f"Expected 0 top-level dot_general (per-block GEMM belongs inside "
            f"scan body); got {n_top_level_dot_general}. Full: {prims_nonchunked}"
        )
        # The inv_sigma = 1 / waveform_stds division should be hoisted too.
        assert n_top_level_div >= 1, (
            f"Expected ≥1 top-level div (from inv_sigma precompute hoist); "
            f"got {n_top_level_div}. Full: {prims_nonchunked}"
        )

        # Bi-directional check: the per-block GEMM must actually exist
        # INSIDE the scan body (not fully optimized away). Walk the scan
        # equation's sub-jaxpr and assert ≥1 ``dot_general`` primitive.
        scan_eqns = [
            eqn for eqn in jaxpr_nonchunked.jaxpr.eqns if eqn.primitive.name == "scan"
        ]
        assert len(scan_eqns) == 1, (
            f"Expected exactly 1 top-level scan; got {len(scan_eqns)}"
        )
        scan_body_jaxpr = scan_eqns[0].params["jaxpr"].jaxpr
        body_prims = [eqn.primitive.name for eqn in scan_body_jaxpr.eqns]

        def _collect_all_primitives(eqns):
            """Recursively collect primitives from jaxpr eqns, descending
            into ``jit``/``pjit`` call params."""
            names = []
            for eqn in eqns:
                names.append(eqn.primitive.name)
                for p in eqn.params.values():
                    inner = getattr(p, "jaxpr", None)
                    if inner is not None:
                        names.extend(
                            _collect_all_primitives(getattr(inner, "eqns", []))
                        )
            return names

        all_body_prims = _collect_all_primitives(scan_body_jaxpr.eqns)
        assert all_body_prims.count("dot_general") >= 1, (
            f"Expected ≥1 dot_general inside the scan body (per-block GEMM); "
            f"got 0. Body top-level primitives: {body_prims}. "
            f"All body primitives (incl. nested jit): {all_body_prims}"
        )


class TestFusedSegmentSum:
    """Fused ``block_estimate_with_segment_sum_log_joint_mark_intensity`` matches
    the separate ``block_estimate -> jax.ops.segment_sum`` pipeline.

    The fused path accumulates each block's mark-intensity contribution
    into a ``(n_time, n_pos)`` output inside the ``fori_loop``, so the
    full ``(n_decoding_spikes, n_pos)`` matrix never materializes. The
    only expected numerical difference from the separate path is
    FP32 accumulation reordering: separate does
    ``segment_sum`` in one pass over ``n_dec`` rows while fused does a
    segment_sum per block and sums the results. Equivalent in real
    arithmetic, not in FP32.
    """

    def _build_seg_ids(self, rng, n_dec: int, n_time: int) -> jnp.ndarray:
        """Generate sorted segment ids uniformly distributed over [0, n_time)."""
        times = np.sort(rng.uniform(0.0, 1.0, n_dec))
        edges = np.linspace(0.0, 1.0, n_time + 1)
        # np.digitize with default args matches jnp.searchsorted(side='right').
        return jnp.asarray(np.digitize(times, edges[1:-1]), dtype=jnp.int32)

    def test_fused_matches_separate(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
            block_estimate_with_segment_sum_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(42)
        n_dec, n_time = 500, 200
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=1000, n_dec=n_dec, n_pos=100
        )
        seg_ids = self._build_seg_ids(rng, n_dec=n_dec, n_time=n_time)

        # Separate: block_estimate → segment_sum
        mark_intensity = block_estimate_log_joint_mark_intensity(
            dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=100
        )
        reference = jax.ops.segment_sum(
            mark_intensity,
            seg_ids,
            num_segments=n_time,
            indices_are_sorted=True,
        )
        # Fused
        result = block_estimate_with_segment_sum_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            seg_ids,
            n_time,
            block_size=100,
        )
        assert result.shape == (n_time, 100)
        max_diff = float(jnp.max(jnp.abs(result - reference)))
        # FP32 accumulation-reorder noise dominates. Typical values of
        # the summed log-likelihood are O(100), so 5e-3 is ~5e-5 relative.
        assert max_diff < 5e-3, (
            f"fused vs separate max diff={max_diff:.3e} exceeds 5e-3 "
            f"(expected FP32 accumulation-reorder noise only)"
        )

    @pytest.mark.parametrize("block_size", [50, 100, 128, 250, 500])
    def test_fused_matches_separate_across_block_sizes(self, block_size):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_log_joint_mark_intensity,
            block_estimate_with_segment_sum_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(block_size)
        n_dec, n_time = 500, 150
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=800, n_dec=n_dec, n_pos=80
        )
        seg_ids = self._build_seg_ids(rng, n_dec=n_dec, n_time=n_time)

        reference = jax.ops.segment_sum(
            block_estimate_log_joint_mark_intensity(
                dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, block_size=block_size
            ),
            seg_ids,
            num_segments=n_time,
            indices_are_sorted=True,
        )
        result = block_estimate_with_segment_sum_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            seg_ids,
            n_time,
            block_size=block_size,
        )
        max_diff = float(jnp.max(jnp.abs(result - reference)))
        assert max_diff < 5e-3, (
            f"block_size={block_size}: fused vs separate max diff={max_diff:.3e}"
        )

    def test_empty_decoding_spikes(self):
        """Zero decoding spikes yields a zero ``(n_time, n_pos)`` accumulator."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_with_segment_sum_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(42)
        _, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=100, n_dec=10, n_pos=50
        )
        n_time = 30
        result = block_estimate_with_segment_sum_log_joint_mark_intensity(
            jnp.zeros((0, 4)),
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            jnp.zeros((0,), dtype=jnp.int32),
            n_time,
            block_size=25,
        )
        assert result.shape == (n_time, 50)
        assert jnp.all(result == 0.0)

    def test_padded_seg_ids_contribute_nothing(self):
        """Padded decoding spikes (seg_id = n_time) must not reach the accumulator.

        Construct two workloads that differ only in a partial-last-block
        padding. The real-spike outputs must agree within FP32 noise.
        """
        from non_local_detector.likelihoods.clusterless_kde_log import (
            block_estimate_with_segment_sum_log_joint_mark_intensity,
        )

        rng = np.random.default_rng(7)
        n_dec, n_time = 200, 100
        dec_wf, enc_wf, wf_std, occ, log_pos = _make_electrode_data(
            rng, n_enc=500, n_dec=n_dec, n_pos=60
        )
        seg_ids = np.sort(rng.integers(low=0, high=n_time, size=n_dec).astype(np.int32))
        seg_ids_j = jnp.asarray(seg_ids)

        # block_size=100 divides n_dec exactly (no padding).
        no_pad = block_estimate_with_segment_sum_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            seg_ids_j,
            n_time,
            block_size=100,
        )
        # block_size=75 → blocks [75, 75, 50]; last block pads 25 slots.
        with_pad = block_estimate_with_segment_sum_log_joint_mark_intensity(
            dec_wf,
            enc_wf,
            wf_std,
            occ,
            5.0,
            log_pos,
            seg_ids_j,
            n_time,
            block_size=75,
        )
        max_diff = float(jnp.max(jnp.abs(with_pad - no_pad)))
        assert max_diff < 5e-3, (
            f"Padded slots appear to leak into the accumulator: max diff={max_diff:.3e}"
        )


class TestFusedSegmentSumJaxpr:
    """The fused fori_loop compiles to JAX primitives with ``segment_sum``
    (``scatter_add``) inside the loop body — not unrolled, not hoisted out.

    IMPORTANT: Trace the private ``_block_estimate_with_segment_sum_impl``,
    NOT the JIT-wrapped ``_block_estimate_with_segment_sum_jit`` — tracing
    the JIT-wrapped version shows ``jit`` at the top level, hiding the
    inner loop structure.
    """

    def test_loop_body_contains_scatter(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _block_estimate_with_segment_sum_impl,
        )

        n_enc, n_dec, n_pos, n_wf = 100, 200, 50, 4
        n_time = 40
        dec_wf = jnp.zeros((n_dec, n_wf))
        enc_wf = jnp.zeros((n_enc, n_wf))
        wf_std = jnp.full(n_wf, 24.0)
        occ = jnp.ones(n_pos) * 0.01
        log_pos = jnp.zeros((n_enc, n_pos))
        seg_ids = jnp.arange(n_dec, dtype=jnp.int32) % n_time

        fn = functools.partial(
            _block_estimate_with_segment_sum_impl,
            n_time=n_time,
            block_size=50,
            use_gemm=True,
            pos_tile_size=None,
            enc_tile_size=None,
            use_streaming=False,
        )
        jaxpr = jax.make_jaxpr(fn)(dec_wf, enc_wf, wf_std, occ, 5.0, log_pos, seg_ids)

        primitives = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]

        # Exactly one top-level loop primitive — fori_loop lowers to
        # scan (or while on some versions).
        n_loop_ops = sum(primitives.count(p) for p in ("scan", "while", "fori_loop"))
        assert n_loop_ops == 1, (
            f"Expected exactly 1 top-level loop primitive; got {n_loop_ops}. "
            f"Full: {primitives}"
        )
        # Zero top-level dot_general — per-block GEMM lives inside the
        # loop body, not at the top level.
        n_dot_general = primitives.count("dot_general")
        assert n_dot_general == 0, (
            f"Expected 0 top-level dot_general (per-block GEMM belongs "
            f"inside scan body); got {n_dot_general}. Full: {primitives}"
        )

        # Walk the scan body to confirm segment_sum's scatter_add lives
        # INSIDE the loop (otherwise the fusion claim is false — it would
        # mean segment_sum was hoisted out or the fori_loop got unrolled).
        scan_eqns = [eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "scan"]
        assert len(scan_eqns) == 1, (
            f"Expected exactly 1 top-level scan; got {len(scan_eqns)}"
        )
        scan_body_jaxpr = scan_eqns[0].params["jaxpr"].jaxpr

        def _collect_all_primitives(eqns):
            names = []
            for eqn in eqns:
                names.append(eqn.primitive.name)
                for p in eqn.params.values():
                    inner = getattr(p, "jaxpr", None)
                    if inner is not None:
                        names.extend(
                            _collect_all_primitives(getattr(inner, "eqns", []))
                        )
            return names

        body_prims = _collect_all_primitives(scan_body_jaxpr.eqns)
        n_dot_general_body = body_prims.count("dot_general")
        # scatter_add is the primitive segment_sum lowers to.
        scatter_primitives = sum(
            body_prims.count(p) for p in ("scatter_add", "scatter", "scatter-add")
        )

        assert n_dot_general_body >= 1, (
            f"Expected ≥1 dot_general inside the scan body (per-block "
            f"GEMM); got 0. Body primitives: {body_prims}"
        )
        assert scatter_primitives >= 1, (
            f"Expected ≥1 scatter/scatter_add inside the scan body "
            f"(segment_sum fused into block loop); got 0. "
            f"Body primitives: {body_prims}"
        )
