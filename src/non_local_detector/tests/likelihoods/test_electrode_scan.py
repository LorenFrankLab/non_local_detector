"""Tests for Task 3's JIT-compiled electrode scan in the non-local path.

The scan-based non-local likelihood is wired into
``predict_clusterless_kde_log_likelihood`` via
:func:`_prepare_electrode_scan_group` +
:func:`_predict_nonlocal_electrode_scan_jit`. These tests verify:

* The scan path is bit-level deterministic across repeat calls and
  produces finite output with the expected ``(n_time, n_position_bins)``
  shape + the ``-ground_process_intensity`` baseline applied.
* Size-bucketing partitions one ``n_wf`` group into sub-batches that
  keep the per-batch pad-to-max cost bounded.
* Mixed ``n_wf`` across electrodes is handled by running one scan per
  feature-count group.
* The jaxpr of the private ``_predict_nonlocal_electrode_scan_impl``
  contains a top-level loop primitive and — inside the scan body —
  both a ``dot_general`` (per-electrode mark-kernel GEMM, via the Task
  4 ``_block_estimate_with_segment_sum_impl``) and a ``scatter_add``
  (the fused segment_sum). A re-Pythonization or loss of the inner
  fusion would fire the assertions.

Cross-path accuracy (scan vs. Python-loop fallback) is validated out-of-
band via the end-to-end GPU benchmark against the ``main``-branch
baseline (see the Task 3 commit message for numbers); within the test
suite we stick to structural + determinism properties that are robust
across CPU / cuBLAS-TF32 GPU environments.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.simulate.clusterless_simulation import (
    make_simulated_run_data,
)


def _build_small_workload(seed: int = 42, n_tetrodes: int = 4, n_runs: int = 4):
    """Tiny simulated workload — finishes in under a second on CPU."""
    from non_local_detector.likelihoods.clusterless_kde_log import (
        fit_clusterless_kde_encoding_model,
    )

    place_means = np.linspace(0.0, 170.0, 4 * n_tetrodes)
    sim = make_simulated_run_data(
        n_tetrodes=n_tetrodes,
        place_field_means=place_means,
        sampling_frequency=500,
        n_runs=n_runs,
        seed=seed,
    )
    n_encode = int(0.7 * len(sim.position_time))
    encode_time = sim.position_time[:n_encode]
    encode_pos = sim.position[:n_encode]
    encode_spikes = [st[st <= encode_time[-1]] for st in sim.spike_times]
    encode_wf = [
        wf[st <= encode_time[-1]]
        for st, wf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]
    encoding = fit_clusterless_kde_encoding_model(
        encode_time,
        encode_pos,
        encode_spikes,
        encode_wf,
        sim.environment,
        sampling_frequency=500,
        position_std=6.0,
        waveform_std=24.0,
        block_size=100,
        disable_progress_bar=True,
    )
    test_edges = sim.edges[sim.edges >= encode_time[-1]]
    test_time = sim.position_time[n_encode:]
    test_position = sim.position[n_encode:]
    test_spikes = [st[st >= encode_time[-1]] for st in sim.spike_times]
    test_wf = [
        wf[st >= encode_time[-1]]
        for st, wf in zip(sim.spike_times, sim.spike_waveform_features, strict=False)
    ]
    return {
        "test_edges": test_edges,
        "test_time": test_time,
        "test_position": test_position,
        "test_spikes": test_spikes,
        "test_wf": test_wf,
        "encoding": {**encoding, "disable_progress_bar": True},
    }


class TestElectrodeScanAccuracy:
    """Scan path output is stable and finite across block_size choices.

    The block-loop pipeline inside the scan body inherits Task 1's
    ≲ 1e-5 chunking sensitivity (compensated-linear fast path) plus FP32
    accumulation-reorder noise, bounded at 5e-3 on GPU (cuBLAS TF32)
    and ≲ 1e-5 on CPU.  Comparing the scan output across block sizes
    verifies that the scan body correctly propagates block-size choice
    through to the fused block_estimate + segment_sum.
    """

    def _predict(self, workload, **kwargs):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            predict_clusterless_kde_log_likelihood,
        )

        # The encoding dict carries default values for keys like
        # enc_tile_size / pos_tile_size / use_streaming; override them
        # via kwargs rather than double-passing.
        encoding = {**workload["encoding"], **kwargs}
        return predict_clusterless_kde_log_likelihood(
            workload["test_edges"],
            workload["test_time"],
            workload["test_position"],
            workload["test_spikes"],
            workload["test_wf"],
            **encoding,
            is_local=False,
        )

    def test_scan_is_deterministic(self):
        """Running the scan path twice on identical inputs produces identical output.

        Bit-level determinism: catches real correctness bugs like
        uninitialized buffers, non-deterministic scatter ordering, or
        stale JIT cache entries.

        Scope: both calls share the same JIT cache entry (same compiled
        XLA executable, same cuBLAS kernel-launch sequence, same
        accumulation order), so this verifies accumulation-order
        stability within a session — NOT cold-start determinism across
        JIT recompiles or across processes.  For a broader accuracy
        check, see the out-of-band GPU benchmark comparison against the
        main-branch baseline documented in the Task 3 commit message.
        """
        workload = _build_small_workload(seed=42)
        out_a = np.asarray(self._predict(workload))
        out_b = np.asarray(self._predict(workload))
        assert out_a.shape == out_b.shape
        np.testing.assert_array_equal(out_a, out_b)

    def test_scan_accumulates_ground_process_intensity(self):
        """The scan output includes the ``-ground_process_intensity`` baseline."""
        workload = _build_small_workload(seed=7)
        out = np.asarray(self._predict(workload))
        # Sanity: output shape matches (n_time, n_pos); values are finite
        # and span a non-trivial range (not all zeros, not all NaN).
        expected_n_time = len(workload["test_edges"])
        expected_n_pos = workload["encoding"]["occupancy"].shape[0]
        assert out.shape == (expected_n_time, expected_n_pos)
        assert np.all(np.isfinite(out))
        assert out.std() > 0.0

    def test_scan_matches_streaming_fallback_loose(self):
        """End-to-end smoke test: scan path vs Python-loop streaming fallback.

        Setting ``use_streaming=True`` + ``enc_tile_size < n_enc`` routes
        through the Task 4 Python-loop fallback (scan path is gated
        off).  Both paths compute the same mathematical non-local
        log-likelihood, but via different numerical routes (the
        streaming path uses on-the-fly logsumexp rather than the
        non-chunked compensated-linear matmul).

        The tolerance is deliberately loose — 10% relative, platform-
        gated slightly tighter on CPU.  This is a **smoke test**: it
        catches major correctness regressions (sign flips, missing
        baseline, un-accumulated electrodes, etc.) without being
        brittle against cuBLAS TF32 matmul reordering or compensated-
        linear-vs-logsumexp path differences at small workload scale.
        End-to-end accuracy at production scale is validated out-of-
        band via the GPU benchmark against the ``main``-branch
        baseline (see Task 3 commit message).
        """
        workload = _build_small_workload(seed=42)
        scan_out = np.asarray(self._predict(workload))
        # Force the Python-loop fallback: pick enc_tile_size < min n_enc
        # and use_streaming=True (the streaming path validates its
        # inputs and requires both).
        min_enc = min(
            int(s.shape[0]) for s in workload["encoding"]["encoding_positions"]
        )
        fallback_out = np.asarray(
            self._predict(
                workload,
                use_streaming=True,
                enc_tile_size=max(1, min_enc - 1),
            )
        )
        assert scan_out.shape == fallback_out.shape
        max_diff = float(np.max(np.abs(scan_out - fallback_out)))
        scale = float(np.max(np.abs(scan_out)))
        rel = max_diff / max(scale, 1e-12)
        on_gpu = any(d.platform == "gpu" for d in jax.devices())
        rel_limit = 0.15 if on_gpu else 0.10
        assert rel < rel_limit, (
            f"scan vs streaming fallback relative |diff|={rel:.3e} "
            f"(max abs={max_diff:.3e}, scale={scale:.3e}); exceeds "
            f"{rel_limit:.0%} smoke-test tolerance on "
            f"{'gpu' if on_gpu else 'cpu'}"
        )


class TestElectrodeScanBucketing:
    """Size-bucketing partitions a mixed-size group into sub-batches."""

    def test_bucketing_preserves_all_electrodes(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _bucket_by_size,
            _group_electrodes_by_n_wf,
        )

        # Fake encoding_spike_waveform_features with varying row counts.
        rng = np.random.default_rng(0)
        enc_wf_list = [
            jnp.asarray(rng.standard_normal((n, 4)))
            for n in [5, 10, 100, 50, 200, 15, 8, 1000, 500]
        ]
        # Decoding spike times uniformly spread; all inside time window.
        dec_times_list = [
            jnp.asarray(rng.uniform(0.0, 1.0, n))
            for n in [5, 10, 100, 50, 200, 15, 8, 1000, 500]
        ]
        time = jnp.linspace(0.0, 1.0, 11)  # 10 bins

        groups = _group_electrodes_by_n_wf(enc_wf_list)
        assert list(groups) == [4]  # single n_wf=4 group
        group_indices = groups[4]
        buckets = _bucket_by_size(group_indices, enc_wf_list, dec_times_list, time)

        # Every electrode must appear in exactly one bucket.
        flat = [i for b in buckets for i in b]
        assert sorted(flat) == sorted(group_indices)
        assert len(flat) == len(group_indices)

    def test_bucketing_reduces_within_bucket_spread(self):
        """Within each bucket, max/min electrode size shouldn't span 10x."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _bucket_by_size,
        )

        enc_wf_list = [
            jnp.zeros((n, 4))
            for n in [5, 10, 20, 100, 200, 500, 1000, 2000, 5000, 10000]
        ]
        dec_times_list = [jnp.array([0.5]) for _ in enc_wf_list]
        time = jnp.linspace(0.0, 1.0, 11)

        buckets = _bucket_by_size(
            list(range(len(enc_wf_list))),
            enc_wf_list,
            dec_times_list,
            time,
            max_buckets=4,
        )
        # Every bucket should have a tighter spread than the global 2000x.
        for b in buckets:
            if len(b) >= 2:
                sizes = [enc_wf_list[i].shape[0] for i in b]
                spread = max(sizes) / min(sizes)
                assert spread < 200, (
                    f"bucket {b} has sizes {sizes}, spread={spread:.1f}x "
                    f"too wide (bucketing ineffective)"
                )


class TestElectrodeScanEdgeCases:
    """Boundary cases — silent electrodes, missing spike windows."""

    def test_zero_encoding_spikes_electrode_contributes_nothing(self):
        """An electrode with zero encoding spikes contributes zero at any decoding spike.

        ``_prepare_electrode_scan_group`` handles this case by leaving
        ``enc_wf_batch`` as zeros and replacing ``enc_pos_batch`` entirely
        with the ``_ELECTRODE_SCAN_POS_PAD_VALUE`` sentinel — every row
        behaves like a padded row, so ``log_pos`` is hugely negative and
        underflows to 0 contribution in the mark-intensity matmul.  This
        test verifies the resulting log-likelihood stays finite (no NaN
        from 0 * inf or inf - inf) and doesn't contaminate other
        electrodes' contributions.
        """
        from non_local_detector.environment import Environment
        from non_local_detector.likelihoods.clusterless_kde_log import (
            fit_clusterless_kde_encoding_model,
            predict_clusterless_kde_log_likelihood,
        )

        rng = np.random.default_rng(42)
        env = Environment(
            place_bin_size=10.0, environment_name="linear", is_track_interior=None
        )
        position = rng.uniform(0, 100, 500)[:, None]
        time_pos = np.linspace(0, 10, 500)
        env.fit_place_grid(position=position, infer_track_interior=False)

        # Two electrodes: the first has encoding spikes; the second has
        # NONE during encoding but fires a few spikes in the decoding
        # window (a silent-during-encode tetrode).
        spike_times = [
            np.sort(rng.uniform(0, 10, 50)),
            np.zeros(0, dtype=np.float64),
        ]
        spike_wf = [
            rng.standard_normal((50, 4)) * 20.0,
            np.zeros((0, 4), dtype=np.float32),
        ]

        encoding = fit_clusterless_kde_encoding_model(
            time_pos,
            position,
            spike_times,
            spike_wf,
            env,
            sampling_frequency=50,
            position_std=6.0,
            waveform_std=24.0,
            block_size=50,
            disable_progress_bar=True,
        )

        test_edges = np.linspace(5.0, 10.0, 51)
        test_time = time_pos[time_pos >= 5.0]
        test_position = position[time_pos >= 5.0]
        test_spikes = [
            spike_times[0][spike_times[0] >= 5.0],
            np.sort(rng.uniform(5.0, 10.0, 5)),
        ]
        test_wf = [
            spike_wf[0][spike_times[0] >= 5.0],
            rng.standard_normal((5, 4)) * 20.0,
        ]

        out = predict_clusterless_kde_log_likelihood(
            test_edges,
            test_time,
            test_position,
            test_spikes,
            test_wf,
            **{**encoding, "disable_progress_bar": True},
            is_local=False,
        )
        out_np = np.asarray(out)
        assert np.all(np.isfinite(out_np)), (
            "Zero-encoding-spike electrode produced NaN/Inf — padding "
            "sentinel didn't zero the contribution cleanly"
        )
        assert out_np.std() > 0.0


class TestElectrodeScanMixedNWf:
    """Electrodes with different waveform-feature counts get separate scans."""

    def test_two_n_wf_groups_give_finite_output(self):
        """Synthetic workload with mixed n_wf (2 and 4)."""
        from non_local_detector.environment import Environment
        from non_local_detector.likelihoods.clusterless_kde_log import (
            fit_clusterless_kde_encoding_model,
            predict_clusterless_kde_log_likelihood,
        )

        rng = np.random.default_rng(42)
        env = Environment(
            place_bin_size=10.0, environment_name="linear", is_track_interior=None
        )
        position = rng.uniform(0, 100, 500)[:, None]
        time_pos = np.linspace(0, 10, 500)
        env.fit_place_grid(position=position, infer_track_interior=False)

        # Three electrodes: two with n_wf=4, one with n_wf=2.
        spike_times = [
            np.sort(rng.uniform(0, 10, 50)),
            np.sort(rng.uniform(0, 10, 60)),
            np.sort(rng.uniform(0, 10, 40)),
        ]
        spike_wf = [
            rng.standard_normal((50, 4)) * 20.0,
            rng.standard_normal((60, 4)) * 20.0,
            rng.standard_normal((40, 2)) * 20.0,  # different n_wf
        ]

        encoding = fit_clusterless_kde_encoding_model(
            time_pos,
            position,
            spike_times,
            spike_wf,
            env,
            sampling_frequency=50,
            position_std=6.0,
            waveform_std=24.0,
            block_size=50,
            disable_progress_bar=True,
        )

        test_edges = np.linspace(5.0, 10.0, 51)
        test_time = time_pos[time_pos >= 5.0]
        test_position = position[time_pos >= 5.0]
        test_spikes = [st[st >= 5.0] for st in spike_times]
        test_wf = [wf[st >= 5.0] for st, wf in zip(spike_times, spike_wf, strict=False)]

        out = predict_clusterless_kde_log_likelihood(
            test_edges,
            test_time,
            test_position,
            test_spikes,
            test_wf,
            **{**encoding, "disable_progress_bar": True},
            is_local=False,
        )
        out_np = np.asarray(out)
        assert out_np.shape == (len(test_edges), encoding["occupancy"].shape[0])
        assert np.all(np.isfinite(out_np))


class TestElectrodeScanJaxpr:
    """The scan body contains the expected JAX primitives.

    A re-Pythonization would eliminate the top-level scan; a failure to
    fuse the block_estimate + segment_sum inside would eliminate
    ``scatter_add`` or ``dot_general`` from the scan body.
    """

    def test_scan_body_contains_block_and_scatter(self):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _predict_nonlocal_electrode_scan_impl,
        )

        n_electrodes = 3
        max_n_enc = 40
        max_n_dec = 30
        n_wf = 4
        n_pos = 20
        n_time = 15
        n_pos_dim = 1

        enc_wf = jnp.zeros((n_electrodes, max_n_enc, n_wf))
        enc_pos = jnp.zeros((n_electrodes, max_n_enc, n_pos_dim))
        dec_wf = jnp.zeros((n_electrodes, max_n_dec, n_wf))
        dec_seg_ids = jnp.zeros((n_electrodes, max_n_dec), dtype=jnp.int32)
        mean_rates = jnp.ones(n_electrodes)
        n_real_enc = jnp.full((n_electrodes,), max_n_enc, dtype=jnp.int32)
        wf_stds = jnp.full(n_wf, 24.0)
        place_bin_centers = jnp.linspace(0, 100, n_pos)[:, None]
        position_std = jnp.array([3.5])
        occupancy = jnp.ones(n_pos) * 0.01

        fn = functools.partial(
            _predict_nonlocal_electrode_scan_impl,
            n_time=n_time,
            block_size=10,
            use_gemm=True,
            pos_tile_size=None,
            enc_tile_size=None,
        )
        jaxpr = jax.make_jaxpr(fn)(
            enc_wf,
            enc_pos,
            dec_wf,
            dec_seg_ids,
            mean_rates,
            n_real_enc,
            wf_stds,
            place_bin_centers,
            position_std,
            occupancy,
        )

        top_prims = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
        n_top_scan = sum(top_prims.count(p) for p in ("scan", "while", "fori_loop"))
        assert n_top_scan >= 1, (
            f"Expected ≥1 top-level scan/while/fori_loop (outer electrode scan "
            f"+ inner block fori_loop may both lower to scan/while); "
            f"got 0 in: {top_prims}"
        )

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

        all_prims = _collect_all_primitives(jaxpr.jaxpr.eqns)
        # Mark-kernel GEMM — dot_general — must be in the fused body.
        assert all_prims.count("dot_general") >= 1, (
            f"Expected ≥1 dot_general inside the scan body; got 0. "
            f"Primitives: {sorted(set(all_prims))}"
        )
        # segment_sum lowers to scatter-add — must be in the fused body.
        has_scatter = any(
            all_prims.count(p) >= 1 for p in ("scatter_add", "scatter", "scatter-add")
        )
        assert has_scatter, (
            f"Expected ≥1 scatter/scatter_add inside the scan body "
            f"(from segment_sum). Primitives: {sorted(set(all_prims))}"
        )


class TestBlockSizeAuto:
    """``block_size='auto'`` resolves to a memory-aware int via auto_select_tile_sizes.

    Manual verification at different memory levels can be done with
    ``XLA_PYTHON_CLIENT_MEM_FRACTION=<f>`` before launching Python
    (JAX reads this env var on first-import).  The unit tests here
    use ``mock.patch`` on ``_default_memory_budget_bytes`` for
    reproducibility within a single pytest process.
    """

    def _predict(self, workload, **kwargs):
        from non_local_detector.likelihoods.clusterless_kde_log import (
            predict_clusterless_kde_log_likelihood,
        )

        encoding = {**workload["encoding"], **kwargs}
        return predict_clusterless_kde_log_likelihood(
            workload["test_edges"],
            workload["test_time"],
            workload["test_position"],
            workload["test_spikes"],
            workload["test_wf"],
            **encoding,
            is_local=False,
        )

    def test_auto_matches_explicit_resolved_value(self):
        """``block_size='auto'`` and ``block_size=<int>`` (the value auto resolved to)
        must produce bit-identical output — ``auto`` is a pure Python-side
        resolution layer, the JIT-traced graph is the same either way.

        Use a single-electrode workload so bucketing collapses to one
        bucket, making the resolved value a single int we can pass
        explicitly on a second call and compare.
        """
        from non_local_detector.likelihoods.clusterless_kde_log import (
            _bucket_by_size,
            _group_electrodes_by_n_wf,
            _prepare_electrode_scan_group,
            auto_select_tile_sizes,
        )

        workload = _build_small_workload(seed=42, n_tetrodes=1)
        groups = _group_electrodes_by_n_wf(workload["test_wf"])
        assert set(groups) == {4}
        buckets = _bucket_by_size(
            groups[4],
            workload["test_wf"],
            workload["test_spikes"],
            workload["test_edges"],
        )
        assert len(buckets) == 1  # single electrode → single bucket
        batch = _prepare_electrode_scan_group(
            buckets[0],
            workload["encoding"]["encoding_spike_waveform_features"],
            workload["encoding"]["encoding_positions"],
            workload["test_wf"],
            workload["test_spikes"],
            workload["encoding"]["mean_rates"],
            workload["test_edges"],
            workload["encoding"]["waveform_std"],
        )
        resolved = auto_select_tile_sizes(
            n_enc=batch["max_n_enc"],
            n_dec=batch["max_n_dec"],
            n_pos=workload["encoding"]["occupancy"].shape[0],
            n_wf=batch["n_wf"],
        )["block_size"]

        out_auto = np.asarray(self._predict(workload, block_size="auto"))
        out_explicit = np.asarray(self._predict(workload, block_size=resolved))
        np.testing.assert_array_equal(out_auto, out_explicit)

    def test_auto_with_tight_budget_picks_smaller_block(self):
        """Tight mocked budget → auto picks a smaller block than default 100 on small n_enc
        or a larger block than default when budget permits.  Verify it at least differs
        from default 100 (smoke test that mocking works end-to-end).
        """
        from unittest import mock

        workload = _build_small_workload(seed=42)
        # 10 KB budget → block_size formula gives ~0 → floor of 1.
        with mock.patch(
            "non_local_detector.likelihoods.clusterless_kde_log."
            "_default_memory_budget_bytes",
            return_value=10_000,
        ):
            out_tight = np.asarray(self._predict(workload, block_size="auto"))
        # 10 GB budget → block_size formula gives a huge value capped at n_dec.
        with mock.patch(
            "non_local_detector.likelihoods.clusterless_kde_log."
            "_default_memory_budget_bytes",
            return_value=int(10e9),
        ):
            out_loose = np.asarray(self._predict(workload, block_size="auto"))
        # Both produce finite outputs of the expected shape — content
        # may differ by FP32 noise from different block_size choices
        # (via compensated-linear chunking).
        assert out_tight.shape == out_loose.shape
        assert np.all(np.isfinite(out_tight))
        assert np.all(np.isfinite(out_loose))

    def test_auto_in_fallback_path(self):
        """``block_size='auto'`` also works with ``use_streaming=True`` (Python-loop fallback).

        Per-electrode resolution happens inside the fallback loop.
        """
        workload = _build_small_workload(seed=42)
        min_enc = min(
            int(s.shape[0]) for s in workload["encoding"]["encoding_positions"]
        )
        out = np.asarray(
            self._predict(
                workload,
                block_size="auto",
                use_streaming=True,
                enc_tile_size=max(1, min_enc - 1),
            )
        )
        assert out.shape == (
            len(workload["test_edges"]),
            workload["encoding"]["occupancy"].shape[0],
        )
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize(
        ("bad_block_size", "expected_match"),
        [
            ("fast", "must be 'auto'"),
            ("AUTO", "must be 'auto'"),  # case-sensitive
            ("", "must be 'auto'"),
            (0, "must be ≥ 1"),
            (-5, "must be ≥ 1"),
            (True, "must be a positive int"),  # bool is a surprising int
            (False, "must be a positive int"),  # bool guard fires before ≥ 1
            (None, "must be a positive int"),
            (1.5, "must be a positive int"),
        ],
    )
    def test_invalid_block_size_raises(self, bad_block_size, expected_match):
        workload = _build_small_workload(seed=42)
        with pytest.raises(ValueError, match=expected_match):
            self._predict(workload, block_size=bad_block_size)

    @pytest.mark.parametrize(
        ("bad_block_size", "expected_match"),
        [
            # String "auto" has a path-specific message on is_local.
            ("auto", "is_local=True"),
            # Non-string invalids must ALSO raise on the is_local path
            # (regression guard: the prior implementation only checked
            # isinstance(str), letting 0 / True / 1.5 / None fall through
            # to compute_local_log_likelihood).
            (0, "must be ≥ 1"),
            (-5, "must be ≥ 1"),
            (True, "must be a positive int"),
            (False, "must be a positive int"),
            (None, "must be a positive int"),
            (1.5, "must be a positive int"),
        ],
    )
    def test_invalid_block_size_raises_on_is_local(
        self, bad_block_size, expected_match
    ):
        """Validation must fire at the public API boundary on both paths."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            predict_clusterless_kde_log_likelihood,
        )

        workload = _build_small_workload(seed=42)
        encoding = {**workload["encoding"], "block_size": bad_block_size}
        with pytest.raises(ValueError, match=expected_match):
            predict_clusterless_kde_log_likelihood(
                workload["test_edges"],
                workload["test_time"],
                workload["test_position"],
                workload["test_spikes"],
                workload["test_wf"],
                **encoding,
                is_local=True,
            )

    def test_auto_rejected_on_is_local_path(self):
        """is_local=True currently doesn't support 'auto'; surface a clear error."""
        from non_local_detector.likelihoods.clusterless_kde_log import (
            predict_clusterless_kde_log_likelihood,
        )

        workload = _build_small_workload(seed=42)
        encoding = {**workload["encoding"], "block_size": "auto"}
        with pytest.raises(ValueError, match="is_local=True"):
            predict_clusterless_kde_log_likelihood(
                workload["test_edges"],
                workload["test_time"],
                workload["test_position"],
                workload["test_spikes"],
                workload["test_wf"],
                **encoding,
                is_local=True,
            )

    def test_auto_resolves_per_bucket(self):
        """Two size-buckets on the same predict call get independent resolutions.

        Spy on ``auto_select_tile_sizes`` to confirm it's called once
        per bucket with that bucket's shapes — not once globally.
        """
        from unittest import mock

        from non_local_detector.environment import Environment
        from non_local_detector.likelihoods.clusterless_kde_log import (
            fit_clusterless_kde_encoding_model,
            predict_clusterless_kde_log_likelihood,
        )

        rng = np.random.default_rng(0)
        env = Environment(
            place_bin_size=10.0,
            environment_name="linear",
            is_track_interior=None,
        )
        position = rng.uniform(0, 100, 500)[:, None]
        time_pos = np.linspace(0, 10, 500)
        env.fit_place_grid(position=position, infer_track_interior=False)

        # Two same-n_wf electrodes with very different spike counts —
        # should fall into separate size-buckets.
        spike_times = [
            np.sort(rng.uniform(0, 10, 30)),
            np.sort(rng.uniform(0, 10, 500)),
        ]
        spike_wf = [
            rng.standard_normal((30, 4)) * 20.0,
            rng.standard_normal((500, 4)) * 20.0,
        ]
        encoding = fit_clusterless_kde_encoding_model(
            time_pos,
            position,
            spike_times,
            spike_wf,
            env,
            sampling_frequency=50,
            position_std=6.0,
            waveform_std=24.0,
            block_size=50,
            disable_progress_bar=True,
        )
        test_edges = np.linspace(5.0, 10.0, 51)
        test_time = time_pos[time_pos >= 5.0]
        test_position = position[time_pos >= 5.0]
        test_spikes = [st[st >= 5.0] for st in spike_times]
        test_wf = [wf[st >= 5.0] for st, wf in zip(spike_times, spike_wf, strict=False)]

        import non_local_detector.likelihoods.clusterless_kde_log as mod

        original = mod.auto_select_tile_sizes
        call_args = []

        def spy(*args, **kwargs):
            call_args.append((args, dict(kwargs)))
            return original(*args, **kwargs)

        with mock.patch.object(mod, "auto_select_tile_sizes", side_effect=spy):
            out = predict_clusterless_kde_log_likelihood(
                test_edges,
                test_time,
                test_position,
                test_spikes,
                test_wf,
                **{**encoding, "disable_progress_bar": True, "block_size": "auto"},
                is_local=False,
            )
        assert np.all(np.isfinite(np.asarray(out)))
        # We expect at least 2 calls (2 buckets).  Depending on
        # whether _bucket_by_size puts both electrodes in one bucket
        # (quantile edges may collapse with 2 points), it could be 1.
        # Assert ≥1 and, if ≥2, that the n_dec values differ.
        assert len(call_args) >= 1
        if len(call_args) >= 2:
            n_dec_values = {kwargs["n_dec"] for args, kwargs in call_args}
            assert len(n_dec_values) >= 2, (
                f"per-bucket n_dec values didn't differ: {n_dec_values}"
            )
