"""Tests for optimized clusterless KDE log-space functions.

This module tests the vectorized implementations of kde_distance and log_kde_distance
in clusterless_kde_log.py to ensure numerical equivalence and performance improvements.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde_log import (
    _compensated_linear_marginal,
    _compute_log_mark_kernel_gemm,
    kde_distance,
    log_kde_distance,
)
from non_local_detector.likelihoods.common import (
    EPS,
    gaussian_pdf,
    log_gaussian_pdf,
    safe_log,
)


class TestKdeDistanceVectorized:
    """Tests for optimized linear-space KDE distance function."""

    def test_numerical_equivalence_2d(self):
        """Optimized version matches manual loop computation for 2D features."""
        rng = np.random.default_rng(42)

        # Test data
        n_eval = 50
        n_samples = 100
        n_dims = 2

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.array([1.0, 1.5])

        # Reference: manual loop computation
        distance_ref = jnp.ones((n_samples, n_eval))
        for dim_eval, dim_samp, dim_std in zip(
            eval_points.T, samples.T, std, strict=False
        ):
            distance_ref *= gaussian_pdf(
                jnp.expand_dims(dim_eval, axis=0),
                jnp.expand_dims(dim_samp, axis=1),
                dim_std,
            )

        # Test: optimized version
        distance_opt = kde_distance(eval_points, samples, std)

        # Verify numerical equivalence
        assert jnp.allclose(distance_opt, distance_ref, rtol=1e-5, atol=1e-8)

    def test_numerical_equivalence_high_dim(self):
        """Vectorized version works for high-dimensional features (4D, 8D, 10D)."""
        rng = np.random.default_rng(123)

        for n_dims in [4, 8, 10]:
            eval_points = jnp.array(rng.standard_normal((30, n_dims)))
            samples = jnp.array(rng.standard_normal((50, n_dims)))
            std = jnp.array(rng.uniform(0.5, 2.0, n_dims))

            # Reference
            distance_ref = jnp.ones((samples.shape[0], eval_points.shape[0]))
            for dim_eval, dim_samp, dim_std in zip(
                eval_points.T, samples.T, std, strict=False
            ):
                distance_ref *= gaussian_pdf(
                    jnp.expand_dims(dim_eval, axis=0),
                    jnp.expand_dims(dim_samp, axis=1),
                    dim_std,
                )

            # Test
            distance_opt = kde_distance(eval_points, samples, std)

            # Verify
            assert jnp.allclose(distance_opt, distance_ref, rtol=1e-5, atol=1e-8)

    def test_numerical_stability(self):
        """Optimized version is numerically stable with small std values."""
        eval_points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        samples = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        std = jnp.array([0.1, 0.1])

        result = kde_distance(eval_points, samples, std)

        # Should be finite and non-negative
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_output_shape(self):
        """Optimized version produces correct output shape."""
        rng = np.random.default_rng(0)
        n_eval, n_samples, n_dims = 25, 40, 3

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.ones(n_dims)

        result = kde_distance(eval_points, samples, std)

        assert result.shape == (n_samples, n_eval)

    def test_positive_output(self):
        """KDE distance values are non-negative."""
        eval_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        samples = jnp.array([[0.5, 0.5], [1.5, 1.5]])
        std = jnp.array([1.0, 1.0])

        result = kde_distance(eval_points, samples, std)

        assert jnp.all(result >= 0)


class TestLogKdeDistanceVectorized:
    """Tests for optimized log-space KDE distance function."""

    def test_numerical_equivalence_2d(self):
        """Optimized log version matches manual loop computation for 2D features."""
        rng = np.random.default_rng(42)

        n_eval = 50
        n_samples = 100
        n_dims = 2

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.array([1.0, 1.5])

        # Reference: manual loop computation
        log_dist_ref = jnp.zeros((n_samples, n_eval))
        for dim_eval, dim_samp, dim_std in zip(
            eval_points.T, samples.T, std, strict=False
        ):
            log_dist_ref += log_gaussian_pdf(
                jnp.expand_dims(dim_eval, axis=0),
                jnp.expand_dims(dim_samp, axis=1),
                dim_std,
            )

        # Test: optimized version
        log_dist_opt = log_kde_distance(eval_points, samples, std)

        # Verify numerical equivalence
        assert jnp.allclose(log_dist_opt, log_dist_ref, rtol=1e-5, atol=1e-8)

    def test_numerical_equivalence_high_dim(self):
        """Optimized log version works for high-dimensional features."""
        rng = np.random.default_rng(456)

        for n_dims in [4, 8, 10]:
            eval_points = jnp.array(rng.standard_normal((30, n_dims)))
            samples = jnp.array(rng.standard_normal((50, n_dims)))
            std = jnp.array(rng.uniform(0.5, 2.0, n_dims))

            # Reference
            log_dist_ref = jnp.zeros((samples.shape[0], eval_points.shape[0]))
            for dim_eval, dim_samp, dim_std in zip(
                eval_points.T, samples.T, std, strict=False
            ):
                log_dist_ref += log_gaussian_pdf(
                    jnp.expand_dims(dim_eval, axis=0),
                    jnp.expand_dims(dim_samp, axis=1),
                    dim_std,
                )

            # Test
            log_dist_opt = log_kde_distance(eval_points, samples, std)

            # Verify
            assert jnp.allclose(log_dist_opt, log_dist_ref, rtol=1e-5, atol=1e-8)

    def test_log_linear_consistency(self):
        """Log-space and linear-space versions are consistent (log of linear = log version)."""
        rng = np.random.default_rng(789)

        eval_points = jnp.array(rng.standard_normal((20, 3)))
        samples = jnp.array(rng.standard_normal((40, 3)))
        std = jnp.array([1.0, 1.5, 0.8])

        # Linear-space
        linear_dist = kde_distance(eval_points, samples, std)

        # Log-space
        log_dist = log_kde_distance(eval_points, samples, std)

        # Should satisfy: log(linear) ≈ log_version
        log_of_linear = jnp.log(linear_dist)

        assert jnp.allclose(log_of_linear, log_dist, rtol=1e-5, atol=1e-8)

    def test_output_shape(self):
        """Log optimized version produces correct output shape."""
        rng = np.random.default_rng(0)
        n_eval, n_samples, n_dims = 25, 40, 3

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.ones(n_dims)

        result = log_kde_distance(eval_points, samples, std)

        assert result.shape == (n_samples, n_eval)

    def test_finite_output(self):
        """Log KDE distance values are finite."""
        eval_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        samples = jnp.array([[0.5, 0.5], [1.5, 1.5]])
        std = jnp.array([1.0, 1.0])

        result = log_kde_distance(eval_points, samples, std)

        assert jnp.all(jnp.isfinite(result))


class TestJitCompiledFunctions:
    """Tests for JIT-compiled optimized functions."""

    def test_kde_distance_consistency(self):
        """kde_distance() produces consistent results across multiple calls."""
        rng = np.random.default_rng(101)

        eval_points = jnp.array(rng.standard_normal((20, 2)))
        samples = jnp.array(rng.standard_normal((40, 2)))
        std = jnp.array([1.0, 1.5])

        # Multiple calls should give identical results (JIT caching)
        result1 = kde_distance(eval_points, samples, std)
        result2 = kde_distance(eval_points, samples, std)

        # Should be identical
        assert jnp.allclose(result1, result2, rtol=1e-10, atol=1e-14)

    def test_log_kde_distance_consistency(self):
        """log_kde_distance() produces consistent results across multiple calls."""
        rng = np.random.default_rng(202)

        eval_points = jnp.array(rng.standard_normal((20, 2)))
        samples = jnp.array(rng.standard_normal((40, 2)))
        std = jnp.array([1.0, 1.5])

        # Multiple calls should give identical results
        result1 = log_kde_distance(eval_points, samples, std)
        result2 = log_kde_distance(eval_points, samples, std)

        # Should be identical
        assert jnp.allclose(result1, result2, rtol=1e-10, atol=1e-14)

    def test_different_dimensions(self):
        """Optimized functions work for various feature dimensions."""
        rng = np.random.default_rng(303)

        for n_dims in [2, 4, 6, 8]:
            eval_points = jnp.array(rng.standard_normal((15, n_dims)))
            samples = jnp.array(rng.standard_normal((30, n_dims)))
            std = jnp.ones(n_dims)

            # Both should work without errors
            result_linear = kde_distance(eval_points, samples, std)
            result_log = log_kde_distance(eval_points, samples, std)

            # Verify shapes
            assert result_linear.shape == (30, 15)
            assert result_log.shape == (30, 15)

            # Verify consistency
            assert jnp.allclose(
                jnp.log(result_linear), result_log, rtol=1e-5, atol=1e-8
            )


class TestPerformanceRegression:
    """Tests to ensure JIT compilation and caching work correctly."""

    def test_jit_compilation_caching(self):
        """JIT-compiled functions cache compilation and are fast on repeated calls."""
        rng = np.random.default_rng(404)

        # Use larger arrays to get more reliable timing
        n_eval = 100
        n_samples = 200
        n_dims = 4

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.ones(n_dims)

        # Warmup call (includes JIT compilation overhead)
        _ = kde_distance(eval_points, samples, std)
        _.block_until_ready()

        # Timed calls - should all be fast now that compilation is cached
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = kde_distance(eval_points, samples, std)
            result.block_until_ready()
            times.append(time.perf_counter() - start)

        # All calls should be reasonably fast (< 10ms average)
        avg_time = np.mean(times)
        assert avg_time < 0.01, f"Average time {avg_time:.4f}s exceeds 10ms threshold"

    def test_log_jit_compilation_caching(self):
        """JIT-compiled log functions cache compilation correctly."""
        rng = np.random.default_rng(505)

        n_eval = 100
        n_samples = 200
        n_dims = 4

        eval_points = jnp.array(rng.standard_normal((n_eval, n_dims)))
        samples = jnp.array(rng.standard_normal((n_samples, n_dims)))
        std = jnp.ones(n_dims)

        # Warmup
        _ = log_kde_distance(eval_points, samples, std)
        _.block_until_ready()

        # Timed calls
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = log_kde_distance(eval_points, samples, std)
            result.block_until_ready()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        assert avg_time < 0.01, f"Average time {avg_time:.4f}s exceeds 10ms threshold"


class TestCompensatedLinearMarginal:
    """Tests for the compensated-linear matmul fast path.

    Validates that _compensated_linear_marginal produces results matching
    the logsumexp reference to within float32 precision for ≤ 8 waveform
    feature dimensions.
    """

    @staticmethod
    def _logsumexp_reference(logK_mark, logK_pos, log_w, occupancy, mean_rate):
        """Pure logsumexp reference (ground-truth implementation).

        This is an independent reimplementation of the logsumexp marginal,
        used as the correctness oracle for the compensated-linear fast path.
        """

        def for_one_dec(wf_col):
            return jax.nn.logsumexp(log_w + logK_pos + wf_col[:, None], axis=0)

        log_marginal = jax.vmap(for_one_dec, in_axes=1)(logK_mark)
        log_mean_rate = safe_log(mean_rate, eps=EPS)
        log_occ = safe_log(occupancy, eps=EPS)
        return jnp.where(
            occupancy[None, :] > 0.0,
            log_mean_rate + log_marginal - log_occ[None, :],
            jnp.log(0.0),
        )

    @staticmethod
    def _make_test_data(rng, n_enc, n_dec, n_pos, n_wf):
        """Create realistic test kernels from simulated spike data."""
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)
        enc_pos = jnp.array(rng.uniform(0, 200, (n_enc, 1)))
        eval_pos = jnp.linspace(0, 200, n_pos)[:, None]

        logK_pos = log_kde_distance(eval_pos, enc_pos, jnp.array([3.5]))
        logK_mark = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
        log_w = -jnp.log(float(n_enc))
        occupancy = jnp.ones(n_pos) * 0.01
        return logK_mark, logK_pos, log_w, occupancy

    def test_matches_logsumexp_4d(self):
        """Compensated linear matches logsumexp for 4D waveform features."""
        rng = np.random.default_rng(42)
        logK_mark, logK_pos, log_w, occupancy = self._make_test_data(
            rng, n_enc=500, n_dec=50, n_pos=100, n_wf=4
        )

        result_comp = _compensated_linear_marginal(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )
        result_ref = self._logsumexp_reference(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )

        assert jnp.allclose(result_comp, result_ref, atol=1e-4, rtol=1e-4), (
            f"Max abs diff: {float(jnp.max(jnp.abs(result_comp - result_ref))):.2e}"
        )

    def test_matches_logsumexp_8d(self):
        """Compensated linear matches logsumexp for 8D waveform features."""
        rng = np.random.default_rng(42)
        logK_mark, logK_pos, log_w, occupancy = self._make_test_data(
            rng, n_enc=500, n_dec=50, n_pos=100, n_wf=8
        )

        result_comp = _compensated_linear_marginal(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )
        result_ref = self._logsumexp_reference(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )

        assert jnp.allclose(result_comp, result_ref, atol=1e-4, rtol=1e-4), (
            f"Max abs diff: {float(jnp.max(jnp.abs(result_comp - result_ref))):.2e}"
        )

    def test_matches_across_multiple_seeds(self):
        """Accuracy holds across 5 random seeds for 4D features at larger scale."""
        for seed in range(5):
            rng = np.random.default_rng(seed * 100)
            logK_mark, logK_pos, log_w, occupancy = self._make_test_data(
                rng, n_enc=1000, n_dec=100, n_pos=200, n_wf=4
            )

            result_comp = _compensated_linear_marginal(
                logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
            )
            result_ref = self._logsumexp_reference(
                logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
            )

            max_diff = float(jnp.max(jnp.abs(result_comp - result_ref)))
            assert max_diff < 1e-4, (
                f"Seed {seed}: max abs diff {max_diff:.2e} exceeds 1e-4"
            )

    def test_accuracy_degrades_above_threshold_9d(self):
        """Document that 9D features can produce large errors (justifies threshold).

        This test documents the empirical observation that the compensated-linear
        approach degrades beyond _COMPENSATED_LINEAR_MAX_FEATURES=8 dimensions.
        At 9D, mark kernel underflow (~23%) causes occasional large errors via
        catastrophic cancellation in the sqrt(scale) decomposition.
        """
        # Use a seed known to trigger degraded accuracy at 9D
        rng = np.random.default_rng(900)
        logK_mark, logK_pos, log_w, occupancy = self._make_test_data(
            rng, n_enc=1000, n_dec=100, n_pos=200, n_wf=9
        )

        result_comp = _compensated_linear_marginal(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )

        # At 9D, either the error is still small (seed-dependent) or it
        # exceeds the 1e-4 bound that the ≤8D tests enforce.  We just
        # verify the result is not NaN and document that it *can* exceed
        # the bound — this justifies the threshold.
        # Note: max abs diff may be >> 1e-4 for some seeds (observed up to ~3.0)
        assert jnp.all(jnp.isfinite(result_comp) | (result_comp == -jnp.inf))

    def test_output_shape_and_finite_nonzero_occupancy(self):
        """Output has correct shape and all-finite values when occupancy > 0."""
        rng = np.random.default_rng(42)
        n_dec, n_pos = 30, 50
        logK_mark, logK_pos, log_w, occupancy = self._make_test_data(
            rng, n_enc=200, n_dec=n_dec, n_pos=n_pos, n_wf=4
        )

        result = _compensated_linear_marginal(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )

        assert result.shape == (n_dec, n_pos)
        # All occupancy is non-zero, so all outputs should be finite (no -inf mask)
        assert jnp.all(jnp.isfinite(result))

    def test_zero_occupancy_produces_neg_inf(self):
        """Zero-occupancy bins produce -inf in output."""
        rng = np.random.default_rng(42)
        n_enc, n_dec, n_pos, n_wf = 100, 10, 20, 4

        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)

        logK_pos = jnp.zeros((n_enc, n_pos))
        logK_mark = _compute_log_mark_kernel_gemm(dec_wf, enc_wf, wf_std)
        log_w = -jnp.log(float(n_enc))

        # Set some occupancy bins to zero
        occupancy = jnp.ones(n_pos) * 0.01
        occupancy = occupancy.at[5].set(0.0)
        occupancy = occupancy.at[15].set(0.0)

        result = _compensated_linear_marginal(
            logK_mark, logK_pos, log_w, occupancy, mean_rate=5.0
        )

        # Zero-occupancy bins should be -inf (from the occupancy mask)
        assert jnp.all(jnp.isinf(result[:, 5]) & (result[:, 5] < 0))
        assert jnp.all(jnp.isinf(result[:, 15]) & (result[:, 15] < 0))
        # Non-zero occupancy bins should be finite
        assert jnp.all(jnp.isfinite(result[:, 0]))
