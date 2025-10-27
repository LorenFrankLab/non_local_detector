"""Tests for optimized clusterless KDE log-space functions.

This module tests the vectorized implementations of kde_distance and log_kde_distance
in clusterless_kde_log.py to ensure numerical equivalence and performance improvements.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.clusterless_kde_log import (
    kde_distance,
    log_kde_distance,
)
from non_local_detector.likelihoods.common import gaussian_pdf, log_gaussian_pdf


class TestKdeDistanceVectorized:
    """Tests for optimized linear-space KDE distance function."""

    def test_numerical_equivalence_2d(self):
        """Optimized version matches manual loop computation for 2D features."""
        np.random.seed(42)

        # Test data
        n_eval = 50
        n_samples = 100
        n_dims = 2

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
        np.random.seed(123)

        for n_dims in [4, 8, 10]:
            eval_points = jnp.array(np.random.randn(30, n_dims))
            samples = jnp.array(np.random.randn(50, n_dims))
            std = jnp.array(np.random.uniform(0.5, 2.0, n_dims))

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
        n_eval, n_samples, n_dims = 25, 40, 3

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
        np.random.seed(42)

        n_eval = 50
        n_samples = 100
        n_dims = 2

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
        np.random.seed(456)

        for n_dims in [4, 8, 10]:
            eval_points = jnp.array(np.random.randn(30, n_dims))
            samples = jnp.array(np.random.randn(50, n_dims))
            std = jnp.array(np.random.uniform(0.5, 2.0, n_dims))

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
        np.random.seed(789)

        eval_points = jnp.array(np.random.randn(20, 3))
        samples = jnp.array(np.random.randn(40, 3))
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
        n_eval, n_samples, n_dims = 25, 40, 3

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
        np.random.seed(101)

        eval_points = jnp.array(np.random.randn(20, 2))
        samples = jnp.array(np.random.randn(40, 2))
        std = jnp.array([1.0, 1.5])

        # Multiple calls should give identical results (JIT caching)
        result1 = kde_distance(eval_points, samples, std)
        result2 = kde_distance(eval_points, samples, std)

        # Should be identical
        assert jnp.allclose(result1, result2, rtol=1e-10, atol=1e-14)

    def test_log_kde_distance_consistency(self):
        """log_kde_distance() produces consistent results across multiple calls."""
        np.random.seed(202)

        eval_points = jnp.array(np.random.randn(20, 2))
        samples = jnp.array(np.random.randn(40, 2))
        std = jnp.array([1.0, 1.5])

        # Multiple calls should give identical results
        result1 = log_kde_distance(eval_points, samples, std)
        result2 = log_kde_distance(eval_points, samples, std)

        # Should be identical
        assert jnp.allclose(result1, result2, rtol=1e-10, atol=1e-14)

    def test_different_dimensions(self):
        """Optimized functions work for various feature dimensions."""
        np.random.seed(303)

        for n_dims in [2, 4, 6, 8]:
            eval_points = jnp.array(np.random.randn(15, n_dims))
            samples = jnp.array(np.random.randn(30, n_dims))
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
        np.random.seed(404)

        # Use larger arrays to get more reliable timing
        n_eval = 100
        n_samples = 200
        n_dims = 4

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
        np.random.seed(505)

        n_eval = 100
        n_samples = 200
        n_dims = 4

        eval_points = jnp.array(np.random.randn(n_eval, n_dims))
        samples = jnp.array(np.random.randn(n_samples, n_dims))
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
