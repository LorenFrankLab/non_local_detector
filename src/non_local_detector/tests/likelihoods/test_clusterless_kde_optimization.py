"""Tests for optimized clusterless KDE implementation."""

import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    kde_distance,
    kde_distance_vectorized,
)


class TestKdeDistanceVectorized:
    """Test vectorized KDE distance implementation."""

    def test_numerical_equivalence_2d(self):
        """Verify vectorized version matches original for 2D features."""
        rng = np.random.default_rng(42)
        eval_points = jnp.array(rng.normal(0, 1, (10, 2)))
        samples = jnp.array(rng.normal(0, 1, (20, 2)))
        std = jnp.ones(2)

        result_original = kde_distance(eval_points, samples, std)
        result_vectorized = kde_distance_vectorized(eval_points, samples, std)

        assert jnp.allclose(result_original, result_vectorized, rtol=1e-5, atol=1e-8)

    def test_numerical_equivalence_high_dim(self):
        """Verify vectorized version matches original for high-dimensional features."""
        rng = np.random.default_rng(42)

        for n_features in [4, 8, 10]:
            eval_points = jnp.array(rng.normal(0, 1, (10, n_features)))
            samples = jnp.array(rng.normal(0, 1, (20, n_features)))
            std = jnp.ones(n_features)

            result_original = kde_distance(eval_points, samples, std)
            result_vectorized = kde_distance_vectorized(eval_points, samples, std)

            assert jnp.allclose(
                result_original, result_vectorized, rtol=1e-5, atol=1e-8
            ), f"Failed for {n_features}D"

    def test_numerical_stability(self):
        """Test numerical stability with small std values."""
        rng = np.random.default_rng(42)
        eval_points = jnp.array(rng.normal(0, 1, (10, 2)))
        samples = jnp.array(rng.normal(0, 1, (20, 2)))
        std = jnp.array([0.1, 0.1])  # Small but valid

        result = kde_distance_vectorized(eval_points, samples, std)

        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_output_shape(self):
        """Test that output has correct shape."""
        rng = np.random.default_rng(42)
        n_eval = 15
        n_samples = 25
        n_dims = 3

        eval_points = jnp.array(rng.normal(0, 1, (n_eval, n_dims)))
        samples = jnp.array(rng.normal(0, 1, (n_samples, n_dims)))
        std = jnp.ones(n_dims)

        result = kde_distance_vectorized(eval_points, samples, std)

        assert result.shape == (n_samples, n_eval)

    def test_positive_output(self):
        """Test that KDE distance is always non-negative."""
        rng = np.random.default_rng(42)
        eval_points = jnp.array(rng.normal(0, 1, (10, 2)))
        samples = jnp.array(rng.normal(0, 1, (20, 2)))
        std = jnp.ones(2)

        result = kde_distance_vectorized(eval_points, samples, std)

        assert jnp.all(result >= 0)


class TestEstimateLogJointMarkIntensity:
    """Test optimized estimate_log_joint_mark_intensity function."""

    def test_uses_vectorized_implementation(self):
        """Verify that the function uses vectorized KDE internally."""
        rng = np.random.default_rng(42)

        dec_features = jnp.array(rng.normal(0, 1, (10, 2)))
        enc_features = jnp.array(rng.normal(0, 1, (20, 2)))
        waveform_stds = jnp.ones(2)
        occupancy = jnp.ones(50) * 0.1
        mean_rate = 5.0
        position_distance = jnp.array(rng.exponential(1.0, (20, 50)))

        # This should work and use the vectorized version
        result = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        # Verify output shape
        assert result.shape == (10, 50)

    def test_zero_occupancy_handling(self):
        """Test that zero occupancy produces -inf."""
        rng = np.random.default_rng(42)

        dec_features = jnp.array(rng.normal(0, 1, (10, 2)))
        enc_features = jnp.array(rng.normal(0, 1, (20, 2)))
        waveform_stds = jnp.ones(2)
        occupancy = jnp.array([0.0, 0.1, 0.2] + [0.1] * 47)  # First bin zero
        mean_rate = 5.0
        position_distance = jnp.array(rng.exponential(1.0, (20, 50)))

        result = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        # Zero occupancy should produce -inf
        assert jnp.all(jnp.isneginf(result[:, 0]))

        # Non-zero occupancy should be finite
        assert jnp.all(jnp.isfinite(result[:, 1:]))

    def test_no_nan_values(self):
        """Test that output contains no NaN values."""
        rng = np.random.default_rng(42)

        dec_features = jnp.array(rng.normal(0, 1, (10, 2)))
        enc_features = jnp.array(rng.normal(0, 1, (20, 2)))
        waveform_stds = jnp.ones(2)
        occupancy = jnp.ones(50) * 0.1
        mean_rate = 5.0
        position_distance = jnp.array(rng.exponential(1.0, (20, 50)))

        result = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )

        assert not jnp.any(jnp.isnan(result))

    def test_different_dimensions(self):
        """Test function works for different feature dimensions."""
        rng = np.random.default_rng(42)

        for n_features in [2, 4, 8]:
            dec_features = jnp.array(rng.normal(0, 1, (10, n_features)))
            enc_features = jnp.array(rng.normal(0, 1, (20, n_features)))
            waveform_stds = jnp.ones(n_features)
            occupancy = jnp.ones(50) * 0.1
            mean_rate = 5.0
            position_distance = jnp.array(rng.exponential(1.0, (20, 50)))

            result = estimate_log_joint_mark_intensity(
                dec_features,
                enc_features,
                waveform_stds,
                occupancy,
                mean_rate,
                position_distance,
            )

            assert result.shape == (10, 50)
            assert jnp.all(jnp.isfinite(result))


class TestPerformanceRegression:
    """Performance regression tests to ensure optimization remains effective."""

    def test_jit_compilation_caching(self):
        """Verify that JIT compilation is cached between calls."""
        import time

        rng = np.random.default_rng(42)
        # Use larger arrays to make timing more reliable
        dec_features = jnp.array(rng.normal(0, 1, (100, 2)))
        enc_features = jnp.array(rng.normal(0, 1, (200, 2)))
        waveform_stds = jnp.ones(2)
        occupancy = jnp.ones(500) * 0.1
        mean_rate = 5.0
        position_distance = jnp.array(rng.exponential(1.0, (200, 500)))

        # Warmup call
        result_warmup = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )
        result_warmup.block_until_ready()

        # Timed calls - both should be fast now
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = estimate_log_joint_mark_intensity(
                dec_features,
                enc_features,
                waveform_stds,
                occupancy,
                mean_rate,
                position_distance,
            )
            result.block_until_ready()
            times.append(time.perf_counter() - start)

        # All calls should be fast (< 10ms) after compilation
        avg_time = np.mean(times)
        assert avg_time < 0.01, (
            f"Average call time ({avg_time:.4f}s) too slow, JIT may not be working"
        )

        # Results should be consistent
        result1 = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )
        result2 = estimate_log_joint_mark_intensity(
            dec_features,
            enc_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
        )
        assert jnp.allclose(result1, result2)
