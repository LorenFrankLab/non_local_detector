"""Tests to verify chunked algorithms match non-chunked versions.

Critical: The chunked versions are memory-efficient implementations that
should produce identical results to the standard versions. These tests
ensure numerical parity.

Testing philosophy:
1. Chunked algorithms should match standard algorithms exactly
2. Test with various chunk sizes
3. Test with edge cases (single chunk, many chunks)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother,
    filter,
    smoother,
)
from non_local_detector.tests.conftest import assert_probability_distribution


def run_chunked(init, trans, log_likes, n_chunks, **kwargs):
    """Run chunked_filter_smoother on precomputed log likelihoods.

    Returns the full 8-tuple from ``chunked_filter_smoother``.
    """
    n_time, n_states = log_likes.shape

    def log_likelihood_func(time_idx, *args):
        return log_likes[time_idx]

    return chunked_filter_smoother(
        time=np.arange(n_time),
        state_ind=np.arange(n_states),
        initial_distribution=np.array(init),
        transition_matrix=np.array(trans),
        log_likelihood_func=log_likelihood_func,
        log_likelihood_args=(),
        n_chunks=n_chunks,
        log_likelihoods=np.array(log_likes),
        **kwargs,
    )


@pytest.mark.unit
class TestChunkedFilterSmootherParity:
    """Verify chunked_filter_smoother matches filter + smoother.

    The chunked version processes data in blocks for memory efficiency
    but should give identical results to the standard implementation.
    """

    def create_simple_test_data(self, n_time=50, n_states=10):
        """Helper to create simple test data."""
        rng = np.random.default_rng(42)

        init = jnp.ones(n_states) / n_states
        trans = (
            jnp.eye(n_states) * 0.8 + jnp.ones((n_states, n_states)) * 0.2 / n_states
        )
        trans = trans / trans.sum(axis=1, keepdims=True)

        # Simple log likelihood function
        log_likes = jnp.array(rng.standard_normal((n_time, n_states)))

        return init, trans, log_likes

    @pytest.mark.parametrize(
        ("n_time", "n_states", "n_chunks", "rtol", "atol"),
        [
            pytest.param(50, 5, 1, 1e-5, 1e-6, id="single-chunk"),
            pytest.param(50, 5, 2, 1e-5, 1e-6, id="2-chunks"),
            pytest.param(50, 5, 5, 1e-5, 1e-6, id="5-chunks"),
            pytest.param(50, 5, 10, 1e-5, 1e-6, id="10-chunks"),
            pytest.param(100, 8, 5, 1e-4, 1e-5, id="multiple-chunks"),
            # n_chunks == n_time: one timestep per chunk
            pytest.param(20, 3, 20, 1e-4, 1e-5, id="one-step-chunks"),
        ],
    )
    def test_chunked_equals_standard(self, n_time, n_states, n_chunks, rtol, atol):
        """Chunked filter/smoother should match standard for any chunk count."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(
            n_time=n_time, n_states=n_states
        )

        # Act - Standard version
        (_, (filtered_std, _)) = filter(init, trans, log_likes)
        smoothed_std = smoother(trans, filtered_std)

        # Act - Chunked version
        _, acausal_state_probs, _, causal_state_probs, *_ = run_chunked(
            init, trans, log_likes, n_chunks
        )

        # Assert - shapes should match exactly (no extra initial row)
        assert jnp.allclose(smoothed_std, acausal_state_probs, rtol=rtol, atol=atol), (
            f"Smoothed posteriors don't match with n_chunks={n_chunks}"
        )
        assert jnp.allclose(filtered_std, causal_state_probs, rtol=rtol, atol=atol), (
            f"Filtered posteriors don't match with n_chunks={n_chunks}"
        )

    def test_chunked_log_likelihood_matches_standard(self):
        """Chunked should return same total log likelihood as standard."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=5)

        # Act - Standard version
        (log_marginals, _), _ = filter(init, trans, log_likes)
        total_log_like_std = log_marginals.sum()

        # Act - Chunked version
        _, _, log_like_chunked, *_ = run_chunked(init, trans, log_likes, n_chunks=3)

        # Assert
        assert jnp.allclose(
            total_log_like_std, log_like_chunked, rtol=1e-4, atol=1e-5
        ), f"Log likelihoods don't match: {total_log_like_std} vs {log_like_chunked}"

    def test_chunked_preserves_probability_normalization(self):
        """Chunked results should be valid probability distributions."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=8)

        # Act
        _, acausal_state_probs, _, causal_state_probs, *_ = run_chunked(
            init, trans, log_likes, n_chunks=5
        )

        # Assert - All probabilities sum to 1
        assert_probability_distribution(acausal_state_probs, rtol=1e-5, atol=1e-5)
        assert_probability_distribution(causal_state_probs, rtol=1e-5, atol=1e-5)

    def test_chunked_with_different_dtypes(self, x64):
        """Chunked should compute in the requested dtype.

        Runs with float64 enabled; otherwise a float64 request is silently
        truncated to float32 and the comparison below is float32 vs float32.
        """
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=20, n_states=4)

        # Act
        result_f32 = run_chunked(init, trans, log_likes, n_chunks=2, dtype=jnp.float32)
        result_f64 = run_chunked(init, trans, log_likes, n_chunks=2, dtype=jnp.float64)

        # Assert - the acausal, causal and predictive posteriors carry the
        # requested dtype, and the two runs agree to float32 precision.
        for index in (0, 6, 7):
            assert result_f32[index].dtype == np.float32
            assert result_f64[index].dtype == np.float64
        assert jnp.allclose(result_f32[0], result_f64[0], rtol=1e-5, atol=1e-6)


@pytest.mark.unit
class TestChunkedEdgeCases:
    """Test edge cases specific to chunked implementation."""

    def test_chunked_with_more_chunks_than_timesteps(self):
        """Should raise ValueError when n_chunks > n_time."""
        # Arrange
        n_time = 5
        n_states = 3
        init = jnp.ones(n_states) / n_states
        trans = jnp.eye(n_states) * 0.8 + 0.2 / n_states
        log_likes = jnp.zeros((n_time, n_states))

        # Act & Assert - Should raise ValueError
        with pytest.raises(ValueError, match="n_chunks .* cannot exceed n_time"):
            run_chunked(init, trans, log_likes, n_chunks=10)  # More than n_time

    def test_chunked_with_single_timestep(self):
        """Should handle single timestep."""
        # Arrange
        n_states = 4
        init = jnp.ones(n_states) / n_states
        trans = jnp.eye(n_states) * 0.8 + 0.2 / n_states
        log_likes = jnp.zeros((1, n_states))

        # Act
        _, acausal_state_probs, _, causal_state_probs, *_ = run_chunked(
            init, trans, log_likes, n_chunks=1
        )

        # Assert
        assert acausal_state_probs.shape == (1, n_states)
        # For single timestep, filtered and smoothed should be identical
        assert jnp.allclose(acausal_state_probs, causal_state_probs)
