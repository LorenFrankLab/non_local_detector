"""Tests to verify chunked algorithms match non-chunked versions.

Critical: The chunked versions are memory-efficient implementations that
should produce identical results to the standard versions. These tests
ensure numerical parity.

Testing philosophy:
1. Chunked algorithms should match standard algorithms exactly
2. Test with various chunk sizes
3. Test with edge cases (single chunk, many chunks)
4. Test with missing data handling
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother,
    filter,
    smoother,
)


class TestChunkedFilterSmootherParity:
    """Verify chunked_filter_smoother matches filter + smoother.

    The chunked version processes data in blocks for memory efficiency
    but should give identical results to the standard implementation.
    """

    def create_simple_test_data(self, n_time=50, n_states=10):
        """Helper to create simple test data."""
        np.random.seed(42)

        init = jnp.ones(n_states) / n_states
        trans = (
            jnp.eye(n_states) * 0.8 + jnp.ones((n_states, n_states)) * 0.2 / n_states
        )
        trans = trans / trans.sum(axis=1, keepdims=True)

        # Simple log likelihood function
        log_likes = jnp.array(np.random.randn(n_time, n_states))

        return init, trans, log_likes

    def test_chunked_equals_standard_single_chunk(self):
        """With n_chunks=1, chunked should exactly match standard."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=5)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        # Simple likelihood function that just returns precomputed likelihoods
        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Standard version
        (_, (filtered_std, _)) = filter(init, trans, log_likes)
        smoothed_std = smoother(trans, filtered_std)

        # Act - Chunked version
        (
            acausal_posterior,
            acausal_state_probabilities,
            log_likelihood,
            causal_state_probabilities,  # causal/filtered probabilities
            predictive_state_probabilities,  # one-step-ahead predictions
            _,
            causal_posterior,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=1,
            log_likelihoods=np.array(log_likes),
        )

        # Assert - shapes should match exactly (no extra initial row)
        assert jnp.allclose(
            smoothed_std, acausal_state_probabilities, rtol=1e-5, atol=1e-6
        ), "Smoothed posteriors don't match"

        assert jnp.allclose(
            filtered_std, causal_state_probabilities, rtol=1e-5, atol=1e-6
        ), "Filtered posteriors don't match"

    def test_chunked_equals_standard_multiple_chunks(self):
        """With multiple chunks, should still match standard."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=100, n_states=8)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Standard version
        (_, (filtered_std, _)) = filter(init, trans, log_likes)
        smoothed_std = smoother(trans, filtered_std)

        # Act - Chunked version with 5 chunks
        (
            _,
            acausal_state_probs,
            _,
            causal_state_probs,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=5,
            log_likelihoods=np.array(log_likes),
        )

        # Assert
        assert jnp.allclose(smoothed_std, acausal_state_probs, rtol=1e-4, atol=1e-5), (
            "Smoothed posteriors don't match with multiple chunks"
        )

        assert jnp.allclose(filtered_std, causal_state_probs, rtol=1e-4, atol=1e-5), (
            "Filtered posteriors don't match with multiple chunks"
        )

    @pytest.mark.parametrize("n_chunks", [1, 2, 5, 10])
    def test_chunked_consistent_across_chunk_sizes(self, n_chunks):
        """Results should be identical regardless of chunk size."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=5)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Get reference with n_chunks=1
        (
            _,
            acausal_ref,
            _,
            causal_ref,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=1,
            log_likelihoods=np.array(log_likes),
        )

        # Act - Test with different n_chunks
        (
            _,
            acausal_test,
            _,
            causal_test,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=n_chunks,
            log_likelihoods=np.array(log_likes),
        )

        # Assert
        assert jnp.allclose(acausal_ref, acausal_test, rtol=1e-5, atol=1e-6), (
            f"Acausal differs for n_chunks={n_chunks}"
        )

        assert jnp.allclose(causal_ref, causal_test, rtol=1e-5, atol=1e-6), (
            f"Causal differs for n_chunks={n_chunks}"
        )

    def test_chunked_log_likelihood_matches_standard(self):
        """Chunked should return same total log likelihood as standard."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=5)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Standard version
        (log_marginals, _), _ = filter(init, trans, log_likes)
        total_log_like_std = log_marginals.sum()

        # Act - Chunked version
        (_, _, log_like_chunked, _, _, _, _) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=3,
            log_likelihoods=np.array(log_likes),
        )

        # Assert
        assert jnp.allclose(
            total_log_like_std, log_like_chunked, rtol=1e-4, atol=1e-5
        ), f"Log likelihoods don't match: {total_log_like_std} vs {log_like_chunked}"

    def test_chunked_with_very_small_chunks(self):
        """Should work even with very small chunk sizes."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=20, n_states=3)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Standard
        (_, (filtered_std, _)) = filter(init, trans, log_likes)
        smoothed_std = smoother(trans, filtered_std)

        # Act - Chunked with n_chunks = n_time (1 timestep per chunk)
        (
            _,
            acausal_chunked,
            _,
            causal_chunked,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=n_time,
            log_likelihoods=np.array(log_likes),
        )

        # Assert
        assert jnp.allclose(smoothed_std, acausal_chunked, rtol=1e-4, atol=1e-5)
        assert jnp.allclose(filtered_std, causal_chunked, rtol=1e-4, atol=1e-5)

    def test_chunked_preserves_probability_normalization(self):
        """Chunked results should be valid probability distributions."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=50, n_states=8)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act
        (
            _,
            acausal_state_probs,
            _,
            causal_state_probs,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=5,
            log_likelihoods=np.array(log_likes),
        )

        # Assert - All probabilities sum to 1
        for t in range(n_time):
            assert jnp.allclose(acausal_state_probs[t].sum(), 1.0, atol=1e-5), (
                f"Acausal probs don't sum to 1 at t={t}"
            )
            assert jnp.allclose(causal_state_probs[t].sum(), 1.0, atol=1e-5), (
                f"Causal probs don't sum to 1 at t={t}"
            )

    def test_chunked_with_missing_data(self):
        """Chunked should handle missing data (if supported)."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=30, n_states=5)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        # Mark some timesteps as missing
        is_missing = np.zeros(n_time, dtype=bool)
        is_missing[[5, 10, 15]] = True

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act
        (
            _,
            acausal_state_probs,
            _,
            causal_state_probs,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            is_missing=is_missing,
            n_chunks=3,
            log_likelihoods=np.array(log_likes),
        )

        # Assert - Should complete without error and produce valid probabilities
        assert jnp.all(jnp.isfinite(acausal_state_probs))
        assert jnp.all(jnp.isfinite(causal_state_probs))
        for t in range(n_time):
            assert jnp.allclose(acausal_state_probs[t].sum(), 1.0, atol=1e-5)
            assert jnp.allclose(causal_state_probs[t].sum(), 1.0, atol=1e-5)

    def test_chunked_with_different_dtypes(self):
        """Chunked should respect dtype argument."""
        # Arrange
        init, trans, log_likes = self.create_simple_test_data(n_time=20, n_states=4)
        n_time, n_states = log_likes.shape
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - float32
        (
            _,
            acausal_f32,
            _,
            _,
            _,
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=2,
            log_likelihoods=np.array(log_likes),
            dtype=jnp.float32,
        )

        # Act - float64
        (
            _,
            acausal_f64,
            _,
            _,
            _,
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=2,
            log_likelihoods=np.array(log_likes),
            dtype=jnp.float64,
        )

        # Assert - Results should be similar (within float32 precision)
        assert jnp.allclose(acausal_f32, acausal_f64, rtol=1e-5, atol=1e-6)


class TestChunkedEdgeCases:
    """Test edge cases specific to chunked implementation."""

    def test_chunked_with_more_chunks_than_timesteps(self):
        """Should handle n_chunks > n_time gracefully."""
        # Arrange
        n_time = 5
        n_states = 3
        init = jnp.ones(n_states) / n_states
        trans = jnp.eye(n_states) * 0.8 + 0.2 / n_states
        log_likes = jnp.zeros((n_time, n_states))
        time = np.arange(n_time)
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act - Request more chunks than timesteps
        (
            _,
            acausal_state_probs,
            _,
            causal_state_probs,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=10,  # More than n_time
            log_likelihoods=np.array(log_likes),
        )

        # Assert - Should still work
        assert acausal_state_probs.shape == (n_time, n_states)
        assert jnp.all(jnp.isfinite(acausal_state_probs))

    def test_chunked_with_single_timestep(self):
        """Should handle single timestep."""
        # Arrange
        n_states = 4
        init = jnp.ones(n_states) / n_states
        trans = jnp.eye(n_states) * 0.8 + 0.2 / n_states
        log_likes = jnp.zeros((1, n_states))
        time = np.array([0])
        state_ind = np.arange(n_states)

        def log_likelihood_func(time_idx, *args):
            return log_likes[time_idx]

        # Act
        (
            _,
            acausal_state_probs,
            _,
            causal_state_probs,  # causal_state_probabilities (filtered)
            _,  # predictive_state_probabilities
            _,
            _,
        ) = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=np.array(init),
            transition_matrix=np.array(trans),
            log_likelihood_func=log_likelihood_func,
            log_likelihood_args=(),
            n_chunks=1,
            log_likelihoods=np.array(log_likes),
        )

        # Assert
        assert acausal_state_probs.shape == (1, n_states)
        # For single timestep, filtered and smoothed should be identical
        assert jnp.allclose(acausal_state_probs, causal_state_probs)
