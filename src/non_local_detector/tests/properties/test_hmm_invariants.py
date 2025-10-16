"""Property-based tests for HMM algorithm invariants.

These tests verify that HMM algorithms (filter, smoother, viterbi) satisfy
mathematical properties and invariants that should hold for all valid inputs.
"""

# Import custom strategies from probability properties
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from non_local_detector.core import filter, smoother, viterbi

# Add parent directory to import the strategies
sys.path.insert(0, str(Path(__file__).parent))
from test_probability_properties import probability_distribution, stochastic_matrix


def entropy(p):
    """Compute Shannon entropy of probability distribution."""
    # Avoid log(0) by adding small epsilon
    return -jnp.sum(p * jnp.log(p + 1e-10), axis=-1)


@pytest.mark.property
class TestHMMInvariants:
    """Test mathematical invariants of HMM algorithms."""

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=1, max_value=20),
    )
    def test_filter_preserves_probability_normalization(self, n_states, n_timesteps):
        """Property: filter output should always be normalized probability distributions."""
        # Generate matching-sized distributions
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        # Generate random observations
        log_likes = np.random.randn(n_timesteps, n_states)

        # Run filter
        (_, (filtered, _)) = filter(
            jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes)
        )

        # Check all timesteps are valid probability distributions
        for t in range(n_timesteps):
            assert jnp.allclose(filtered[t].sum(), 1.0, rtol=1e-5)
            assert jnp.all(filtered[t] >= 0)
            assert jnp.all(filtered[t] <= 1)

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=1, max_value=15),
    )
    def test_smoother_at_final_timestep_equals_filter(self, n_states, n_timesteps):
        """Property: at final timestep T, smoother and filter should be identical."""
        # Generate matching-sized init and trans
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        log_likes = np.random.randn(n_timesteps, n_states)

        (_, (filtered, _)) = filter(
            jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes)
        )
        smoothed = smoother(jnp.asarray(trans), filtered)

        # At final time, no future information, so filter == smoother
        assert jnp.allclose(smoothed[-1], filtered[-1], rtol=1e-5)

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=1, max_value=15),
    )
    def test_viterbi_returns_valid_state_sequence(self, n_states, n_timesteps):
        """Property: viterbi should return a valid sequence of state indices."""
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        log_likes = np.random.randn(n_timesteps, n_states)

        states = viterbi(jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes))

        # Check output shape
        assert states.shape == (n_timesteps,)

        # Check all states are valid indices
        assert jnp.all((states >= 0) & (states < n_states))

    @given(st.integers(min_value=2, max_value=5))
    def test_uniform_likelihood_preserves_transition_dynamics(self, n_states):
        """Property: with uniform likelihood, filter should follow pure transition dynamics."""
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        # Uniform likelihood (all zeros in log space)
        n_timesteps = 5
        log_likes = jnp.zeros((n_timesteps, n_states))

        (_, (filtered, _)) = filter(jnp.asarray(init), jnp.asarray(trans), log_likes)

        # First filtered state should match initial distribution
        assert jnp.allclose(filtered[0], jnp.asarray(init), rtol=1e-5)

        # Each subsequent state should follow from previous via transition matrix
        for t in range(1, n_timesteps):
            # predicted[t] = filtered[t-1] @ trans
            # filtered[t] = condition_on(predicted[t], uniform_ll) = predicted[t]
            predicted = filtered[t - 1] @ jnp.asarray(trans)
            # Normalize to account for numerical precision
            predicted = predicted / predicted.sum()
            assert jnp.allclose(filtered[t], predicted, rtol=1e-4)

    @given(
        stochastic_matrix(min_size=2, max_size=5),
        st.integers(min_value=2, max_value=10),  # Need at least 2 timesteps
    )
    def test_deterministic_initial_state_with_uniform_likelihood(
        self, trans, n_timesteps
    ):
        """Property: starting from deterministic state with uniform likelihood."""
        n_states = trans.shape[0]

        # Start deterministically in state 0
        init = jnp.zeros(n_states)
        init = init.at[0].set(1.0)

        # Uniform likelihood
        log_likes = jnp.zeros((n_timesteps, n_states))

        (_, (filtered, _)) = filter(init, jnp.asarray(trans), log_likes)

        # First timestep should be deterministic in state 0
        assert jnp.allclose(filtered[0, 0], 1.0, rtol=1e-5)
        assert jnp.allclose(filtered[0, 1:], 0.0, atol=1e-5)

        # Second timestep should match first row of transition matrix
        assert jnp.allclose(filtered[1], trans[0, :], rtol=1e-3)

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=2, max_value=10),
    )
    def test_filter_marginal_likelihood_is_real_valued(self, n_states, n_timesteps):
        """Property: marginal log likelihood should be real-valued (not NaN or inf)."""
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        log_likes = np.random.randn(n_timesteps, n_states)

        ((log_marginals, _), _) = filter(
            jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes)
        )

        # Marginal likelihoods should be finite real numbers
        assert jnp.all(jnp.isfinite(log_marginals))
        # Sum should be finite
        total_log_likelihood = log_marginals.sum()
        assert jnp.isfinite(total_log_likelihood)

    @given(st.integers(min_value=2, max_value=5))
    def test_smoother_preserves_probability_normalization(self, n_states):
        """Property: smoother output should always be normalized probability distributions."""
        init = np.random.rand(n_states)
        init = init / init.sum()

        trans = np.random.rand(n_states, n_states)
        trans = trans / trans.sum(axis=1, keepdims=True)

        log_likes = np.random.randn(5, n_states)

        (_, (filtered, _)) = filter(
            jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes)
        )
        smoothed = smoother(jnp.asarray(trans), filtered)

        # Check all timesteps are valid probability distributions
        for t in range(len(smoothed)):
            assert jnp.allclose(smoothed[t].sum(), 1.0, rtol=1e-5)
            assert jnp.all(smoothed[t] >= 0)
            assert jnp.all(smoothed[t] <= 1)

    @given(
        probability_distribution(min_size=3, max_size=5),
        stochastic_matrix(min_size=3, max_size=5),
    )
    def test_viterbi_with_deterministic_observations_follows_evidence(
        self, init, trans
    ):
        """Property: with very strong evidence, viterbi should follow observations."""
        n_states = len(init)
        assume(trans.shape[0] == n_states)

        # Create sequence that strongly indicates states: 0, 1, 2, 0, 1
        n_timesteps = min(5, n_states)
        log_likes = np.ones((n_timesteps, n_states)) * -100.0  # Very low likelihood

        # Set high likelihood for specific states
        for t in range(n_timesteps):
            state = t % n_states
            log_likes[t, state] = 0.0  # High likelihood

        states = viterbi(jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes))

        # With very strong evidence (deterministic), viterbi should follow it
        # (unless transition probabilities are extremely contradictory)
        expected_states = jnp.array([t % n_states for t in range(n_timesteps)])

        # Check that most states match (allow for transition matrix influence)
        match_ratio = jnp.mean(states == expected_states)
        assert match_ratio >= 0.6  # At least 60% should match
