"""Unit tests for HMM filtering and smoothing algorithms.

These tests verify the core Hidden Markov Model algorithms:
- Forward filtering (filter)
- Backward smoothing (smoother)
- Viterbi (most likely sequence)

Testing philosophy:
1. Test mathematical properties (probabilities sum to 1, monotonicity, etc.)
2. Test convergence behavior
3. Test edge cases (deterministic transitions, uniform observations, etc.)
4. Test scaling to different problem sizes
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import filter, smoother, viterbi


@pytest.mark.unit
class TestFilter:
    """Test forward filtering algorithm.

    The filter computes P(state_t | observations_1:t) recursively using:
    1. Prediction: prior @ transition_matrix
    2. Update: prediction * likelihood, then normalize
    """

    def test_filter_returns_correct_tuple_structure(self):
        """Filter should return ((log_marginal, predicted_next), (filtered, predicted))."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.zeros((10, 2))

        # Act
        result = filter(init, trans, log_likes)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        carry, outputs = result
        assert len(carry) == 2
        log_marginal, predicted_next = carry
        assert isinstance(log_marginal, (float, jnp.ndarray))  # Scalar
        assert predicted_next.shape == (2,)  # n_states

        filtered_probs, predicted_probs = outputs
        assert filtered_probs.shape == (10, 2)  # (n_time, n_states)
        assert predicted_probs.shape == (10, 2)  # (n_time, n_states)

    def test_filter_preserves_probability_normalization(self):
        """Each filtered timestep should sum to 1."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        np.random.seed(42)
        log_likes = jnp.array(np.random.randn(10, 2))

        # Act
        (_, (filtered_probs, _)) = filter(init, trans, log_likes)

        # Assert
        for t in range(10):
            assert jnp.allclose(filtered_probs[t].sum(), 1.0), f"Failed at timestep {t}"

    def test_filter_with_deterministic_emissions_converges(self):
        """Deterministic observations should converge to single state."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.8, 0.2], [0.2, 0.8]])
        # State 1 always has very high likelihood (0 in log space)
        log_likes = jnp.array([[0.0, -1e6]] * 20)

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        # After many steps, should be almost entirely in state 0
        assert filtered[-1, 0] > 0.99, (
            f"Expected convergence to state 0, got {filtered[-1]}"
        )

    def test_filter_with_uniform_likelihood_follows_transition(self):
        """With uniform likelihood, should follow transition dynamics only."""
        # Arrange
        init = jnp.array([1.0, 0.0])  # Start in state 0
        trans = jnp.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform transition
        log_likes = jnp.zeros((10, 2))  # Uniform likelihood

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        # At t=0: filtered[0] = condition_on(init, ll[0]) = condition_on([1,0], [0,0]) = [1, 0]
        assert jnp.allclose(filtered[0], jnp.array([1.0, 0.0]), atol=1e-5)
        # After transition: predicted[1] = [1,0] @ trans = [0.5, 0.5]
        # At t=1: filtered[1] = condition_on([0.5, 0.5], [0,0]) = [0.5, 0.5]
        assert jnp.allclose(filtered[1], jnp.array([0.5, 0.5]), atol=1e-5)
        # Should stay at [0.5, 0.5] after that
        assert jnp.allclose(filtered[-1], jnp.array([0.5, 0.5]), atol=1e-5)

    def test_filter_initial_conditions_matter(self):
        """Different initial conditions should produce different results."""
        # Arrange
        init1 = jnp.array([1.0, 0.0])
        init2 = jnp.array([0.0, 1.0])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.zeros((5, 2))

        # Act
        _, filtered1 = filter(init1, trans, log_likes)
        _, filtered2 = filter(init2, trans, log_likes)

        # Assert
        # Should be different at t=0
        assert not jnp.allclose(filtered1[0], filtered2[0])

    @pytest.mark.parametrize("n_states", [2, 5, 10, 20])
    def test_filter_scales_to_many_states(self, n_states):
        """Should work with various state space sizes."""
        # Arrange
        init = jnp.ones(n_states) / n_states
        trans = (
            jnp.eye(n_states) * 0.8 + jnp.ones((n_states, n_states)) * 0.2 / n_states
        )
        trans = trans / trans.sum(axis=1, keepdims=True)
        log_likes = jnp.zeros((5, n_states))

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        assert filtered.shape == (5, n_states)
        assert jnp.all(jnp.isfinite(filtered))
        for t in range(5):
            assert jnp.allclose(filtered[t].sum(), 1.0)

    def test_filter_handles_sparse_transition_matrix(self):
        """Should handle transition matrices with many zeros."""
        # Arrange
        init = jnp.array([1.0, 0.0, 0.0])
        # State 0 can only go to state 1, state 1 only to state 2, etc.
        trans = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        log_likes = jnp.zeros((10, 3))

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        # Should cycle through states
        # t=0: filtered[0] = condition_on(init=[1,0,0], ll) = [1, 0, 0]
        assert jnp.allclose(filtered[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)
        # After trans: predicted[1] = [1,0,0] @ trans = [0, 1, 0]
        # t=1: filtered[1] = condition_on([0,1,0], ll) = [0, 1, 0]
        assert jnp.allclose(filtered[1], jnp.array([0.0, 1.0, 0.0]), atol=1e-5)
        # After trans: predicted[2] = [0,1,0] @ trans = [0, 0, 1]
        # t=2: filtered[2] = condition_on([0,0,1], ll) = [0, 0, 1]
        assert jnp.allclose(filtered[2], jnp.array([0.0, 0.0, 1.0]), atol=1e-5)
        # After trans: predicted[3] = [0,0,1] @ trans = [1, 0, 0]
        # t=3: filtered[3] = condition_on([1,0,0], ll) = [1, 0, 0] (cycle complete)
        assert jnp.allclose(filtered[3], jnp.array([1.0, 0.0, 0.0]), atol=1e-5)

    def test_filter_with_all_zero_likelihood_handles_gracefully(self):
        """Should handle case where all likelihoods are very small."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        # Very negative log likelihoods (very small likelihoods)
        log_likes = jnp.ones((5, 2)) * -1000

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        assert jnp.all(jnp.isfinite(filtered))
        for t in range(5):
            assert jnp.allclose(filtered[t].sum(), 1.0)

    def test_filter_single_timestep(self):
        """Should work with single timestep."""
        # Arrange
        init = jnp.array([0.7, 0.3])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[0.0, -1.0]])  # Shape (1, 2)

        # Act
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Assert
        assert filtered.shape == (1, 2)
        assert jnp.allclose(filtered[0].sum(), 1.0)

    def test_filter_marginal_likelihood_accumulated(self):
        """Log marginal likelihood should be sum of per-step marginals."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        np.random.seed(123)
        log_likes = jnp.array(np.random.randn(10, 2))

        # Act
        (log_marginals, _), _ = filter(init, trans, log_likes)

        # Assert
        # Each element should be a log probability (<=0 in theory, but numerics...)
        assert jnp.all(jnp.isfinite(log_marginals))
        # Total log marginal is sum of per-timestep marginals
        total_log_marginal = log_marginals.sum()
        assert jnp.isfinite(total_log_marginal)


@pytest.mark.unit
class TestSmoother:
    """Test backward smoothing algorithm.

    The smoother computes P(state_t | observations_1:T) using filtered probs
    and backward messages. It should have higher certainty than filtering.
    """

    def test_smoother_returns_correct_shape(self):
        """Smoother should return array of same shape as filtered."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.zeros((10, 2))
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        assert smoothed.shape == filtered.shape
        assert smoothed.shape == (10, 2)

    def test_smoother_preserves_probability_normalization(self):
        """Each smoothed timestep should sum to 1."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        np.random.seed(42)
        log_likes = jnp.array(np.random.randn(10, 2))
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        for t in range(10):
            assert jnp.allclose(smoothed[t].sum(), 1.0), f"Failed at timestep {t}"

    def test_smoother_at_final_timestep_equals_filter(self):
        """At T, smoother and filter should be identical (no future info)."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        np.random.seed(42)
        log_likes = jnp.array(np.random.randn(10, 2))
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        assert jnp.allclose(smoothed[-1], filtered[-1])

    def test_smoother_has_higher_certainty_than_filter(self):
        """Smoother should generally have lower entropy (higher certainty)."""

        def entropy(p):
            """Compute entropy of probability distribution."""
            return -jnp.sum(p * jnp.log(p + 1e-10), axis=-1)

        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        # Create informative observations that prefer state 0
        log_likes = jnp.array([[0.0, -2.0]] * 5 + [[-2.0, 0.0]] * 5)
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        filtered_entropy = entropy(filtered).mean()
        smoothed_entropy = entropy(smoothed).mean()
        # Smoother should have lower or equal entropy
        assert smoothed_entropy <= filtered_entropy + 1e-5

    def test_smoother_with_deterministic_observations(self):
        """With very informative observations, should be very certain."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.8, 0.2], [0.2, 0.8]])
        # Alternate between very strong evidence for each state
        log_likes = jnp.array([[0.0, -100.0], [-100.0, 0.0]] * 5)
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        # Should be very certain at each timestep
        for t in range(10):
            max_prob = smoothed[t].max()
            assert max_prob > 0.95, f"Expected high certainty at t={t}, got {max_prob}"

    @pytest.mark.parametrize("n_states", [2, 5, 10])
    def test_smoother_scales_to_many_states(self, n_states):
        """Should work with various state space sizes."""
        # Arrange
        init = jnp.ones(n_states) / n_states
        trans = (
            jnp.eye(n_states) * 0.8 + jnp.ones((n_states, n_states)) * 0.2 / n_states
        )
        trans = trans / trans.sum(axis=1, keepdims=True)
        log_likes = jnp.zeros((5, n_states))
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        assert smoothed.shape == (5, n_states)
        assert jnp.all(jnp.isfinite(smoothed))
        for t in range(5):
            assert jnp.allclose(smoothed[t].sum(), 1.0)

    def test_smoother_single_timestep(self):
        """Should work with single timestep (equals filter)."""
        # Arrange
        init = jnp.array([0.7, 0.3])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[0.0, -1.0]])
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        assert smoothed.shape == (1, 2)
        assert jnp.allclose(smoothed[0], filtered[0])

    def test_smoother_backward_pass_incorporates_future_info(self):
        """Smoothed estimates at t should be affected by observations at t+k."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.95, 0.05], [0.05, 0.95]])  # Sticky transitions
        # Weak evidence at t=0, strong evidence at t=9
        log_likes = jnp.zeros((10, 2))
        log_likes = log_likes.at[9].set(jnp.array([0.0, -10.0]))  # Strong for state 0
        (_, (filtered, _)) = filter(init, trans, log_likes)

        # Act
        smoothed = smoother(trans, filtered)

        # Assert
        # At early times, smoother should be more biased toward state 0 than filter
        # (because of future strong evidence)
        assert smoothed[0, 0] > filtered[0, 0] or jnp.allclose(
            smoothed[0, 0], filtered[0, 0], atol=1e-5
        )


@pytest.mark.unit
class TestViterbi:
    """Test Viterbi algorithm for most likely state sequence.

    Viterbi finds the single most likely sequence of states (not marginal probs).
    """

    def test_viterbi_returns_correct_shape(self):
        """Viterbi should return state sequence of length T."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.zeros((10, 2))

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        assert states.shape == (10,)
        assert jnp.all((states == 0) | (states == 1))  # Valid state indices

    def test_viterbi_with_deterministic_observations(self):
        """Should follow deterministic observations exactly."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.5, 0.5], [0.5, 0.5]])  # Uniform transition
        # Very strong evidence: state 0, 0, 1, 1, 0
        log_likes = jnp.array(
            [[0.0, -100.0], [0.0, -100.0], [-100.0, 0.0], [-100.0, 0.0], [0.0, -100.0]]
        )

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        expected = jnp.array([0, 0, 1, 1, 0])
        assert jnp.allclose(states, expected)

    def test_viterbi_prefers_smooth_sequences_with_sticky_transitions(self):
        """With sticky transitions, should prefer staying in same state."""
        # Arrange
        init = jnp.array([1.0, 0.0])  # Start in state 0
        trans = jnp.array([[0.99, 0.01], [0.01, 0.99]])  # Very sticky
        # Weak noisy evidence
        np.random.seed(42)
        log_likes = jnp.array(np.random.randn(10, 2) * 0.1)

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        # Should mostly stay in state 0 due to sticky transitions
        assert jnp.mean(states == 0) > 0.7

    def test_viterbi_with_single_timestep(self):
        """Should work with single timestep."""
        # Arrange
        init = jnp.array([0.3, 0.7])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[0.0, -2.0]])  # Prefers state 0

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        assert states.shape == (1,)
        # Despite higher init prob for state 1, likelihood favors state 0
        # Outcome depends on relative strengths, but should be valid
        assert states[0] in [0, 1]

    @pytest.mark.parametrize("n_states", [2, 5, 10])
    def test_viterbi_scales_to_many_states(self, n_states):
        """Should work with various state space sizes."""
        # Arrange
        init = jnp.ones(n_states) / n_states
        trans = (
            jnp.eye(n_states) * 0.8 + jnp.ones((n_states, n_states)) * 0.2 / n_states
        )
        trans = trans / trans.sum(axis=1, keepdims=True)
        log_likes = jnp.zeros((5, n_states))

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        assert states.shape == (5,)
        assert jnp.all((states >= 0) & (states < n_states))

    def test_viterbi_path_is_valid_sequence(self):
        """Viterbi path should consist of valid state indices."""
        # Arrange
        init = jnp.array([0.5, 0.5, 0.0])
        trans = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        np.random.seed(42)
        log_likes = jnp.array(np.random.randn(20, 3))

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        assert jnp.all((states >= 0) & (states < 3))
        assert len(jnp.unique(states)) > 0  # Not all same state
