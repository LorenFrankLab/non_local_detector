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

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import core as core_module
from non_local_detector.core import (
    _condition_on,
    chunked_filter_smoother,
    filter,
    filter_covariate_dependent,
    smoother,
    viterbi,
)


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
        assert isinstance(log_marginal, float | jnp.ndarray)  # Scalar
        assert predicted_next.shape == (2,)  # n_states

        filtered_probs, predicted_probs = outputs
        assert filtered_probs.shape == (10, 2)  # (n_time, n_states)
        assert predicted_probs.shape == (10, 2)  # (n_time, n_states)

    def test_filter_preserves_probability_normalization(self):
        """Each filtered timestep should sum to 1."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        rng = np.random.default_rng(42)
        log_likes = jnp.array(rng.standard_normal((10, 2)))

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
        rng = np.random.default_rng(123)
        log_likes = jnp.array(rng.standard_normal((10, 2)))

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
        rng = np.random.default_rng(42)
        log_likes = jnp.array(rng.standard_normal((10, 2)))
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
        rng = np.random.default_rng(42)
        log_likes = jnp.array(rng.standard_normal((10, 2)))
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
        assert smoothed[0, 0] > filtered[0, 0] + 1e-6, (
            f"Smoother should incorporate future evidence: "
            f"smoothed={float(smoothed[0, 0]):.6f}, filtered={float(filtered[0, 0]):.6f}"
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
        rng = np.random.default_rng(42)
        log_likes = jnp.array(rng.standard_normal((10, 2)) * 0.1)

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
        rng = np.random.default_rng(42)
        log_likes = jnp.array(rng.standard_normal((20, 3)))

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert
        assert jnp.all((states >= 0) & (states < 3))
        assert len(jnp.unique(states)) > 1  # Path should visit multiple states


@pytest.mark.unit
class TestConditionOnDegenerateLikelihoods:
    """Behaviour of ``_condition_on`` when every state has -inf log-likelihood.

    A degenerate timestep means no state has any finite evidence; rather than
    return an all-zero (unnormalized) posterior, the predicted distribution is
    returned unchanged and ``log_norm`` is set to -inf so callers can detect
    the situation on the host.
    """

    def test_condition_on_all_inf_falls_back_to_predicted(self):
        """All -inf log-likelihoods leave predicted probabilities unchanged."""
        # Arrange
        probs = jnp.array([0.3, 0.7])
        ll = jnp.array([-jnp.inf, -jnp.inf])

        # Act
        new_probs, log_norm = _condition_on(probs, ll)

        # Assert
        assert jnp.allclose(new_probs, probs, atol=1e-7)
        assert jnp.isneginf(log_norm)
        # The predicted distribution is preserved (it already sums to 1).
        assert jnp.allclose(new_probs.sum(), 1.0, atol=1e-7)

    def test_condition_on_partial_inf_still_normalizes(self):
        """A single finite state still yields a valid normalized posterior."""
        # Arrange: state 0 is impossible, state 1 has finite log-likelihood.
        probs = jnp.array([0.4, 0.6])
        ll = jnp.array([-jnp.inf, -0.5])

        # Act
        new_probs, log_norm = _condition_on(probs, ll)

        # Assert
        assert jnp.all(new_probs >= 0)
        assert jnp.allclose(new_probs.sum(), 1.0, atol=1e-7)
        # The -inf state must receive zero posterior mass.
        assert float(new_probs[0]) == 0.0
        assert float(new_probs[1]) == pytest.approx(1.0, abs=1e-7)
        assert jnp.isfinite(log_norm)

    def test_condition_on_all_finite_matches_legacy_normalization(self):
        """Well-behaved input is unchanged from the pre-fallback path."""
        # Arrange
        probs = jnp.array([0.4, 0.6])
        ll = jnp.array([-1.0, -2.0])

        # Act
        new_probs, log_norm = _condition_on(probs, ll)

        # Assert: reproduce the math by hand.
        weights = probs * jnp.exp(ll - ll.max())
        expected_probs = weights / weights.sum()
        expected_log_norm = jnp.log(weights.sum()) + ll.max()
        assert jnp.allclose(new_probs, expected_probs, atol=1e-7)
        assert jnp.isclose(log_norm, expected_log_norm, atol=1e-7)

    def test_condition_on_partial_nan_propagates_not_masked(self):
        """A NaN in one state must NOT be silently replaced by the prior.

        The degenerate fallback is for *all-impossible* (every state ``-inf``)
        timesteps only. A NaN log-likelihood signals a bug in the likelihood
        computation (e.g. a non-converged encoding model), so it must remain
        visible in the posterior rather than being laundered into the
        predicted distribution.
        """
        # Arrange: state 0 has valid finite evidence, state 1 is NaN.
        probs = jnp.array([0.4, 0.6])
        ll = jnp.array([-0.5, jnp.nan])

        # Act
        new_probs, log_norm = _condition_on(probs, ll)

        # Assert: NaN propagates (not the prior), so the bug stays visible.
        assert jnp.any(jnp.isnan(new_probs)), (
            "NaN log-likelihood was silently masked by the predicted prior; "
            f"got new_probs={new_probs}"
        )
        assert not jnp.allclose(new_probs, probs, equal_nan=False)

    def test_condition_on_all_inf_with_nan_elsewhere_is_not_treated_as_degenerate(self):
        """A mix of ``-inf`` and ``NaN`` is a NaN case, not an all-impossible one."""
        # Arrange
        probs = jnp.array([0.5, 0.5])
        ll = jnp.array([-jnp.inf, jnp.nan])

        # Act
        new_probs, _ = _condition_on(probs, ll)

        # Assert: NaN dominates -> not the clean prior fallback.
        assert jnp.any(jnp.isnan(new_probs))

    def test_condition_on_zero_normalizer_from_support_mismatch_falls_back(self):
        """A zero normalizer with finite evidence must not yield all-zero posterior.

        If the predicted distribution places zero probability on every state
        that has *finite* log-likelihood, the Bayes update is 0/0. This is not
        an all-``-inf`` (impossible-data) step -- the likelihood is finite -- so
        an ``ll_max == -inf`` guard does not fire. Without a normalizer check the
        posterior silently becomes all zeros and propagates forward as an
        invalid (non-summing-to-one) distribution. The fallback must instead
        return the predicted distribution and mark ``log_norm`` as ``-inf``.
        """
        # Arrange: all predicted mass on state 0, but only state 1 is possible.
        probs = jnp.array([1.0, 0.0])
        ll = jnp.array([-jnp.inf, 0.0])

        # Act
        new_probs, log_norm = _condition_on(probs, ll)

        # Assert: still a valid distribution, equal to the prediction, and the
        # step is marked via log_norm = -inf.
        assert jnp.allclose(new_probs.sum(), 1.0, atol=1e-7), (
            f"posterior must remain a valid distribution; got {new_probs}"
        )
        assert jnp.allclose(new_probs, probs, atol=1e-7)
        assert jnp.isneginf(log_norm)


@pytest.mark.unit
class TestFilterDegenerateTimestepWarning:
    """Host-side warning when the filter encounters all-impossible likelihoods."""

    def test_filter_degenerate_step_warns(self, caplog):
        """A time step with all -inf log-likelihoods triggers a logger.warning.

        The posterior at that step should fall back to the predicted
        distribution, and the remainder of the filter should complete normally.
        """
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        # Three timesteps: middle one has all -inf log-likelihoods.
        log_likes = jnp.array(
            [
                [0.0, -1.0],
                [-jnp.inf, -jnp.inf],
                [0.0, -1.0],
            ]
        )

        # Act
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            (_, (filtered, predicted)) = filter(init, trans, log_likes)

        # Assert: warning records present and mention all-impossible likelihoods.
        warning_records = [
            r
            for r in caplog.records
            if "all-impossible" in r.getMessage() and r.levelno == logging.WARNING
        ]
        assert warning_records, (
            "Expected a logger.warning about degenerate timesteps; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )

        # The posterior at the degenerate step equals the predicted distribution.
        assert jnp.allclose(filtered[1], predicted[1], atol=1e-7)
        assert jnp.allclose(filtered[1].sum(), 1.0, atol=1e-7)

        # The first and third timesteps complete normally.
        assert jnp.all(jnp.isfinite(filtered[0]))
        assert jnp.all(jnp.isfinite(filtered[2]))
        assert jnp.allclose(filtered[0].sum(), 1.0, atol=1e-6)
        assert jnp.allclose(filtered[2].sum(), 1.0, atol=1e-6)

    def test_filter_well_behaved_input_emits_no_warning(self, caplog):
        """A finite-everywhere filter run should not emit the degenerate warning."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        rng = np.random.default_rng(7)
        log_likes = jnp.array(rng.standard_normal((10, 2)))

        # Act
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            filter(init, trans, log_likes)

        # Assert: no degenerate-step warnings.
        warning_records = [
            r for r in caplog.records if "all-impossible" in r.getMessage()
        ]
        assert not warning_records, (
            f"Did not expect degenerate-step warnings; got: "
            f"{[r.getMessage() for r in warning_records]}"
        )

    def test_filter_nan_step_warns_distinctly_from_all_impossible(self, caplog):
        """A NaN log-likelihood step warns about NaN, not 'all-impossible'.

        A NaN signals a likelihood-computation bug, which must be reported
        differently from a legitimately impossible (all ``-inf``) timestep so
        the user is not misdirected.
        """
        # Arrange: middle step has a NaN (state 0 is otherwise finite).
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array(
            [
                [0.0, -1.0],
                [0.0, jnp.nan],
                [0.0, -1.0],
            ]
        )

        # Act
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            filter(init, trans, log_likes)

        # Assert: a NaN-specific warning fired...
        nan_records = [r for r in caplog.records if "NaN" in r.getMessage()]
        assert nan_records, (
            "Expected a NaN-specific warning; "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )
        # ...and the step was NOT mislabeled as all-impossible.
        impossible_records = [
            r for r in caplog.records if "all-impossible" in r.getMessage()
        ]
        assert not impossible_records, (
            "A NaN step was misreported as all-impossible: "
            f"{[r.getMessage() for r in impossible_records]}"
        )


@pytest.mark.unit
class TestChunkedFilterDegenerateWarning:
    """Degenerate-step tally across chunks in ``chunked_filter_smoother``."""

    def test_chunked_filter_counts_degenerate_across_chunks(self, caplog):
        """Degenerate steps in different chunks are summed into one warning."""
        # Arrange: 4 timesteps, 2 chunks -> [0, 1] and [2, 3]. Put one
        # all-impossible step in each chunk (index 1 and index 2).
        time = np.arange(4.0)
        state_ind = np.array([0, 1])
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        log_likelihoods = np.array(
            [
                [0.0, -1.0],
                [-np.inf, -np.inf],
                [-np.inf, -np.inf],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )

        # Act
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            chunked_filter_smoother(
                time=time,
                state_ind=state_ind,
                initial_distribution=initial_distribution,
                transition_matrix=transition_matrix,
                log_likelihood_func=lambda *a, **k: None,
                log_likelihood_args=(),
                n_chunks=2,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=False,
                dtype=jnp.float64,
            )

        # Assert: a single warning reporting 2 of 4 degenerate steps.
        records = [r for r in caplog.records if "all-impossible" in r.getMessage()]
        assert len(records) == 1, (
            f"Expected exactly one degenerate-step warning; "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )
        assert "2/4" in records[0].getMessage(), records[0].getMessage()

    def test_chunked_filter_collects_global_degenerate_indices(self):
        """``degenerate_indices_out`` receives the GLOBAL indices of all-impossible
        steps, even when likelihoods are uncached (so they cannot be recovered
        from the returned ``log_likelihoods``, which is then ``None``)."""
        # Arrange: degenerate steps at global indices 1 and 2, across 2 chunks.
        time = np.arange(4.0)
        state_ind = np.array([0, 1])
        initial_distribution = np.array([0.5, 0.5])
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        all_ll = np.array(
            [
                [0.0, -1.0],
                [-np.inf, -np.inf],
                [-np.inf, -np.inf],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )

        # Compute likelihoods per chunk (uncached) so the driver returns None
        # for log_likelihoods -- the indices then survive only via the collector.
        def ll_func(time_chunk, *args, is_missing=None):
            return all_ll[np.asarray(time_chunk).astype(int)]

        collected: list[int] = []

        # Act
        result = chunked_filter_smoother(
            time=time,
            state_ind=state_ind,
            initial_distribution=initial_distribution,
            transition_matrix=transition_matrix,
            log_likelihood_func=ll_func,
            log_likelihood_args=(),
            n_chunks=2,
            log_likelihoods=None,
            cache_log_likelihoods=False,
            dtype=jnp.float64,
            degenerate_indices_out=collected,
        )

        # Assert: global indices collected; the returned log_likelihoods is None
        # (uncached), so this is the only place those indices survive.
        assert sorted(collected) == [1, 2]
        assert result[5] is None


@pytest.mark.unit
class TestFilterJitComposable:
    """``filter`` and ``filter_covariate_dependent`` stay JAX-transformable.

    The host-side degenerate/NaN diagnostics must be skipped under tracing so a
    caller can still do ``jax.jit(filter)(...)`` (or call the filter inside an
    outer transformation) without a ConcretizationTypeError.
    """

    def test_filter_composes_under_jit(self):
        import jax

        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        rng = np.random.default_rng(0)
        log_likes = jnp.array(rng.standard_normal((8, 2)))

        # Act: jit-compose the public wrapper (raised ConcretizationTypeError
        # before the tracer guard).
        (_, (filtered_jit, _)) = jax.jit(filter)(init, trans, log_likes)
        (_, (filtered_ref, _)) = filter(init, trans, log_likes)

        # Assert: identical result to the un-jitted call.
        assert jnp.allclose(filtered_jit, filtered_ref, atol=1e-6)
        assert jnp.allclose(filtered_jit.sum(axis=-1), 1.0, atol=1e-6)

    def test_filter_covariate_dependent_composes_under_jit(self):
        import jax

        n_time, n_states = 6, 2
        init = jnp.ones(n_states) / n_states
        state_ind = jnp.arange(n_states)
        c_tm = jnp.eye(n_states)
        d_tm = jnp.broadcast_to(
            jnp.array([[0.9, 0.1], [0.1, 0.9]]), (n_time, n_states, n_states)
        )
        rng = np.random.default_rng(1)
        ll = jnp.array(rng.standard_normal((n_time, n_states)))

        (_, (filtered_jit, _)) = jax.jit(filter_covariate_dependent)(
            init, d_tm, c_tm, state_ind, ll
        )
        (_, (filtered_ref, _)) = filter_covariate_dependent(
            init, d_tm, c_tm, state_ind, ll
        )

        assert jnp.allclose(filtered_jit, filtered_ref, atol=1e-6)
        assert jnp.allclose(filtered_jit.sum(axis=-1), 1.0, atol=1e-6)


@pytest.mark.unit
class TestDegenerateMarginalLikelihoodPropagation:
    """End-to-end behaviour of the marginal LL and smoother at a degenerate step."""

    def test_marginal_likelihood_is_neg_inf_and_smoother_stays_valid(self):
        """A degenerate step drives the marginal LL to -inf but the smoother
        still returns finite, normalized posteriors at every step."""
        # Arrange
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array(
            [
                [0.0, -1.0],
                [-jnp.inf, -jnp.inf],
                [0.0, -1.0],
            ]
        )

        # Act
        (marginal, _), (filtered, _) = filter(init, trans, log_likes)
        smoothed = smoother(trans, filtered)

        # Assert: the impossible step makes the total marginal LL -inf.
        assert jnp.isneginf(marginal)
        # The smoother output is finite and normalized everywhere.
        assert jnp.all(jnp.isfinite(smoothed))
        assert jnp.allclose(smoothed.sum(axis=-1), 1.0, atol=1e-6)
