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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import core as core_module
from non_local_detector.core import (
    _condition_on,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    filter,
    filter_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
    row_slice_aware,
    smoother,
    viterbi,
    viterbi_covariate_dependent,
)
from non_local_detector.exceptions import DataError
from tests.conftest import precision_mode


@pytest.mark.unit
class TestFilter:
    """Test forward filtering algorithm.

    The filter computes P(state_t | observations_1:t) recursively using:
    1. Prediction: prior @ transition_matrix
    2. Update: prediction * likelihood, then normalize
    """

    def test_filter_carry_predicts_the_row_after_the_last(self):
        """The returned carry is the evidence and ``filtered[-1] @ trans``."""
        rng = np.random.default_rng(0)
        init = jnp.array([0.3, 0.7])
        trans = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        log_likes = jnp.asarray(rng.standard_normal((10, 2)))

        (log_marginal, predicted_next), (filtered, _) = filter(init, trans, log_likes)

        # Measured error 4e-8 (float32); a carry one row stale differs by ~1e-1.
        np.testing.assert_allclose(
            predicted_next, np.asarray(filtered[-1]) @ np.asarray(trans), atol=1e-6
        )
        assert np.ndim(log_marginal) == 0

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
        np.testing.assert_allclose(filtered.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-8)

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
        np.testing.assert_allclose(filtered.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-8)

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


@pytest.mark.unit
class TestSmoother:
    """Test backward smoothing algorithm.

    The smoother computes P(state_t | observations_1:T) using filtered probs
    and backward messages.
    """

    @pytest.mark.parametrize("n_states", [2, 5])
    @pytest.mark.parametrize("n_timesteps", [1, 10, 15])
    def test_smoother_at_final_timestep_equals_filter(self, n_states, n_timesteps):
        """At the final row there is no future evidence, so smoothed == filtered."""
        rng = np.random.default_rng(0)
        init = rng.random(n_states)
        init = init / init.sum()
        trans = rng.random((n_states, n_states))
        trans = trans / trans.sum(axis=1, keepdims=True)
        log_likes = rng.standard_normal((n_timesteps, n_states))

        (_, (filtered, _)) = filter(
            jnp.asarray(init), jnp.asarray(trans), jnp.asarray(log_likes)
        )
        smoothed = smoother(jnp.asarray(trans), filtered)

        # Measured error 0 (float32) and 1e-16 (float64); pulling the final
        # row back through the transition changes it by ~1e-1.
        assert smoothed.shape == filtered.shape
        np.testing.assert_allclose(smoothed[-1], filtered[-1], rtol=0, atol=1e-6)

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
        np.testing.assert_allclose(smoothed.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-8)

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
        """A single step picks the argmax of log initial + log likelihood."""
        # Arrange
        init = jnp.array([0.3, 0.7])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[0.0, -2.0]])  # Prefers state 0

        # Act
        states = viterbi(init, trans, log_likes)

        # Assert: log 0.3 + 0 = -1.20 beats log 0.7 - 2 = -2.36, so the
        # likelihood outweighs the higher initial probability of state 1.
        assert states.shape == (1,)
        assert int(states[0]) == 0

    @pytest.mark.parametrize("covariate", [False, True])
    def test_viterbi_scan_keeps_float32_with_x64(self, covariate):
        """With 64-bit mode on, float32 inputs keep the scan in float32.

        The backward carry used to start as ``jnp.zeros(n_state_bins)``, which
        is float64 under x64, so every step promoted the (n_state_bins,
        n_state_bins) max-plus scores to float64.
        """
        with precision_mode(True):
            n_bins, n_time = 4, 6
            arguments = [
                jnp.full(n_bins, 1 / n_bins, jnp.float32),
                jnp.full((n_bins, n_bins), 1 / n_bins, jnp.float32),
                jnp.zeros((n_time, n_bins), jnp.float32),
            ]
            if covariate:
                function = core_module._viterbi_covariate_dependent_impl
                arguments[1:2] = [
                    jnp.full((n_time, 2, 2), 0.5, jnp.float32),
                    arguments[1],
                    jnp.array([0, 0, 1, 1]),
                ]
            else:
                function = core_module._viterbi_impl
            jaxpr = jax.make_jaxpr(function)(*arguments).jaxpr
            scans = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
            float_dtypes = {
                var.aval.dtype
                for eqn in scans
                for var in (*eqn.invars, *eqn.outvars)
                if jnp.issubdtype(var.aval.dtype, jnp.floating)
            }
        assert scans
        assert float_dtypes == {jnp.dtype(jnp.float32)}, float_dtypes

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


@pytest.mark.unit
class TestConditionOnDegenerateLikelihoods:
    """``_condition_on`` when some, but not all, states are impossible.

    The all-impossible, support-mismatch and NaN cases are checked against a
    float64 reference in ``test_conditioning_reference.py``.
    """

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

    def test_filter_nan_step_raises(self, caplog):
        """A NaN log-likelihood step raises instead of being logged.

        A NaN signals a likelihood-computation bug, which must not be taken
        for a legitimately impossible (all ``-inf``) timestep.
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

        with (
            caplog.at_level(logging.WARNING, logger=core_module.logger.name),
            pytest.raises(DataError, match=r"filtering .* 1/3 time step\(s\) with NaN"),
        ):
            filter(init, trans, log_likes)

        assert not [r for r in caplog.records if "all-impossible" in r.getMessage()]

    def test_filter_reports_impossible_steps_before_raising_on_nan(self, caplog):
        """Both problems are reported: the log is not lost to the error."""
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[-jnp.inf, -jnp.inf], [0.0, jnp.nan], [0.0, -1.0]])

        with (
            caplog.at_level(logging.WARNING, logger=core_module.logger.name),
            pytest.raises(DataError, match="NaN"),
        ):
            filter(init, trans, log_likes)

        assert [r for r in caplog.records if "all-impossible" in r.getMessage()]

    def test_transformed_filter_cannot_raise_and_propagates_nan(self):
        """Under ``jax.jit`` the check is skipped and the NaN reaches the output."""
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        log_likes = jnp.array([[0.0, -1.0], [0.0, jnp.nan], [0.0, -1.0]])

        _, (filtered, _) = jax.jit(filter)(init, trans, log_likes)

        assert np.all(np.isnan(np.asarray(filtered)[1:]))


@pytest.mark.unit
class TestChunkedFilterDegenerateWarning:
    """Degenerate-step tally across chunks in ``chunked_filter_smoother``."""

    def test_chunked_filter_counts_degenerate_across_chunks(self, caplog):
        """Degenerate steps in different chunks are summed into one warning."""
        # Arrange: 4 timesteps, 2 chunks -> [0, 1] and [2, 3]. Put one
        # all-impossible step in each chunk (index 1 and index 2).
        time_edges = np.arange(5.0)
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
                time_edges=time_edges,
                state_ind=state_ind,
                initial_distribution=initial_distribution,
                transition_matrix=transition_matrix,
                log_likelihood_func=lambda *a, **k: None,
                log_likelihood_args=(),
                n_chunks=2,
                log_likelihoods=log_likelihoods,
                cache_log_likelihoods=False,
            )

        # Assert: a single warning reporting 2 of 4 degenerate steps.
        records = [r for r in caplog.records if "all-impossible" in r.getMessage()]
        assert len(records) == 1, (
            f"Expected exactly one degenerate-step warning; "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )
        assert "2/4" in records[0].getMessage(), records[0].getMessage()

    def test_chunked_filter_collects_global_degenerate_indices(self):
        """``degenerate_indices`` holds the GLOBAL indices of all-impossible
        steps, even when likelihoods are uncached (so they cannot be recovered
        from the returned ``log_likelihoods``, which is then ``None``)."""
        # Arrange: degenerate steps at global indices 1 and 2, across 2 chunks.
        time_edges = np.arange(5.0)
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
        @row_slice_aware
        def ll_func(*args, time_edges, is_missing=None, row_slice=None):
            return all_ll[row_slice]

        # Act
        result = chunked_filter_smoother(
            time_edges=time_edges,
            state_ind=state_ind,
            initial_distribution=initial_distribution,
            transition_matrix=transition_matrix,
            log_likelihood_func=ll_func,
            log_likelihood_args=(),
            n_chunks=2,
            log_likelihoods=None,
            cache_log_likelihoods=False,
        )

        # Assert: global indices returned; the returned log_likelihoods is None
        # (uncached), so this is the only place those indices survive.
        np.testing.assert_array_equal(result.degenerate_indices, [1, 2])
        assert result.degenerate_indices.dtype.kind == "i"
        assert result.log_likelihoods is None


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


@pytest.mark.unit
class TestViterbiDegenerateTimestepWarning:
    """The Viterbi drivers emit degeneracy warnings with Viterbi-appropriate
    wording (no posterior / predicted-distribution fallback)."""

    @staticmethod
    def _run(log_likes, caplog):
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            states, _ = most_likely_sequence(
                time_edges=np.arange(log_likes.shape[0] + 1),
                initial_distribution=init,
                transition_matrix=trans,
                log_likelihood_func=None,
                log_likelihood_args=(),
                log_likelihoods=log_likes,
            )
        return states

    def test_viterbi_degenerate_step_warns_with_viterbi_wording(self, caplog):
        """An all-(-inf) step warns, and the message describes Viterbi behavior
        (the step is treated as uninformative), NOT the filter's posterior
        fallback."""
        log_likes = jnp.array([[0.0, -1.0], [-jnp.inf, -jnp.inf], [0.0, -1.0]])
        self._run(log_likes, caplog)
        msgs = [
            r.getMessage() for r in caplog.records if "all-impossible" in r.getMessage()
        ]
        assert msgs, (
            f"expected a degenerate warning; got {[r.getMessage() for r in caplog.records]}"
        )
        joined = " ".join(msgs)
        assert "Viterbi decoding" in joined
        assert "uninformative" in joined
        # Must not use the filter-only phrasing.
        assert "posterior" not in joined
        assert "predicted distribution" not in joined

    def test_viterbi_nan_step_raises(self, caplog):
        """A NaN step raises, with Viterbi wording."""
        log_likes = jnp.array([[0.0, -1.0], [jnp.nan, 0.0], [0.0, -1.0]])
        with pytest.raises(DataError, match="Viterbi decoding"):
            self._run(log_likes, caplog)

    def test_viterbi_well_behaved_input_emits_no_warning(self, caplog):
        """A finite-everywhere Viterbi run emits no degeneracy warning."""
        rng = np.random.default_rng(3)
        log_likes = jnp.array(rng.standard_normal((10, 2)))
        self._run(log_likes, caplog)
        assert not [r for r in caplog.records if "all-impossible" in r.getMessage()]


def _decode_with_log_likelihoods(path, log_likes, is_missing=None):
    """Run one decoding path on a two-state stationary HMM; return its result."""
    init = np.array([0.5, 0.5])
    trans = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_ind = np.array([0, 1])
    discrete = np.broadcast_to(trans, (log_likes.shape[0], 2, 2))
    continuous = np.eye(2)
    driver = {
        "time_edges": np.arange(float(log_likes.shape[0]) + 1),
        "initial_distribution": init,
        "log_likelihood_func": None,
        "log_likelihood_args": (),
        "log_likelihoods": log_likes,
        "is_missing": is_missing,
    }
    covariate = {
        "state_ind": state_ind,
        "discrete_transition_matrix": discrete,
        "continuous_transition_matrix": continuous,
    }
    if path == "filter":
        return filter(init, trans, log_likes)
    if path == "filter_covariate":
        return filter_covariate_dependent(
            init, discrete, continuous, state_ind, log_likes
        )
    if path == "chunked_filter_smoother":
        return chunked_filter_smoother(
            **driver, state_ind=state_ind, transition_matrix=trans, n_chunks=2
        )
    if path == "chunked_covariate":
        return chunked_filter_smoother_covariate_dependent(
            **driver, **covariate, n_chunks=2
        )
    if path == "most_likely_sequence_covariate":
        return most_likely_sequence_covariate_dependent(**driver, **covariate)
    if path == "viterbi":
        return viterbi(init, trans, log_likes)
    if path == "viterbi_covariate":
        return viterbi_covariate_dependent(
            init, discrete, continuous, state_ind, log_likes
        )
    return most_likely_sequence(**driver, transition_matrix=trans)


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "chunked_filter_smoother",
        "chunked_covariate",
        "most_likely_sequence",
        "most_likely_sequence_covariate",
    ],
)
def test_a_missing_row_has_no_likelihood_to_refuse(path):
    """The drivers neutralize it, as ``checkpointed_forward_backward`` does:
    the result equals a decode of the same rows with the missing row set to 0
    (no evidence), and the NaN in that row is not refused."""
    is_missing = np.array([False, True, False, False])
    # The path stays in state 1 when the missing row carries no evidence, but
    # the row's finite entry (-6 against a switching cost of 2 log 0.1 = -4.6)
    # would move it to state 0 at row 1 if a driver used the row.
    log_likes = np.array([[-1.0, 0.0], [np.nan, -6.0], [-1.0, 0.0], [-1.0, 0.0]])
    without_evidence = log_likes.copy()
    without_evidence[is_missing] = 0.0

    actual = _decode_with_log_likelihoods(path, log_likes, is_missing=is_missing)
    expected = _decode_with_log_likelihoods(path, without_evidence)

    if path.startswith("most_likely_sequence"):
        np.testing.assert_array_equal(actual[0], expected[0])
    else:
        acausal, _, marginal, *_ = actual
        np.testing.assert_array_equal(acausal, expected[0])
        assert marginal == expected[2]


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "filter",
        "filter_covariate",
        "chunked_filter_smoother",
        "most_likely_sequence",
        "most_likely_sequence_covariate",
        "viterbi",
        "viterbi_covariate",
    ],
)
def test_positive_infinite_log_likelihood_raises(path, caplog):
    """A +inf log likelihood is refused like a NaN one, not passed silently.

    ``log_likelihood - max`` is ``inf - inf = NaN`` at that step, so the
    posterior is NaN from there on (and the Viterbi path is undefined); the
    step is counted with the NaN steps.
    """
    log_likes = np.array([[0.0, -1.0], [np.inf, 0.0], [0.0, -1.0], [0.0, -1.0]])
    with (
        caplog.at_level(logging.WARNING, logger=core_module.logger.name),
        pytest.raises(DataError, match=r"1/4 time step\(s\) with NaN or \+inf"),
    ):
        _decode_with_log_likelihoods(path, log_likes)
    assert not any("all-impossible" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize("path", ["viterbi", "viterbi_covariate"])
def test_public_viterbi_warns_on_impossible_steps(path, caplog):
    """The jit-composable Viterbi passes warn about all-(-inf) steps, as the
    filters do, with the Viterbi wording."""
    log_likes = np.array([[0.0, -1.0], [-np.inf, -np.inf], [0.0, -1.0]])
    with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
        _decode_with_log_likelihoods(path, log_likes)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "1/3 time step(s)" in m and "all-impossible" in m and "Viterbi decoding" in m
        for m in messages
    ), messages


@pytest.mark.unit
def test_public_viterbi_stays_jit_composable(caplog):
    """Under ``jit`` the inputs are tracers, so no diagnostics run or warn."""
    init = jnp.array([0.5, 0.5])
    trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
    log_likes = jnp.array([[0.0, -1.0], [-jnp.inf, -jnp.inf], [0.0, -1.0]])
    with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
        jitted = jax.jit(viterbi)(init, trans, log_likes)
    assert not caplog.records
    np.testing.assert_array_equal(jitted, viterbi(init, trans, log_likes))


@pytest.mark.unit
class TestFilterMarginalUnderflowWarning:
    """A -inf marginal with no all-(-inf) step (support mismatch / underflow)
    is surfaced by its own warning."""

    def test_support_mismatch_warns_marginal_underflow(self, caplog):
        """With an identity transition pinned to state 0, a step whose only
        finite likelihood is on state 1 has a zero Bayes normalizer despite
        having a finite likelihood -- not an all-(-inf) step. The marginal is
        -inf and the dedicated support-mismatch/underflow warning fires."""
        init = jnp.array([1.0, 0.0])
        trans = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # identity: mass stays on state 0
        log_likes = jnp.array([[0.0, -1.0], [-jnp.inf, 0.0], [0.0, -1.0]])

        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            (marginal, _), _ = filter(init, trans, log_likes)

        assert jnp.isneginf(marginal)
        underflow_msgs = [
            r.getMessage()
            for r in caplog.records
            if "support mismatch or underflow" in r.getMessage()
        ]
        assert underflow_msgs, (
            "expected a marginal-underflow warning; got "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        # No step is all-(-inf), so the all-impossible warning must NOT fire.
        assert not [r for r in caplog.records if "all-impossible" in r.getMessage()]

    def test_normal_filter_emits_no_marginal_underflow_warning(self, caplog):
        """A finite marginal must not trigger the underflow warning."""
        init = jnp.array([0.5, 0.5])
        trans = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        rng = np.random.default_rng(11)
        log_likes = jnp.array(rng.standard_normal((10, 2)))
        with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
            filter(init, trans, log_likes)
        assert not [
            r
            for r in caplog.records
            if "support mismatch or underflow" in r.getMessage()
        ]


@pytest.mark.unit
class TestViterbiGlobalOptimum:
    """Viterbi must honor transition structure over greedy per-step likelihood."""

    def test_viterbi_follows_forced_transitions_against_greedy_likelihood(self):
        """A deterministic transition chain forces a path the likelihood disfavors.

        Per-timestep argmax of the log-likelihoods would pick state 0 at every
        step, but the transition matrix forms a forced chain 0 -> 1 -> 2 from
        the initial state. A correct Viterbi decode returns [0, 1, 2]; a greedy
        decode that ignored the global path constraint would return [0, 0, 0].
        """
        initial = jnp.array([1.0, 0.0, 0.0])
        transition = jnp.array(
            [
                [0.0, 1.0, 0.0],  # state 0 -> 1
                [0.0, 0.0, 1.0],  # state 1 -> 2
                [0.0, 0.0, 1.0],  # state 2 -> 2 (absorbing)
            ]
        )
        # Likelihood prefers state 0 at every timestep.
        log_likelihoods = jnp.array(
            [
                [0.0, -20.0, -20.0],
                [0.0, -20.0, -20.0],
                [0.0, -20.0, -20.0],
            ]
        )
        path = viterbi(initial, transition, log_likelihoods)
        assert tuple(int(s) for s in path) == (0, 1, 2)
        # Greedy per-step argmax would have been (0, 0, 0).
        greedy = tuple(int(s) for s in jnp.argmax(log_likelihoods, axis=1))
        assert greedy == (0, 0, 0)
        assert tuple(int(s) for s in path) != greedy

    def test_viterbi_smooths_over_likelihood_blip_with_soft_transitions(self):
        """All transitions are feasible; backtracking must discriminate.

        Unlike the forced-chain test above (where only one path has nonzero
        probability), here every transition is strictly positive, so multiple
        paths are feasible and the backward pass must compare finite scores to
        pick the optimal predecessor at each step. The likelihood flip-flops --
        it prefers state 1 only at the middle step -- so a greedy per-step
        decode returns [0, 1, 0]. Sticky transitions make the two flips
        (0->1->0) cost far more than the middle-step likelihood gain, so the
        global Viterbi optimum is the smooth path [0, 0, 0].
        """
        initial = jnp.array([0.5, 0.5])
        transition = jnp.array([[0.95, 0.05], [0.05, 0.95]])  # all entries > 0
        log_likelihoods = jnp.array(
            [
                [0.0, -1.0],  # prefer 0
                [0.0, 0.5],  # blip: prefer 1
                [0.0, -1.0],  # prefer 0
            ]
        )
        path = viterbi(initial, transition, log_likelihoods)
        greedy = tuple(int(s) for s in jnp.argmax(log_likelihoods, axis=1))
        assert greedy == (0, 1, 0)
        assert tuple(int(s) for s in path) == (0, 0, 0)


def _simulate_place_cell_decoding(n_time: int) -> dict[str, np.ndarray]:
    """A random-walk trajectory and the log likelihoods of 40 Poisson place cells.

    60 position bins, a Gaussian random-walk transition with a 1.5-bin
    standard deviation, 2 ms bins and a fixed seed.

    Returns
    -------
    model : dict
        ``time_edges`` (n_time + 1,), ``initial_distribution`` (n_bins,),
        ``transition_matrix`` (n_bins, n_bins) and ``log_likelihoods``
        (n_time, n_bins), all float64.
    """
    n_bins, n_cells, dt = 60, 40, 0.002
    rng = np.random.default_rng(0)
    bins = np.arange(n_bins)
    transition = np.exp(-0.5 * ((bins[:, None] - bins[None, :]) / 1.5) ** 2)
    transition /= transition.sum(axis=1, keepdims=True)

    position = np.empty(n_time, dtype=int)
    position[0] = n_bins // 2
    cumulative = np.cumsum(transition, axis=1)
    draws = rng.random(n_time)
    for t in range(1, n_time):
        position[t] = min(
            np.searchsorted(cumulative[position[t - 1]], draws[t]), n_bins - 1
        )
    centers = rng.uniform(0, n_bins, n_cells)
    rates = 0.5 + 15 * np.exp(-0.5 * ((bins[None] - centers[:, None]) / 4) ** 2)
    counts = rng.poisson(rates[:, position].T * dt)  # (n_time, n_cells)
    log_likelihoods = counts @ np.log(rates * dt) - (rates * dt).sum(axis=0)
    return {
        "time_edges": np.arange(n_time + 1) * dt,
        "initial_distribution": np.full(n_bins, 1 / n_bins),
        "transition_matrix": transition,
        "log_likelihoods": log_likelihoods,
    }


@pytest.fixture
def place_cell_decoding(request):
    """:func:`_simulate_place_cell_decoding` for ``request.param`` time bins."""
    return _simulate_place_cell_decoding(request.param)


def _float32_and_float64_paths(model, covariate):
    """Decode ``model`` with both dtypes through one Viterbi driver.

    The covariate-dependent driver gets one discrete state with a constant
    discrete transition, so its continuous transition is the model's.
    """
    common = {
        "time_edges": model["time_edges"],
        "initial_distribution": model["initial_distribution"],
        "log_likelihood_func": None,
        "log_likelihood_args": (),
        "log_likelihoods": model["log_likelihoods"],
    }
    if covariate:
        n_time, n_bins = model["log_likelihoods"].shape
        common |= {
            "state_ind": np.zeros(n_bins, dtype=int),
            "discrete_transition_matrix": np.ones((n_time, 1, 1)),
            "continuous_transition_matrix": model["transition_matrix"],
        }
        decode = most_likely_sequence_covariate_dependent
    else:
        common["transition_matrix"] = model["transition_matrix"]
        decode = most_likely_sequence
    with precision_mode(True):
        return tuple(
            decode(**common, dtype=dtype)[0] for dtype in (np.float32, np.float64)
        )


@pytest.mark.unit
class TestViterbiFloat32Precision:
    """The float32 path equals the float64 path however long the recording.

    The best score of the remaining bins grows linearly with the number of
    bins: about 4.8e5 nats at 200,000 bins, where float32 spacing is 0.03,
    coarser than the transition and likelihood differences that choose
    between neighbouring bins. Without re-centring, 12.8% of those bins
    differed from the float64 path, 0.8% at 20,000 bins.
    """

    @pytest.mark.parametrize("place_cell_decoding", [2_000], indirect=True)
    def test_short_recording_matches_float64(self, place_cell_decoding):
        path32, path64 = _float32_and_float64_paths(place_cell_decoding, False)
        np.testing.assert_array_equal(path32, path64)

    @pytest.mark.slow
    @pytest.mark.parametrize("place_cell_decoding", [200_000], indirect=True)
    def test_long_recording_matches_float64(self, place_cell_decoding):
        path32, path64 = _float32_and_float64_paths(place_cell_decoding, False)
        np.testing.assert_array_equal(path32, path64)

    @pytest.mark.parametrize("place_cell_decoding", [20_000], indirect=True)
    def test_covariate_dependent_matches_float64(self, place_cell_decoding):
        path32, path64 = _float32_and_float64_paths(place_cell_decoding, True)
        np.testing.assert_array_equal(path32, path64)


def _wandering_decoding(n_bins=20, n_time=400):
    """A sinusoidal path over ``n_bins`` with sharp Gaussian likelihoods.

    Returns ``(time_edges, initial, transition, log_likelihoods)``.
    """
    bins = np.arange(n_bins)
    transition = np.exp(-0.5 * (bins[:, None] - bins[None, :]) ** 2)
    transition /= transition.sum(axis=1, keepdims=True)
    true = (10 + 6 * np.sin(np.arange(n_time) / 30)).round().astype(int)
    log_likelihoods = -0.5 * (bins[None, :] - true[:, None]) ** 2 - 5.0
    initial = np.full(n_bins, 1 / n_bins)
    return np.arange(n_time + 1) * 0.002, initial, transition, log_likelihoods


def _viterbi_path(time_edges, initial, transition, log_likelihoods, caplog):
    with caplog.at_level(logging.WARNING, logger=core_module.logger.name):
        path, _ = most_likely_sequence(
            time_edges=time_edges,
            initial_distribution=initial,
            transition_matrix=transition,
            log_likelihood_func=None,
            log_likelihood_args=(),
            log_likelihoods=log_likelihoods,
        )
    return path


@pytest.mark.unit
class TestViterbiImpossibleRow:
    """A row no path can explain changes the path only near that row.

    Its scores are ``-inf`` for every state, so before it was treated as
    uninformative every earlier back-pointer was the argmax of a constant
    and the path sat in bin 0 from the start of the recording to that row.
    """

    far_rows = np.r_[0:291, 310:400]

    def _assert_local(self, log_likelihoods, transition, caplog):
        time_edges, initial, _, _ = _wandering_decoding()
        without_evidence = log_likelihoods.copy()
        without_evidence[300] = 0.0
        expected = _viterbi_path(
            time_edges, initial, transition, without_evidence, caplog
        )
        caplog.clear()
        path = _viterbi_path(time_edges, initial, transition, log_likelihoods, caplog)
        assert np.mean(expected[:300] == 0) < 0.5
        np.testing.assert_array_equal(path[self.far_rows], expected[self.far_rows])
        return [r.getMessage() for r in caplog.records]

    def test_all_impossible_row_is_local(self, caplog):
        _, _, transition, log_likelihoods = _wandering_decoding()
        log_likelihoods[300] = -np.inf
        messages = self._assert_local(log_likelihoods, transition, caplog)
        assert any(
            "1/400 time step(s)" in m and "uninformative" in m for m in messages
        ), messages

    def test_row_finite_only_on_an_unreachable_bin_is_local(self, caplog):
        """No transition enters bin 19, the one bin row 300 allows."""
        _, _, transition, log_likelihoods = _wandering_decoding()
        transition[:, 19] = 0.0
        transition /= transition.sum(axis=1, keepdims=True)
        log_likelihoods[300] = -np.inf
        log_likelihoods[300, 19] = 0.0
        messages = self._assert_local(log_likelihoods, transition, caplog)
        assert any("1/400 time step(s)" in m and "reach" in m for m in messages), (
            messages
        )

    def test_rows_no_path_joins_are_local(self, caplog):
        """Rows 300 and 301 each allow one bin, 13 bins apart, and a step
        moves at most 2 bins: no path explains both, and neither row is
        all-impossible."""
        _, _, transition, log_likelihoods = _wandering_decoding()
        bins = np.arange(len(transition))
        transition[np.abs(bins[:, None] - bins[None, :]) > 2] = 0.0
        transition /= transition.sum(axis=1, keepdims=True)
        log_likelihoods[300:302] = -np.inf
        log_likelihoods[300, 19] = 0.0
        log_likelihoods[301, 6] = 0.0
        messages = self._assert_local(log_likelihoods, transition, caplog)
        assert any("1/400 time step(s)" in m and "reach" in m for m in messages), (
            messages
        )


@pytest.mark.unit
class TestSingleStateHMM:
    """Degenerate n_states=1 case: filtering is trivial but must be exact."""

    def test_filter_single_state_is_certain_and_marginal_is_sum(self):
        """With one state, filtered prob is 1.0 everywhere and the marginal
        log-likelihood equals the sum of the per-timestep log-likelihoods."""
        n_time = 5
        initial = jnp.array([1.0])
        transition = jnp.array([[1.0]])
        log_likelihoods = (jnp.arange(n_time, dtype=jnp.float32) * -0.1).reshape(
            n_time, 1
        )

        (marginal, _), (filtered, _) = filter(initial, transition, log_likelihoods)

        np.testing.assert_allclose(np.asarray(filtered), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            float(marginal), float(log_likelihoods.sum()), atol=1e-5
        )
