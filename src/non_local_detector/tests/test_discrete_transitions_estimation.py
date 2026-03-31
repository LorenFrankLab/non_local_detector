"""High-priority tests for discrete state transition estimation functions.

Tests the core EM algorithm functions that are currently untested:
- estimate_non_stationary_state_transition
- estimate_stationary_state_transition
- _estimate_discrete_transition
"""

import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    _estimate_discrete_transition,
    estimate_non_stationary_state_transition,
    estimate_stationary_state_transition,
)
from non_local_detector.tests.conftest import assert_stochastic_matrix


@pytest.mark.unit
class TestEstimateNonStationaryStateTransition:
    """Test non-stationary state transition estimation (EM algorithm)."""

    def test_returns_correct_shapes(self, posterior_data, design_matrix_data):
        """Verify coefficient and transition matrix shapes are correct."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Act
        coeffs, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # No stickiness (uniform prior)
            transition_regularization=1e-5,
            maxiter=10,  # Limit iterations for speed
        )

        # Assert
        n_coeffs, n_states = dm["n_coefficients"], post["n_states"]
        assert coeffs.shape == (n_coeffs, n_states, n_states - 1)
        assert trans_matrix.shape == (post["n_time"], n_states, n_states)

    def test_produces_valid_probabilities(self, posterior_data, design_matrix_data):
        """Check all transition matrices are valid stochastic matrices."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Act
        _, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Assert - check each time step
        for t in range(trans_matrix.shape[0]):
            assert_stochastic_matrix(trans_matrix[t])

    def test_with_different_concentrations(self, posterior_data, design_matrix_data):
        """Test with different concentration (prior strength) values."""
        post = posterior_data
        dm = design_matrix_data

        # Test with weak prior (concentration=1.0)
        _, trans_weak = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Test with strong prior (concentration=10.0)
        _, trans_strong = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=10.0,
            stickiness=0.0,  # uniform prior
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Both should be valid
        for t in range(min(5, trans_weak.shape[0])):  # Check first 5 timesteps
            assert_stochastic_matrix(trans_weak[t])
            assert_stochastic_matrix(trans_strong[t])

    def test_with_diagonal_stickiness(self, posterior_data, design_matrix_data):
        """Test with diagonal stickiness prior."""
        post = posterior_data
        dm = design_matrix_data

        # Act
        _, trans_matrix = estimate_non_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_coefficients=dm["transition_coefficients"],
            concentration=1.0,
            stickiness=1.0,  # Diagonal stickiness - favors self-transitions
            transition_regularization=1e-5,
            maxiter=10,
        )

        # Assert - diagonal should be larger than off-diagonal on average
        diagonal_mean = np.mean(
            [
                trans_matrix[t, i, i]
                for t in range(trans_matrix.shape[0])
                for i in range(post["n_states"])
            ]
        )
        off_diagonal_mean = np.mean(
            [
                trans_matrix[t, i, j]
                for t in range(trans_matrix.shape[0])
                for i in range(post["n_states"])
                for j in range(post["n_states"])
                if i != j
            ]
        )

        assert diagonal_mean > off_diagonal_mean, (
            "Diagonal stickiness should favor self-transitions"
        )


@pytest.mark.unit
class TestEstimateStationaryStateTransition:
    """Test stationary state transition estimation."""

    def test_returns_stochastic_matrix(self, posterior_data):
        """Verify output is a valid stochastic matrix."""
        # Arrange
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert
        assert trans_matrix.shape == (post["n_states"], post["n_states"])
        assert_stochastic_matrix(trans_matrix)

    def test_respects_uniform_prior(self, posterior_data):
        """Test with uniform prior (concentration=1.0)."""
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert - should be a valid stochastic matrix
        assert_stochastic_matrix(trans_matrix)
        # All probabilities should be reasonable (not extreme)
        assert np.all(trans_matrix > 1e-6), (
            "No probability should be exactly zero with uniform prior"
        )

    def test_respects_diagonal_prior(self, posterior_data):
        """Test with diagonal stickiness prior."""
        post = posterior_data

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=post["causal_posterior"],
            predictive_distribution=post["predictive_distribution"],
            acausal_posterior=post["acausal_posterior"],
            transition_matrix=post["transition_matrix"],
            concentration=1.0,
            stickiness=2.0,  # Diagonal stickiness
        )

        # Assert
        assert_stochastic_matrix(trans_matrix)
        # Diagonal should be favored
        diagonal_mean = np.mean(np.diag(trans_matrix))
        off_diagonal_mean = np.mean(trans_matrix[~np.eye(post["n_states"], dtype=bool)])
        assert diagonal_mean > off_diagonal_mean

    def test_numerical_stability_with_small_probabilities(self):
        """Test with very small posterior probabilities."""
        # Arrange - create extreme case with very small probabilities
        n_time, n_states = 20, 4
        causal_posterior = np.ones((n_time, n_states)) * 1e-8
        causal_posterior[:, 0] = 1.0 - 3e-8
        acausal_posterior = causal_posterior.copy()

        transition_matrix = np.eye(n_states) * 0.9 + 0.025
        predictive_distribution = np.zeros((n_time, n_states))
        for t in range(n_time):
            predictive_distribution[t] = causal_posterior[t] @ transition_matrix

        # Act
        trans_matrix = estimate_stationary_state_transition(
            causal_posterior=causal_posterior,
            predictive_distribution=predictive_distribution,
            acausal_posterior=acausal_posterior,
            transition_matrix=transition_matrix,
            concentration=1.0,
            stickiness=0.0,  # uniform prior
        )

        # Assert - should not contain NaN or inf
        assert np.all(np.isfinite(trans_matrix))
        assert_stochastic_matrix(trans_matrix)


@pytest.mark.unit
class TestEstimateDiscreteTransition:
    """Test _estimate_discrete_transition wrapper function."""

    def test_stationary_diagonal(self, posterior_data):
        """Test with stationary diagonal transition type."""
        # Arrange
        post = posterior_data

        # Act - _estimate_discrete_transition uses transition matrix directly
        new_trans, _ = _estimate_discrete_transition(
            causal_state_probabilities=post["causal_posterior"],
            predictive_state_probabilities=post["predictive_distribution"],
            acausal_state_probabilities=post["acausal_posterior"],
            discrete_transition=post["transition_matrix"],
            discrete_transition_coefficients=None,
            discrete_transition_design_matrix=None,
            transition_concentration=1.0,
            transition_stickiness=1.0,  # Diagonal stickiness
            transition_regularization=1e-5,
        )

        # Assert
        assert new_trans.shape == (post["n_states"], post["n_states"])
        assert_stochastic_matrix(new_trans)

    def test_non_stationary_diagonal(self, posterior_data, design_matrix_data):
        """Test with non-stationary diagonal transition type."""
        # Arrange
        post = posterior_data
        dm = design_matrix_data

        # Create time-varying transition matrix
        time_varying_trans = np.tile(post["transition_matrix"], (post["n_time"], 1, 1))

        # Act - returns (transition_matrix, coefficients)
        new_trans, new_coeffs = _estimate_discrete_transition(
            causal_state_probabilities=post["causal_posterior"],
            predictive_state_probabilities=post["predictive_distribution"],
            acausal_state_probabilities=post["acausal_posterior"],
            discrete_transition=time_varying_trans,
            discrete_transition_coefficients=dm["transition_coefficients"],
            discrete_transition_design_matrix=dm["design_matrix"][: post["n_time"]],
            transition_concentration=1.0,
            transition_stickiness=1.0,  # Diagonal stickiness
            transition_regularization=1e-5,
        )

        # Assert
        assert new_trans.shape == (post["n_time"], post["n_states"], post["n_states"])
        assert new_coeffs.shape == dm["transition_coefficients"].shape
        for t in range(min(5, new_trans.shape[0])):
            assert_stochastic_matrix(new_trans[t])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
