"""Unit tests for discrete state transitions module.

Tests cover the core public functions that can be tested in isolation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    centered_softmax_forward,
    centered_softmax_inverse,
    estimate_joint_distribution,
    jax_centered_log_softmax_forward,
    make_transition_from_diag,
)

from non_local_detector.tests.conftest import assert_stochastic_matrix


class TestCenteredSoftmax:
    """Test centered softmax transformation and its inverse."""

    def test_centered_softmax_forward_1d_sums_to_one(self):
        """Forward transform should produce probabilities summing to 1."""
        # Arrange
        y = np.log([2.0, 3.0, 4.0])

        # Act
        result = centered_softmax_forward(y)

        # Assert
        assert np.allclose(result.sum(), 1.0)
        assert len(result) == 4  # Should add one more element

    def test_centered_softmax_forward_example_case(self):
        """Test documented example case."""
        # Arrange
        y = np.log([2.0, 3.0, 4.0])

        # Act
        result = centered_softmax_forward(y)

        # Assert
        expected = [0.2, 0.3, 0.4, 0.1]
        assert np.allclose(result, expected)

    def test_centered_softmax_forward_2d_array(self):
        """Should handle 2D arrays properly."""
        # Arrange
        y = np.array([[np.log(2), np.log(3)], [np.log(4), np.log(5)]])

        # Act
        result = centered_softmax_forward(y)

        # Assert
        assert result.shape == (2, 3)  # Adds one column
        for i in range(2):
            assert np.allclose(result[i].sum(), 1.0)

    def test_centered_softmax_inverse_recovers_original(self):
        """Inverse should recover original values (up to constant)."""
        # Arrange
        y = np.asarray([0.2, 0.3, 0.4, 0.1])

        # Act
        result = centered_softmax_inverse(y)

        # Assert
        # Result should be log of [2, 3, 4] up to additive constant
        expected_ratio = np.exp(result)
        assert np.allclose(expected_ratio, [2.0, 3.0, 4.0])

    def test_centered_softmax_roundtrip(self):
        """Forward then inverse should be identity."""
        # Arrange
        y = np.log([1.5, 2.5, 3.5])

        # Act
        forward = centered_softmax_forward(y)
        inverse = centered_softmax_inverse(forward)

        # Assert
        # Should recover original up to additive constant
        assert np.allclose(inverse - inverse[0], y - y[0])

    def test_jax_centered_log_softmax_forward(self):
        """JAX version should match numpy version in log space."""
        # Arrange
        y = jnp.log(jnp.array([2.0, 3.0, 4.0]))

        # Act
        result = jax_centered_log_softmax_forward(y)

        # Assert
        # Result should be log of softmax
        expected_softmax = centered_softmax_forward(np.asarray(y))
        expected_log_softmax = np.log(expected_softmax)
        assert jnp.allclose(result, expected_log_softmax)


class TestTransitionMatrixConstruction:
    """Test functions for constructing transition matrices."""

    def test_make_transition_from_diag_2states(self):
        """Construct 2x2 transition matrix from diagonal."""
        # Arrange
        diag = np.array([0.9, 0.8])

        # Act
        trans = make_transition_from_diag(diag)

        # Assert
        assert trans.shape == (2, 2)
        assert_stochastic_matrix(trans)
        assert np.allclose(trans[0, 0], 0.9)
        assert np.allclose(trans[0, 1], 0.1)
        assert np.allclose(trans[1, 1], 0.8)
        assert np.allclose(trans[1, 0], 0.2)

    def test_make_transition_from_diag_many_states(self):
        """Should work with many states."""
        # Arrange
        n_states = 10
        diag = np.linspace(0.6, 0.95, n_states)

        # Act
        trans = make_transition_from_diag(diag)

        # Assert
        assert trans.shape == (n_states, n_states)
        assert_stochastic_matrix(trans)
        # Check diagonal values
        for i in range(n_states):
            assert np.allclose(trans[i, i], diag[i])

    def test_make_transition_from_diag_extreme_values(self):
        """Handle extreme diagonal values (0 and 1)."""
        # Arrange
        diag = np.array([0.0, 1.0, 0.5])

        # Act
        trans = make_transition_from_diag(diag)

        # Assert
        assert_stochastic_matrix(trans)
        assert np.allclose(trans[0, 0], 0.0)
        assert np.allclose(trans[1, 1], 1.0)


class TestEstimateJointDistribution:
    """Test joint distribution estimation for EM algorithm."""

    def test_estimate_joint_distribution_shape(self):
        """Joint distribution should have correct shape."""
        # Arrange
        n_time = 10
        n_states = 3
        causal_posterior = np.random.rand(n_time, n_states)
        causal_posterior = causal_posterior / causal_posterior.sum(axis=1, keepdims=True)

        predictive = np.random.rand(n_time, n_states)
        predictive = predictive / predictive.sum(axis=1, keepdims=True)

        trans = np.eye(n_states) * 0.8 + 0.2 / n_states

        acausal_posterior = np.random.rand(n_time, n_states)
        acausal_posterior = acausal_posterior / acausal_posterior.sum(
            axis=1, keepdims=True
        )

        # Act
        joint = estimate_joint_distribution(
            causal_posterior, predictive, trans, acausal_posterior
        )

        # Assert
        assert joint.shape == (n_time - 1, n_states, n_states)

    def test_estimate_joint_distribution_stationary_transition(self):
        """Test with stationary transition matrix."""
        # Arrange
        n_time = 5
        n_states = 2

        # Simple uniform distributions
        causal = np.ones((n_time, n_states)) / n_states
        predictive = np.ones((n_time, n_states)) / n_states
        acausal = np.ones((n_time, n_states)) / n_states
        trans = np.ones((n_states, n_states)) / n_states

        # Act
        joint = estimate_joint_distribution(causal, predictive, trans, acausal)

        # Assert
        # Each joint distribution should sum to 1
        for t in range(n_time - 1):
            assert np.allclose(joint[t].sum(), 1.0)

    def test_estimate_joint_distribution_nonstationary_transition(self):
        """Test with non-stationary transition matrix."""
        # Arrange
        n_time = 5
        n_states = 2

        causal = np.ones((n_time, n_states)) / n_states
        predictive = np.ones((n_time, n_states)) / n_states
        acausal = np.ones((n_time, n_states)) / n_states

        # Time-varying transition matrix
        trans = np.zeros((n_time, n_states, n_states))
        for t in range(n_time):
            trans[t] = np.eye(n_states) * (0.5 + 0.1 * t) + 0.1

        # Normalize
        trans = trans / trans.sum(axis=-1, keepdims=True)

        # Act
        joint = estimate_joint_distribution(causal, predictive, trans, acausal)

        # Assert
        for t in range(n_time - 1):
            assert np.allclose(joint[t].sum(), 1.0)
