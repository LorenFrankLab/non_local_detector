"""Unit tests for discrete state transitions module.

Tests cover the core public functions that can be tested in isolation.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
    centered_softmax_forward,
    centered_softmax_inverse,
    dirichlet_neg_log_likelihood,
    estimate_joint_distribution,
    get_transition_prior,
    jax_centered_log_softmax_forward,
    make_transition_from_diag,
    multinomial_neg_log_likelihood,
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
        causal_posterior = causal_posterior / causal_posterior.sum(
            axis=1, keepdims=True
        )

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


# Tests for jax_centered_log_softmax_forward with 2D input
class TestJaxCenteredLogSoftmax2D:
    def test_jax_centered_log_softmax_forward_2d_input(self):
        """Test 2D input case (batched)."""
        y = jnp.array([[0.0, 1.0], [2.0, 3.0]])  # 2 samples, 2 features
        result = jax_centered_log_softmax_forward(y)

        # Should have shape (2, 3) after appending zeros
        assert result.shape == (2, 3)
        # Each row should sum to 1 in probability space
        probs = jnp.exp(result)
        assert jnp.allclose(probs.sum(axis=1), 1.0)


# Tests for multinomial_neg_log_likelihood
class TestMultinomialNegLogLikelihood:
    def test_multinomial_neg_log_likelihood_basic(self):
        """Test basic multinomial negative log likelihood computation."""
        # Arrange
        n_samples, n_coefficients, n_states = 10, 3, 4
        rng = np.random.default_rng(42)

        # Coefficients for n_states-1 outcomes
        coefficients = jnp.array(rng.normal(size=n_coefficients * (n_states - 1)))
        design_matrix = jnp.array(rng.normal(size=(n_samples, n_coefficients)))

        # Response: expected counts/probabilities
        response = jnp.array(rng.dirichlet(np.ones(n_states), size=n_samples))

        # Act
        nll = multinomial_neg_log_likelihood(
            coefficients, design_matrix, response, l2_penalty=1e-3
        )

        # Assert
        assert isinstance(nll, (float, jnp.ndarray))
        assert nll > 0  # Negative log likelihood should be positive

    def test_multinomial_neg_log_likelihood_with_penalty(self):
        """Test that L2 penalty increases the loss."""
        n_samples, n_coefficients, n_states = 10, 3, 3
        rng = np.random.default_rng(43)

        coefficients = jnp.array(rng.normal(size=n_coefficients * (n_states - 1)))
        design_matrix = jnp.array(rng.normal(size=(n_samples, n_coefficients)))
        response = jnp.array(rng.dirichlet(np.ones(n_states), size=n_samples))

        nll_no_penalty = multinomial_neg_log_likelihood(
            coefficients, design_matrix, response, l2_penalty=0.0
        )
        nll_with_penalty = multinomial_neg_log_likelihood(
            coefficients, design_matrix, response, l2_penalty=1.0
        )

        # With penalty should be larger
        assert nll_with_penalty > nll_no_penalty


# Tests for get_transition_prior
class TestGetTransitionPrior:
    def test_get_transition_prior_scalar_stickiness(self):
        """Test with scalar stickiness parameter."""
        n_states = 3
        # Note: parameter order is (concentration, stickiness, n_states)
        prior = get_transition_prior(
            concentration=1.0, stickiness=2.0, n_states=n_states
        )

        assert prior.shape == (n_states, n_states)
        # Diagonal should have stickiness added
        assert np.allclose(np.diag(prior), np.ones(n_states) + 2.0)
        # Off-diagonal should be concentration
        assert np.allclose(prior[0, 1], 1.0)

    def test_get_transition_prior_array_stickiness(self):
        """Test with per-state stickiness array."""
        n_states = 3
        stickiness = np.array([1.0, 2.0, 3.0])
        prior = get_transition_prior(
            concentration=0.5, stickiness=stickiness, n_states=n_states
        )

        assert prior.shape == (n_states, n_states)
        # Each diagonal element should have its own stickiness
        expected_diag = np.array([1.5, 2.5, 3.5])  # concentration + stickiness
        assert np.allclose(np.diag(prior), expected_diag)


# Tests for dirichlet_neg_log_likelihood
class TestDirichletNegLogLikelihood:
    def test_dirichlet_neg_log_likelihood_basic(self):
        """Test Dirichlet negative log likelihood computation."""
        # Arrange
        n_samples, n_coefficients, n_states = 10, 3, 3
        rng = np.random.default_rng(44)

        coefficients = jnp.array(rng.normal(size=n_coefficients * (n_states - 1)))
        design_matrix = jnp.array(rng.normal(size=(n_samples, n_coefficients)))
        response = jnp.array(rng.dirichlet(np.ones(n_states), size=n_samples))
        alpha = 1.0  # Uniform prior

        # Act
        nll = dirichlet_neg_log_likelihood(
            coefficients, design_matrix, response, alpha=alpha, l2_penalty=1e-5
        )

        # Assert
        assert isinstance(nll, (float, jnp.ndarray))
        assert nll > 0  # Should be positive

    def test_dirichlet_neg_log_likelihood_with_strong_prior(self):
        """Test that stronger priors affect the likelihood."""
        n_samples, n_coefficients, n_states = 10, 3, 2
        rng = np.random.default_rng(45)

        coefficients = jnp.array(rng.normal(size=n_coefficients * (n_states - 1)))
        design_matrix = jnp.array(rng.normal(size=(n_samples, n_coefficients)))
        response = jnp.array(rng.dirichlet(np.ones(n_states), size=n_samples))

        nll_weak = dirichlet_neg_log_likelihood(
            coefficients, design_matrix, response, alpha=1.0, l2_penalty=0.0
        )
        nll_strong = dirichlet_neg_log_likelihood(
            coefficients, design_matrix, response, alpha=10.0, l2_penalty=0.0
        )

        # Different priors should give different likelihoods
        assert not np.isclose(nll_weak, nll_strong)


# Tests for DiscreteStationaryDiagonal class
class TestDiscreteStationaryDiagonal:
    def test_make_state_transition_basic(self):
        """Test basic discrete state transition matrix creation."""
        # Arrange
        model = DiscreteStationaryDiagonal(diagonal_values=[0.9, 0.8])

        # Act
        trans_mat, coeffs, design = model.make_state_transition()

        # Assert
        assert trans_mat.shape == (2, 2)
        assert_stochastic_matrix(trans_mat)
        assert coeffs is None  # Stationary models return None
        assert design is None


# Tests for DiscreteStationaryCustom class
class TestDiscreteStationaryCustom:
    def test_make_state_transition_custom_matrix(self):
        """Test custom transition matrix."""
        # Arrange
        custom_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])
        model = DiscreteStationaryCustom(values=custom_matrix)

        # Act
        trans_mat, coeffs, design = model.make_state_transition()

        # Assert
        assert trans_mat.shape == (2, 2)
        assert np.allclose(trans_mat, custom_matrix)
        assert_stochastic_matrix(trans_mat)
        assert coeffs is None
        assert design is None


# ============================================================================
# SNAPSHOT TESTS
# ============================================================================


def serialize_transition_matrix_summary(trans_mat: np.ndarray) -> dict:
    """Serialize transition matrix to summary for snapshot comparison.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transition matrix

    Returns
    -------
    summary : dict
        Summary statistics suitable for snapshot comparison
    """
    return {
        "shape": trans_mat.shape,
        "dtype": str(trans_mat.dtype),
        "diagonal": np.diag(trans_mat).tolist() if trans_mat.ndim == 2 else None,
        "row_sums": np.sum(trans_mat, axis=-1).tolist(),
        "mean": float(np.mean(trans_mat)),
        "std": float(np.std(trans_mat)),
        "min": float(np.min(trans_mat)),
        "max": float(np.max(trans_mat)),
        "values": trans_mat.tolist() if trans_mat.size <= 100 else "too_large",
    }


@pytest.mark.snapshot
def test_make_transition_from_diag_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for making transition matrix from diagonal values."""
    diag = np.array([0.9, 0.85, 0.8, 0.95])
    trans_mat = make_transition_from_diag(diag)

    assert serialize_transition_matrix_summary(trans_mat) == snapshot


@pytest.mark.snapshot
def test_centered_softmax_forward_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for centered softmax forward transformation."""
    y = np.array([np.log(2), np.log(3), np.log(4), np.log(5)])
    result = centered_softmax_forward(y)

    assert {
        "shape": result.shape,
        "dtype": str(result.dtype),
        "sum": float(np.sum(result)),
        "values": result.tolist(),
    } == snapshot


@pytest.mark.snapshot
def test_get_transition_prior_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for Dirichlet prior generation."""
    n_states = 4
    concentration = 1.5
    stickiness = np.array([0.5, 1.0, 1.5, 2.0])

    prior = get_transition_prior(
        concentration=concentration, stickiness=stickiness, n_states=n_states
    )

    assert serialize_transition_matrix_summary(prior) == snapshot


@pytest.mark.snapshot
def test_discrete_stationary_diagonal_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for DiscreteStationaryDiagonal."""
    diag_values = np.array([0.92, 0.88, 0.90, 0.96])
    model = DiscreteStationaryDiagonal(diagonal_values=diag_values)

    trans_mat, coeffs, design = model.make_state_transition()

    assert serialize_transition_matrix_summary(trans_mat) == snapshot


@pytest.mark.snapshot
def test_discrete_stationary_custom_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for DiscreteStationaryCustom."""
    custom_matrix = np.array([[0.7, 0.2, 0.1], [0.15, 0.75, 0.1], [0.1, 0.2, 0.7]])
    model = DiscreteStationaryCustom(values=custom_matrix)

    trans_mat, coeffs, design = model.make_state_transition()

    assert serialize_transition_matrix_summary(trans_mat) == snapshot


@pytest.mark.snapshot
def test_discrete_nonstationary_diagonal_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for DiscreteNonStationaryDiagonal."""
    import pandas as pd

    diag_values = np.array([0.9, 0.85, 0.9, 0.95])
    model = DiscreteNonStationaryDiagonal(
        diagonal_values=diag_values,
        formula="1 + bs(speed, knots=[2.0, 10.0, 30.0])",
    )

    # Create covariate data
    speed = np.array([1.0, 5.0, 15.0, 25.0, 35.0, 10.0, 20.0])
    covariate_data = pd.DataFrame({"speed": speed})

    trans_mat, coeffs, design = model.make_state_transition(covariate_data)

    # Serialize transition matrix and coefficients
    summary = {
        "transition_matrix": serialize_transition_matrix_summary(trans_mat),
        "coefficients_shape": coeffs.shape,
        "design_matrix_shape": design.shape,
        "transition_matrix_time_0": trans_mat[0].tolist(),
        "transition_matrix_time_last": trans_mat[-1].tolist(),
    }

    assert summary == snapshot


@pytest.mark.snapshot
def test_estimate_joint_distribution_snapshot(snapshot: SnapshotAssertion):
    """Snapshot test for joint distribution estimation."""
    np.random.seed(123)
    n_time, n_states = 10, 3

    # Create probability distributions
    causal_posterior = np.random.dirichlet(np.ones(n_states), size=n_time)
    predictive = np.random.dirichlet(np.ones(n_states), size=n_time)
    acausal_posterior = np.random.dirichlet(np.ones(n_states), size=n_time)
    trans = make_transition_from_diag(np.array([0.8, 0.85, 0.9]))

    joint = estimate_joint_distribution(
        causal_posterior, predictive, trans, acausal_posterior
    )

    summary = {
        "shape": joint.shape,
        "dtype": str(joint.dtype),
        "sum_per_timestep": [float(joint[t].sum()) for t in range(joint.shape[0])],
        "mean": float(np.mean(joint)),
        "std": float(np.std(joint)),
        "first_timestep": joint[0].tolist(),
        "last_timestep": joint[-1].tolist(),
    }

    assert summary == snapshot
