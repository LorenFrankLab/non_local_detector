"""Unit tests for discrete state transitions module.

Tests cover the core public functions that can be tested in isolation.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from patsy import PatsyError, build_design_matrices
from scipy.special import softmax

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
    centered_softmax_forward,
    centered_softmax_inverse,
    dirichlet_neg_log_likelihood,
    get_transition_prior,
    jax_centered_log_softmax_forward,
    make_transition_from_diag,
    predict_discrete_state_transitions,
)
from non_local_detector.tests.conftest import assert_stochastic_matrix

pytestmark = pytest.mark.unit


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

    def test_centered_softmax_inverse_with_zero_entry(self):
        """Zero entries should produce finite output, not -inf or NaN."""
        # Arrange - row with a zero probability (e.g., impossible transition)
        y = np.asarray([0.5, 0.0, 0.3, 0.2])

        # Act
        result = centered_softmax_inverse(y)

        # Assert - all values should be finite (no -inf from log(0))
        assert np.all(np.isfinite(result)), f"Got non-finite values: {result}"

    def test_centered_softmax_inverse_with_zero_last_entry(self):
        """Zero in the last (centering) entry should produce finite output."""
        # Arrange - the last entry is used as the centering denominator
        y = np.asarray([0.5, 0.3, 0.2, 0.0])

        # Act
        result = centered_softmax_inverse(y)

        # Assert - should be finite (clipped to tiny before log)
        assert np.all(np.isfinite(result)), f"Got non-finite values: {result}"

    def test_centered_softmax_inverse_with_zero_preserves_dtype(self):
        """Guard should respect input dtype (float32 vs float64)."""
        for dtype in [np.float32, np.float64]:
            y = np.asarray([0.5, 0.0, 0.3, 0.2], dtype=dtype)
            result = centered_softmax_inverse(y)
            assert np.all(np.isfinite(result))
            assert result.dtype == dtype

    def test_centered_softmax_inverse_accepts_integer_input(self):
        """Integer-typed input should not raise from np.finfo."""
        # Arrange - an integer identity-like matrix (e.g. from a list of ints)
        y = np.array([[1, 0], [0, 1]])
        assert not np.issubdtype(y.dtype, np.floating)

        # Act - previously raised "ValueError: data type <int> not inexact"
        result = centered_softmax_inverse(y)

        # Assert - finite and matches the float computation
        assert np.all(np.isfinite(result))
        expected = centered_softmax_inverse(y.astype(float))
        assert np.allclose(result, expected)

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

    @pytest.mark.parametrize("diag_value", [1.0, 0.9, 0.0])
    def test_make_transition_from_diag_single_state(self, diag_value):
        """A single state has only the [[1.0]] stochastic matrix.

        Regression guard: the old special case set off_diag = 1.0, yielding
        [[diag]] whose row sums to `diag` rather than 1 for diag != 1.
        """
        trans = make_transition_from_diag(np.array([diag_value]))

        assert trans.shape == (1, 1)
        assert_stochastic_matrix(trans)
        assert np.allclose(trans, np.ones((1, 1)))


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

    @pytest.mark.parametrize(
        "stickiness",
        [np.float32(2.0), np.float64(2.0), np.int64(2), np.array(2.0)],
        ids=["float32", "float64", "int64", "zero_d_array"],
    )
    def test_get_transition_prior_numpy_scalar_stickiness(self, stickiness):
        """NumPy scalars and 0-d arrays are treated as scalar stickiness.

        Regression guard: `isinstance(stickiness, int | float)` missed
        np.float32/np.int64/0-d arrays, which then hit `np.diag(scalar)` and
        raised.
        """
        n_states = 3
        prior = get_transition_prior(
            concentration=1.0, stickiness=stickiness, n_states=n_states
        )

        assert prior.shape == (n_states, n_states)
        assert np.allclose(np.diag(prior), np.ones(n_states) + 2.0)
        assert np.allclose(prior[0, 1], 1.0)


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
        assert isinstance(nll, float | jnp.ndarray)
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


class TestPredictDiscreteStateTransitions:
    """Test the predict-time covariate transition path."""

    @pytest.fixture
    def fitted_nonstationary_model(self):
        """Design matrix of a ``1 + speed`` model and coefficients with
        non-zero speed slopes, so predictions depend on the covariate.

        ``make_state_transition`` only fits the intercept (its slopes are zero),
        so the slopes are set here as an M-step would.
        """
        model = DiscreteNonStationaryDiagonal(
            diagonal_values=np.array([0.8, 0.85, 0.9]),
            formula="1 + speed",
        )
        _, coefficients, design_matrix = model.make_state_transition(
            {"speed": np.linspace(0, 10, 20)}
        )
        coefficients = np.array(coefficients, dtype=np.float64)
        coefficients[1] = np.random.default_rng(3).normal(
            0.0, 0.3, coefficients.shape[1:]
        )
        return coefficients, design_matrix

    def test_predict_matches_softmax_of_design_times_coefficients(
        self, fitted_nonstationary_model
    ):
        """Each predicted row is the softmax of ``[1, speed] @ coefficients``
        with a zero logit for the reference (last) state, as a NumPy array."""
        coefficients, design_matrix = fitted_nonstationary_model
        speed = np.array([0.0, 2.5, 5.0, 10.0, 15.0])

        result = predict_discrete_state_transitions(
            design_matrix, coefficients, {"speed": speed}
        )

        new_design = np.column_stack((np.ones_like(speed), speed))
        logits = np.einsum("tc,csk->tsk", new_design, coefficients)
        logits = np.concatenate((logits, np.zeros(logits.shape[:-1] + (1,))), axis=-1)
        expected = softmax(logits, axis=-1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (speed.size, 3, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)

    def test_predictions_vary_with_the_covariate(self, fitted_nonstationary_model):
        """Different speeds give different transition matrices."""
        coefficients, design_matrix = fitted_nonstationary_model

        result = predict_discrete_state_transitions(
            design_matrix, coefficients, {"speed": np.array([0.0, 10.0])}
        )

        assert np.max(np.abs(result[1] - result[0])) > 0.05


class TestNonStationaryNaNCovariates:
    """Non-stationary transition construction must reject NaN covariates."""

    @pytest.mark.parametrize(
        "model",
        [
            DiscreteNonStationaryDiagonal(
                diagonal_values=np.array([0.9, 0.8]), formula="1 + speed"
            ),
            DiscreteNonStationaryCustom(
                values=np.array([[0.9, 0.1], [0.2, 0.8]]), formula="1 + speed"
            ),
        ],
        ids=["diagonal", "custom"],
    )
    def test_nan_covariate_raises_instead_of_silently_dropping(self, model):
        """NaN covariate rows must raise, not silently shorten the design matrix.

        patsy's default NA_action="drop" would drop the NaN row, leaving a
        design matrix shorter than the posteriors and misaligning the M-step.
        """
        covariate_data = {"speed": np.array([1.0, np.nan, 3.0, 4.0])}

        with pytest.raises(PatsyError):
            model.make_state_transition(covariate_data)


class TestDefaultCovariateSpline:
    """The default covariate formula places spline knots where the data are
    and accepts covariates outside the fitted range."""

    @staticmethod
    def _design(speed):
        _, _, design_matrix = DiscreteNonStationaryDiagonal(
            diagonal_values=np.array([0.99, 0.98, 0.98, 0.98])
        ).make_state_transition({"speed": speed})
        return design_matrix

    def test_speed_below_fixed_knots(self):
        """Speeds that never reach the largest fixed knot (64 cm/s) fit with
        full rank."""
        design_matrix = np.asarray(self._design(np.linspace(0.0, 50.0, 1000)))
        assert np.linalg.matrix_rank(design_matrix) == design_matrix.shape[1]

    def test_zero_heavy_speed_keeps_every_column_supported(self):
        """Ties at zero (an animal at rest) do not stack knots: every basis
        column has data and the design stays well conditioned."""
        rng = np.random.default_rng(0)
        speed = rng.gamma(1.5, 12.0, 20_000)
        speed[rng.uniform(size=speed.size) < 0.7] = 0.0
        design_matrix = np.asarray(self._design(speed))
        singular_values = np.linalg.svd(design_matrix, compute_uv=False)

        assert design_matrix.shape[1] == 9
        assert np.all(design_matrix.sum(axis=0) > 10.0)
        assert singular_values[0] / singular_values[-1] < 1e4

    def test_prediction_outside_fitted_range_uses_boundary(self):
        """New covariates beyond the fitted range evaluate at the nearest
        boundary instead of raising."""
        design_matrix = self._design(np.linspace(0.0, 50.0, 1000))
        new_rows = np.asarray(
            build_design_matrices(
                [design_matrix.design_info],
                {"speed": np.array([-5.0, 0.0, 50.0, 80.0])},
            )[0]
        )
        np.testing.assert_allclose(new_rows[0], new_rows[1])
        np.testing.assert_allclose(new_rows[3], new_rows[2])

    def test_explicit_knots_still_supported(self):
        _, _, design_matrix = DiscreteNonStationaryDiagonal(
            diagonal_values=np.array([0.99, 0.98]),
            formula="1 + bs(speed, knots=[10.0, 20.0])",
        ).make_state_transition({"speed": np.linspace(0.0, 50.0, 100)})
        assert design_matrix.shape == (100, 6)
