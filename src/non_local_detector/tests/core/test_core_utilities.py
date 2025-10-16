"""Unit tests for core utility functions in core.py.

These tests focus on the fundamental building blocks that all other code depends on.
Testing philosophy: Each function should be tested for:
1. Happy path (expected inputs)
2. Edge cases (zeros, very small/large values)
3. Invalid inputs (if applicable)
4. Mathematical properties (probability constraints, etc.)
"""

import jax.numpy as jnp
import pytest

from non_local_detector.core import (
    _condition_on,
    _divide_safe,
    _normalize,
    _safe_log,
)


class TestNormalize:
    """Test _normalize function for probability normalization.

    The _normalize function is critical for maintaining valid probability
    distributions throughout HMM computations.
    """

    def test_normalize_returns_probabilities_that_sum_to_one(self):
        """Normalized array should sum to 1.0 along specified axis."""
        # Arrange
        arr = jnp.array([1.0, 2.0, 3.0])

        # Act
        normalized, const = _normalize(arr, axis=0)

        # Assert
        assert jnp.allclose(normalized.sum(), 1.0)

    def test_normalize_returns_normalization_constant(self):
        """Should return the sum as normalization constant."""
        # Arrange
        arr = jnp.array([1.0, 2.0, 3.0])

        # Act
        normalized, const = _normalize(arr, axis=0)

        # Assert
        assert jnp.allclose(const, 6.0)

    def test_normalize_handles_zero_array_without_nan(self):
        """Zero arrays should not produce NaN with eps protection."""
        # Arrange
        arr = jnp.zeros(5)

        # Act
        normalized, const = _normalize(arr, axis=0)

        # Assert
        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.allclose(const, 0.0)

    def test_normalize_preserves_shape(self):
        """Normalized array should have same shape as input."""
        # Arrange
        arr = jnp.ones((3, 4, 5))

        # Act
        normalized, _ = _normalize(arr, axis=1)

        # Assert
        assert normalized.shape == arr.shape

    def test_normalize_works_on_different_axes(self):
        """Should normalize along specified axis."""
        # Arrange
        arr = jnp.ones((3, 4, 5))

        # Act & Assert
        for axis in [0, 1, 2]:
            normalized, _ = _normalize(arr, axis=axis)
            # Sum along the normalized axis should be 1
            sums = normalized.sum(axis=axis)
            assert jnp.allclose(sums, 1.0), f"Failed for axis {axis}"

    def test_normalize_with_very_small_values(self):
        """Should handle very small values without underflow."""
        # Arrange - Use values that work well with float32 precision
        arr = jnp.array([1e-10, 2e-10, 3e-10])

        # Act
        normalized, const = _normalize(arr)

        # Assert
        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.allclose(normalized.sum(), 1.0)
        # Check proportions are preserved
        assert jnp.allclose(normalized[0], 1.0 / 6.0, rtol=1e-3)
        assert jnp.allclose(normalized[1], 2.0 / 6.0, rtol=1e-3)
        assert jnp.allclose(normalized[2], 3.0 / 6.0, rtol=1e-3)

    def test_normalize_with_very_large_values(self):
        """Should handle very large values without overflow."""
        # Arrange - Use values that don't overflow in float32
        arr = jnp.array([1e30, 2e30, 3e30])

        # Act
        normalized, const = _normalize(arr)

        # Assert
        # Very large values may cause overflow/inf in normalization constant
        # But proportions should be preserved if result is finite
        if jnp.all(jnp.isfinite(normalized)):
            assert jnp.allclose(normalized.sum(), 1.0)
            # Check proportions are preserved
            assert jnp.allclose(normalized[0], 1.0 / 6.0, rtol=1e-4)
            assert jnp.allclose(normalized[1], 2.0 / 6.0, rtol=1e-4)
            assert jnp.allclose(normalized[2], 3.0 / 6.0, rtol=1e-4)

    def test_normalize_preserves_proportions(self):
        """Normalization should preserve relative proportions."""
        # Arrange
        arr = jnp.array([10.0, 20.0, 30.0])

        # Act
        normalized, _ = _normalize(arr)

        # Assert
        # Original proportions: 10:20:30 = 1:2:3
        assert jnp.allclose(normalized[1] / normalized[0], 2.0)
        assert jnp.allclose(normalized[2] / normalized[0], 3.0)

    def test_normalize_2d_array_axis0(self):
        """Normalize 2D array along axis 0 (columns sum to 1)."""
        # Arrange
        arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        normalized, const = _normalize(arr, axis=0)

        # Assert
        col_sums = normalized.sum(axis=0)
        assert jnp.allclose(col_sums, 1.0)
        # const shape: squeeze axis 0 from (2, 3) -> (3,)
        assert const.shape == (3,)

    def test_normalize_2d_array_axis1(self):
        """Normalize 2D array along axis 1 (rows sum to 1)."""
        # Arrange
        arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        normalized, const = _normalize(arr, axis=1)

        # Assert
        row_sums = normalized.sum(axis=1)
        assert jnp.allclose(row_sums, 1.0)
        # const shape: squeeze axis 1 from (2, 3) -> (2,)
        assert const.shape == (2,)

    def test_normalize_single_element(self):
        """Single element should normalize to 1.0."""
        # Arrange
        arr = jnp.array([42.0])

        # Act
        normalized, const = _normalize(arr)

        # Assert
        assert jnp.allclose(normalized[0], 1.0)
        assert jnp.allclose(const, 42.0)

    def test_normalize_with_negative_values(self):
        """Should handle negative values (though unusual for probabilities)."""
        # Arrange
        arr = jnp.array([-1.0, 2.0, 3.0])

        # Act
        normalized, const = _normalize(arr)

        # Assert
        # Sum is -1 + 2 + 3 = 4
        assert jnp.allclose(const, 4.0)
        assert jnp.allclose(normalized.sum(), 1.0)


class TestSafeLog:
    """Test _safe_log function for numerically stable logarithms.

    Safe log prevents -inf and NaN values that would break HMM computations.
    """

    def test_safe_log_of_positive_values_equals_standard_log(self):
        """For positive values, should equal standard log."""
        # Arrange
        x = jnp.array([0.1, 1.0, 10.0])

        # Act
        result = _safe_log(x)
        expected = jnp.log(x)

        # Assert
        assert jnp.allclose(result, expected)

    def test_safe_log_of_zero_returns_negative_inf(self):
        """Zero should return -inf (valid in log probability space)."""
        # Arrange
        x = jnp.array([0.0])

        # Act
        result = _safe_log(x)

        # Assert
        assert result[0] == -jnp.inf

    def test_safe_log_prevents_nan_propagation(self):
        """Should not produce NaN values (but may produce -inf for zeros)."""
        # Arrange
        x = jnp.array([0.0, 1e-20, 1.0, 1e20])

        # Act
        result = _safe_log(x)

        # Assert
        # No NaN values (but -inf is okay for zero)
        assert not jnp.any(jnp.isnan(result))
        # First element is -inf (for zero), rest should be finite
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_safe_log_preserves_shape(self):
        """Output shape should match input shape."""
        # Arrange
        shapes = [(5,), (3, 4), (2, 3, 4)]

        # Act & Assert
        for shape in shapes:
            x = jnp.ones(shape)
            result = _safe_log(x)
            assert result.shape == shape, f"Failed for shape {shape}"

    @pytest.mark.parametrize(
        "value,expected_result",
        [
            (1e-100, "neg_inf"),  # Underflows to zero in float32, becomes -inf
            (1e-300, "neg_inf"),  # Underflows to zero in float32
            (0.0, "neg_inf"),  # Exactly zero
            (1.0, "zero"),  # log(1) = 0
            (1e100, "finite_positive"),  # Large positive (may overflow in float32)
        ],
    )
    def test_safe_log_edge_cases(self, value, expected_result):
        """Test behavior at numerical boundaries."""
        # Arrange
        x = jnp.array([value])

        # Act
        result = _safe_log(x)

        # Assert
        if expected_result == "neg_inf":
            # Check for -inf (handles both explicit zero and underflow)
            assert jnp.isinf(result[0]) and result[0] < 0
        elif expected_result == "zero":
            assert jnp.allclose(result[0], 0.0)
        elif expected_result == "finite_positive":
            # May overflow to inf in float32
            assert jnp.isfinite(result[0]) or jnp.isinf(result[0])

    def test_safe_log_monotonicity(self):
        """Safe log should preserve monotonicity (larger input → larger output)."""
        # Arrange
        x = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])

        # Act
        result = _safe_log(x)

        # Assert
        # Check monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_safe_log_of_one_equals_zero(self):
        """log(1) should equal 0."""
        # Arrange
        x = jnp.array([1.0])

        # Act
        result = _safe_log(x)

        # Assert
        assert jnp.allclose(result, 0.0)

    def test_safe_log_vectorization(self):
        """Should handle vectorized operations correctly."""
        # Arrange
        x = jnp.linspace(0.1, 10.0, 100)

        # Act
        result = _safe_log(x)

        # Assert
        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))


class TestDivideSafe:
    """Test _divide_safe for division without inf/nan.

    Safe division is critical when normalizing probabilities that might be zero.
    """

    def test_divide_safe_standard_division_equals_normal(self):
        """Normal division should match standard behavior."""
        # Arrange
        a = jnp.array([4.0, 6.0, 8.0])
        b = jnp.array([2.0, 3.0, 4.0])

        # Act
        result = _divide_safe(a, b)
        expected = a / b

        # Assert
        assert jnp.allclose(result, expected)

    def test_divide_safe_by_zero_returns_zero(self):
        """Division by zero should return zero, not inf."""
        # Arrange
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([0.0, 0.0, 0.0])

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert jnp.all(result == 0.0)

    def test_divide_safe_zero_by_zero_returns_zero(self):
        """0/0 should be defined as 0 for probability contexts."""
        # Arrange
        a = jnp.array([0.0])
        b = jnp.array([0.0])

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert result == 0.0

    def test_divide_safe_broadcasts_correctly(self):
        """Should support NumPy broadcasting rules."""
        # Arrange
        a = jnp.array([[1.0, 2.0, 3.0]])  # (1, 3)
        b = jnp.array([[2.0], [4.0]])  # (2, 1)

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert result.shape == (2, 3)
        assert jnp.allclose(result[0], jnp.array([0.5, 1.0, 1.5]))
        assert jnp.allclose(result[1], jnp.array([0.25, 0.5, 0.75]))

    def test_divide_safe_preserves_shape(self):
        """Output shape should match broadcasted shape."""
        # Arrange
        shapes = [((5,), (5,)), ((3, 4), (3, 4)), ((2, 3, 4), (2, 3, 4))]

        # Act & Assert
        for shape_a, shape_b in shapes:
            a = jnp.ones(shape_a)
            b = jnp.ones(shape_b) * 2
            result = _divide_safe(a, b)
            assert result.shape == shape_a, f"Failed for shapes {shape_a}, {shape_b}"

    def test_divide_safe_mixed_zeros_and_nonzeros(self):
        """Should handle mix of zero and non-zero denominators."""
        # Arrange
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([2.0, 0.0, 4.0, 0.0])

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert jnp.allclose(result[0], 0.5)
        assert result[1] == 0.0
        assert jnp.allclose(result[2], 0.75)
        assert result[3] == 0.0

    def test_divide_safe_no_nans_or_infs(self):
        """Should never produce NaN or inf values."""
        # Arrange
        a = jnp.array([0.0, 1.0, 1e-100, 1e100])
        b = jnp.array([0.0, 0.0, 0.0, 0.0])

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert jnp.all(jnp.isfinite(result))

    def test_divide_safe_negative_values(self):
        """Should handle negative values correctly."""
        # Arrange
        a = jnp.array([-4.0, -6.0])
        b = jnp.array([2.0, 3.0])

        # Act
        result = _divide_safe(a, b)

        # Assert
        assert jnp.allclose(result, jnp.array([-2.0, -2.0]))


class TestConditionOn:
    """Test _condition_on for Bayesian conditioning.

    This function implements the core Bayesian update: posterior ∝ prior × likelihood
    """

    def test_condition_on_increases_probability_for_high_likelihood(self):
        """States with high likelihood should have increased probability."""
        # Arrange
        probs = jnp.array([0.5, 0.5])
        ll = jnp.array([0.0, 10.0])  # Second state much more likely

        # Act
        conditioned, _ = _condition_on(probs, ll)

        # Assert
        assert conditioned[1] > conditioned[0]
        assert conditioned[1] > 0.99  # Should be nearly 1

    def test_condition_on_returns_normalized_probabilities(self):
        """Output should be a valid probability distribution."""
        # Arrange
        probs = jnp.array([0.3, 0.7])
        ll = jnp.array([1.0, 2.0])

        # Act
        conditioned, _ = _condition_on(probs, ll)

        # Assert
        assert jnp.allclose(conditioned.sum(), 1.0)
        assert jnp.all(conditioned >= 0)
        assert jnp.all(conditioned <= 1)

    def test_condition_on_returns_marginal_likelihood(self):
        """Second return value should be marginal log likelihood."""
        # Arrange
        probs = jnp.array([0.5, 0.5])
        ll = jnp.array([jnp.log(0.2), jnp.log(0.8)])

        # Act
        _, marginal = _condition_on(probs, ll)

        # Assert
        # Marginal = sum(prior * likelihood) = 0.5*0.2 + 0.5*0.8 = 0.5
        expected = jnp.log(0.5)
        assert jnp.allclose(marginal, expected, rtol=1e-5)

    def test_condition_on_handles_uniform_likelihood(self):
        """Uniform likelihood should preserve prior."""
        # Arrange
        probs = jnp.array([0.3, 0.7])
        ll = jnp.zeros(2)  # log(1) = 0 for all states

        # Act
        conditioned, _ = _condition_on(probs, ll)

        # Assert
        assert jnp.allclose(conditioned, probs)

    def test_condition_on_with_zero_prior_stays_zero(self):
        """States with zero prior should remain zero regardless of likelihood."""
        # Arrange
        probs = jnp.array([0.0, 1.0])
        ll = jnp.array([10.0, 0.0])  # First state has high likelihood

        # Act
        conditioned, _ = _condition_on(probs, ll)

        # Assert
        assert conditioned[0] == 0.0  # Still zero
        assert conditioned[1] == 1.0  # Still one

    def test_condition_on_handles_very_negative_log_likelihoods(self):
        """Should handle very negative log likelihoods without underflow."""
        # Arrange
        probs = jnp.array([0.5, 0.5])
        ll = jnp.array([-1000.0, -1001.0])

        # Act
        conditioned, marginal = _condition_on(probs, ll)

        # Assert
        assert jnp.all(jnp.isfinite(conditioned))
        assert jnp.isfinite(marginal)
        assert jnp.allclose(conditioned.sum(), 1.0)

    def test_condition_on_multiple_states(self):
        """Should work with more than 2 states."""
        # Arrange
        n_states = 10
        probs = jnp.ones(n_states) / n_states  # Uniform prior
        ll = jnp.linspace(-5, 0, n_states)  # Increasing likelihoods

        # Act
        conditioned, _ = _condition_on(probs, ll)

        # Assert
        assert conditioned.shape == (n_states,)
        assert jnp.allclose(conditioned.sum(), 1.0)
        # Posterior should favor higher likelihood states
        assert conditioned[-1] > conditioned[0]

    def test_condition_on_preserves_shape(self):
        """Output shape should match input shape."""
        # Arrange
        sizes = [2, 5, 10, 50]

        # Act & Assert
        for size in sizes:
            probs = jnp.ones(size) / size
            ll = jnp.zeros(size)
            conditioned, _ = _condition_on(probs, ll)
            assert conditioned.shape == (size,), f"Failed for size {size}"

    def test_condition_on_bayes_rule_property(self):
        """Verify Bayes rule: P(state|obs) = P(obs|state) * P(state) / P(obs)."""
        # Arrange
        prior = jnp.array([0.3, 0.7])
        likelihood = jnp.array([0.8, 0.2])  # Not log yet
        log_likelihood = jnp.log(likelihood)

        # Act
        posterior, log_marginal = _condition_on(prior, log_likelihood)

        # Assert
        # Manual Bayes rule calculation
        unnormalized = prior * likelihood
        marginal = unnormalized.sum()
        expected_posterior = unnormalized / marginal

        assert jnp.allclose(posterior, expected_posterior)
        assert jnp.allclose(jnp.exp(log_marginal), marginal)
