"""Unit tests for core utility functions in core.py.

These tests focus on the fundamental building blocks that all other code depends on.
Testing philosophy: Each function should be tested for:
1. Happy path (expected inputs)
2. Edge cases (zeros, very small/large values)
3. Invalid inputs (if applicable)
4. Mathematical properties (probability constraints, etc.)
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    _condition_on,
    _divide_safe,
    _normalize,
    _safe_log,
    check_converged,
)


@pytest.mark.unit
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

    def test_normalize_with_very_large_values(self):
        """Should handle very large values without overflow."""
        # Arrange - Use values that don't overflow in float32
        arr = jnp.array([1e30, 2e30, 3e30])

        # Act
        normalized, const = _normalize(arr)

        # Assert: the 6e30 total is representable in float32, so the result
        # must be finite with the proportions preserved.
        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.allclose(const, 6e30, rtol=1e-6)
        assert jnp.allclose(normalized.sum(), 1.0)
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

    @pytest.mark.parametrize(("axis", "const_shape"), [(0, (3,)), (1, (2,))])
    def test_normalize_2d_array(self, axis, const_shape):
        """Normalize 2D array along axis 0 (columns) or 1 (rows) to sum to 1."""
        # Arrange
        arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        normalized, const = _normalize(arr, axis=axis)

        # Assert
        assert jnp.allclose(normalized.sum(axis=axis), 1.0)
        # const shape: the normalized axis is squeezed from (2, 3)
        assert const.shape == const_shape

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


@pytest.mark.unit
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
            # Must not be NaN; may overflow to +inf in float32
            assert not jnp.isnan(result[0])
            assert result[0] > 0

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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestCheckConverged:
    """``check_converged`` returns (is_converged, is_increasing).

    Monotonicity allows float rounding relative to the objective's magnitude.
    Convergence is the objective's gain per time bin: it does not depend on
    the magnitude, which the spike or mark likelihood sets (and a change of
    mark or rate units shifts), nor on recording length.
    """

    def test_tiny_relative_decrease_on_large_ll_is_not_flagged(self):
        """A decrease that is large in absolute terms but negligible relative
        to a large log-likelihood must NOT be flagged as a monotonicity
        violation."""
        # |LL| ~ 50000; an absolute drop of 0.01 is a ~2e-7 relative change,
        # float noise, and 1e-6 nats per bin over 10000 bins.
        is_converged, is_increasing = check_converged(
            jnp.asarray(-50000.01),
            jnp.asarray(-50000.0),
            tolerance=1e-4,
            n_time_bins=10_000,
        )
        assert is_increasing, (
            "A negligible relative decrease on a large log-likelihood was "
            "incorrectly flagged as a monotonicity violation"
        )
        assert is_converged

    def test_large_relative_decrease_is_flagged(self):
        """A decrease that exceeds the tolerance fraction of the magnitude is
        flagged as non-increasing."""
        # avg magnitude ~50; a drop of 10 is a 0.2 relative change >> 1e-4.
        _, is_increasing = check_converged(
            jnp.asarray(-60.0), jnp.asarray(-50.0), tolerance=1e-4
        )
        assert not is_increasing

    def test_increase_is_never_a_violation(self):
        """An increasing log-likelihood is always monotonic."""
        _, is_increasing = check_converged(
            jnp.asarray(-49000.0), jnp.asarray(-50000.0), tolerance=1e-4
        )
        assert is_increasing

    @pytest.mark.parametrize(
        ("gain_per_bin", "expected"), [(0.9e-5, True), (1.1e-5, False)]
    )
    def test_converges_on_gain_per_time_bin(self, gain_per_bin, expected):
        """EM has converged when the objective rose by less than
        ``tolerance`` nats per time bin."""
        n_time_bins = 300_000
        previous = -1.5e6
        is_converged, is_increasing = check_converged(
            previous + gain_per_bin * n_time_bins,
            previous,
            tolerance=1e-5,
            n_time_bins=n_time_bins,
        )
        assert is_increasing
        assert is_converged is expected

    @pytest.mark.parametrize("offset", [0.0, -5e7])
    def test_convergence_ignores_the_objective_magnitude(self, offset):
        """Shifting the objective by a constant (as mark or rate units do)
        does not change whether the same gain counts as converged."""
        # A 3-nat gain over 300000 bins is 1e-5 nats per bin; relative to
        # |objective| it would be 2e-6 or 6e-8.
        decisions = [
            check_converged(
                -1.5e6 + offset + 3.0,
                -1.5e6 + offset,
                tolerance=tolerance,
                n_time_bins=300_000,
            )[0]
            for tolerance in (0.9e-5, 1.1e-5)
        ]
        assert decisions == [False, True]

    def test_convergence_does_not_depend_on_recording_length(self):
        """The same gain per bin on a recording twice as long gives the same
        decision."""
        short = check_converged(-1e6 + 2.0, -1e6, tolerance=1e-5, n_time_bins=250_000)
        long = check_converged(-2e6 + 4.0, -2e6, tolerance=1e-5, n_time_bins=500_000)
        assert short == long == (True, True)

    def test_slack_does_not_scale_with_tolerance(self):
        """The monotonicity slack is a fixed float-noise allowance, so a loose
        convergence tolerance does not hide a real decrease."""
        # Relative change = -0.02 / ~100 = -2e-4.
        curr, prev = jnp.asarray(-100.02), jnp.asarray(-100.0)
        _, tight = check_converged(curr, prev, tolerance=1e-5)
        _, loose = check_converged(curr, prev, tolerance=1e-3)
        assert not tight
        assert not loose

    def test_decrease_beyond_float_noise_is_not_convergence(self):
        """A decrease smaller than ``tolerance`` but larger than float noise is
        a monotonicity violation and must not end EM as converged."""
        # Relative change -1e-5: inside tolerance=1e-4, far beyond float32 noise.
        is_converged, is_increasing = check_converged(
            -1_000_010.0, -1_000_000.0, tolerance=1e-4
        )
        assert not is_increasing
        assert not is_converged

    def test_float_noise_decrease_counts_as_converged(self):
        """A decrease at float32 rounding level is noise, not a violation."""
        # Relative change -5e-7, below the ~1.2e-6 float-noise allowance;
        # 2.5e-6 nats per bin over 200000 bins.
        is_converged, is_increasing = check_converged(
            -1_000_000.5, -1_000_000.0, tolerance=1e-4, n_time_bins=200_000
        )
        assert is_increasing
        assert is_converged

    def test_leaving_negative_infinity_is_an_increase(self):
        """A finite objective after a ``-inf`` one (e.g. an initial transition
        matrix with a zero entry the prior penalizes) is an increase, not a
        NaN comparison."""
        is_converged, is_increasing = check_converged(-50.0, -np.inf, tolerance=1e-4)
        assert is_increasing
        assert not is_converged

    def test_falling_to_negative_infinity_is_a_violation(self):
        """A ``-inf`` objective after a finite one (e.g. a refit that makes a
        timestep impossible) is a decrease, decided without a NaN ratio."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            is_converged, is_increasing = check_converged(
                -np.inf, -50.0, tolerance=1e-4
            )
        assert not is_increasing
        assert not is_converged

    def test_staying_at_negative_infinity_is_not_a_violation(self):
        """A degenerate timestep makes every E-step's marginal log-likelihood
        ``-inf``. That is no decrease, but there is no change to judge
        convergence by."""
        is_converged, is_increasing = check_converged(-np.inf, -np.inf, tolerance=1e-4)
        assert is_increasing
        assert not is_converged
