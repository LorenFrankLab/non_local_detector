"""Tests for validation utilities."""

import numpy as np
import pytest

from non_local_detector import _validation as val
from non_local_detector.exceptions import DataError, ValidationError


class TestProbabilityDistribution:
    """Test ensure_probability_distribution."""

    def test_valid_distribution(self):
        """Test that valid distribution passes."""
        arr = np.array([0.25, 0.25, 0.25, 0.25])
        val.ensure_probability_distribution(arr, "probs")  # Should not raise

    def test_invalid_distribution_sum_too_high(self):
        """Test that distribution summing > 1 raises error."""
        arr = np.array([0.6, 0.6])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_probability_distribution(arr, "probs")
        assert "sum = 1.0" in str(exc_info.value)
        assert "sum = 1.200000" in str(exc_info.value)

    def test_invalid_distribution_sum_too_low(self):
        """Test that distribution summing < 1 raises error."""
        arr = np.array([0.2, 0.2])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_probability_distribution(arr, "probs")
        assert "sum = 0.400000" in str(exc_info.value)


class TestPositiveScalar:
    """Test ensure_positive_scalar."""

    def test_valid_positive_strict(self):
        """Test positive value with strict=True."""
        val.ensure_positive_scalar(1.0, "param", strict=True)  # Should not raise

    def test_zero_strict_raises(self):
        """Test that zero raises with strict=True."""
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_positive_scalar(0.0, "param", strict=True)
        assert "param > 0" in str(exc_info.value)

    def test_zero_non_strict_passes(self):
        """Test that zero passes with strict=False."""
        val.ensure_positive_scalar(0.0, "param", strict=False)  # Should not raise

    def test_negative_raises(self):
        """Test that negative value raises."""
        with pytest.raises(ValidationError):
            val.ensure_positive_scalar(-1.0, "param")


class TestArray1D:
    """Test ensure_array_1d."""

    def test_valid_1d_array(self):
        """Test that 1D array passes."""
        arr = np.array([1, 2, 3])
        val.ensure_array_1d(arr, "arr")  # Should not raise

    def test_2d_array_raises(self):
        """Test that 2D array raises."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_array_1d(arr, "arr")
        assert "1-dimensional" in str(exc_info.value)
        assert "(2, 2)" in str(exc_info.value)


class TestAllFinite:
    """Test ensure_all_finite."""

    def test_valid_finite_array(self):
        """Test that finite array passes."""
        arr = np.array([1.0, 2.0, 3.0])
        val.ensure_all_finite(arr, "arr")  # Should not raise

    def test_nan_raises(self):
        """Test that NaN raises DataError."""
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(DataError) as exc_info:
            val.ensure_all_finite(arr, "arr")
        assert "NaN" in str(exc_info.value)

    def test_inf_raises(self):
        """Test that Inf raises DataError."""
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(DataError) as exc_info:
            val.ensure_all_finite(arr, "arr")
        assert "Inf" in str(exc_info.value)


class TestAllNonNegative:
    """Test ensure_all_non_negative."""

    def test_valid_non_negative(self):
        """Test that non-negative array passes."""
        arr = np.array([0.0, 1.0, 2.0])
        val.ensure_all_non_negative(arr, "arr")  # Should not raise

    def test_negative_raises(self):
        """Test that negative values raise."""
        arr = np.array([1.0, -0.5, 2.0])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_all_non_negative(arr, "arr")
        assert "non-negative" in str(exc_info.value)
        assert "-0.5" in str(exc_info.value)


class TestInRange:
    """Test ensure_in_range."""

    def test_valid_range(self):
        """Test that values in range pass."""
        arr = np.array([0.2, 0.5, 0.8])
        val.ensure_in_range(arr, "arr", 0.0, 1.0)  # Should not raise

    def test_value_too_high_raises(self):
        """Test that value > high raises."""
        arr = np.array([0.5, 1.5])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_in_range(arr, "arr", 0.0, 1.0)
        assert "[0.0, 1.0]" in str(exc_info.value)

    def test_value_too_low_raises(self):
        """Test that value < low raises."""
        arr = np.array([-0.5, 0.5])
        with pytest.raises(ValidationError):
            val.ensure_in_range(arr, "arr", 0.0, 1.0)


class TestSquareMatrix:
    """Test ensure_square_matrix."""

    def test_valid_square_matrix(self):
        """Test that square matrix passes."""
        matrix = np.eye(3)
        val.ensure_square_matrix(matrix, "matrix")  # Should not raise

    def test_non_square_raises(self):
        """Test that non-square matrix raises."""
        matrix = np.zeros((3, 4))
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_square_matrix(matrix, "matrix")
        assert "square" in str(exc_info.value)

    def test_1d_array_raises(self):
        """Test that 1D array raises."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValidationError):
            val.ensure_square_matrix(arr, "arr")


class TestStochasticMatrix:
    """Test ensure_stochastic_matrix."""

    def test_valid_stochastic_matrix(self):
        """Test that row-stochastic matrix passes."""
        matrix = np.array([[1.0, 0.0], [0.3, 0.7]])
        val.ensure_stochastic_matrix(matrix, "matrix")  # Should not raise

    def test_invalid_row_sum_raises(self):
        """Test that invalid row sum raises."""
        matrix = np.array([[0.5, 0.5], [0.6, 0.3]])  # Second row sums to 0.9
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_stochastic_matrix(matrix, "matrix")
        assert "row-stochastic" in str(exc_info.value)
        # Floating point representation might be 0.8999... instead of 0.9
        assert "0.9" in str(exc_info.value) or "0.899" in str(exc_info.value)


class TestNdarray:
    """Test ensure_ndarray."""

    def test_valid_ndarray(self):
        """Test that ndarray passes."""
        arr = np.array([1, 2, 3])
        val.ensure_ndarray(arr, "arr")  # Should not raise

    def test_list_raises(self):
        """Test that list raises."""
        lst = [1, 2, 3]
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_ndarray(lst, "arr")
        assert "numpy array" in str(exc_info.value)
        assert "list" in str(exc_info.value)


class TestMonotonicIncreasing:
    """Test ensure_monotonic_increasing."""

    def test_valid_increasing(self):
        """Test that increasing array passes."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        val.ensure_monotonic_increasing(arr, "time")  # Should not raise

    def test_equal_values_non_strict(self):
        """Test that equal consecutive values pass with strict=False."""
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        val.ensure_monotonic_increasing(arr, "time", strict=False)  # Should not raise

    def test_equal_values_strict_raises(self):
        """Test that equal values raise with strict=True."""
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        with pytest.raises(DataError):
            val.ensure_monotonic_increasing(arr, "time", strict=True)

    def test_decreasing_raises(self):
        """Test that decreasing values raise."""
        arr = np.array([1.0, 3.0, 2.0, 4.0])
        with pytest.raises(DataError) as exc_info:
            val.ensure_monotonic_increasing(arr, "time")
        assert "index 1" in str(exc_info.value)


class TestMatchingLengths:
    """Test ensure_matching_lengths."""

    def test_matching_lengths(self):
        """Test that matching lengths pass."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        val.ensure_matching_lengths(arr1, arr2, "arr1", "arr2")  # Should not raise

    def test_mismatched_lengths_raise(self):
        """Test that mismatched lengths raise."""
        arr1 = np.array([1, 2])
        arr2 = np.array([4, 5, 6])
        with pytest.raises(ValidationError) as exc_info:
            val.ensure_matching_lengths(arr1, arr2, "arr1", "arr2")
        assert "Length mismatch" in str(exc_info.value)
        assert "2" in str(exc_info.value)
        assert "3" in str(exc_info.value)


class TestIntegration:
    """Test validation in actual model usage."""

    def test_invalid_initial_conditions_sum(self):
        """Test that model initialization catches invalid probability sum."""
        from non_local_detector import NonLocalClusterlessDetector

        with pytest.raises(ValidationError) as exc_info:
            NonLocalClusterlessDetector(
                discrete_initial_conditions=np.array([0.5, 0.5, 0.5, 0.5])  # Sums to 2!
            )
        assert "sum = 1.0" in str(exc_info.value)
        assert "sum = 2.0" in str(exc_info.value)

    def test_negative_concentration_raises(self):
        """Test that negative concentration parameter raises."""
        from non_local_detector import NonLocalClusterlessDetector

        with pytest.raises(ValidationError) as exc_info:
            NonLocalClusterlessDetector(discrete_transition_concentration=-1.0)
        assert "discrete_transition_concentration" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)

    def test_invalid_transition_matrix_raises(self):
        """Test that invalid custom transition matrix raises."""
        from non_local_detector import NonLocalClusterlessDetector
        from non_local_detector.discrete_state_transitions import (
            DiscreteStationaryCustom,
        )

        # Matrix with row that doesn't sum to 1
        bad_matrix = np.array(
            [
                [0.9, 0.05, 0.03, 0.02],
                [0.1, 0.8, 0.05, 0.04],  # Sums to 0.99
                [0.05, 0.05, 0.85, 0.05],
                [0.02, 0.02, 0.02, 0.94],
            ]
        )

        with pytest.raises(ValidationError) as exc_info:
            NonLocalClusterlessDetector(
                discrete_transition_type=DiscreteStationaryCustom(values=bad_matrix)
            )
        assert "row-stochastic" in str(exc_info.value)

    def test_2d_initial_conditions_raises(self):
        """Test that 2D initial conditions array raises."""
        from non_local_detector import NonLocalClusterlessDetector

        with pytest.raises(ValidationError) as exc_info:
            NonLocalClusterlessDetector(
                discrete_initial_conditions=np.array([[0.25, 0.25], [0.25, 0.25]])
            )
        # The validation catches this as a mismatch first (2D arrays are seen as having shape[0] rows)
        # before checking dimensionality, so we get the mismatch error
        assert "Mismatch" in str(exc_info.value) or "1-dimensional" in str(
            exc_info.value
        )
