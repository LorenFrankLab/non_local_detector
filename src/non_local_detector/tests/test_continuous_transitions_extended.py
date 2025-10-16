"""Extended unit tests for continuous state transitions module.

These tests complement existing tests in transitions/test_continuous_transitions.py
by adding coverage for helper functions and edge cases.
"""

import numpy as np
import pytest

from non_local_detector.continuous_state_transitions import (
    _normalize_row_probability,
    estimate_movement_var,
)


class TestNormalizeRowProbability:
    """Test _normalize_row_probability helper function."""

    def test_normalize_row_probability_standard_case(self):
        """Standard case should normalize rows to sum to 1."""
        # Arrange
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Act
        result = _normalize_row_probability(x)

        # Assert
        assert np.allclose(result.sum(axis=1), 1.0)
        # Check proportions preserved
        assert np.allclose(result[0, 1] / result[0, 0], 2.0)

    def test_normalize_row_probability_with_zero_rows(self):
        """Zero rows should remain zero without NaN."""
        # Arrange
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])

        # Act
        result = _normalize_row_probability(x)

        # Assert
        assert np.all(np.isfinite(result))
        # Non-zero rows should sum to 1
        assert np.allclose(result[0].sum(), 1.0)
        assert np.allclose(result[2].sum(), 1.0)
        # Zero row should remain zero
        assert np.allclose(result[1], 0.0)

    def test_normalize_row_probability_single_row(self):
        """Should work with single row."""
        # Arrange
        x = np.array([[2.0, 3.0, 5.0]])

        # Act
        result = _normalize_row_probability(x)

        # Assert
        assert np.allclose(result.sum(), 1.0)

    def test_normalize_row_probability_all_zeros(self):
        """All-zero matrix should remain all zeros."""
        # Arrange
        x = np.zeros((3, 4))

        # Act
        result = _normalize_row_probability(x)

        # Assert
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))

    def test_normalize_row_probability_very_small_values(self):
        """Should handle very small values without underflow."""
        # Arrange
        x = np.array([[1e-100, 2e-100], [3e-10, 4e-10]])

        # Act
        result = _normalize_row_probability(x)

        # Assert
        assert np.all(np.isfinite(result))
        for row in result:
            if row.sum() > 0:
                assert np.allclose(row.sum(), 1.0)

    def test_normalize_row_probability_preserves_proportions(self):
        """Normalization should preserve relative proportions within rows."""
        # Arrange
        x = np.array([[10.0, 20.0, 30.0]])

        # Act
        result = _normalize_row_probability(x)

        # Assert
        # 10:20:30 = 1:2:3 should be preserved
        assert np.allclose(result[0, 1] / result[0, 0], 2.0)
        assert np.allclose(result[0, 2] / result[0, 0], 3.0)


class TestEstimateMovementVar:
    """Test estimate_movement_var function."""

    def test_estimate_movement_var_1d_position(self):
        """Should estimate variance for 1D position."""
        # Arrange - linear motion with some noise
        np.random.seed(42)
        position = np.linspace(0, 10, 100) + np.random.randn(100) * 0.1

        # Act
        var = estimate_movement_var(position)

        # Assert
        assert isinstance(var, (float, np.ndarray))
        assert var > 0

    def test_estimate_movement_var_2d_position(self):
        """Should estimate covariance for 2D position."""
        # Arrange
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        position = np.column_stack([
            t + np.random.randn(100) * 0.1,
            t + np.random.randn(100) * 0.1
        ])

        # Act
        var = estimate_movement_var(position)

        # Assert
        assert var.shape == (2, 2)
        # Should be symmetric
        assert np.allclose(var, var.T)
        # Diagonal should be positive
        assert np.all(np.diag(var) > 0)

    def test_estimate_movement_var_with_nans(self):
        """Should handle NaN values by excluding them."""
        # Arrange
        position = np.linspace(0, 10, 100)
        position[20:30] = np.nan  # Add NaN segment

        # Act
        var = estimate_movement_var(position)

        # Assert
        assert np.isfinite(var)
        assert var > 0

    def test_estimate_movement_var_constant_position(self):
        """Constant position should give near-zero variance."""
        # Arrange
        position = np.ones(100) * 5.0

        # Act
        var = estimate_movement_var(position)

        # Assert
        assert np.allclose(var, 0.0, atol=1e-10)

    def test_estimate_movement_var_flat_array(self):
        """1D array should be handled (converted to 2D internally)."""
        # Arrange
        position = np.linspace(0, 10, 50)

        # Act
        var = estimate_movement_var(position)

        # Assert
        assert isinstance(var, (float, np.ndarray))
        assert var > 0
