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


@pytest.mark.unit
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


@pytest.mark.unit
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
        assert isinstance(var, float | np.ndarray)
        assert var > 0

    def test_estimate_movement_var_2d_position(self):
        """Should estimate covariance for 2D position."""
        # Arrange
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        position = np.column_stack(
            [t + np.random.randn(100) * 0.1, t + np.random.randn(100) * 0.1]
        )

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
        assert isinstance(var, float | np.ndarray)
        assert var > 0


# Tests for RandomWalk class error handling
class TestRandomWalkErrorHandling:
    """Test RandomWalk error handling for missing place bin centers."""

    def test_random_walk_error_without_place_bin_centers(self):
        """Test that RandomWalk raises error when environment lacks place bin centers."""
        from non_local_detector.continuous_state_transitions import RandomWalk
        from non_local_detector.environment import Environment

        env = Environment(environment_name="test")
        envs = (env,)

        rw = RandomWalk(environment_name="test", movement_var=1.0)

        with pytest.raises(ValueError, match="must have defined place bin centers"):
            rw.make_state_transition(envs)


# Tests for Uniform class comprehensive coverage
class TestUniformComprehensive:
    """Comprehensive tests for Uniform transition class."""

    def test_uniform_error_without_place_bin_centers(self):
        """Test that Uniform raises error when environment lacks place bin centers."""
        from non_local_detector.continuous_state_transitions import Uniform
        from non_local_detector.environment import Environment

        env = Environment(environment_name="test")
        envs = (env,)

        uniform = Uniform(environment_name="test")

        with pytest.raises(ValueError, match="must have defined place bin centers"):
            uniform.make_state_transition(envs)

    def test_uniform_with_no_track_interior(self):
        """Test Uniform when environment has no track interior mask."""
        from non_local_detector.continuous_state_transitions import Uniform
        from non_local_detector.environment import Environment

        env = Environment(
            environment_name="test", place_bin_size=1.0, position_range=((0.0, 5.0),)
        )
        pos = np.linspace(0.0, 5.0, 6)[:, None]
        env = env.fit_place_grid(position=pos, infer_track_interior=False)
        n_bins = env.place_bin_centers_.shape[0]
        env.is_track_interior_ = None
        envs = (env,)

        uniform = Uniform(environment_name="test")
        trans = uniform.make_state_transition(envs)

        assert trans.shape == (n_bins, n_bins)
        assert np.allclose(trans, 1.0 / n_bins)


# Tests for Identity class comprehensive coverage
class TestIdentityComprehensive:
    """Comprehensive tests for Identity transition class."""

    def test_identity_error_without_place_bin_centers(self):
        """Test that Identity raises error when environment lacks place bin centers."""
        from non_local_detector.continuous_state_transitions import Identity
        from non_local_detector.environment import Environment

        env = Environment(environment_name="test")
        envs = (env,)

        identity = Identity(environment_name="test")

        with pytest.raises(ValueError, match="must have defined place bin centers"):
            identity.make_state_transition(envs)

    def test_identity_with_no_track_interior(self):
        """Test Identity when environment has no track interior mask."""
        from non_local_detector.continuous_state_transitions import Identity
        from non_local_detector.environment import Environment

        env = Environment(
            environment_name="test", place_bin_size=1.0, position_range=((0.0, 5.0),)
        )
        pos = np.linspace(0.0, 5.0, 6)[:, None]
        env = env.fit_place_grid(position=pos, infer_track_interior=False)
        n_bins = env.place_bin_centers_.shape[0]
        env.is_track_interior_ = None
        envs = (env,)

        identity = Identity(environment_name="test")
        trans = identity.make_state_transition(envs)

        assert trans.shape == (n_bins, n_bins)
        assert np.allclose(trans, np.eye(n_bins))


# Tests for EmpiricalMovement class
# Tests for RandomWalkDirection1 class
class TestRandomWalkDirection1Comprehensive:
    """Comprehensive tests for RandomWalkDirection1 transition class."""

    def test_random_walk_direction1_error_without_place_bin_centers(self):
        """Test that RandomWalkDirection1 raises error when environment lacks place bin centers."""
        from non_local_detector.continuous_state_transitions import RandomWalkDirection1
        from non_local_detector.environment import Environment

        env = Environment(environment_name="test")
        envs = (env,)

        rwd1 = RandomWalkDirection1(environment_name="test", movement_var=1.0)

        with pytest.raises(ValueError, match="must have defined place bin centers"):
            rwd1.make_state_transition(envs)


# Tests for RandomWalkDirection2 class
class TestRandomWalkDirection2Comprehensive:
    """Comprehensive tests for RandomWalkDirection2 transition class."""

    def test_random_walk_direction2_error_without_place_bin_centers(self):
        """Test that RandomWalkDirection2 raises error when environment lacks place bin centers."""
        from non_local_detector.continuous_state_transitions import RandomWalkDirection2
        from non_local_detector.environment import Environment

        env = Environment(environment_name="test")
        envs = (env,)

        rwd2 = RandomWalkDirection2(environment_name="test", movement_var=1.0)

        with pytest.raises(ValueError, match="must have defined place bin centers"):
            rwd2.make_state_transition(envs)


# Tests for Discrete class
class TestDiscreteComprehensive:
    """Comprehensive tests for Discrete transition class."""

    def test_discrete_returns_single_element(self):
        """Test that Discrete returns a 1x1 matrix of ones."""
        from non_local_detector.continuous_state_transitions import Discrete

        discrete = Discrete()
        trans = discrete.make_state_transition()

        assert trans.shape == (1, 1)
        assert np.isclose(trans[0, 0], 1.0)


# Additional tests to reach 80%+ coverage
class TestRandomWalkWithTrackGraph:
    """Test RandomWalk with track graphs to cover _random_walk_on_track_graph."""

    def test_random_walk_with_1d_environment(self):
        """Test RandomWalk on simple 1D environment (uses Euclidean)."""
        from non_local_detector.continuous_state_transitions import RandomWalk
        from non_local_detector.environment import Environment

        env = Environment(
            environment_name="test", place_bin_size=2.0, position_range=((0.0, 10.0),)
        )
        pos = np.linspace(0.0, 10.0, 20)[:, None]
        env = env.fit_place_grid(position=pos, infer_track_interior=False)
        envs = (env,)

        rw = RandomWalk(environment_name="test", movement_var=4.0)
        trans = rw.make_state_transition(envs)

        # Verify it's a valid stochastic matrix
        assert trans.ndim == 2
        assert np.all(trans >= 0)
        assert np.allclose(trans.sum(axis=1), 1.0)


# Tests for EmpiricalMovement class
