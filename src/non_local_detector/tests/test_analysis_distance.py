"""Tests for analysis/distance2D.py — velocity, speed, distance, similarity."""

import numpy as np
import pytest

from non_local_detector.analysis.distance2D import (
    get_2D_distance,
    get_ahead_behind_distance2D,
    get_bin_ind,
    get_speed,
    get_velocity,
    head_direction_simliarity,
)


@pytest.mark.unit
class TestGetVelocity:
    """Test velocity estimation."""

    def test_constant_velocity_trajectory(self):
        """Linear trajectory should have approximately constant velocity."""
        n = 200
        time = np.linspace(0, 1, n)
        position = np.column_stack([10 * time, 5 * time])

        # Use very small sigma to minimize smoothing distortion
        velocity = get_velocity(position, time=time, sigma=0.001)

        # Interior points should be close to true velocity
        # (boundary effects excluded by slicing)
        mid = slice(20, 180)
        np.testing.assert_allclose(velocity[mid, 0], 10.0, atol=0.1)
        np.testing.assert_allclose(velocity[mid, 1], 5.0, atol=0.1)

    def test_output_shape(self):
        """Output shape should match input."""
        rng = np.random.default_rng(0)
        position = rng.standard_normal((50, 2))
        time = np.linspace(0, 1, 50)

        velocity = get_velocity(position, time=time, sigma=0.001)

        assert velocity.shape == position.shape


@pytest.mark.unit
class TestGetSpeed:
    """Test speed computation."""

    def test_known_velocity(self):
        """Speed of [3, 4] velocity should be 5."""
        velocity = np.array([[3.0, 4.0]] * 10)

        speed = get_speed(velocity)

        np.testing.assert_allclose(speed, 5.0)

    def test_non_negative(self):
        """Speed should always be non-negative."""
        rng = np.random.default_rng(0)
        velocity = rng.standard_normal((20, 2))

        speed = get_speed(velocity)

        assert np.all(speed >= 0)

    def test_1d_velocity(self):
        """1D velocity magnitude should be absolute value."""
        velocity = np.array([[-3.0], [5.0], [0.0]])

        speed = get_speed(velocity)

        np.testing.assert_allclose(speed, [3.0, 5.0, 0.0])


@pytest.mark.unit
class TestGet2DDistance:
    """Test 2D distance computation."""

    def test_euclidean_without_graph(self):
        """Without a graph, should return Euclidean distance."""
        pos1 = np.array([[0.0, 0.0]])
        pos2 = np.array([[3.0, 4.0]])

        dist = get_2D_distance(pos1, pos2, track_graph=None)

        np.testing.assert_allclose(dist, 5.0)

    def test_same_point_zero_distance(self):
        """Distance from a point to itself should be 0."""
        pos = np.array([[7.0, 3.0]])

        dist = get_2D_distance(pos, pos, track_graph=None)

        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_multiple_points(self):
        """Should compute element-wise distances."""
        pos1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        pos2 = np.array([[1.0, 0.0], [1.0, 1.0]])

        dist = get_2D_distance(pos1, pos2, track_graph=None)

        np.testing.assert_allclose(dist, [1.0, 1.0])


@pytest.mark.unit
class TestHeadDirectionSimilarity:
    """Test cosine similarity between head direction angle and decoded position.

    Note: head_direction is an angle in radians (not a unit vector).
    The function computes cos(head_direction - angle_to_map).
    """

    def test_same_direction(self):
        """Head direction angle pointing toward MAP should give similarity ~1."""
        head_pos = np.array([[0.0, 0.0]])
        # angle to (5, 0) from (0, 0) is 0 radians
        head_dir = np.array([0.0])  # pointing right (0 radians)
        map_est = np.array([[5.0, 0.0]])

        sim = head_direction_simliarity(head_pos, head_dir, map_est)

        assert sim[0] == pytest.approx(1.0, abs=0.01)

    def test_opposite_direction(self):
        """Head direction pointing away from MAP should give similarity ~-1."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([np.pi])  # pointing left (pi radians)
        map_est = np.array([[5.0, 0.0]])  # MAP is to the right

        sim = head_direction_simliarity(head_pos, head_dir, map_est)

        assert sim[0] == pytest.approx(-1.0, abs=0.01)

    def test_perpendicular(self):
        """Perpendicular head direction should give similarity ~0."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([np.pi / 2])  # pointing up (pi/2 radians)
        map_est = np.array([[5.0, 0.0]])  # MAP is to the right

        sim = head_direction_simliarity(head_pos, head_dir, map_est)

        assert sim[0] == pytest.approx(0.0, abs=0.01)


@pytest.mark.unit
class TestGetAheadBehindDistance2D:
    """Test signed distance (ahead positive, behind negative)."""

    def test_ahead_is_positive(self):
        """MAP in front of head direction should give positive distance."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([0.0])  # pointing right
        map_pos = np.array([[5.0, 0.0]])  # ahead

        dist = get_ahead_behind_distance2D(head_pos, head_dir, map_pos)

        assert dist[0] > 0
        assert dist[0] == pytest.approx(5.0, abs=0.01)

    def test_behind_is_negative(self):
        """MAP behind head direction should give negative distance."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([0.0])  # pointing right
        map_pos = np.array([[-5.0, 0.0]])  # behind

        dist = get_ahead_behind_distance2D(head_pos, head_dir, map_pos)

        assert dist[0] < 0
        assert dist[0] == pytest.approx(-5.0, abs=0.01)

    def test_same_position_zero(self):
        """Same position should give zero distance."""
        pos = np.array([[3.0, 4.0]])
        head_dir = np.array([0.0])

        dist = get_ahead_behind_distance2D(pos, head_dir, pos)

        assert dist[0] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.unit
class TestGetBinInd:
    """Test bin index computation."""

    def test_center_of_bin(self):
        """Points at bin centers should map to sequential indices."""
        edges = [np.array([0.0, 1.0, 2.0, 3.0])]  # 3 bins
        sample = np.array([[0.5], [1.5], [2.5]])

        indices = get_bin_ind(sample, edges)

        # get_bin_ind uses np.digitize-style 1-based indexing
        np.testing.assert_array_equal(indices, [1, 2, 3])

    def test_right_edge(self):
        """Point on rightmost edge should map to last bin, not overflow."""
        edges = [np.array([0.0, 1.0, 2.0])]  # 2 bins
        sample = np.array([[2.0]])  # exactly on right edge

        indices = get_bin_ind(sample, edges)

        # Right edge is clamped to last bin (index n_bins, i.e., 2)
        assert indices[0] == 2
