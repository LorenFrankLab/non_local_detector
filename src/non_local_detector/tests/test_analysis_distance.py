"""Tests for analysis/distance2D.py — velocity, speed, distance, similarity."""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.analysis.distance1D import get_map_speed
from non_local_detector.analysis.distance2D import (
    get_2D_distance,
    get_ahead_behind_distance2D,
    get_bin_ind,
    get_speed,
    get_velocity,
    head_direction_simliarity,
)


def _make_linear_track(n_nodes: int, edge_length: float = 1.0) -> nx.Graph:
    """Build a linear chain track graph with uniform edge distances.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (>= 2). Nodes are integer-labeled 0..n_nodes-1.
    edge_length : float, optional
        Distance attribute placed on every edge, by default 1.0.

    Returns
    -------
    networkx.Graph
        Path graph with the ``distance`` edge attribute set on every edge.
    """
    graph = nx.path_graph(n_nodes)
    for u, v in graph.edges():
        graph[u][v]["distance"] = edge_length
    return graph


def _posterior_from_node_sequence(
    node_ids: np.ndarray, place_bin_center_ind_to_node: np.ndarray
) -> np.ndarray:
    """Construct a one-hot posterior whose argmax recovers ``node_ids``.

    Parameters
    ----------
    node_ids : np.ndarray, shape (n_time,)
        Desired sequence of MAP node IDs at each time step.
    place_bin_center_ind_to_node : np.ndarray, shape (n_bins,)
        Mapping from bin index to node ID.

    Returns
    -------
    posterior : np.ndarray, shape (n_time, n_bins)
        One-hot posterior with mass on the bin whose node matches ``node_ids[t]``.
    """
    n_bins = len(place_bin_center_ind_to_node)
    posterior = np.zeros((len(node_ids), n_bins), dtype=float)
    for t, node in enumerate(node_ids):
        bin_ind = int(np.where(place_bin_center_ind_to_node == node)[0][0])
        posterior[t, bin_ind] = 1.0
    return posterior


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


@pytest.fixture
def _identity_gaussian_smooth(monkeypatch):
    """Replace ``_gaussian_smooth`` with the identity so tests can assert raw
    boundary/interior speed values without Gaussian-smoothing distortion.
    """
    from non_local_detector.analysis import distance1D

    monkeypatch.setattr(
        distance1D,
        "_gaussian_smooth",
        lambda data, sigma, sampling_frequency, axis=0, truncate=8: data,
    )


@pytest.mark.unit
class TestGetMapSpeedBoundaryHandling:
    """Validate the trailing-boundary speed of ``get_map_speed``.

    The interior speeds use a 2-step central difference on shortest-path
    distance. The first and last samples use one-sided forward/backward
    differences. The earlier implementation used ``np.insert(speed, -1, ...)``
    for the trailing sample, which inserts *before* the last element and
    silently swaps the final two samples.
    """

    @pytest.mark.parametrize("n_time", [3, 4, 5, 100])
    def test_get_map_speed_boundary_appends_trailing(
        self, n_time, _identity_gaussian_smooth
    ):
        """Trailing speed is the backward difference between the last two nodes."""
        n_nodes = max(n_time, 3)
        track_graph = _make_linear_track(n_nodes, edge_length=1.0)
        place_bin_center_ind_to_node = np.arange(n_nodes)
        sampling_frequency = 500.0
        dt = 1.0 / sampling_frequency

        node_ids = np.arange(n_time) % n_nodes
        posterior = _posterior_from_node_sequence(
            node_ids, place_bin_center_ind_to_node
        )

        # Disable smoothing so we can check the underlying speed values.
        speed = get_map_speed(
            posterior=posterior,
            track_graph_with_bin_centers_edges=track_graph,
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0,
        )

        assert speed.shape == (n_time,)

        # First sample: forward difference (node 0 -> node 1).
        expected_first = (
            nx.shortest_path_length(
                track_graph, source=node_ids[0], target=node_ids[1], weight="distance"
            )
            / dt
        )
        np.testing.assert_allclose(speed[0], expected_first, atol=1e-10)

        # Last sample: backward difference (node[-2] -> node[-1]).
        expected_last = (
            nx.shortest_path_length(
                track_graph,
                source=node_ids[-2],
                target=node_ids[-1],
                weight="distance",
            )
            / dt
        )
        np.testing.assert_allclose(speed[-1], expected_last, atol=1e-10)

        # Interior samples: central difference (node[t-1] -> node[t+1]) / (2*dt).
        for t in range(1, n_time - 1):
            expected_interior = nx.shortest_path_length(
                track_graph,
                source=node_ids[t - 1],
                target=node_ids[t + 1],
                weight="distance",
            ) / (2.0 * dt)
            np.testing.assert_allclose(speed[t], expected_interior, atol=1e-10)

    def test_get_map_speed_demonstrates_pre_fix_misordering(
        self, _identity_gaussian_smooth
    ):
        """The trailing sample matches an append, not an insert-before-last.

        The earlier implementation produced ``speed[-2]`` = trailing boundary
        speed and ``speed[-1]`` = the central difference at the original last
        position. This test constructs a sequence in which the central
        difference at the original final position differs from the trailing
        backward difference by a known amount, and asserts the trailing
        sample equals the backward difference (the correct value).
        """
        # Use unequal edge distances so the boundary and the original final
        # central difference disagree.
        track_graph = nx.Graph()
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 7.0)]
        for u, v, d in edges:
            track_graph.add_edge(u, v, distance=d)

        place_bin_center_ind_to_node = np.array([0, 1, 2, 3, 4])
        node_ids = np.array([0, 1, 2, 3, 4])
        sampling_frequency = 500.0
        dt = 1.0 / sampling_frequency

        posterior = _posterior_from_node_sequence(
            node_ids, place_bin_center_ind_to_node
        )

        speed = get_map_speed(
            posterior=posterior,
            track_graph_with_bin_centers_edges=track_graph,
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0,
        )

        # Correct trailing value: backward difference between nodes 3 and 4.
        expected_correct_last = 7.0 / dt
        # Buggy trailing value would have been the central difference at the
        # original last position (between nodes 3 and 4 via central diff on
        # the pre-padded array); concretely, with the old np.insert the final
        # entry was the central diff at position -2, which here is
        # shortest_path_length(2, 4) / (2*dt) = (1 + 7) / (2*dt).
        buggy_last = (1.0 + 7.0) / (2.0 * dt)

        np.testing.assert_allclose(speed[-1], expected_correct_last, atol=1e-10)
        assert not np.isclose(speed[-1], buggy_last, atol=1e-10), (
            "Trailing speed matches the pre-fix np.insert(-1) value; "
            "the trailing sample must be appended, not inserted before the last."
        )

    def test_get_map_speed_monotone_on_constant_velocity(
        self, _identity_gaussian_smooth
    ):
        """A linear trajectory with constant velocity yields equal interior speeds.

        Boundary samples use one-sided differences but, for a uniform-spacing
        linear trajectory, they equal the interior central differences exactly.
        """
        n_time = 6
        track_graph = _make_linear_track(n_time, edge_length=1.0)
        place_bin_center_ind_to_node = np.arange(n_time)
        sampling_frequency = 500.0

        node_ids = np.arange(n_time)
        posterior = _posterior_from_node_sequence(
            node_ids, place_bin_center_ind_to_node
        )

        speed = get_map_speed(
            posterior=posterior,
            track_graph_with_bin_centers_edges=track_graph,
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
            sampling_frequency=sampling_frequency,
            smooth_sigma=0.0,
        )

        # All interior values are central differences of step 2 over uniform
        # spacing of 1; all equal 1 / dt.
        interior = speed[1:-1]
        np.testing.assert_allclose(
            interior, interior[0] * np.ones_like(interior), atol=1e-10
        )

        # For a constant-velocity uniform-spacing trajectory the boundary
        # forward/backward differences equal the interior central differences.
        np.testing.assert_allclose(speed[0], interior[0], atol=1e-10)
        np.testing.assert_allclose(speed[-1], interior[0], atol=1e-10)


def _make_positioned_track(n_nodes: int, edge_length: float = 1.0) -> nx.Graph:
    """Linear chain track graph with 2D node positions along the x-axis.

    Adds both ``distance`` edge attributes and ``pos`` node attributes so
    ``_setup_track_graph`` (which projects positions and computes
    node-to-node distances) can operate on it.
    """
    graph = nx.path_graph(n_nodes)
    for node in graph.nodes():
        graph.nodes[node]["pos"] = np.array([float(node) * edge_length, 0.0])
    for u, v in graph.edges():
        graph[u][v]["distance"] = edge_length
    return graph


@pytest.mark.unit
class TestSetupTrackGraphIsolation:
    """``_setup_track_graph`` must not mutate its input graph.

    The same-edge branch adds a node_ahead<->node_behind shortcut whose
    endpoints are real track nodes that the caller never removes. If the
    function mutates the passed-in graph, that shortcut persists across
    time steps and distorts later shortest-path distances.
    """

    def test_setup_track_graph_does_not_mutate_input(self):
        """The input graph's nodes and edges are unchanged after the call."""
        from non_local_detector.analysis.distance1D import _setup_track_graph

        track_graph = _make_positioned_track(3, edge_length=1.0)
        nodes_before = set(track_graph.nodes())
        edges_before = {frozenset(e) for e in track_graph.edges()}

        # Same-edge case (actual_edge == mental_edge) triggers the shortcut.
        _setup_track_graph(
            track_graph,
            actual_pos=np.array([0.5, 0.0]),
            actual_edge=np.array([0, 1]),
            head_direction=0.0,  # points toward node 1 (+x)
            mental_pos=np.array([0.7, 0.0]),
            mental_edge=np.array([0, 1]),
        )

        assert set(track_graph.nodes()) == nodes_before
        assert {frozenset(e) for e in track_graph.edges()} == edges_before

    def test_setup_track_graph_no_cross_call_leak(self):
        """Repeated calls on the same input give identical mental-position distance.

        A first call on a different edge pair must not leave a shortcut that
        shortens the path computed by a later call on the original graph.
        """
        from non_local_detector.analysis.distance1D import (
            _calculate_distance,
            _setup_track_graph,
        )

        track_graph = _make_positioned_track(4, edge_length=1.0)

        def distance_for(actual_edge, mental_edge, mental_pos):
            g = _setup_track_graph(
                track_graph,
                actual_pos=np.array([actual_edge[0] + 0.5, 0.0]),
                actual_edge=np.array(actual_edge),
                head_direction=0.0,
                mental_pos=np.array(mental_pos),
                mental_edge=np.array(mental_edge),
            )
            return _calculate_distance(
                g, source="actual_position", target="mental_position"
            )

        # Reference distance computed in isolation.
        ref = distance_for([0, 1], [0, 1], [0.7, 0.0])

        # Run a different edge pair first (would add a persistent shortcut
        # pre-fix), then recompute the reference case.
        _ = distance_for([2, 3], [2, 3], [2.7, 0.0])
        after = distance_for([0, 1], [0, 1], [0.7, 0.0])

        np.testing.assert_allclose(after, ref, atol=1e-12)
