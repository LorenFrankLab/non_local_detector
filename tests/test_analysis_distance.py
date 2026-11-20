"""Tests for analysis/distance2D.py — velocity, speed, distance, similarity."""

import pickle
import warnings
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from track_linearization import make_track_graph

from non_local_detector import analysis, graph_distances
from non_local_detector.analysis.distance1D import get_map_speed
from non_local_detector.analysis.distance2D import (
    get_2D_distance,
    get_ahead_behind_distance2D,
    get_map_estimate_direction_from_track_graph,
    get_speed,
    get_velocity,
    head_direction_similarity,
    make_2D_track_graph_from_environment,
)
from non_local_detector.analysis.posterior import get_spatial_posterior
from non_local_detector.environment import Environment
from non_local_detector.models.base import ClusterlessDetector


@pytest.mark.unit
@pytest.mark.parametrize("boundary_interior", [False, True])
def test_2d_track_graph_matches_grid_geometry_at_boundaries(boundary_interior):
    """Graph construction keeps diagonal neighbors and clips grid boundaries."""
    from non_local_detector.environment import Environment

    env = Environment(place_bin_size=1.0).fit_place_grid(
        np.array([[0.0, 0.0], [2.0, 2.0]]), infer_track_interior=False
    )
    assert env.is_track_interior_ is not None
    assert env.place_bin_centers_ is not None
    if not boundary_interior:
        env.is_track_interior_[:] = False
        env.is_track_interior_[1:3, 1:3] = True
    graph = make_2D_track_graph_from_environment(env)
    interior = env.is_track_interior_.ravel()
    expected_edges = set()
    for left in range(16):
        for right in range(left + 1, 16):
            x_coords, y_coords = np.unravel_index([left, right], (4, 4))
            if (
                interior[left]
                and interior[right]
                and abs(x_coords[0] - x_coords[1]) <= 1
                and abs(y_coords[0] - y_coords[1]) <= 1
            ):
                expected_edges.add((left, right))

    assert set(graph.nodes) == set(range(16))
    assert set(graph.edges) == expected_edges
    assert {data["edge_id"] for _, _, data in graph.edges(data=True)} == set(
        range(len(expected_edges))
    )
    for node, data in graph.nodes(data=True):
        assert data["pos"] == tuple(env.place_bin_centers_[node])
        assert data["is_track_interior"] == interior[node]
    for left, right, data in graph.edges(data=True):
        assert data["distance"] == np.linalg.norm(
            env.place_bin_centers_[left] - env.place_bin_centers_[right]
        )
    # The shortest path crosses two diagonal edges on the all-interior grid.
    if boundary_interior:
        expected_distance = 2 * np.sqrt(2)
        assert (
            nx.shortest_path_length(graph, 0, 10, weight="distance")
            == expected_distance
        )


def _make_linear_track(
    n_nodes: int, edge_length: float = 1.0, last_edge_length: float | None = None
) -> nx.Graph:
    """Build a linear chain track graph with per-edge distances.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (>= 2). Nodes are integer-labeled 0..n_nodes-1.
    edge_length : float, optional
        Distance attribute placed on every edge, by default 1.0.
    last_edge_length : float or None, optional
        If given, overrides the distance on the final edge
        ``(n_nodes - 2, n_nodes - 1)``. Use this to make the trailing backward
        difference distinguishable from the interior central differences so a
        misordered trailing speed sample is observable, by default None.

    Returns
    -------
    networkx.Graph
        Path graph with the ``distance`` edge attribute set on every edge.
    """
    graph = nx.path_graph(n_nodes)
    for u, v in graph.edges():
        graph[u][v]["distance"] = edge_length
    if last_edge_length is not None:
        graph[n_nodes - 2][n_nodes - 1]["distance"] = last_edge_length
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

    def test_constant_velocity_is_exact_at_the_ends(self):
        """Smoothing does not pull the first and last samples toward zero."""
        sampling_frequency = 500.0
        time = np.arange(50) / sampling_frequency
        position = np.column_stack([100.0 * time, -40.0 * time])

        velocity = get_velocity(
            position, time=time, sampling_frequency=sampling_frequency
        )

        np.testing.assert_allclose(velocity[:, 0], 100.0, rtol=0, atol=1e-9)
        np.testing.assert_allclose(velocity[:, 1], -40.0, rtol=0, atol=1e-9)

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

        sim = head_direction_similarity(head_pos, head_dir, map_est)

        assert sim[0] == pytest.approx(1.0, abs=0.01)

    def test_opposite_direction(self):
        """Head direction pointing away from MAP should give similarity ~-1."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([np.pi])  # pointing left (pi radians)
        map_est = np.array([[5.0, 0.0]])  # MAP is to the right

        sim = head_direction_similarity(head_pos, head_dir, map_est)

        assert sim[0] == pytest.approx(-1.0, abs=0.01)

    def test_perpendicular(self):
        """Perpendicular head direction should give similarity ~0."""
        head_pos = np.array([[0.0, 0.0]])
        head_dir = np.array([np.pi / 2])  # pointing up (pi/2 radians)
        map_est = np.array([[5.0, 0.0]])  # MAP is to the right

        sim = head_direction_similarity(head_pos, head_dir, map_est)

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
    differences. Using ``np.insert(speed, -1, ...)`` for the trailing sample
    would insert *before* the last element and silently swap the final two
    samples.
    """

    @pytest.mark.usefixtures("_identity_gaussian_smooth")
    def test_every_posterior_layout_gives_the_same_map_speed(self):
        """A raw ``state_bins`` posterior is reduced to position, not read as
        if its Local bin were a place bin."""
        n_bins = 5
        track_graph = _make_linear_track(n_bins, edge_length=2.0)
        nodes = np.arange(n_bins)
        positions = 2.0 * nodes
        position_only = _posterior_from_node_sequence(np.array([0, 2, 4, 1]), nodes)
        # The Local state, which holds most of the mass, is the first bin.
        raw = np.column_stack((np.full(len(position_only), 0.9), 0.1 * position_only))
        index = pd.MultiIndex.from_tuples(
            [("Local", np.nan), *(("NL", position) for position in positions)],
            names=["state", "position"],
        )
        raw_posterior = xr.DataArray(
            raw,
            dims=["time", "state_bins"],
            coords={"time": np.arange(4) * 0.002, "state_bins": index},
        )
        spatial_posterior = xr.DataArray(
            0.1 * position_only,
            dims=["time", "position"],
            coords={"position": positions},
        )

        speeds = [
            get_map_speed(
                posterior,
                track_graph,
                nodes,
                sampling_frequency=500.0,
                smooth_sigma=0.0,
            )
            for posterior in (
                raw_posterior,
                get_spatial_posterior(raw_posterior),
                spatial_posterior,
                position_only,
            )
        ]

        assert np.all(speeds[0] > 0)
        np.testing.assert_array_equal(speeds[0], speeds[1])
        np.testing.assert_array_equal(speeds[0], speeds[2])

    @pytest.mark.usefixtures("_identity_gaussian_smooth")
    def test_a_row_with_no_position_mass_has_no_speed(self):
        """A detector that starts in Local has no MAP there, and no jump."""
        n_bins = 5
        track_graph = _make_linear_track(n_bins, edge_length=2.0)
        nodes = np.arange(n_bins)
        posterior = _posterior_from_node_sequence(np.full(4, 2), nodes)
        posterior[0] = 0.0

        speed = get_map_speed(
            posterior, track_graph, nodes, sampling_frequency=500.0, smooth_sigma=0.0
        )

        assert np.isnan(speed[0])
        np.testing.assert_array_equal(speed[1:], 0.0)

    @pytest.mark.usefixtures("_identity_gaussian_smooth")
    @pytest.mark.parametrize("position_dim", ["linear_position", "bin"])
    def test_map_speed_accepts_any_name_for_the_position_dimension(self, position_dim):
        """A (time, position) array needs no reducing, whatever its dim is named."""
        n_bins = 5
        track_graph = _make_linear_track(n_bins, edge_length=2.0)
        nodes = np.arange(n_bins)
        posterior = _posterior_from_node_sequence(np.array([0, 2, 4, 1]), nodes)

        speeds = [
            get_map_speed(
                values,
                track_graph,
                nodes,
                sampling_frequency=500.0,
                smooth_sigma=0.0,
            )
            for values in (
                xr.DataArray(posterior, dims=["time", position_dim]),
                posterior,
            )
        ]

        np.testing.assert_array_equal(speeds[0], speeds[1])

    @pytest.mark.parametrize("n_time", [3, 4, 5, 100])
    @pytest.mark.usefixtures("_identity_gaussian_smooth")
    def test_get_map_speed_boundary_appends_trailing(self, n_time):
        """Trailing speed is the backward difference between the last two nodes.

        The final edge is given a distinct weight so the trailing backward
        difference differs from the interior central differences. Without this,
        a uniform-spacing track hides an ``np.insert(speed, -1, ...)``
        swap (the two misordered samples would be equal), so the asymmetric
        edge makes the regression observable at every ``n_time``.
        """
        n_nodes = max(n_time, 3)
        last_edge_length = 7.0
        track_graph = _make_linear_track(
            n_nodes, edge_length=1.0, last_edge_length=last_edge_length
        )
        place_bin_center_ind_to_node = np.arange(n_nodes)
        sampling_frequency = 500.0
        dt = 1.0 / sampling_frequency

        # Walk along every node so the final (heavy) edge is traversed.
        node_ids = np.arange(n_time)
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

        # Last sample: backward difference (node[-2] -> node[-1]) — the heavy
        # edge, so this equals last_edge_length / dt.
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
        np.testing.assert_allclose(speed[-1], last_edge_length / dt, atol=1e-10)

        # np.insert(speed, -1, ...) would put the *interior* central
        # difference at node[-2] into the final slot instead. Assert the
        # trailing sample is NOT that swapped value.
        buggy_last = nx.shortest_path_length(
            track_graph,
            source=node_ids[-3],
            target=node_ids[-1],
            weight="distance",
        ) / (2.0 * dt)
        assert not np.isclose(speed[-1], buggy_last, atol=1e-10), (
            "Trailing speed matches the np.insert(-1) swapped value; "
            "the trailing sample must be appended, not inserted before the last."
        )

        # Interior samples: central difference (node[t-1] -> node[t+1]) / (2*dt).
        for t in range(1, n_time - 1):
            expected_interior = nx.shortest_path_length(
                track_graph,
                source=node_ids[t - 1],
                target=node_ids[t + 1],
                weight="distance",
            ) / (2.0 * dt)
            np.testing.assert_allclose(speed[t], expected_interior, atol=1e-10)

    def test_smoothed_constant_speed_is_exact_at_the_ends(self):
        """A MAP moving 0.2 cm per sample at 500 Hz is 100 cm/s at every sample,
        including the first and last, with the default smoothing."""
        n_time, sampling_frequency = 50, 500.0
        track_graph = _make_linear_track(n_time, edge_length=0.2)
        nodes = np.arange(n_time)
        posterior = _posterior_from_node_sequence(nodes, nodes)

        speed = get_map_speed(
            posterior, track_graph, nodes, sampling_frequency=sampling_frequency
        )

        np.testing.assert_allclose(speed, 100.0, rtol=0, atol=1e-9)

    @pytest.mark.usefixtures("_identity_gaussian_smooth")
    def test_get_map_speed_monotone_on_constant_velocity(self):
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


# networkx functions that search a graph for shortest paths.
_NETWORKX_SEARCHES = (
    "shortest_path",
    "shortest_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "dijkstra_path",
    "dijkstra_path_length",
    "bidirectional_dijkstra",
)


@contextmanager
def _graph_searches_recorded():
    """Make networkx searches raise; yield the sources scipy Dijkstra runs from.

    Each scipy ``dijkstra`` call made through ``graph_distances`` appends its
    ``indices`` (None for an all-pairs search) to the yielded list.
    """

    def refuse(*args, **kwargs):
        msg = "networkx graph search"
        raise AssertionError(msg)

    real_dijkstra = graph_distances.dijkstra
    sources = []

    def recording_dijkstra(graph, *args, indices=None, **kwargs):
        sources.append(indices)
        return real_dijkstra(graph, *args, indices=indices, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for name in _NETWORKX_SEARCHES:
            patch.setattr(nx, name, refuse)
        patch.setattr(graph_distances, "dijkstra", recording_dijkstra)
        yield sources


def _graph_arguments(env, path):
    """Keyword arguments that give the 2-D functions ``env``'s bin graph."""
    if path == "track_graph":
        return {"track_graph": env.track_graphDD, "edges": env.edges_}
    return {"environment": env}


def _nearest_bin(graph, position):
    """Node whose ``pos`` (bin center) is nearest each position, shape (n_time,)."""
    nodes = np.asarray(sorted(graph.nodes))
    centers = np.asarray([graph.nodes[node]["pos"] for node in nodes])
    return nodes[
        np.argmin(
            np.linalg.norm(position[:, np.newaxis] - centers[np.newaxis], axis=-1),
            axis=1,
        )
    ]


def _brute_force(graph, head_position, map_position):
    """Graph distance and the directions of all shortest first steps, by networkx.

    Returns the distance from each head bin to its MAP bin (inf without a
    path) and, per time point, the set of directions (radians) of every
    neighbor of the head bin that starts a shortest path: {0.0} in the same
    bin, empty without a path. A neighbor starts a shortest path when its edge
    plus its distance to the MAP bin is within 1e-9 (relative) of the distance.
    """
    head_bins = _nearest_bin(graph, head_position)
    map_bins = _nearest_bin(graph, map_position)
    distances, directions = [], []
    for head_bin, map_bin in zip(head_bins, map_bins, strict=True):
        from_map = nx.single_source_dijkstra_path_length(
            graph, int(map_bin), weight="distance"
        )
        distance = from_map.get(int(head_bin), np.inf)
        head = np.asarray(graph.nodes[head_bin]["pos"])
        if head_bin == map_bin:
            first_steps = {0.0}
        elif np.isinf(distance):
            first_steps = set()
        else:
            first_steps = set()
            # The head bin's neighbors are connected to the MAP bin too.
            for neighbor in graph[head_bin]:
                via = graph.edges[head_bin, neighbor]["distance"] + from_map[neighbor]
                if abs(via - distance) <= 1e-9 * distance:
                    step = np.asarray(graph.nodes[neighbor]["pos"]) - head
                    first_steps.add(float(np.arctan2(step[1], step[0])))
        distances.append(distance)
        directions.append(first_steps)
    return np.asarray(distances), directions


def _u_shaped_environment(*, compute_all_pairs_distances, **fit_kwargs):
    """A U-shaped arena in 10 cm bins: 76 of 144 bins on the track.

    Shortest paths between the arms go around the bottom, and the bins in the
    gap between the arms and around the edge are off the track (isolated
    nodes, unreachable from the track). ``fit_kwargs`` go to
    ``fit_place_grid``.
    """
    rng = np.random.default_rng(0)
    position = rng.uniform(0, 100, (6000, 2))
    in_gap = (position[:, 0] > 30) & (position[:, 0] < 70) & (position[:, 1] > 30)
    return Environment(place_bin_size=10.0).fit_place_grid(
        position[~in_gap],
        compute_all_pairs_distances=compute_all_pairs_distances,
        **fit_kwargs,
    )


def _random_pairs(env, n_time, seed):
    """Head and MAP positions anywhere on the grid, and headings."""
    rng = np.random.default_rng(seed)
    lower = [edge[0] for edge in env.edges_]
    upper = [edge[-1] for edge in env.edges_]
    head = rng.uniform(lower, upper, (n_time, 2))
    map_position = rng.uniform(lower, upper, (n_time, 2))
    heading = rng.uniform(-np.pi, np.pi, n_time)
    return head, heading, map_position


@pytest.mark.unit
class TestAheadBehindDistance2DTrackGraph:
    """The track-graph path of the 2-D distance, direction and ahead/behind.

    The graph is given as a fitted ``environment`` or as ``track_graph`` and
    ``edges``. Covers the 1-D heading-angle ``head_direction`` contract.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def env_2d(cls):
        """A fitted 2D environment with a connected bin grid (built once)."""
        rng = np.random.default_rng(0)
        env = Environment(place_bin_size=10.0)
        # Dense, fully-covered square so the bin grid is connected.
        env.fit_place_grid(rng.random((1500, 2)) * 100.0)
        return env

    @pytest.fixture(scope="class")
    @classmethod
    def u_environments(cls):
        """The U-shaped arena with an all-pairs and with a deferred table."""
        return {
            table: _u_shaped_environment(
                compute_all_pairs_distances=table == "all_pairs"
            )
            for table in ("all_pairs", "deferred")
        }

    @pytest.mark.parametrize(
        "call",
        [
            lambda head, direction, map_pos, **kw: get_2D_distance(head, map_pos, **kw),
            lambda head, direction, map_pos, **kw: head_direction_similarity(
                head, direction, map_pos, **kw
            ),
            get_ahead_behind_distance2D,
            lambda head, direction, map_pos, **kw: (
                get_map_estimate_direction_from_track_graph(head, map_pos, **kw)
            ),
        ],
        ids=[
            "get_2D_distance",
            "head_direction_similarity",
            "ahead_behind",
            "map_estimate_direction",
        ],
    )
    def test_track_graph_without_edges_raises(self, env_2d, call):
        """The track graph's nodes are indexed by the environment's bin grid,
        so a track graph without ``edges`` is rejected with a message naming
        the missing argument, not an IndexError or TypeError."""
        with pytest.raises(ValueError, match="edges"):
            call(
                np.array([[10.0, 10.0]]),
                np.array([0.0]),
                np.array([[50.0, 50.0]]),
                track_graph=env_2d.track_graphDD,
                edges=None,
            )

    @pytest.mark.parametrize(
        ("arguments", "match"),
        [
            ({}, "track_graph and edges, or environment"),
            ({"environment": "env", "track_graph": "graph"}, "not both"),
            ({"environment": "env", "edges": "edges"}, "not both"),
            ({"environment": "linearized"}, "track_graphDD"),
        ],
        ids=["neither", "environment_and_track_graph", "environment_and_edges", "1d"],
    )
    def test_graph_arguments_are_checked(self, env_2d, arguments, match):
        """The direction needs a graph, given one way only, and an
        environment must have the 2-D bin graph (not a linearized track)."""
        linearized = Environment(
            place_bin_size=5.0,
            track_graph=make_track_graph([(0, 0), (20, 0)], [(0, 1)]),
            edge_order=[(0, 1)],
            edge_spacing=0.0,
        ).fit_place_grid()
        values = {
            "env": env_2d,
            "graph": env_2d.track_graphDD,
            "edges": env_2d.edges_,
            "linearized": linearized,
        }
        with pytest.raises(ValueError, match=match):
            get_map_estimate_direction_from_track_graph(
                np.array([[10.0, 10.0]]),
                np.array([[50.0, 50.0]]),
                **{name: values[value] for name, value in arguments.items()},
            )

    @pytest.mark.parametrize(
        "path", ["environment", "deferred_environment", "track_graph"]
    )
    def test_matches_brute_force_networkx(self, u_environments, path):
        """Distance, direction, similarity and ahead/behind on the U-shaped
        arena equal a networkx computation from the bin centers: the shortest
        path length (inf without a path) and the direction of a first step on
        a shortest path (0 in the same bin, NaN without a path)."""
        env = u_environments[
            "deferred" if path == "deferred_environment" else "all_pairs"
        ]
        kw = _graph_arguments(env, path)
        head, heading, map_position = _random_pairs(env, 80, seed=3)
        map_position[:5] = head[:5]  # same bin
        expected_distance, first_step_directions = _brute_force(
            env.track_graphDD, head, map_position
        )
        # The draw covers same-bin, unreachable and tied pairs.
        assert np.sum(expected_distance == 0) >= 5
        assert np.sum(np.isinf(expected_distance)) >= 5
        assert sum(len(steps) > 1 for steps in first_step_directions) >= 5

        distance = get_2D_distance(head, map_position, **kw)
        direction = get_map_estimate_direction_from_track_graph(
            head, map_position, **kw
        )
        similarity = head_direction_similarity(head, heading, map_position, **kw)
        ahead_behind = get_ahead_behind_distance2D(head, heading, map_position, **kw)

        np.testing.assert_allclose(distance, expected_distance, rtol=1e-12, atol=0)
        for t, steps in enumerate(first_step_directions):
            if steps:
                assert np.isclose(
                    direction[t], sorted(steps), rtol=0, atol=1e-12
                ).any(), (
                    t,
                    direction[t],
                    steps,
                )
            else:
                assert np.isnan(direction[t]), (t, direction[t])
        np.testing.assert_array_equal(similarity, np.cos(heading - direction))
        sign = np.sign(np.cos(heading - direction))
        sign[sign == 0] = 1.0
        np.testing.assert_allclose(
            ahead_behind, sign * expected_distance, rtol=1e-12, atol=0
        )

    @pytest.mark.parametrize(
        "path", ["environment", "deferred_environment", "track_graph"]
    )
    def test_tied_first_steps_take_the_lowest_node_id(self, path):
        """Of the head bin's neighbors that start a shortest path with exactly
        the same length, the direction is toward the one with the lowest node
        id (flat bin index)."""
        # All 25 bins of 1 cm (centers -0.5 ... 3.5) are on the track, so the
        # edges are exactly 1 and sqrt(2).
        env = Environment(
            place_bin_size=1.0, position_range=[(0, 3), (0, 3)]
        ).fit_place_grid(
            np.array([[0.0, 0.0], [3.0, 3.0]]),
            infer_track_interior=False,
            compute_all_pairs_distances=path != "deferred_environment",
        )
        graph = env.track_graphDD
        # Head bin 6 (center 0.5, 0.5) to MAP bin 17 (2.5, 1.5): a step to bin
        # 11 (1.5, 0.5) then a diagonal, or a diagonal to bin 12 (1.5, 1.5)
        # then a step, both 1 + sqrt(2) exactly. The reverse pair ties bins 11
        # (diagonal first) and 12.
        assert graph.nodes[6]["pos"] == (0.5, 0.5)
        assert graph.nodes[17]["pos"] == (2.5, 1.5)
        assert (
            graph.edges[6, 11]["distance"] + graph.edges[11, 17]["distance"]
            == graph.edges[6, 12]["distance"] + graph.edges[12, 17]["distance"]
        )
        head = np.array([[0.5, 0.5], [2.5, 1.5]])
        map_position = np.array([[2.5, 1.5], [0.5, 0.5]])
        kw = _graph_arguments(env, path)

        direction = get_map_estimate_direction_from_track_graph(
            head, map_position, **kw
        )
        # Toward bin 11 both times: +x from bin 6, and down-left from bin 17.
        np.testing.assert_allclose(direction, [0.0, -3 * np.pi / 4], rtol=0, atol=1e-15)
        # Both headings are ahead of the step toward bin 11 and behind the
        # step toward bin 12.
        ahead_behind = get_ahead_behind_distance2D(
            head, np.array([-3 * np.pi / 8, -np.pi / 3]), map_position, **kw
        )
        np.testing.assert_allclose(ahead_behind, [1 + np.sqrt(2)] * 2, rtol=1e-15)

    def test_deferred_table_budget_does_not_limit_the_lookup(
        self, u_environments, monkeypatch
    ):
        """A deferred table's dense budget (a detector's
        ``max_dense_transition_bytes``) does not limit the lookup: 80 time
        points read 9 distances (the head bin and up to 8 neighbors) for
        each distinct pair of bins, several kB, past a 1,000-byte budget. Rows
        come in blocks of at most what the table's workspace holds, here 3
        rows, and every function returns what the same table without these
        budgets gives."""
        small_budget = _u_shaped_environment(
            compute_all_pairs_distances=False, max_dense_distance_bytes=1_000
        )
        table = small_budget.distance_between_nodes_
        table.max_workspace_bytes = 3 * 8 * table.shape[1]
        block_rows = []
        cross_distances = table.cross_distances

        def recording_cross_distances(rows, columns):
            block_rows.append(len(rows))
            return cross_distances(rows, columns)

        monkeypatch.setattr(table, "cross_distances", recording_cross_distances)
        default_budget = u_environments["deferred"]
        head, heading, map_position = _random_pairs(small_budget, 80, seed=3)
        # The budget refuses a selection of 14 pairs' 9 distances.
        with pytest.raises(graph_distances.GraphDistanceBudgetError):
            table[np.zeros((14, 1), dtype=int), np.zeros((1, 9), dtype=int)]
        calls = {
            "distance": lambda **kw: get_2D_distance(head, map_position, **kw),
            "direction": lambda **kw: get_map_estimate_direction_from_track_graph(
                head, map_position, **kw
            ),
            "similarity": lambda **kw: head_direction_similarity(
                head, heading, map_position, **kw
            ),
            "ahead_behind": lambda **kw: get_ahead_behind_distance2D(
                head, heading, map_position, **kw
            ),
        }

        for name, call in calls.items():
            np.testing.assert_array_equal(
                call(environment=small_budget),
                call(environment=default_budget),
                err_msg=name,
            )
        expected_distance, _ = _brute_force(
            small_budget.track_graphDD, head, map_position
        )
        np.testing.assert_allclose(
            calls["distance"](environment=small_budget),
            expected_distance,
            rtol=1e-12,
            atol=0,
        )
        # Many blocks per call, none above the workspace.
        assert len(block_rows) > 5 * len(calls)
        assert max(block_rows) == 3

    def test_track_graph_smaller_than_the_grid_raises(self, env_2d):
        """A track graph without a node for every bin of the grid of
        ``edges`` raises IndexError at a missing bin, instead of reading the
        distance of another pair of bins."""
        graph = env_2d.track_graphDD
        truncated = graph.subgraph(range(100)).copy()
        # Head in bin 13, which the truncated graph has; MAP in bin 130,
        # which it lacks.
        head = np.array([graph.nodes[13]["pos"]])
        map_position = np.array([graph.nodes[130]["pos"]])

        with pytest.raises(IndexError):
            get_2D_distance(
                head, map_position, track_graph=truncated, edges=env_2d.edges_
            )

    @pytest.mark.parametrize("table", ["all_pairs", "deferred"])
    def test_environment_runs_no_networkx_or_all_pairs_search(self, table):
        """With an environment, distances come from its table: no networkx
        search, no all-pairs search, and only the distinct MAP bins' rows when
        the table is deferred (none when it holds all pairs)."""
        env = _u_shaped_environment(compute_all_pairs_distances=table == "all_pairs")
        head, heading, map_position = _random_pairs(env, 60, seed=4)
        map_bins = np.unique(_nearest_bin(env.track_graphDD, map_position))

        with _graph_searches_recorded() as sources:
            get_2D_distance(head, map_position, environment=env)
            get_map_estimate_direction_from_track_graph(
                head, map_position, environment=env
            )
            head_direction_similarity(head, heading, map_position, environment=env)
            get_ahead_behind_distance2D(head, heading, map_position, environment=env)

        if table == "all_pairs":
            assert sources == []
        else:
            assert all(indices is not None for indices in sources)
            np.testing.assert_array_equal(np.unique(np.concatenate(sources)), map_bins)

    def test_track_graph_and_edges_runs_no_networkx_search(self, u_environments):
        """With ``track_graph`` and ``edges``, ahead/behind computes the
        shortest-path row of each distinct MAP bin once, with scipy, and
        nothing else."""
        env = u_environments["all_pairs"]
        head, heading, map_position = _random_pairs(env, 60, seed=5)
        map_bins = np.unique(_nearest_bin(env.track_graphDD, map_position))

        with _graph_searches_recorded() as sources:
            get_ahead_behind_distance2D(
                head,
                heading,
                map_position,
                track_graph=env.track_graphDD,
                edges=env.edges_,
            )

        assert all(indices is not None for indices in sources)
        np.testing.assert_array_equal(np.sort(np.concatenate(sources)), map_bins)

    @pytest.mark.parametrize("value", [True, False])
    @pytest.mark.parametrize(
        ("call", "path"),
        [
            (call, path)
            for call in ("distance", "similarity", "ahead_behind")
            for path in ("euclidean", "environment", "track_graph")
        ]
        + [("direction", "environment"), ("direction", "track_graph")],
    )
    def test_precomputed_distance_warns_and_changes_nothing(
        self, env_2d, call, path, value
    ):
        """``precomputed_distance`` is deprecated: any value gives one
        ``FutureWarning``, attributed to the caller, and the same result."""
        functions = {
            "distance": lambda **kw: get_2D_distance(head, map_position, **kw),
            "direction": lambda **kw: get_map_estimate_direction_from_track_graph(
                head, map_position, **kw
            ),
            "similarity": lambda **kw: head_direction_similarity(
                head, heading, map_position, **kw
            ),
            "ahead_behind": lambda **kw: get_ahead_behind_distance2D(
                head, heading, map_position, **kw
            ),
        }
        head, heading, map_position = _random_pairs(env_2d, 15, seed=1)
        kw = {} if path == "euclidean" else _graph_arguments(env_2d, path)
        expected = functions[call](**kw)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = functions[call](precomputed_distance=value, **kw)

        assert [warning.category for warning in caught] == [FutureWarning]
        assert "precomputed_distance" in str(caught[0].message)
        assert caught[0].filename == __file__
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("path", ["environment", "track_graph"])
    def test_head_direction_angle_contract(self, env_2d, path):
        """``head_direction`` is a 1-D angle array; a heading toward the MAP
        position yields a positive (ahead) signed distance.
        """
        env = env_2d
        n_time = 8
        head_position = np.full((n_time, 2), 50.0)
        # MAP is to the +x side of the head; heading 0 rad points +x -> ahead.
        map_position = head_position + np.array([20.0, 0.0])
        head_direction = np.zeros(n_time)  # radians, pointing +x

        ahead_behind = get_ahead_behind_distance2D(
            head_position,
            head_direction,
            map_position,
            **_graph_arguments(env, path),
        )
        assert ahead_behind.shape == (n_time,)
        assert np.all(ahead_behind > 0)

    @pytest.mark.parametrize("path", ["environment", "track_graph"])
    def test_disconnected_graph_gives_inf_distance_and_nan_direction(self, path):
        """An unreachable head/MAP pair on a disconnected track graph gives an
        infinite distance and a NaN direction, not a wrong heading or a crash.
        """
        rng = np.random.default_rng(0)
        # Two dense clusters separated by a wide empty gap -> the bins between
        # them are off-track, so the manifold grid is disconnected.
        cluster_a = rng.random((400, 2)) * 20.0
        cluster_b = rng.random((400, 2)) * 20.0 + 200.0
        env = Environment(place_bin_size=10.0)
        env.fit_place_grid(np.vstack([cluster_a, cluster_b]))
        assert nx.number_connected_components(env.track_graphDD) > 1, (
            "test setup requires a disconnected track graph"
        )

        head_position = np.array([[10.0, 10.0]])  # in cluster A
        map_estimate = np.array([[210.0, 210.0]])  # in cluster B (unreachable)
        kw = _graph_arguments(env, path)

        direction = get_map_estimate_direction_from_track_graph(
            head_position, map_estimate, **kw
        )
        distance = get_2D_distance(head_position, map_estimate, **kw)
        np.testing.assert_array_equal(direction, [np.nan])
        np.testing.assert_array_equal(distance, [np.inf])

    @pytest.mark.parametrize("path", ["environment", "track_graph"])
    def test_nan_position_gives_nan(self, env_2d, path):
        """A row with a NaN head or MAP position gives NaN, as the Euclidean
        branch does, not a distance from the last grid bin; other rows keep
        the values they have alone."""
        env = env_2d
        nan = np.nan
        # Row 1: NaN head; row 2: NaN MAP; row 3: one NaN coordinate;
        # row 4: both NaN; rows 0 and 5 are finite.
        head = np.array(
            [
                [10.0, 10.0],
                [nan, nan],
                [50.0, 60.0],
                [80.0, nan],
                [nan, nan],
                [80.0, 20.0],
            ]
        )
        map_pos = np.array(
            [
                [90.0, 90.0],
                [20.0, 30.0],
                [nan, nan],
                [40.0, 70.0],
                [nan, nan],
                [40.0, 70.0],
            ]
        )
        head_dir = np.array([0.3, 0.0, -1.0, 0.0, 0.0, 2.0])
        nan_rows = np.array([False, True, True, True, True, False])
        keep = np.flatnonzero(~nan_rows)
        kw = _graph_arguments(env, path)

        distance = get_2D_distance(head, map_pos, **kw)
        ahead_behind = get_ahead_behind_distance2D(head, head_dir, map_pos, **kw)
        direction = get_map_estimate_direction_from_track_graph(head, map_pos, **kw)

        assert np.isnan(distance[nan_rows]).all(), distance
        assert np.isnan(ahead_behind[nan_rows]).all(), ahead_behind
        assert np.isnan(direction[nan_rows]).all(), direction
        np.testing.assert_array_equal(
            distance[keep], get_2D_distance(head[keep], map_pos[keep], **kw)
        )
        np.testing.assert_array_equal(
            ahead_behind[keep],
            get_ahead_behind_distance2D(
                head[keep], head_dir[keep], map_pos[keep], **kw
            ),
        )
        # All rows NaN: nothing to look up.
        np.testing.assert_array_equal(
            get_ahead_behind_distance2D(
                head[[1, 4]], head_dir[[1, 4]], head[[1, 4]], **kw
            ),
            [nan, nan],
        )


COMPAT_DATA = Path(__file__).parent / "compat_data"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("version", "artifact"),
    [
        ("main-11d9b3de", "contfrag_clusterless_2d_structured"),
        ("main-d5f85823", "nonlocal_clusterless_2d"),
    ],
)
def test_spyglass_2d_ahead_behind_call_with_a_stored_model(version, artifact):
    """The 2-D call of Spyglass's ``get_ahead_behind_distance`` in the
    companion patches (``docs/spyglass_migration``), with its arguments built
    as the patch builds them from a stored model and its results.

    Spyglass has the loaded model there, so it passes
    ``environment=classifier.environments_[0]``, the fitted environment: a
    model fitted by this version keeps only the configuration in
    ``environments``, as the first artifact does. The track graph without
    ``edges``, the call the patch made before, is refused.
    """
    classifier = ClusterlessDetector.load_model(
        COMPAT_DATA / version / f"{artifact}_model.pkl"
    )
    results = ClusterlessDetector.load_results(
        COMPAT_DATA / version / f"{artifact}_results.nc"
    )
    inputs = pickle.loads((COMPAT_DATA / "inputs.pkl").read_bytes())
    time = inputs["time"]
    columns, orientation = ["x", "y"], "head_orientation"
    tracking = pd.DataFrame(
        {
            "x": inputs["field_position"][:, 0],
            "y": inputs["field_position"][:, 1],
            orientation: np.random.default_rng(0).uniform(-np.pi, np.pi, time.size),
        },
        index=pd.Index(time, name="time"),
    )

    # As the patch builds the call.
    posterior = analysis.get_position_posterior(results.acausal_posterior)
    distance = np.full(results.sizes["time"], np.nan)
    aligned, valid = analysis.align_tracking_to_results(
        tracking,
        results,
        position_columns=columns,
        valid_position_intervals=[[time[0], time[-1]]],
        circular_columns=[orientation],
    )
    valid &= np.isfinite(aligned[[*columns, orientation]].to_numpy(dtype=float)).all(
        axis=1
    )
    assert valid.any()
    map_position = analysis.maximum_a_posteriori_estimate(posterior.isel(time=valid))
    distance[valid] = analysis.get_ahead_behind_distance2D(
        aligned.loc[valid, columns].to_numpy(),
        aligned.loc[valid, orientation].to_numpy(),
        map_position,
        environment=classifier.environments_[0],
    )

    graph = classifier.environments_[0].track_graphDD
    head = aligned.loc[valid, columns].to_numpy()
    expected, _ = _brute_force(graph, head, map_position)
    # Without a path there is no direction, so no sign: NaN.
    np.testing.assert_allclose(
        np.abs(distance[valid]),
        np.where(np.isinf(expected), np.nan, expected),
        rtol=1e-12,
        atol=0,
    )
    assert np.isnan(distance[~valid]).all()
    # In the non-local artifact every valid MAP is the last grid bin, off the
    # track: unstacking puts the Local and No-Spike mass, which has no
    # position, there. So only the continuous/fragmented artifact has paths.
    if artifact.startswith("contfrag"):
        assert np.isfinite(distance[valid]).sum() >= 10
    with pytest.raises(ValueError, match="edges"):
        analysis.get_ahead_behind_distance2D(
            head, aligned.loc[valid, orientation].to_numpy(), map_position, graph
        )
