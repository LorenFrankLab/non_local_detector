from typing import Tuple

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from non_local_detector.environment import (
    Environment,
    _create_1d_track_grid_data,
    _create_grid,
    _extract_bin_info_from_track_graph,
    _get_distance_between_bins,
    _get_node_pos,
    _infer_track_interior,
    _make_nd_track_graph,
    _make_track_graph_bin_centers,
    _make_track_graph_bin_centers_edges,
    get_centers,
    get_n_bins,
)

# --- Fixtures for Test Data ---


@pytest.fixture
def position_data_2d_simple() -> NDArray[np.float64]:
    """Simple 2D position data forming roughly a square path."""
    pos = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],  # Bottom edge
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],  # Right edge
            [4, 5],
            [3, 5],
            [2, 5],
            [1, 5],
            [0, 5],  # Top edge
            [0, 4],
            [0, 3],
            [0, 2],
            [0, 1],  # Left edge
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],  # Diagonalish inside
            [np.nan, np.nan],  # Add a NaN point
        ]
    )
    # Repeat to simulate more time
    return np.tile(pos, (5, 1))


@pytest.fixture
def position_data_square_nd() -> NDArray[np.float64]:
    """User-provided square path data with noise."""
    x = np.linspace(0, 30, 50)  # Increased points for better coverage
    position = np.concatenate(
        (
            np.stack((np.zeros_like(x), x[::-1]), axis=1),  # Left edge (30 -> 0)
            np.stack((x, np.zeros_like(x)), axis=1),  # Bottom edge (0 -> 30)
            np.stack((np.ones_like(x) * 30, x), axis=1),  # Right edge (0 -> 30)
            # Removed the partial path from original example for clarity
        )
    )
    # Add Gaussian noise
    rng = np.random.default_rng(seed=42)  # for reproducibility
    noise = rng.multivariate_normal(
        [0, 0], [[0.5, 0], [0, 0.5]], size=position.shape[0]
    )
    return position + noise


@pytest.fixture
def track_graph_u_shape() -> nx.Graph:
    """User-provided U-shaped graph structure."""
    node_positions = {
        0: (0, 0),
        1: (30, 0),
        2: (30, 30),
        3: (0, 30),
    }
    edges = [(0, 1), (0, 3), (1, 2)]  # U-shape: 3-0-1-2

    graph = nx.Graph()
    graph.add_nodes_from(node_positions.keys())
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, node_positions, "pos")
    # Add edge_id and distance (optional but good practice for the class)
    for i, edge in enumerate(edges):
        pos1 = np.array(node_positions[edge[0]])
        pos2 = np.array(node_positions[edge[1]])
        dist = np.linalg.norm(pos1 - pos2)
        graph.edges[edge]["distance"] = dist
        graph.edges[edge]["edge_id"] = i
    return graph


@pytest.fixture
def position_data_1d_linear() -> NDArray[np.float64]:
    """Position data roughly along a line (for testing 1D mapping)."""
    # Corresponds roughly to the linear_track_graph fixture
    pos = np.array(
        [
            [0.5, 0],
            [1.5, 0],
            [2.5, 0],
            [3.5, 0],
            [4.5, 0],
            [5.5, 0],
            [6.5, 0],
            [7.5, 0],
            [8.5, 0],
            [9.5, 0],
        ]
    )
    return np.tile(pos, (10, 1))


@pytest.fixture
def linear_track_graph() -> Tuple[nx.Graph, list, float]:
    """Creates a simple linear track graph for 1D tests."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    node_positions = {
        0: (0, 0),
        1: (5, 0),
        2: (10, 0),
        3: (15, 0),
    }
    # Define edge_order for linearization
    edge_order = [(0, 1), (1, 2), (2, 3)]
    edge_spacing = 0.0

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, node_positions, "pos")
    # Add edge_id and distance
    for i, edge in enumerate(edge_order):
        pos1 = np.array(node_positions[edge[0]])
        pos2 = np.array(node_positions[edge[1]])
        dist = np.linalg.norm(pos1 - pos2)
        graph.edges[edge]["distance"] = dist
        graph.edges[edge]["edge_id"] = i

    return graph, edge_order, edge_spacing


# --- Test Helper Functions ---


def test_get_centers():
    edges = np.array([0, 2, 4, 6])
    centers = get_centers(edges)
    np.testing.assert_array_equal(centers, np.array([1, 3, 5]))


def test_get_n_bins():
    pos = np.array([[0, 0], [10, 20]])
    n_bins = get_n_bins(pos, bin_size=2.0)
    np.testing.assert_array_equal(n_bins, np.array([5, 10]))
    n_bins = get_n_bins(pos, bin_size=[2.0, 5.0])
    np.testing.assert_array_equal(n_bins, np.array([5, 4]))
    n_bins = get_n_bins(pos, bin_size=1.0, position_range=[(0, 5), (0, 10)])
    np.testing.assert_array_equal(n_bins, np.array([5, 10]))
    # Test zero extent
    pos_zero = np.array([[1, 1], [1, 1]])
    n_bins = get_n_bins(pos_zero, bin_size=1.0)
    np.testing.assert_array_equal(n_bins, np.array([1, 1]))


def test_create_grid(position_data_2d_simple):
    """Test _create_grid helper function."""
    pos = position_data_2d_simple
    bin_size = 1.0

    # Test without boundary bins (default)
    edges_no_b, _, centers_no_b, shape_no_b = _create_grid(
        position=pos, bin_size=bin_size, add_boundary_bins=False
    )
    assert shape_no_b == (5, 5)  # Based on data range 0-5
    assert len(edges_no_b) == 2
    assert len(edges_no_b[0]) == shape_no_b[0] + 1
    assert len(edges_no_b[1]) == shape_no_b[1] + 1
    assert centers_no_b.shape == (np.prod(shape_no_b), 2)
    np.testing.assert_allclose(edges_no_b[0][0], 0.0)  # Check tight range
    np.testing.assert_allclose(edges_no_b[0][-1], 5.0)

    # Test with boundary bins
    edges_b, _, centers_b, shape_b = _create_grid(
        position=pos, bin_size=bin_size, add_boundary_bins=True
    )
    assert shape_b == (7, 7)  # Adds one bin each side
    assert len(edges_b) == 2
    assert len(edges_b[0]) == shape_b[0] + 1
    assert len(edges_b[1]) == shape_b[1] + 1
    assert centers_b.shape == (np.prod(shape_b), 2)
    np.testing.assert_allclose(edges_b[0][0], -1.0)  # Check extended range
    np.testing.assert_allclose(edges_b[0][-1], 6.0)

    # Test with position_range
    pos_range = [(0.0, 4.0), (0.0, 6.0)]
    edges_r, _, centers_r, shape_r = _create_grid(
        position_range=pos_range, bin_size=bin_size, add_boundary_bins=False
    )
    assert shape_r == (4, 6)
    np.testing.assert_allclose(edges_r[0][0], 0.0)
    np.testing.assert_allclose(edges_r[0][-1], 4.0)
    np.testing.assert_allclose(edges_r[1][0], 0.0)
    np.testing.assert_allclose(edges_r[1][-1], 6.0)


def test_infer_track_interior(position_data_2d_simple):
    """Test _infer_track_interior helper function."""
    # Create a grid first
    edges, _, _, shape = _create_grid(
        position=position_data_2d_simple, bin_size=1.0, add_boundary_bins=False
    )

    # Basic inference
    interior = _infer_track_interior(
        position_data_2d_simple, edges, boundary_exists=False
    )
    assert interior.shape == shape
    assert interior.dtype == bool
    assert np.sum(interior) > 0  # Some bins should be true
    assert np.sum(~interior) > 0  # Some bins should be false (corners of square grid)

    # Test threshold
    interior_thresh = _infer_track_interior(
        position_data_2d_simple, edges, bin_count_threshold=1000, boundary_exists=False
    )
    assert np.sum(interior_thresh) < np.sum(interior)  # Higher threshold = fewer bins

    # Test dilation
    interior_dilate = _infer_track_interior(
        position_data_2d_simple, edges, dilate=True, boundary_exists=False
    )
    assert np.sum(interior_dilate) > np.sum(interior)

    # Test boundary_exists=True (should remove perimeter)
    interior_boundary = _infer_track_interior(
        position_data_2d_simple, edges, boundary_exists=True
    )
    assert np.sum(interior_boundary) < np.sum(interior)
    # Check if perimeter is False (for 2D)
    assert not np.any(interior_boundary[0, :])
    assert not np.any(interior_boundary[-1, :])
    assert not np.any(interior_boundary[:, 0])
    assert not np.any(interior_boundary[:, -1])


def test_make_nd_track_graph():
    """Test _make_nd_track_graph helper function."""
    # Simple 2x2 interior
    shape = (3, 3)
    centers = np.array(
        [[x + 0.5, y + 0.5] for x in range(shape[0]) for y in range(shape[1])]
    )
    interior = np.zeros(shape, dtype=bool)
    interior[0:2, 0:2] = True  # Top-left 2x2 square

    graph = _make_nd_track_graph(centers, interior, shape)

    assert isinstance(graph, nx.Graph)
    # Nodes: 0, 1, 3, 4 should be interior
    assert graph.nodes[0]["is_track_interior"]
    assert graph.nodes[1]["is_track_interior"]
    assert not graph.nodes[2]["is_track_interior"]
    assert graph.nodes[3]["is_track_interior"]
    assert graph.nodes[4]["is_track_interior"]
    assert graph.number_of_nodes() == 9  # All nodes added

    # Edges should only connect adjacent interior nodes (0,1), (0,3), (1,4), (3,4), (0,4-diag), (1,3-diag)
    expected_edges = {(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4)}
    actual_edges = set(tuple(sorted(e)) for e in graph.edges())
    assert actual_edges == expected_edges

    # Check distance attribute
    dist_0_1 = np.linalg.norm(centers[0] - centers[1])
    assert np.isclose(graph.edges[(0, 1)]["distance"], dist_0_1)
    dist_0_4 = np.linalg.norm(centers[0] - centers[4])  # Diagonal
    assert np.isclose(graph.edges[(0, 4)]["distance"], dist_0_4)


def test_get_distance_between_bins():
    """Test _get_distance_between_bins helper function."""
    # Simple path graph
    graph = nx.path_graph(4)
    # Add positions and distances
    pos = {i: (i, 0) for i in range(4)}
    nx.set_node_attributes(graph, pos, "pos")
    nx.set_node_attributes(
        graph, {i: i for i in range(4)}, "bin_ind_flat"
    )  # Need this attr
    for u, v in graph.edges():
        graph.edges[u, v]["distance"] = 1.0

    distances = _get_distance_between_bins(graph)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(distances, expected)

    # Test disconnected graph
    graph.remove_edge(1, 2)
    distances_disconnected = _get_distance_between_bins(graph)
    assert np.isinf(distances_disconnected[0, 2])
    assert np.isinf(distances_disconnected[3, 1])


def test_make_track_graph_bin_centers_edges(linear_track_graph):
    """Test _make_track_graph_bin_centers_edges helper function."""
    graph, _, _ = linear_track_graph
    bin_size = 2.0
    new_graph = _make_track_graph_bin_centers_edges(graph, bin_size)

    assert isinstance(new_graph, nx.Graph)
    assert new_graph.number_of_nodes() > graph.number_of_nodes()
    assert (
        new_graph.number_of_edges() > graph.number_of_edges()
    )  # Original edges replaced by chains

    # Check attributes of a new node (find one that is not an original node)
    original_nodes = set(graph.nodes())
    new_node = next(n for n in new_graph.nodes() if n not in original_nodes)
    assert "pos" in new_graph.nodes[new_node]
    assert "edge_id" in new_graph.nodes[new_node]
    assert "is_bin_edge" in new_graph.nodes[new_node]

    # Check edge distances in a chain
    # Find path between original node 0 and 1
    path_nodes = nx.shortest_path(new_graph, source=0, target=1, weight="distance")
    path_dist = 0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = new_graph.get_edge_data(u, v)
        assert "distance" in edge_data
        assert edge_data["distance"] > 0  # Should be positive for this graph
        path_dist += edge_data["distance"]

    original_dist_0_1 = np.linalg.norm(
        np.array(graph.nodes[0]["pos"]) - np.array(graph.nodes[1]["pos"])
    )
    assert np.isclose(path_dist, original_dist_0_1)


# --- Test Environment Class Methods ---


# Fixtures for fitted environments using new data
@pytest.fixture
def fitted_env_square_nd(position_data_square_nd) -> Environment:
    """Fixture for a fitted 2D environment using square data."""
    env = Environment(place_bin_size=3.0, infer_track_interior=True, dilate=True)
    env.fit(position_data_square_nd)
    return env


@pytest.fixture
def fitted_env_u_shape_1d(track_graph_u_shape) -> Environment:
    """Fixture for a fitted 1D environment using U-shape graph."""
    graph = track_graph_u_shape
    # Define a plausible edge order for linearization (e.g., 3->0->1->2)
    edge_order = [(3, 0), (0, 1), (1, 2)]
    edge_spacing = 0.0
    env = Environment(
        track_graph=graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        place_bin_size=2.0,
    )
    env.fit()
    return env


def test_fit_square_nd(fitted_env_square_nd):
    """Check basic fitting results for square N-D data."""
    env = fitted_env_square_nd
    assert env._is_fitted
    assert not env.is_1d
    assert env.place_bin_centers_ is not None
    assert env.is_track_interior_ is not None
    assert np.sum(env.is_track_interior_) > 0
    assert env.track_graph_nd_ is not None
    assert env.track_graph_nd_.number_of_nodes() > 0
    assert env.track_graph_nd_.number_of_edges() > 0


def test_fit_u_shape_1d(fitted_env_u_shape_1d):
    """Check basic fitting results for U-shape 1D data."""
    env = fitted_env_u_shape_1d
    assert env._is_fitted
    assert env.is_1d
    assert env.place_bin_centers_ is not None
    assert env.is_track_interior_ is not None
    assert np.all(env.is_track_interior_)  # Assuming 0 spacing
    assert env.track_graph_bin_centers_ is not None
    assert env.track_graph_bin_centers_.number_of_nodes() > 0
    assert env.track_graph_bin_centers_.number_of_edges() > 0


# --- Test Initialization ---
def test_environment_init_nd_defaults():
    """Test N-D environment initialization uses new defaults."""
    env = Environment()
    # Check defaults related to boundary handling if they were added to __init__
    # If changes were only in _fit_nd, this test remains simple.
    assert not env.is_1d


# --- Test N-D Fitting ---
def test_fit_nd_default_no_boundary(position_data_2d_simple):
    """Test default N-D fit has no enforced boundary."""
    env = Environment(place_bin_size=1.0, infer_track_interior=False)
    env.fit(position_data_2d_simple)
    assert env._is_fitted
    # Expect all bins to be True by default now when not inferring
    assert np.all(env.is_track_interior_)

    env_infer = Environment(place_bin_size=1.0, infer_track_interior=True)
    env_infer.fit(position_data_2d_simple)
    assert env_infer._is_fitted
    # Check perimeter - it might be False due to data occupancy, but shouldn't be *forced* False
    # A weak check: assert interior exists near the data range min/max
    assert np.any(env_infer.is_track_interior_[0, :]) or np.any(
        env_infer.is_track_interior_[:, 0]
    )


def test_get_node_pos():
    graph = nx.Graph()
    graph.add_node(0, pos=(1, 2))
    graph.add_node(1)  # No pos attribute
    pos = _get_node_pos(graph, 0)
    np.testing.assert_array_equal(pos, np.array([1, 2]))
    with pytest.raises(KeyError):
        _get_node_pos(graph, 1)  # Missing 'pos'
    with pytest.raises(KeyError):
        _get_node_pos(graph, 2)  # Missing node
