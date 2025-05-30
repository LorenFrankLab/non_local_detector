import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.layout import hex_grid as hgu


def test_create_hex_grid_basic():
    data_samples = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    (
        bin_centers,
        centers_shape,
        hex_radius,
        hex_orientation,
        min_x,
        min_y,
        dimension_range,
    ) = hgu._create_hex_grid(data_samples, hexagon_width=1.0)
    assert bin_centers.shape[1] == 2
    assert isinstance(centers_shape, tuple)
    assert hex_radius > 0
    assert hex_orientation == 0.0
    assert min_x == 0.0
    assert min_y == 0.0
    assert isinstance(dimension_range, list) or isinstance(dimension_range, tuple)


def test_create_hex_grid_with_dimension_range():
    data_samples = np.empty((0, 2))
    (
        bin_centers,
        centers_shape,
        hex_radius,
        hex_orientation,
        min_x,
        min_y,
        dimension_range,
    ) = hgu._create_hex_grid(
        data_samples, dimension_range=[(0, 2), (0, 2)], hexagon_width=1.0
    )
    assert bin_centers.shape[1] == 2
    assert centers_shape[0] > 0 and centers_shape[1] > 0
    assert min_x == 0.0
    assert min_y == 0.0


def test_create_hex_grid_invalid_hexagon_width():
    data_samples = np.array([[0, 0], [1, 1]], dtype=np.float64)
    with pytest.raises(ValueError):
        hgu._create_hex_grid(data_samples, hexagon_width=0)


def test_cartesian_to_fractional_cube_and_round_trip():
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    hex_radius = 1.0
    q, r, s = hgu._cartesian_to_fractional_cube(points[:, 0], points[:, 1], hex_radius)
    assert q.shape == (3,)
    assert r.shape == (3,)
    assert s.shape == (3,)
    q_ax, r_ax = hgu._round_fractional_cube_to_integer_axial(q, r, s)
    assert q_ax.shape == (3,)
    assert r_ax.shape == (3,)
    # Check that q_ax + r_ax + s_ax == 0 for all points
    s_ax = -q_ax - r_ax
    assert np.all(q_ax + r_ax + s_ax == 0)


def test_axial_to_offset_bin_indices_and_points_to_hex_bin_ind():
    q_axial = np.array([0, 1, 2])
    r_axial = np.array([0, 0, 0])
    n_hex_x, n_hex_y = 3, 1
    indices = hgu._axial_to_offset_bin_indices(q_axial, r_axial, n_hex_x, n_hex_y)
    assert np.all(indices == np.array([0, 1, 2]))
    # Out of bounds
    indices = hgu._axial_to_offset_bin_indices(
        np.array([3]), np.array([0]), n_hex_x, n_hex_y
    )
    assert indices[0] == -1

    # Test _points_to_hex_bin_ind
    bin_centers, centers_shape, hex_radius, _, min_x, min_y, _ = hgu._create_hex_grid(
        np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64), hexagon_width=1.0
    )
    points = np.array([[0, 0], [1, 0], [2, 0], [100, 100]], dtype=np.float64)
    indices = hgu._points_to_hex_bin_ind(
        points, min_x, min_y, hex_radius, centers_shape
    )
    assert indices.shape == (4,)
    assert np.any(indices == -1)  # The last point should be out of bounds


def test_infer_active_bins_from_hex_grid():
    data_samples = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=np.float64)
    bin_centers, centers_shape, hex_radius, _, min_x, min_y, _ = hgu._create_hex_grid(
        data_samples, hexagon_width=1.0
    )
    # Threshold 0: all bins with at least 1 sample
    active_bins = hgu._infer_active_bins_from_hex_grid(
        data_samples, centers_shape, hex_radius, min_x, min_y, bin_count_threshold=0
    )
    assert active_bins.size > 0
    # Threshold 2: only bins with at least 2 samples
    active_bins2 = hgu._infer_active_bins_from_hex_grid(
        data_samples, centers_shape, hex_radius, min_x, min_y, bin_count_threshold=2
    )
    assert np.all(np.isin(active_bins2, active_bins))
    # Threshold high: no bins
    active_bins3 = hgu._infer_active_bins_from_hex_grid(
        data_samples, centers_shape, hex_radius, min_x, min_y, bin_count_threshold=10
    )
    assert active_bins3.size == 0


def test_get_hex_grid_neighbor_deltas():
    even_neighbors = hgu._get_hex_grid_neighbor_deltas(False)
    odd_neighbors = hgu._get_hex_grid_neighbor_deltas(True)
    assert len(even_neighbors) == 6
    assert len(odd_neighbors) == 6
    # Check that neighbor deltas are tuples of ints
    for delta in even_neighbors + odd_neighbors:
        assert isinstance(delta, tuple) and len(delta) == 2


def test_create_hex_connectivity_graph():
    # Create a small grid and mark all bins as active
    data_samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    bin_centers, centers_shape, hex_radius, _, min_x, min_y, _ = hgu._create_hex_grid(
        data_samples, hexagon_width=1.0
    )
    n_bins = bin_centers.shape[0]
    active_indices = np.arange(n_bins, dtype=int)
    G = hgu._create_hex_connectivity_graph(active_indices, bin_centers, centers_shape)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == n_bins
    # Each node should have 'pos', 'source_grid_flat_index', 'original_grid_nd_index'
    for node in G.nodes:
        attrs = G.nodes[node]
        assert (
            "pos" in attrs
            and "source_grid_flat_index" in attrs
            and "original_grid_nd_index" in attrs
        )
    # Edges should have 'distance', 'vector', 'weight', 'angle_2d', 'edge_id'
    for u, v, attrs in G.edges(data=True):
        assert (
            "distance" in attrs
            and "vector" in attrs
            and "weight" in attrs
            and "angle_2d" in attrs
            and "edge_id" in attrs
        )


def test_points_to_hex_bin_ind_nan_handling():
    # Points with NaN should return -1
    data_samples = np.array([[0, 0], [np.nan, 1], [1, np.nan]], dtype=np.float64)
    bin_centers, centers_shape, hex_radius, _, min_x, min_y, _ = hgu._create_hex_grid(
        np.array([[0, 0], [1, 0]], dtype=np.float64), hexagon_width=1.0
    )
    indices = hgu._points_to_hex_bin_ind(
        data_samples, min_x, min_y, hex_radius, centers_shape
    )
    assert indices[1] == -1 and indices[2] == -1
