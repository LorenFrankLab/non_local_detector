import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.layout.helpers.regular_grid import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
    _infer_active_bins_from_regular_grid,
    _points_to_regular_grid_bin_ind,
)
from non_local_detector.environment.layout.helpers.utils import get_centers, get_n_bins


def test_create_regular_grid_2d_basic():
    # Simple 2D data, no boundary bins
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    bin_size = 1.0
    edges, centers, shape = _create_regular_grid(data, bin_size)
    assert len(edges) == 2
    assert all(isinstance(e, np.ndarray) for e in edges)
    assert centers.shape[1] == 2
    assert np.prod(shape) == centers.shape[0]
    # Check that the grid covers the data
    assert np.all(data.min(axis=0) >= [e[0] for e in edges])
    assert np.all(data.max(axis=0) <= [e[-1] for e in edges])


def test_create_regular_grid_with_dimension_range_and_boundary():
    # 1D grid, explicit range, with boundary bins
    edges, centers, shape = _create_regular_grid(
        data_samples=None,
        bin_size=1.0,
        dimension_range=[(0, 2)],
        add_boundary_bins=True,
    )
    # Should have 4 bins (2 core + 2 boundary)
    assert shape == (4,)
    assert edges[0].shape == (5,)
    # Centers should be within the extended range
    assert np.all(centers[:, 0] >= edges[0][0])
    assert np.all(centers[:, 0] <= edges[0][-1])


def test_infer_active_bins_from_regular_grid_simple():
    # 2D: points in a 3x3 grid, threshold=0
    data = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    edges = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    mask = _infer_active_bins_from_regular_grid(data, edges)
    assert mask.shape == (3, 3)
    # Only diagonal bins should be active
    assert np.all(mask.diagonal())
    assert np.sum(mask) == 3


def test_infer_active_bins_with_threshold_and_morphology():
    # 2D: points in a cross, threshold=1, dilate
    data = np.array(
        [
            [1.5, 0.5],
            [1.5, 1.5],
            [1.5, 2.5],  # vertical
            [0.5, 1.5],
            [2.5, 1.5],  # horizontal
        ]
    )
    edges = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    mask = _infer_active_bins_from_regular_grid(
        data, edges, bin_count_threshold=0, dilate=True
    )
    # Dilation should expand the cross
    assert mask.sum() > 5
    # All original cross points should be active
    for pt in data:
        i, j = int(pt[0]), int(pt[1])
        assert mask[i, j]


def test_infer_active_bins_boundary_exists():
    # 1D: boundary bins should be set to False
    data = np.array([[0.5], [1.5], [2.5]])
    edges = (np.array([0, 1, 2, 3]),)
    mask = _infer_active_bins_from_regular_grid(data, edges, boundary_exists=True)
    assert mask.shape == (3,)
    assert not mask[0]
    assert not mask[-1]
    assert mask[1]


def test_create_regular_grid_connectivity_2d_orthogonal():
    # 2x2 grid, all bins active, orthogonal connections
    centers_list = [0.5, 1.5]
    mesh = np.meshgrid(centers_list, centers_list, indexing="ij")
    bin_centers = np.stack([m.ravel() for m in mesh], axis=1)
    active_mask = np.ones((2, 2), dtype=bool)
    graph = _create_regular_grid_connectivity_graph(bin_centers, active_mask, (2, 2))
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 4
    # Each node should have up to 2 orthogonal neighbors (except corners)
    degrees = [d for n, d in graph.degree()]
    assert set(degrees) == {2}
    # Check edge attributes
    for u, v, d in graph.edges(data=True):
        assert "distance" in d
        assert "weight" in d
        assert "vector" in d
        assert "edge_id" in d


def test_create_regular_grid_connectivity_2d_diagonal():
    # 2x2 grid, all bins active, diagonal connections
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    centers_list = [0.5, 1.5]
    mesh = np.meshgrid(centers_list, centers_list, indexing="ij")
    bin_centers = np.stack([m.ravel() for m in mesh], axis=1)
    active_mask = np.ones((2, 2), dtype=bool)
    graph = _create_regular_grid_connectivity_graph(
        bin_centers, active_mask, (2, 2), connect_diagonal=True
    )
    assert graph.number_of_edges() > 4  # Should include diagonals
    # Check angle_2d attribute exists
    for u, v, d in graph.edges(data=True):
        assert "angle_2d" in d


def test_points_to_regular_grid_bin_ind_basic():
    # 2D grid, no active mask
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    shape = (2, 2)
    points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [np.nan, 1]])
    inds = _points_to_regular_grid_bin_ind(points, edges, shape)
    # First two points are in-bounds, third is out, fourth is nan
    assert inds[0] == 0
    assert inds[1] == 3
    assert inds[2] == -1
    assert inds[3] == -1


def test_points_to_regular_grid_bin_ind_with_active_mask():
    # 2D grid, only one bin active
    edges = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    shape = (2, 2)
    points = np.array([[0.5, 0.5], [1.5, 1.5]])
    active_mask = np.zeros((2, 2), dtype=bool)
    active_mask[1, 1] = True
    inds = _points_to_regular_grid_bin_ind(points, edges, shape, active_mask)
    # Only the second point is in the active bin
    assert inds[0] == -1
    assert inds[1] == 0


def test_points_to_regular_grid_bin_ind_dimensionality_mismatch():
    # Should return all -1 if dimensions don't match
    edges = (np.array([0, 1, 2]),)
    shape = (2,)
    points = np.array([[0.5, 0.5]])
    inds = _points_to_regular_grid_bin_ind(points, edges, shape)
    assert np.all(inds == -1)


def test_create_and_connectivity_for_3d_grid():
    """
    Build a small 3D dataset (points forming the 8 corners of a unit cube)
    and verify:

      1) _create_regular_grid produces exactly one bin along each dimension
      2) bin_centers is a single point at (0.5, 0.5, 0.5)
      3) The 3D connectivity graph has exactly 1 node and 0 edges
      4) Per-dimension bin sizes are [1.0, 1.0, 1.0]
    """
    # 1) Eight corner points of [0,1]^3
    cube_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    # 2) Create the grid with bin_size = 1.0 (no boundary bins)
    edges_tuple, bin_centers, centers_shape = _create_regular_grid(
        data_samples=cube_pts,
        bin_size=1.0,
        dimension_range=None,
        add_boundary_bins=False,
    )

    # We expect 3 dimensions, each edge array = [0.0, 1.0]
    assert len(edges_tuple) == 3
    for dim_edges in edges_tuple:
        assert isinstance(dim_edges, np.ndarray)
        assert dim_edges.shape == (2,)
        assert np.allclose(dim_edges, np.array([0.0, 1.0], dtype=float))

    # Confirm get_n_bins returns [1,1,1] given points at the exact endpoints
    n_bins = get_n_bins(
        cube_pts,
        np.array([1.0, 1.0, 1.0]),
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )
    assert np.array_equal(n_bins, np.array([1, 1, 1], dtype=int))

    # bin_centers: exactly one point at (0.5, 0.5, 0.5)
    assert bin_centers.shape == (1, 3)
    assert np.allclose(bin_centers, np.array([[0.5, 0.5, 0.5]]))

    # centers_shape should be (1, 1, 1)
    assert centers_shape == (1, 1, 1)

    # 3) Infer the active_mask without any morphological ops.
    #    Signature: _infer_active_bins_from_regular_grid(data_samples, edges, ...)
    active_mask = _infer_active_bins_from_regular_grid(
        data_samples=cube_pts,
        edges=edges_tuple,
        close_gaps=False,
        fill_holes=False,
        dilate=False,
        bin_count_threshold=0,
        boundary_exists=False,
    )
    # Should be shape (1, 1, 1), with exactly one True
    assert active_mask.ndim == 3
    assert active_mask.shape == (1, 1, 1)
    assert active_mask[0, 0, 0]

    # Build the connectivity graph for this 1×1×1 grid:
    conn_graph = _create_regular_grid_connectivity_graph(
        full_grid_bin_centers=bin_centers,
        active_mask_nd=active_mask,
        grid_shape=centers_shape,
        connect_diagonal=False,
    )
    # There’s exactly 1 active bin → 1 node, 0 edges
    assert conn_graph.number_of_nodes() == 1
    assert conn_graph.number_of_edges() == 0

    # 4) Compute per-dimension bin sizes by differencing each edge array:
    diffs = [float(np.diff(e)[0]) for e in edges_tuple]
    assert diffs == [1.0, 1.0, 1.0]


def test_points_to_regular_grid_bin_ind_3d_and_oob():
    """
    Test that `_points_to_regular_grid_bin_ind` correctly handles 3D points:

      - (0.5,0.5,0.5) should map to some valid bin index ≥ 0.
      - (2.0,2.0,2.0) should map to -1 (out-of-bounds).
      - (NaN, 0.5, 0.5) should map to -1.
    """
    # Edges for a single bin in each of 3 dims:
    edges_tuple = (
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    # Only one bin total (shape = (1,1,1)), flat index = 0
    grid_shape = (1, 1, 1)
    active_mask = np.ones(grid_shape, dtype=bool)

    pts = np.array(
        [
            [0.5, 0.5, 0.5],  # inside
            [2.0, 2.0, 2.0],  # outside
            [np.nan, 0.5, 0.5],  # NaN → out-of-bounds
        ],
        dtype=float,
    )

    # Now call with the correct signature: (points, grid_edges, grid_shape, active_mask)
    idxs = _points_to_regular_grid_bin_ind(pts, edges_tuple, grid_shape, active_mask)

    assert isinstance(idxs, np.ndarray)
    assert idxs.shape == (3,)
    # The first point should map to a valid bin index (≥ 0). The others → -1.
    assert idxs[0] >= 0
    assert idxs[1] == -1
    assert idxs[2] == -1
