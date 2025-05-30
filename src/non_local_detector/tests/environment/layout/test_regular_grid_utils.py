import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.layout.regular_grid_utils import (
    _create_regular_grid,
    _create_regular_grid_connectivity_graph,
    _infer_active_bins_from_regular_grid,
    _points_to_regular_grid_bin_ind,
)


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


def test_create_regular_grid_connectivity_graph_2d_orthogonal():
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


def test_create_regular_grid_connectivity_graph_2d_diagonal():
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
