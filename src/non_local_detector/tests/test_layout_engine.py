import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import layout_engine


def add_edge_distances(graph: nx.Graph) -> nx.Graph:
    for u, v in graph.edges():
        pos_u = np.array(graph.nodes[u]["pos"])
        pos_v = np.array(graph.nodes[v]["pos"])
        graph.edges[u, v]["distance"] = np.linalg.norm(pos_v - pos_u)
    return graph


def test_list_available_layouts():
    layouts = layout_engine.list_available_layouts()
    assert isinstance(layouts, list)
    assert "RegularGrid" in layouts
    assert "Hexagonal" in layouts
    assert "Graph" in layouts
    assert "MaskedGrid" in layouts
    assert "ImageMask" in layouts


def test_get_layout_parameters_regular_grid():
    params = layout_engine.get_layout_parameters("RegularGrid")
    assert "bin_size" in params
    assert "dimension_ranges" in params
    assert params["bin_size"]["annotation"] is not None


def test_create_layout_regular_grid():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
    assert layout.active_mask_ is not None
    assert layout.grid_edges_ is not None
    assert not layout.is_1d


def test_regular_grid_point_to_bin_index_and_neighbors():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.get_bin_neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_regular_grid_bin_area_volume():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "RegularGrid",
        bin_size=2.0,
        data_samples=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    areas = layout.get_bin_area_volume()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers_.shape[0]


def test_create_layout_hexagonal():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
    assert not layout.is_1d


def test_hexagonal_point_to_bin_index_and_neighbors():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.get_bin_neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_hexagonal_bin_area_volume():
    data = np.random.rand(100, 2) * 10
    layout = layout_engine.create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        data_samples=data,
        infer_active_bins=True,
    )
    areas = layout.get_bin_area_volume()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers_.shape[0]


def test_create_layout_masked_grid():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    edges = (
        np.arange(6).astype(float),
        np.arange(6).astype(float),
    )  # Ensure float for grid_edges
    layout = layout_engine.create_layout(
        "MaskedGrid",
        active_mask=mask,
        grid_edges=edges,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.bin_centers_.shape[0] == np.sum(mask)  # Should be 9
    assert layout.connectivity_graph_ is not None
    assert not layout.is_1d


def test_create_layout_image_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = layout_engine.create_layout(
        "ImageMask",
        image_mask=mask,
        bin_size=1.0,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    assert layout.connectivity_graph_ is not None
    assert not layout.is_1d


def test_image_mask_point_to_bin_index_and_neighbors():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = layout_engine.create_layout(
        "ImageMask",
        image_mask=mask,
        bin_size=1.0,
    )
    points = np.array([[2.5, 2.5], [0.5, 0.5]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (2,)
    assert inds[0] >= 0
    neighbors = layout.get_bin_neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_create_layout_graph():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = layout_engine.create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    assert hasattr(layout, "bin_centers_")
    assert layout.bin_centers_.ndim == 2
    # Graph: 0 --1m-- 1 --1m-- 2. Total length 2m. Bin size 0.5m. Expected 4 bins.
    assert layout.bin_centers_.shape[0] == 4
    assert layout.connectivity_graph_ is not None
    assert layout.is_1d


def test_graph_point_to_bin_index_and_neighbors():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = layout_engine.create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    inds = layout.point_to_bin_index(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)
    neighbors = layout.get_bin_neighbors(inds[0])
    assert isinstance(neighbors, list)


def test_graph_bin_area_volume():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = layout_engine.create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    lengths = layout.get_bin_area_volume()
    assert np.allclose(lengths, lengths[0])
    assert lengths.shape[0] == layout.bin_centers_.shape[0]


def test_create_layout_invalid_kind():
    with pytest.raises(ValueError):
        layout_engine.create_layout("NotARealLayout", foo=1)


def test_get_layout_parameters_invalid():
    with pytest.raises(ValueError):
        layout_engine.get_layout_parameters("NotARealLayout")
