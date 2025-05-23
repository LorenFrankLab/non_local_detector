import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.graph_utils import (
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)
from non_local_detector.environment.utils import get_centers


@pytest.fixture
def simple_graph():
    # Linear graph: 0 -- 1 -- 2
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_node(2, pos=(2.0, 0.0))
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    # Add 'distance' attribute for _project_1d_to_2d
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]["pos"])
        pos_v = np.array(G.nodes[v]["pos"])
        G.edges[u, v]["distance"] = np.linalg.norm(pos_v - pos_u)
    return G


def test_get_graph_bins_basic(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    bin_size = 0.5
    edge_spacing = 0.0

    bin_centers, bin_edges_tuple, active_mask, edge_ids = _get_graph_bins(
        simple_graph, edge_order, edge_spacing, bin_size
    )

    bin_edges = bin_edges_tuple[0]
    # There should be 4 bins: [0,0.5,1,1.5,2]
    np.testing.assert_allclose(bin_edges, [0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(bin_centers, [0.25, 0.75, 1.25, 1.75])
    assert np.all(active_mask)
    assert len(edge_ids) == 4
    assert set(edge_ids) == {0, 1}


def test_get_graph_bins_with_gap(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    bin_size = 0.5
    edge_spacing = 0.5

    bin_centers, bin_edges_tuple, active_mask, edge_ids = _get_graph_bins(
        simple_graph, edge_order, edge_spacing, bin_size
    )

    bin_edges = bin_edges_tuple[0]
    # Should be bins for [0,0.5,1.0] (edge 0-1), then gap, then [1.5,2.0] (edge 1-2)
    # So bin_edges: [0,0.5,1.0,1.5,2.0,2.5]
    np.testing.assert_allclose(bin_edges, [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    # active_mask: 2 bins (edge 0-1), 1 gap, 2 bins (edge 1-2)
    assert np.array_equal(active_mask, [True, True, False, True, True])
    assert len(edge_ids) == 4  # Only active bins have edge_ids


def test_get_graph_bins_invalid_bin_size(simple_graph):
    with pytest.raises(ValueError):
        _get_graph_bins(simple_graph, [(0, 1)], 0.0, 0.0)


def test_get_graph_bins_invalid_edge(simple_graph):
    with pytest.raises(ValueError):
        _get_graph_bins(simple_graph, [(0, 3)], 0.0, 0.5)


def test_get_graph_bins_invalid_gap_length(simple_graph):
    with pytest.raises(ValueError):
        _get_graph_bins(simple_graph, [(0, 1), (1, 2)], [0.5, 0.5], 0.5)


def test_project_1d_to_2d_basic(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    # Add 'distance' attribute if not present
    for e in edge_order:
        if "distance" not in simple_graph.edges[e]:
            pos0 = np.array(simple_graph.nodes[e[0]]["pos"])
            pos1 = np.array(simple_graph.nodes[e[1]]["pos"])
            simple_graph.edges[e]["distance"] = np.linalg.norm(pos1 - pos0)
    # Positions: 0.0 (start), 0.5 (mid first edge), 1.0 (junction), 1.5 (mid second edge), 2.0 (end)
    positions = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    coords = _project_1d_to_2d(positions, simple_graph, edge_order)
    expected = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
        ]
    )
    np.testing.assert_allclose(coords, expected)


def test_project_1d_to_2d_with_gap(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    for e_idx, e in enumerate(edge_order):
        if "distance" not in simple_graph.edges[e]:
            pos0 = np.array(simple_graph.nodes[e[0]]["pos"])
            pos1 = np.array(simple_graph.nodes[e[1]]["pos"])
            simple_graph.edges[e]["distance"] = np.linalg.norm(pos1 - pos0)
    positions = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
    coords = _project_1d_to_2d(positions, simple_graph, edge_order, edge_spacing=0.5)
    # The gap is between 1.0 and 1.5, so 1.25 should be projected to the end of first edge or start of second
    expected = np.array(
        [
            [0.0, 0.0],  # lin_pos = 0.0
            [0.5, 0.0],  # lin_pos = 0.5
            [1.0, 0.0],  # lin_pos = 1.0 (end of edge 1)
            [1.0, 0.0],  # lin_pos = 1.25 (in gap, projects to end of edge 1)
            [1.0, 0.0],  # lin_pos = 1.5 (start of edge 2)
            [1.5, 0.0],  # lin_pos = 2.0 (mid of edge 2)
            [2.0, 0.0],  # lin_pos = 2.5 (end of edge 2)
        ]
    )
    np.testing.assert_allclose(coords, expected, atol=1e-7)


def test_project_1d_to_2d_nan(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    for e in edge_order:
        if "distance" not in simple_graph.edges[e]:
            pos0 = np.array(simple_graph.nodes[e[0]]["pos"])
            pos1 = np.array(simple_graph.nodes[e[1]]["pos"])
            simple_graph.edges[e]["distance"] = np.linalg.norm(pos1 - pos0)
    positions = np.array([0.0, np.nan, 2.0])
    coords = _project_1d_to_2d(positions, simple_graph, edge_order)
    assert np.isnan(coords[1]).all()
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[2], [2.0, 0.0])


def test_project_1d_to_2d_invalid_shape(simple_graph):
    edge_order = [(0, 1), (1, 2)]
    with pytest.raises(ValueError):
        _project_1d_to_2d(np.ones((2, 2)), simple_graph, edge_order)


def test_find_bin_for_linear_position_basic():
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.array([0.1, 1.5, 2.9])
    result = _find_bin_for_linear_position(positions, bin_edges)
    assert np.array_equal(result, [0, 1, 2])


def test_find_bin_for_linear_position_scalar():
    bin_edges = np.array([0.0, 1.0, 2.0])
    assert _find_bin_for_linear_position(0.5, bin_edges) == 0
    assert _find_bin_for_linear_position(1.5, bin_edges) == 1
    assert _find_bin_for_linear_position(-1.0, bin_edges) == -1
    assert _find_bin_for_linear_position(2.0, bin_edges) == -1


def test_find_bin_for_linear_position_with_active_mask():
    bin_edges = np.array(
        [0.0, 1.0, 2.0, 3.0]
    )  # Bins: [0,1), [1,2), [2,3) -> Indices 0, 1, 2
    active_mask = np.array([True, False, True])  # Bins 0 and 2 are active
    positions = np.array([0.5, 1.5, 2.5])  # Points in bin 0, bin 1 (inactive), bin 2

    # 1.5 falls into inactive bin, should be handled
    result = _find_bin_for_linear_position(positions, bin_edges, active_mask)
    # Expected: 0.5 -> bin 0 (active)
    #           1.5 -> bin 1 (inactive), should now return -1
    #           2.5 -> bin 2 (active)
    assert np.array_equal(result, [0, -1, 2])
    # Test scalar that is clearly in an inactive bin and not on an edge
    assert _find_bin_for_linear_position(1.5, bin_edges, active_mask) == -1
    # Test scalar on an edge that might be remapped (e.g. point 1.0, if bin 0 is active and bin 1 is inactive)
    # With current logic, 1.0 is start of bin 1. If bin 1 is inactive, and 1.0 is an edge, it's shifted to bin 0.
    active_mask_edge_case = np.array([True, False, True])
    assert _find_bin_for_linear_position(1.0, bin_edges, active_mask_edge_case) == 0

    active_mask_edge_case_2 = np.array([False, True, True])
    # Point 1.0 is start of bin 1 (active). No shift needed.
    assert _find_bin_for_linear_position(1.0, bin_edges, active_mask_edge_case_2) == 1


def test_find_bin_for_linear_position_invalid_active_mask():
    bin_edges = np.array([0.0, 1.0, 2.0])
    active_mask = np.array([True])  # Wrong length
    with pytest.raises(ValueError):
        _find_bin_for_linear_position(0.5, bin_edges, active_mask)
