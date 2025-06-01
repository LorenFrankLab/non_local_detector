import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.layout.helpers.graph import (
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)


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


def test_get_graph_bins_and_project_on_T_junction():
    """
    Build a “T”-shaped graph:

         (0)
          |
         (1)
        /   \
      (2)   (3)

    Node positions:
      0: (0, +1)
      1: (0,  0)
      2: (-1, -1)
      3: (+1, -1)

    Edges (in this order):
      e0: (0,1)   length = 1.0
      e1: (1,2)   length = sqrt(2)
      e2: (1,3)   length = sqrt(2)

    Using bin_size = 1.0:
      - e0 yields 1 active bin   (length 1 → ceil(1/1) = 1)
      - e1 yields 2 active bins  (length ≈1.414 → ceil(1.414/1)=2)
      - e2 yields 2 active bins  (same)
    Total active bins = 1 + 2 + 2 = 5.  With edge_spacing = 0.0, there are no “gap” bins.
    We then project these 1D centers back to 2D and verify each lies exactly on its segment.
    """
    # 1) Build the T-shaped graph:
    G = nx.Graph()
    G.add_node(0, pos=(0.0, +1.0))
    G.add_node(1, pos=(0.0, 0.0))
    G.add_node(2, pos=(-1.0, -1.0))
    G.add_node(3, pos=(+1.0, -1.0))
    # Add 'distance' explicitly so _project_1d_to_2d can use it:
    G.add_edge(0, 1, distance=1.0)  # e0
    G.add_edge(1, 2, distance=np.sqrt(2))  # e1
    G.add_edge(1, 3, distance=np.sqrt(2))  # e2

    # 2) Define edge_order in the sequence we want to linearize:
    edge_order = [(0, 1), (1, 2), (1, 3)]
    bin_size = 1.0
    edge_spacing = 0.0  # no artificial gaps

    # 3) Call _get_graph_bins and unpack exactly four outputs:
    #    - bin_centers_1d: array of all bin centers along the linear track (length = 5)
    #    - bin_edges_tuple: a 1-tuple whose only element is the 1D array of bin edges
    #    - active_mask: boolean mask of length 5 indicating which bins lie on actual edges
    #    - edge_ids: integer array (length 5) saying “for each active bin, which edge idx (0,1,2) it belongs to”
    bin_centers_1d, bin_edges_tuple, active_mask, edge_ids = _get_graph_bins(
        G, edge_order, edge_spacing, bin_size
    )
    # Sanity checks:
    assert active_mask.dtype == np.bool_
    # Exactly five active bins in total:
    assert np.sum(active_mask) == 5
    assert bin_centers_1d.shape[0] == 5

    # Count how many bins landed on each edge (e0→edge_id 0, e1→edge_id 1, e2→edge_id 2):
    counts = {0: 0, 1: 0, 2: 0}
    for eid in edge_ids:
        counts[int(eid)] += 1
    # e0 should have 1 bin; e1 → 2 bins; e2 → 2 bins
    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[2] == 2

    # 4) Extract the sole array of bin edges from the returned tuple:
    bin_edges = bin_edges_tuple[0]
    # There are (total active bins) + 1 = 6 edges when spacing=0, but we only assert it has length 6:
    assert bin_edges.shape[0] == 6

    # 5) Call _project_1d_to_2d using the actual signature:
    #    _project_1d_to_2d(linear_positions, graph, edge_order, edge_spacing)
    projected_nd = _project_1d_to_2d(bin_centers_1d, G, edge_order, edge_spacing)
    # Shape should be (5,2)
    assert projected_nd.shape == (5, 2)

    # 6) Verify: each of the 5 points lies exactly on one of the three line segments.
    #    To do this, we compute the distance from projected_nd[i] to the segment (u→v).
    def point_to_segment_dist(pt, a, b):
        """Euclidean distance from `pt` to line‐segment `ab`."""
        ap = np.array(pt) - np.array(a)
        ab = np.array(b) - np.array(a)
        # Project ap onto ab, then clamp t to [0,1]
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        closest = np.array(a) + t * ab
        return np.linalg.norm(np.array(pt) - closest)

    for idx in range(5):
        eid = int(edge_ids[idx])  # which edge this bin belongs to
        (u, v) = edge_order[eid]  # node IDs of that edge
        a = G.nodes[u]["pos"]
        b = G.nodes[v]["pos"]
        dist = point_to_segment_dist(projected_nd[idx], a, b)
        # Must lie exactly on the segment (within floating‐point tolerance)
        assert pytest.approx(0.0, abs=1e-6) == dist

    # 7) Lastly, test a few key _find_bin_for_linear_position edge cases:
    #    (a) Position exactly at bin_edges[0] → should map to bin 0
    #    (b) Position exactly at bin_edges[1] → is the boundary between bins 0 and 1;
    #        since both are active_mask=True, it should map “backwards” to bin 0.
    #    (c) Position just past the final edge → should return -1.
    # 7a) Scalar at bin_edges[0]:
    first_edge = float(bin_edges[0])
    idx0 = _find_bin_for_linear_position(first_edge, bin_edges, active_mask)
    assert isinstance(idx0, int) and idx0 == 0

    # 7b) Scalar exactly at bin_edges[1] (shared boundary):
    mid_edge = float(bin_edges[1])
    idx_mid = _find_bin_for_linear_position(mid_edge, bin_edges, active_mask)
    # Bins 0 and 1 are both active; by np.searchsorted logic, it should assign to bin 1
    assert isinstance(idx_mid, int) and idx_mid == 1

    # 7c) Just beyond the final edge:
    last_edge = float(bin_edges[-1])
    idx_out = _find_bin_for_linear_position(last_edge + 0.1, bin_edges, active_mask)
    assert idx_out == -1
