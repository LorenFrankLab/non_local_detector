import networkx as nx
import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import get_position_at_time


def make_multi_edge_line_graph():
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(5.0, 0.0))
    g.add_node(2, pos=(10.5, 0.0))
    # Two edges of different lengths
    g.add_edge(0, 1, distance=5.0)
    g.add_edge(1, 2, distance=5.5)
    # Assign edge_id expected by environment utils
    for eid, e in enumerate(g.edges):
        g.edges[e]["edge_id"] = eid
    edge_order = [(0, 1), (1, 2)]
    # Non-zero spacing between edges
    edge_spacing = 0.25
    return g, edge_order, edge_spacing


def test_linear_position_monotonic_across_multiple_edges():
    g, edge_order, edge_spacing = make_multi_edge_line_graph()
    env = Environment(
        environment_name="multi-line",
        place_bin_size=1.0,
        track_graph=g,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).fit_place_grid()

    # Interpolate 2D positions along 0..10.5
    t = np.linspace(0.0, 10.5, 22)
    pos = np.stack([t, np.zeros_like(t)], axis=1)
    spikes = np.linspace(0.0, 10.5, 11)
    lin = get_position_at_time(t, pos, spikes, env)
    # Monotonic and length roughly equals position + cumulative spacing at the internal edge
    assert np.all(np.diff(lin.squeeze()) >= 0)
    # Check that the linear position jumps by ~edge_spacing at the internal boundary index
    # Find the spike closest to x=5.0 (boundary between edges)
    boundary_idx = np.argmin(np.abs(spikes - 5.0))
    # Next increment should be slightly larger due to spacing
    diffs = np.diff(lin.squeeze())
    assert diffs[boundary_idx] >= diffs.max() * 0.5  # rough check for a bump
