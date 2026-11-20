"""A track graph's grid does not depend on how its nodes are labelled.

The bin-edge and bin-center nodes added to the graph get integer ids; they
must never equal a label of the graph's own nodes, whatever those are.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment

pytestmark = pytest.mark.unit

LABELLINGS = {
    "0-based": lambda i: i,
    "1-based": lambda i: i + 1,
    "non-contiguous": lambda i: 5 * i + 2,
    "skips 2": lambda i: i if i < 2 else i + 1,
    "offset 100": lambda i: i + 100,
    "strings": lambda i: f"node{i}",
    "tuples": lambda i: (i, "a"),
}


def _fit_track(positions, edges, relabel, edge_spacing=0.0, place_bin_size=2.0):
    """Fit the track with node ``i`` labelled ``relabel(i)``."""
    graph = nx.Graph()
    for node, pos in enumerate(positions):
        graph.add_node(relabel(node), pos=pos)
    for edge_id, (a, b) in enumerate(edges):
        distance = float(np.linalg.norm(np.subtract(positions[a], positions[b])))
        graph.add_edge(relabel(a), relabel(b), distance=distance, edge_id=edge_id)
    return Environment(
        place_bin_size=place_bin_size,
        track_graph=graph,
        edge_order=[(relabel(a), relabel(b)) for a, b in edges],
        edge_spacing=edge_spacing,
    ).fit_place_grid()


def _center_node(environment, linear_position):
    """Id of the bin-center node at ``linear_position``."""
    centers = environment.place_bin_centers_nodes_df_
    (node_id,) = centers.node_id[np.isclose(centers.linear_position, linear_position)]
    return node_id


@pytest.mark.parametrize("labelling", LABELLINGS)
def test_straight_edge_grid_is_the_closed_form_for_any_labels(labelling):
    """A 10 cm edge in 2 cm bins: centers 1, 3, ..., 9, edges 0, 2, ..., 10."""
    relabel = LABELLINGS[labelling]
    environment = _fit_track([(0.0, 0.0), (10.0, 0.0)], [(0, 1)], relabel)

    np.testing.assert_array_equal(
        environment.place_bin_centers_[:, 0], [1.0, 3.0, 5.0, 7.0, 9.0]
    )
    np.testing.assert_array_equal(
        environment.edges_[0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    )
    assert len(environment.place_bin_centers_nodes_df_) == 5
    distance = environment.distance_between_nodes_[_center_node(environment, 1.0)][
        _center_node(environment, 9.0)
    ]
    assert distance == pytest.approx(8.0, rel=1e-12)
    graph = environment.track_graph_with_bin_centers_edges_
    np.testing.assert_array_equal(graph.nodes[relabel(0)]["pos"], (0.0, 0.0))
    np.testing.assert_array_equal(graph.nodes[relabel(1)]["pos"], (10.0, 0.0))


W_TRACK = (
    [(0.0, 0.0), (0.0, 30.0), (15.0, 0.0), (15.0, 30.0), (30.0, 0.0), (30.0, 30.0)],
    [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)],
)
Y_TRACK = (
    [(0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (20.0, -10.0)],
    [(0, 1), (1, 2), (1, 3)],
)


@pytest.mark.parametrize("edge_spacing", [0.0, 15.0])
@pytest.mark.parametrize("track", [Y_TRACK, W_TRACK], ids=["Y", "W"])
@pytest.mark.parametrize(
    "labelling", [name for name in LABELLINGS if name != "0-based"]
)
def test_track_grid_equals_the_grid_of_the_0_based_track(
    labelling, track, edge_spacing
):
    """Bins, interior, bin-center rows and bin-to-bin distances all match.

    The W track adds more than 100 nodes, so with labels offset by 100 the
    numbers given to added nodes run into the labels.
    """
    reference = _fit_track(*track, LABELLINGS["0-based"], edge_spacing)
    environment = _fit_track(*track, LABELLINGS[labelling], edge_spacing)

    np.testing.assert_array_equal(environment.edges_[0], reference.edges_[0])
    np.testing.assert_array_equal(
        environment.place_bin_centers_, reference.place_bin_centers_
    )
    np.testing.assert_array_equal(
        environment.is_track_interior_, reference.is_track_interior_
    )
    assert len(environment.place_bin_centers_nodes_df_) == len(
        reference.place_bin_centers_nodes_df_
    )
    interior_centers = reference.place_bin_centers_[reference.is_track_interior_flat_]
    np.testing.assert_array_equal(
        environment.get_distances_to_interior_bins(interior_centers),
        reference.get_distances_to_interior_bins(interior_centers),
    )
