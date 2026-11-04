"""``Environment.fit_place_grid_from_edges`` rebuilds a fitted open field.

A saved model keeps an open-field environment's bin edges and interior mask
instead of the position data the grid was fitted on. Fitting a copy of the
configuration from those must give the same grid as the original fit.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.graph_distances import LazyGraphDistances

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(0)


def _assert_same_grid(rebuilt, original):
    """Every PlaceGrid field of ``rebuilt`` equals that of ``original``."""
    for name in ("place_bin_edges_", "place_bin_centers_", "is_track_interior_"):
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(original, name))
    assert rebuilt.centers_shape_ == original.centers_shape_
    assert rebuilt.infer_track_interior_ == original.infer_track_interior_
    assert len(rebuilt.edges_) == len(original.edges_)
    for a, b in zip(rebuilt.edges_, original.edges_, strict=True):
        np.testing.assert_array_equal(a, b)
    if original.is_track_boundary_ is None:
        assert rebuilt.is_track_boundary_ is None
    else:
        np.testing.assert_array_equal(
            rebuilt.is_track_boundary_, original.is_track_boundary_
        )
    assert list(rebuilt.track_graphDD.edges(data=True)) == list(
        original.track_graphDD.edges(data=True)
    )
    assert type(rebuilt.distance_between_nodes_) is type(
        original.distance_between_nodes_
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt.distance_between_nodes_),
        np.asarray(original.distance_between_nodes_),
    )


@pytest.mark.parametrize(
    "position",
    [
        RNG.uniform(0.0, 50.0, size=(400, 1)),
        RNG.uniform(0.0, 30.0, size=(800, 2)),
    ],
    ids=["1d", "2d"],
)
@pytest.mark.parametrize("all_pairs", [True, False])
def test_open_field_rebuilt_from_its_edges_and_mask_matches_the_fit(
    position, all_pairs
):
    original = Environment(place_bin_size=4.0).fit_place_grid(
        position, compute_all_pairs_distances=all_pairs
    )

    rebuilt = Environment(**original.get_params()).fit_place_grid_from_edges(
        original.edges_,
        original.is_track_interior_,
        infer_track_interior=original.infer_track_interior_,
        compute_all_pairs_distances=all_pairs,
    )

    _assert_same_grid(rebuilt, original)
    assert isinstance(rebuilt.distance_between_nodes_, LazyGraphDistances) is (
        not all_pairs
    )


def test_rebuild_refuses_a_track_graph_environment():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    environment = Environment(
        place_bin_size=1.0, track_graph=graph, edge_order=[(0, 1)], edge_spacing=0.0
    )

    with pytest.raises(ValidationError, match="track_graph"):
        environment.fit_place_grid_from_edges([np.arange(11.0)], np.ones(10, bool))


def test_rebuild_refuses_a_mask_of_another_shape():
    with pytest.raises(ValidationError, match="shape"):
        Environment().fit_place_grid_from_edges(
            [np.arange(5.0), np.arange(4.0)], np.ones((3, 4), dtype=bool)
        )


def test_rebuild_refuses_edges_that_do_not_increase():
    with pytest.raises(ValidationError, match="increasing"):
        Environment().fit_place_grid_from_edges(
            [np.array([0.0, 2.0, 1.0])], np.ones(2, dtype=bool)
        )
