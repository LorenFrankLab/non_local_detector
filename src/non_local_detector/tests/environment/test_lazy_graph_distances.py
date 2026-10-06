"""Bounded graph-distance queries against the existing exact graph oracle."""

import pickle

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.graph_distances import (
    GraphDistanceBudgetError,
    LazyGraphDistances,
)


def graph_fixture():
    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_weighted_edges_from(
        [(0, 1, 1.5), (1, 2, 2.0), (2, 3, 0.0), (3, 4, 1.0), (1, 4, 8.0), (4, 5, 3.0)],
        weight="distance",
    )
    dense = np.full((7, 7), np.inf)
    for source, distances in nx.shortest_path_length(graph, weight="distance"):
        for target, distance in distances.items():
            dense[source, target] = distance
    return graph, dense


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(None)),
        (slice(1, 4), slice(2, 6)),
        ([0, 2, 6], [5, 3, 6]),
        np.ix_([0, 2, 6], [1, 5]),
        (2, slice(None)),
        (slice(None), 4),
        (2, 3),
        ([-1, 0], [0, -1]),
        ([], []),
        (np.array([True, False, True, False, False, False, False]), slice(2, 4)),
    ],
)
def test_graph_selections_are_exact(key):
    graph, dense = graph_fixture()
    lazy = LazyGraphDistances.from_graph(
        graph, max_dense_bytes=4096, max_workspace_bytes=112
    )
    np.testing.assert_array_equal(lazy[key], dense[key])
    assert lazy.shape == dense.shape
    assert lazy.dtype == dense.dtype


def test_dense_budget_preflight_and_bounded_source_rows(monkeypatch):
    import non_local_detector.graph_distances as module

    graph, dense = graph_fixture()
    lazy = LazyGraphDistances.from_graph(
        graph, max_dense_bytes=64, max_workspace_bytes=112
    )
    calls = []
    original = module.dijkstra

    def counted(*args, **kwargs):
        calls.append(len(np.atleast_1d(kwargs["indices"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "dijkstra", counted)
    with pytest.raises(GraphDistanceBudgetError, match="392"):
        np.asarray(lazy)
    assert calls == []
    np.testing.assert_array_equal(
        lazy[[0, 1, 2, 3], [5, 5, 5, 5]], dense[[0, 1, 2, 3], [5, 5, 5, 5]]
    )
    assert max(calls) <= 2
    assert "_dense" not in vars(lazy)


def test_pickle_retains_sparse_graph_without_a_dense_cache():
    graph, dense = graph_fixture()
    lazy = LazyGraphDistances.from_graph(graph)
    lazy[:2, :2]
    restored = pickle.loads(pickle.dumps(lazy))
    np.testing.assert_array_equal(np.asarray(restored), dense)
    assert restored.storage_nbytes < dense.nbytes


def test_dense_conversion_does_not_allocate_quadratic_index_metadata(monkeypatch):
    import non_local_detector.graph_distances as module

    graph, dense = graph_fixture()
    lazy = LazyGraphDistances.from_graph(
        graph, max_dense_bytes=4096, max_workspace_bytes=112
    )
    unique = np.unique

    def bounded_unique(values, *args, **kwargs):
        assert np.size(values) <= len(graph), "quadratic source-index metadata"
        return unique(values, *args, **kwargs)

    calls = []
    original = module.dijkstra

    def counted(*args, **kwargs):
        calls.append(len(np.atleast_1d(kwargs["indices"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "unique", bounded_unique)
    monkeypatch.setattr(module, "dijkstra", counted)
    np.testing.assert_array_equal(lazy.to_dense(), dense)
    assert max(calls) <= 2
    np.testing.assert_array_equal(
        lazy.to_dense(dtype=np.float32), dense.astype(np.float32)
    )


def test_deferred_environment_matches_topology_and_queries(monkeypatch):
    kwargs = {
        "place_bin_size": 1.0,
        "position_range": ((0.0, 5.0), (0.0, 6.0)),
        "infer_track_interior": False,
    }
    eager = Environment(**kwargs).fit_place_grid(
        np.array([[0.0, 0.0], [5.0, 6.0]]), infer_track_interior=False
    )
    mask = np.ones(eager.centers_shape_, dtype=bool)
    mask[2, 1:-1] = False
    eager = Environment(
        **kwargs, is_track_interior=mask, is_track_interior_=mask
    ).fit_place_grid(np.array([[0.0, 0.0], [5.0, 6.0]]), infer_track_interior=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("eager all-pairs construction was called")

    monkeypatch.setattr(nx, "shortest_path_length", forbidden)
    lazy_env = Environment(
        **kwargs, is_track_interior=mask, is_track_interior_=mask
    ).fit_place_grid(
        np.array([[0.0, 0.0], [5.0, 6.0]]),
        infer_track_interior=False,
        compute_all_pairs_distances=False,
    )
    assert isinstance(lazy_env.distance_between_nodes_, LazyGraphDistances)
    np.testing.assert_array_equal(
        np.asarray(lazy_env.distance_between_nodes_), eager.distance_between_nodes_
    )
    position = eager.place_bin_centers_[[0, 5, 8]]
    np.testing.assert_array_equal(
        lazy_env.get_distances_to_interior_bins(position),
        eager.get_distances_to_interior_bins(position),
    )
    np.testing.assert_array_equal(
        lazy_env.get_manifold_distances(position, position[::-1]),
        eager.get_manifold_distances(position, position[::-1]),
    )


def test_lazy_graph_rejects_insufficient_workspace_before_dijkstra():
    graph, _ = graph_fixture()
    lazy = LazyGraphDistances.from_graph(graph, max_workspace_bytes=8)
    with pytest.raises(GraphDistanceBudgetError, match="workspace"):
        lazy[0, 1]
