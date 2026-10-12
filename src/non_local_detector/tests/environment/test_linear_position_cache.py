"""Track linearization is cached by value on the environment."""

import pickle

import networkx as nx
import numpy as np
import pytest
from track_linearization import get_linearized_position

import non_local_detector.environment as environment_module
from non_local_detector.environment import Environment

pytestmark = pytest.mark.unit


@pytest.fixture
def track_environment():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(10.0, 0.0))
    graph.add_node(2, pos=(10.0, 8.0))
    graph.add_edge(0, 1, distance=10.0, edge_id=0)
    graph.add_edge(1, 2, distance=8.0, edge_id=1)
    return Environment(
        environment_name="corner",
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=2.0,
    ).fit_place_grid()


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    along = rng.uniform(0.0, 18.0, 500)
    return np.where(
        along[:, None] < 10.0,
        np.column_stack([along, rng.normal(0, 0.2, 500)]),
        np.column_stack([10.0 + rng.normal(0, 0.2, 500), along - 10.0]),
    )


@pytest.fixture
def count_linearizations(monkeypatch):
    calls = []
    original = environment_module.get_linearized_position

    def counted(*args, **kwargs):
        calls.append(np.shape(args[0])[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(environment_module, "get_linearized_position", counted)
    return calls


def test_cached_linear_position_matches_and_reuses_equal_points(
    track_environment, points, count_linearizations
):
    expected = get_linearized_position(
        points,
        track_environment.track_graph,
        edge_order=track_environment.edge_order,
        edge_spacing=track_environment.edge_spacing,
    ).linear_position.to_numpy()

    first = track_environment._linear_position(points)
    # A different array object with equal values (as each EM E-step makes).
    second = track_environment._linear_position(points.copy())

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert count_linearizations == [500]
    # Callers own their result: changing it does not change the cache.
    second[:] = -1.0
    np.testing.assert_array_equal(track_environment._linear_position(points), expected)


def test_changed_points_refit_and_pickle_do_not_reuse_entries(
    track_environment, points, count_linearizations
):
    track_environment._linear_position(points)
    moved = points.copy()
    moved[0, 0] += 0.5
    track_environment._linear_position(moved)
    assert count_linearizations == [500, 500]

    restored = pickle.loads(pickle.dumps(track_environment))
    assert "_linear_position_cache_" not in restored.__dict__

    track_environment.fit_place_grid()
    track_environment._linear_position(points)
    assert count_linearizations == [500, 500, 500]


def test_cache_evicts_oldest_entries_beyond_its_byte_budget(
    track_environment, points, count_linearizations, monkeypatch
):
    # Room for two 500-point float64 results.
    monkeypatch.setattr(environment_module, "_LINEAR_POSITION_CACHE_BYTES", 8_000)
    shifted = [points + [0.0, offset] for offset in (0.0, 0.01, 0.02)]
    for array in shifted:
        track_environment._linear_position(array)
    track_environment._linear_position(shifted[2])  # newest: cached
    track_environment._linear_position(shifted[0])  # oldest: evicted
    assert count_linearizations == [500, 500, 500, 500]


def test_cache_entries_own_their_memory(track_environment, points):
    """Eviction counts each entry's own bytes, so entries must not be views.

    A view of the linearization's DataFrame keeps all its columns alive,
    several times the bytes the cache counts.
    """
    track_environment._linear_position(points)
    (entry,) = track_environment._linear_position_cache_.values()
    assert entry.base is None
