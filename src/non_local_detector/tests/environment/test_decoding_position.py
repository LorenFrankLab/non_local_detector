"""Environment.decoding_position maps points into the place grid's coordinates."""

import networkx as nx
import numpy as np
import pytest
from track_linearization import get_linearized_position

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
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=2.0,
    ).fit_place_grid()


@pytest.fixture
def points():
    along = np.linspace(0.0, 18.0, 50)
    return np.where(
        along[:, None] < 10.0,
        np.column_stack([along, np.zeros_like(along)]),
        np.column_stack([np.full_like(along, 10.0), along - 10.0]),
    )


def test_track_graph_linearizes_multi_dimensional_points(track_environment, points):
    expected = get_linearized_position(
        points,
        track_environment.track_graph,
        edge_order=track_environment.edge_order,
        edge_spacing=track_environment.edge_spacing,
    ).linear_position.to_numpy()

    result = track_environment.decoding_position(points)

    assert result.shape == (len(points), 1)
    np.testing.assert_array_equal(result[:, 0], expected)


@pytest.mark.parametrize("shape", [(50, 1), (50,)])
def test_track_graph_returns_already_linear_points_unchanged(track_environment, shape):
    linear = np.linspace(0.0, 20.0, 50).reshape(shape)

    assert track_environment.decoding_position(linear) is linear


def test_track_graph_empty_points_give_one_column(track_environment):
    result = track_environment.decoding_position(np.empty((0, 2)))

    assert result.shape == (0, 1)
    assert result.dtype == np.float64


@pytest.mark.parametrize("n_points", [0, 50])
def test_without_track_graph_points_are_unchanged(points, n_points):
    environment = Environment(place_bin_size=2.0).fit_place_grid(points)
    sample = points[:n_points]

    assert environment.decoding_position(sample) is sample
