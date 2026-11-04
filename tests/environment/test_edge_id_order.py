"""A track graph's ``edge_id`` values are labels, not positions in ``graph.edges``.

NetworkX lists a graph's edges node by node (``graph.edges``), which is not
the order the edges were added in when an edge joins a node that comes earlier
than one already listed. Hand-built graphs often number ``edge_id`` by their
own edge list, so ``edge_id`` and the position in ``graph.edges`` differ. The
grid, the linearization of the animal's position, the random walk and the
analysis helpers must give the same physical result for any unique labels:
here, the same as for the same graph labeled by ``graph.edges`` position
(``track_linearization.make_track_graph``'s convention), and the linear
positions an edge's offset and length predict.
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from non_local_detector.analysis.distance1D import (
    _get_MAP_estimate_2d_position_edges,
)
from non_local_detector.continuous_state_transitions import RandomWalk
from non_local_detector.environment import Environment
from non_local_detector.models import SortedSpikesDecoder
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = pytest.mark.unit

PLACE_BIN_SIZE = 2.0

# name: (node positions, edges in the order added, edge_order, edge_spacing).
# In each, graph.edges lists the edges in a different order than they were
# added, so numbering edge_id by the edge list differs from graph.edges.
TRACKS = {
    # graph.edges: (0,1), (1,2), (1,4), (2,3), (4,5)
    "w_track": (
        {
            0: (60.0, 0.0),
            1: (60.0, 120.0),
            2: (0.0, 120.0),
            3: (0.0, 0.0),
            4: (120.0, 120.0),
            5: (120.0, 0.0),
        },
        [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)],
        [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)],
        10.0,
    ),
    # graph.edges: (0,1), (1,2), (1,3)
    "t_maze": (
        {0: (50.0, 0.0), 1: (50.0, 60.0), 2: (0.0, 60.0), 3: (100.0, 60.0)},
        [(1, 2), (0, 1), (1, 3)],
        [(0, 1), (1, 2), (1, 3)],
        15.0,
    ),
    # Center node 3; graph.edges: (0,3), (1,3), (2,3). edge_order runs each
    # arm outward from the center, against graph.edges' orientation.
    "y_maze": (
        {0: (0.0, 50.0), 1: (43.0, -25.0), 2: (-43.0, -25.0), 3: (0.0, 0.0)},
        [(3, 1), (3, 0), (3, 2)],
        [(3, 0), (3, 1), (3, 2)],
        20.0,
    ),
    # Edges added from the far end; graph.edges: (0,1), (1,2), (2,3).
    "linear": (
        {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (80.0, 0.0), 3: (130.0, 0.0)},
        [(2, 3), (1, 2), (0, 1)],
        [(0, 1), (1, 2), (2, 3)],
        0.0,
    ),
}

# How each convention numbers an edge: from its index in the edge list, its
# index in graph.edges, or arbitrary (non-contiguous) labels.
CONVENTIONS = {
    "graph_edges_index": lambda edge_list, graph_edges: {
        edge: i for i, edge in enumerate(graph_edges)
    },
    "edge_list_index": lambda edge_list, graph_edges: {
        edge: i for i, edge in enumerate(edge_list)
    },
    "sparse_labels": lambda edge_list, graph_edges: {
        edge: 10 * i + 3 for i, edge in enumerate(reversed(edge_list))
    },
}


def _make_graph(track: str, convention: str) -> nx.Graph:
    node_positions, edge_list, _, _ = TRACKS[track]
    graph = nx.Graph()
    for node, pos in node_positions.items():
        graph.add_node(node, pos=pos)
    for node1, node2 in edge_list:
        distance = float(
            np.linalg.norm(np.subtract(node_positions[node1], node_positions[node2]))
        )
        graph.add_edge(node1, node2, distance=distance)
    edge_ids = CONVENTIONS[convention](edge_list, list(graph.edges))
    for edge, edge_id in edge_ids.items():
        graph.edges[edge]["edge_id"] = edge_id
    return graph


def _make_environment(track: str, convention: str) -> Environment:
    _, _, edge_order, edge_spacing = TRACKS[track]
    return Environment(
        place_bin_size=PLACE_BIN_SIZE,
        track_graph=_make_graph(track, convention),
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).fit_place_grid()


def _edge_of_label(graph: nx.Graph) -> dict[int, frozenset]:
    return {edge_id: frozenset((u, v)) for u, v, edge_id in graph.edges(data="edge_id")}


def _points_along_edges(track: str) -> tuple[np.ndarray, np.ndarray]:
    """Points 1/4, 1/2 and 3/4 along each edge and their expected linear position.

    Returns
    -------
    points : np.ndarray, shape (3 * n_edges, 2)
    expected_linear_position : np.ndarray, shape (3 * n_edges,)
    """
    node_positions, _, edge_order, edge_spacing = TRACKS[track]
    points, expected = [], []
    start = 0.0
    for node1, node2 in edge_order:
        pos1, pos2 = (
            np.asarray(node_positions[node1]),
            np.asarray(node_positions[node2]),
        )
        length = float(np.linalg.norm(pos2 - pos1))
        for fraction in (0.25, 0.5, 0.75):
            points.append(pos1 + fraction * (pos2 - pos1))
            expected.append(start + fraction * length)
        start += length + edge_spacing
    return np.asarray(points), np.asarray(expected)


@pytest.fixture(scope="module")
def reference_environments() -> dict[str, Environment]:
    """Each track labeled by graph.edges position, the convention that always worked."""
    return {track: _make_environment(track, "graph_edges_index") for track in TRACKS}


@pytest.mark.parametrize("convention", list(CONVENTIONS))
@pytest.mark.parametrize("track", list(TRACKS))
def test_grid_is_independent_of_edge_id_labels(
    track, convention, reference_environments
):
    """The grid has one interior bin per place_bin_size of track, on the right edges."""
    environment = _make_environment(track, convention)
    reference = reference_environments[track]
    edge_order = TRACKS[track][2]
    graph = environment.track_graph

    n_interior = sum(
        int(np.ceil(graph.edges[edge]["distance"] / PLACE_BIN_SIZE))
        for edge in edge_order
    )
    assert environment.is_track_interior_.sum() == n_interior
    np.testing.assert_array_equal(
        environment.place_bin_centers_, reference.place_bin_centers_
    )
    np.testing.assert_array_equal(
        environment.is_track_interior_, reference.is_track_interior_
    )
    np.testing.assert_array_equal(
        environment.place_bin_edges_, reference.place_bin_edges_
    )

    nodes_df = environment.place_bin_centers_nodes_df_
    reference_nodes_df = reference.place_bin_centers_nodes_df_
    assert len(nodes_df) == environment.place_bin_centers_.shape[0]
    columns = ["linear_position", "x_position", "y_position"]
    np.testing.assert_allclose(
        nodes_df[columns].to_numpy(), reference_nodes_df[columns].to_numpy()
    )
    # Each bin is labeled with its own edge's edge_id (-1 in the gaps).
    on_edge = np.asarray(nodes_df.edge_id) != -1
    np.testing.assert_array_equal(on_edge, np.asarray(reference_nodes_df.edge_id) != -1)
    edge_of = _edge_of_label(graph)
    reference_edge_of = _edge_of_label(reference.track_graph)
    assert [edge_of[i] for i in nodes_df.edge_id[on_edge]] == [
        reference_edge_of[i] for i in reference_nodes_df.edge_id[on_edge]
    ]
    # The animal at a bin's 2-D center linearizes to that bin.
    np.testing.assert_allclose(
        environment.decoding_position(
            nodes_df.loc[on_edge, ["x_position", "y_position"]].to_numpy()
        ).squeeze(),
        nodes_df.linear_position[on_edge],
        atol=1e-9,
    )


@pytest.mark.parametrize("convention", list(CONVENTIONS))
@pytest.mark.parametrize("track", list(TRACKS))
def test_position_linearizes_onto_its_own_edge(track, convention):
    """A point a fraction f along an edge is at that edge's offset + f * length."""
    environment = _make_environment(track, convention)
    points, expected = _points_along_edges(track)

    np.testing.assert_allclose(
        environment.decoding_position(points).squeeze(), expected, atol=1e-9
    )


@pytest.mark.parametrize("convention", list(CONVENTIONS))
@pytest.mark.parametrize("track", list(TRACKS))
def test_random_walk_is_independent_of_edge_id_labels(
    track, convention, reference_environments
):
    """The random walk builds and equals the one for graph.edges labels."""
    environment = _make_environment(track, convention)

    transition = RandomWalk().make_state_transition([environment])
    reference = RandomWalk().make_state_transition([reference_environments[track]])

    interior = environment.is_track_interior_
    np.testing.assert_allclose(transition[interior].sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(transition, reference, atol=1e-12)


def _w_track_session(environment: Environment):
    """Runs from the center well out each outer arm and back, with place cells.

    Returns
    -------
    time : np.ndarray, shape (n_time,)
    position : np.ndarray, shape (n_time, 2)
    linear_position : np.ndarray, shape (n_time,)
    spike_times : list of np.ndarray
    """
    node_positions, _, _, _ = TRACKS["w_track"]
    sampling_frequency, speed = 250.0, 40.0
    route = [0, 1, 2, 3, 2, 1, 0, 1, 4, 5, 4, 1, 0] * 3
    position = []
    for node1, node2 in itertools.pairwise(route):
        pos1, pos2 = (
            np.asarray(node_positions[node1]),
            np.asarray(node_positions[node2]),
        )
        n_samples = int(np.linalg.norm(pos2 - pos1) / speed * sampling_frequency)
        fraction = np.arange(n_samples)[:, None] / n_samples
        position.append(pos1 + fraction * (pos2 - pos1))
    position = np.concatenate(position)
    time = np.arange(position.shape[0]) / sampling_frequency
    linear_position = environment.decoding_position(position).squeeze()

    rng = np.random.default_rng(7)
    # Fields every 22 cm along the vertical arms and 25 cm along the
    # horizontal ones.
    field_centers = np.concatenate(
        [
            [60.0, 0.0] + np.linspace(5.0, 115.0, 6)[:, None] * [0.0, 1.0],
            [60.0, 120.0] + np.linspace(5.0, 55.0, 3)[:, None] * [-1.0, 0.0],
            [0.0, 120.0] + np.linspace(5.0, 115.0, 6)[:, None] * [0.0, -1.0],
            [60.0, 120.0] + np.linspace(5.0, 55.0, 3)[:, None] * [1.0, 0.0],
            [120.0, 120.0] + np.linspace(5.0, 115.0, 6)[:, None] * [0.0, -1.0],
        ]
    )
    distance = np.linalg.norm(position[:, None] - field_centers[None], axis=-1)
    rate = 1.0 + 40.0 * np.exp(-0.5 * (distance / 8.0) ** 2)
    is_spike = rng.random(rate.shape) < rate / sampling_frequency
    spike_times = [time[is_spike[:, cell]] for cell in range(rate.shape[1])]
    return time, position, linear_position, spike_times


@pytest.mark.parametrize("convention", ["edge_list_index", "sparse_labels"])
def test_decoding_is_independent_of_edge_id_labels(convention):
    """A decoder on simulated W-track data decodes the same with any edge labels.

    Fit on the first two thirds of the session, decode the last third. The
    posterior equals the one for graph.edges labels, its MAP tracks the
    animal's linear position, and the MAP's edge (``distance1D``) is the
    physical edge the reference decoder finds.
    """
    decoders, results = {}, {}
    for name in ("graph_edges_index", convention):
        environment = _make_environment("w_track", name)
        time, position, linear_position, spike_times = _w_track_session(environment)
        is_training = time < time[-1] * 2 / 3
        decoder = SortedSpikesDecoder(
            environments=environment,
            sorted_spikes_algorithm_params={"disable_progress_bar": True},
        ).fit(time, position, spike_times, is_training=is_training)
        test_time = time[~is_training][::5]
        results[name] = decoder.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(test_time),
            position=position[~is_training][::5],
            position_time=test_time,
        )
        decoders[name] = decoder

    posterior = {
        name: result.acausal_posterior.unstack("state_bins").sum("state")
        for name, result in results.items()
    }
    np.testing.assert_allclose(
        posterior[convention].values,
        posterior["graph_edges_index"].values,
        rtol=1e-6,
        atol=1e-10,
    )

    map_position = posterior[convention].idxmax("position").values
    true_position = linear_position[~is_training][::5]
    assert np.median(np.abs(map_position - true_position)) < 5.0

    edges = {
        name: [
            frozenset(edge)
            for edge in _get_MAP_estimate_2d_position_edges(
                posterior[name], decoder.environments_[0].track_graph, decoder
            )[1]
        ]
        for name, decoder in decoders.items()
    }
    assert edges[convention] == edges["graph_edges_index"]
