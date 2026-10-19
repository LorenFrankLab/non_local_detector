"""Sorted-spikes GLM basis that follows the track graph.

A single spline over the linearized coordinate joins the end of one track segment
to the start of the next one in ``edge_order``, even when the two are far apart on
the track, so a field at a segment end leaks into an unrelated arm, and the fit
depends on ``edge_spacing``. On a track graph the GLM instead uses a cubic spline
along each branch of the track (a path between leaves and junctions), sharing its
values at junctions: it depends only on the track's geometry and connections, not
on the linear layout or on how a branch is split into edges.
"""

import pickle

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    TrackGraphBasis,
    fit_sorted_spikes_glm_encoding_model,
    make_spline_predict_matrix,
    make_track_graph_basis,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.time_edges import time_edges_from_centers

KNOT_SPACING = np.sqrt(12.5) * 2


def _track_env(nodes, edges, edge_order, edge_spacing, name="track", bin_size=1.0):
    graph = nx.Graph()
    for node, pos in nodes.items():
        graph.add_node(node, pos=pos)
    for eid, (u, v) in enumerate(edges):
        distance = float(np.linalg.norm(np.subtract(nodes[u], nodes[v])))
        graph.add_edge(u, v, distance=distance, edge_id=eid)
    return Environment(
        environment_name=name,
        place_bin_size=bin_size,
        track_graph=graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).fit_place_grid()


Y_NODES = {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (70.0, 30.0), 3: (70.0, -30.0)}
Y_EDGES = [(0, 1), (1, 2), (1, 3)]


def _y_track(edge_spacing):
    """Y track: stem 0-1, arms 1-2 and 1-3. In linear order the far end of arm 1-2
    (node 2) is followed, after ``edge_spacing``, by the junction end of arm 1-3."""
    return _track_env(Y_NODES, Y_EDGES, Y_EDGES, edge_spacing=edge_spacing, name="y")


@pytest.fixture(scope="module")
def y_track():
    return _y_track(15.0)


def _walk(env, node_sequence, n_per_leg=300, n_laps=12):
    nodes = env.track_graph.nodes
    legs = [
        np.linspace(nodes[a]["pos"], nodes[b]["pos"], n_per_leg)
        for a, b in zip(node_sequence[:-1], node_sequence[1:], strict=True)
    ]
    position = np.tile(np.concatenate(legs), (n_laps, 1))
    return np.arange(position.shape[0]) / 100.0, position


def _fit(env, time, position, spike_times):
    return fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        spike_times,
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
        disable_progress_bar=True,
    )


def test_edge_spans_match_environment_linearization(y_track):
    """Each segment's span in linear position is where the environment puts it."""
    basis = make_track_graph_basis(y_track, KNOT_SPACING)
    nodes = y_track.track_graph.nodes
    for (u, v), start, length in zip(
        y_track.edge_order, basis.edge_starts, basis.edge_lengths, strict=True
    ):
        a, b = np.asarray(nodes[u]["pos"]), np.asarray(nodes[v]["pos"])
        inside = np.stack([a + 0.01 * (b - a), a + 0.99 * (b - a)])
        np.testing.assert_allclose(
            y_track._linear_position(inside),
            [start + 0.01 * length, start + 0.99 * length],
        )


def test_basis_is_cardinal_and_reproduces_constants_and_cubics(y_track):
    """Columns after the intercept are log-rate values at knots, they sum to 1, and
    a cubic along a branch is reproduced exactly: branch ends are not forced flat."""
    basis = make_track_graph_basis(y_track, KNOT_SPACING)
    x = np.linspace(0.0, basis.edge_starts[-1] + basis.edge_lengths[-1], 500)
    np.testing.assert_allclose(
        basis.design_matrix(x)[:, 1:].sum(axis=1), 1.0, atol=1e-12
    )
    # Arm 1-2 is one branch; put points on it at its knots and in between.
    edge = 1
    branch = basis.edge_branches[edge]
    knots = basis.branch_knots[branch]
    to_linear = basis.edge_starts[edge] + (
        knots if basis.edge_forward[edge] else basis.edge_lengths[edge] - knots
    )
    design = basis.design_matrix(to_linear)
    np.testing.assert_allclose(
        design[:, basis.branch_columns[branch]], np.eye(knots.size), atol=1e-12
    )
    along = np.linspace(0.0, knots[-1], 200)
    points = basis.edge_starts[edge] + (
        along if basis.edge_forward[edge] else basis.edge_lengths[edge] - along
    )

    def cubic(s):
        return 0.002 * s**3 - 0.1 * s**2 + s

    coefficients = np.zeros(basis.n_columns)
    coefficients[basis.branch_columns[branch]] = cubic(knots)
    np.testing.assert_allclose(
        basis.design_matrix(points) @ coefficients, cubic(along), atol=1e-8
    )


def test_loop_is_smooth_through_every_node():
    """A circular track drawn as four edges is one periodic spline: no kink where
    the loop closes or at any node."""
    corners = {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (40.0, 40.0), 3: (0.0, 40.0)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    env = _track_env(corners, edges, edges, 5.0, name="loop")
    basis = make_track_graph_basis(env, KNOT_SPACING)
    assert basis.branch_periodic == (True,)
    np.testing.assert_allclose(
        basis.design_matrix(np.linspace(0.0, 175.0, 400))[:, 1:].sum(axis=1), 1.0
    )
    eps = 1e-4
    for start in basis.edge_starts:
        # Each edge's start node is the previous edge's end node.
        previous = (np.flatnonzero(basis.edge_starts == start)[0] - 1) % 4
        end_previous = basis.edge_starts[previous] + basis.edge_lengths[previous]
        rows = basis.design_matrix(
            np.array([end_previous - eps, end_previous, start, start + eps])
        )
        np.testing.assert_allclose(rows[1], rows[2], atol=1e-10)
        np.testing.assert_allclose(
            (rows[1] - rows[0]) / eps, (rows[3] - rows[2]) / eps, atol=1e-3
        )


def test_segments_meeting_at_a_node_share_its_value(y_track):
    """The three segment ends at the junction (node 1) give identical basis rows,
    so every fitted field is continuous through the junction."""
    basis = make_track_graph_basis(y_track, KNOT_SPACING)
    starts, lengths = basis.edge_starts, basis.edge_lengths
    at_junction = np.array([starts[0] + lengths[0], starts[1], starts[2]])
    rows = basis.design_matrix(at_junction)
    np.testing.assert_allclose(rows[1], rows[0], atol=1e-12)
    np.testing.assert_allclose(rows[2], rows[0], atol=1e-12)


def test_splitting_a_branch_into_edges_does_not_change_the_basis():
    """A straight 100 cm track drawn as one edge or as three gives the same basis
    at the same physical points: degree-2 nodes are invisible to it."""
    one = _track_env({0: (0.0, 0.0), 1: (100.0, 0.0)}, [(0, 1)], [(0, 1)], 0.0)
    nodes = {0: (0.0, 0.0), 1: (23.0, 0.0), 2: (61.0, 0.0), 3: (100.0, 0.0)}
    three = _track_env(nodes, [(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)], 10.0)
    points = np.stack([np.linspace(0.5, 99.5, 300), np.zeros(300)], axis=1)
    design_one = make_track_graph_basis(one, KNOT_SPACING).design_matrix(
        one._linear_position(points)
    )
    design_three = make_track_graph_basis(three, KNOT_SPACING).design_matrix(
        three._linear_position(points)
    )
    np.testing.assert_allclose(design_three, design_one, atol=1e-10)


def _node2_field_fit(env, seed=0):
    rng = np.random.default_rng(seed)
    time, position = _walk(env, (0, 1, 2, 1, 3, 1, 0))
    distance = np.linalg.norm(position - np.asarray(Y_NODES[2]), axis=1)
    rate = 0.2 + 20.0 * np.exp(-0.5 * (distance / 4.0) ** 2)
    spikes = time[rng.random(time.size) < rate * 0.01]
    return _fit(env, time, position, [spikes])


@pytest.mark.parametrize("edge_spacing", [0.0, 15.0])
def test_field_at_segment_end_does_not_leak_into_next_segment_in_linear_order(
    edge_spacing,
):
    """A field at the far end of arm 1-2 does not appear at the junction end of arm
    1-3, its neighbor in linear order but 42 cm away along the track."""
    env = _y_track(edge_spacing)
    field = np.asarray(_node2_field_fit(env)["place_fields"])[0]
    centers = env.place_bin_centers_.ravel()
    interior = env.is_track_interior_.ravel()
    basis = make_track_graph_basis(env, KNOT_SPACING)
    arm3_start = basis.edge_starts[2]
    arm3_junction_end = (
        interior & (centers >= arm3_start) & (centers < arm3_start + 6.0)
    )
    assert field[arm3_junction_end].max() < 0.5


def test_field_at_a_leaf_end_keeps_its_peak_there(y_track):
    """A field peaked at the far end of an arm (a reward well) is recovered at the
    end with most of its peak: the branch end is not forced flat."""
    enc = _node2_field_fit(y_track)
    basis = enc["emission_design_info"]
    coefficients = np.asarray(enc["coefficients"])[0]
    end = basis.edge_starts[1] + basis.edge_lengths[1]
    x = np.linspace(end - 20.0, end, 201)
    rate = np.exp(basis.design_matrix(x) @ coefficients)
    assert end - x[np.argmax(rate)] < 2.0
    assert rate.max() > 0.8 * 20.0


def test_fields_do_not_depend_on_edge_spacing():
    """``edge_spacing`` only lays segments out on the linear axis; the same spikes
    give the same field at the same place on the track whatever the gap."""
    rates = []
    for edge_spacing in (0.0, 15.0):
        env = _y_track(edge_spacing)
        enc = _node2_field_fit(env)
        basis = enc["emission_design_info"]
        coefficients = np.asarray(enc["coefficients"])[0]
        # The same physical points on each segment, in this layout's coordinates.
        offsets = np.linspace(0.5, 0.95, 20)
        linear = np.concatenate(
            [
                start + offsets * length
                for start, length in zip(
                    basis.edge_starts, basis.edge_lengths, strict=True
                )
            ]
        )
        rates.append(np.exp(basis.design_matrix(linear) @ coefficients))
    np.testing.assert_allclose(rates[0], rates[1], rtol=1e-4)


def test_field_at_a_junction_spreads_onto_every_arm_that_meets_there(y_track):
    """A field centered on the junction is real on all three segments near it."""
    rng = np.random.default_rng(1)
    time, position = _walk(y_track, (0, 1, 2, 1, 3, 1, 0))
    node1 = np.asarray(y_track.track_graph.nodes[1]["pos"])
    rate = 0.2 + 20.0 * np.exp(
        -0.5 * (np.linalg.norm(position - node1, axis=1) / 4.0) ** 2
    )
    spikes = time[rng.random(time.size) < rate * 0.01]
    enc = _fit(y_track, time, position, [spikes])
    basis = make_track_graph_basis(y_track, KNOT_SPACING)
    coefficients = np.asarray(enc["coefficients"])[0]
    starts, lengths = basis.edge_starts, basis.edge_lengths
    three_cm_in = np.array(
        [starts[0] + lengths[0] - 3.0, starts[1] + 3.0, starts[2] + 3.0]
    )
    rates = np.exp(basis.design_matrix(three_cm_in) @ coefficients)
    true = 0.2 + 20.0 * np.exp(-0.5 * (3.0 / 4.0) ** 2)
    np.testing.assert_allclose(rates, true, rtol=0.3)


def test_glm_predict_and_pickle_with_track_graph_basis(y_track):
    """Local and non-local prediction run on the per-segment basis, the local
    likelihood uses it at the animal's position, and the model pickles."""
    rng = np.random.default_rng(2)
    time, position = _walk(y_track, (0, 1, 2, 1, 3, 1, 0), n_laps=4)
    linear = y_track._linear_position(position)
    spikes = [
        time[rng.random(time.size) < (1.0 + 10.0 * (linear > 60)) * 0.01]
        for _ in range(3)
    ]
    enc = _fit(y_track, time, position, spikes)
    assert isinstance(enc["emission_design_info"], TrackGraphBasis)
    restored = pickle.loads(pickle.dumps(enc))
    edges = time_edges_from_centers(time)
    kwargs = {
        k: v for k, v in enc.items() if k not in ("environment", "disable_progress_bar")
    }
    for is_local in (False, True):
        ll = predict_sorted_spikes_glm_log_likelihood(
            time,
            position,
            spikes,
            y_track,
            **kwargs,
            is_local=is_local,
            time_edges=edges,
            disable_progress_bar=True,
        )
        assert np.all(np.isfinite(np.asarray(ll)))
        restored_kwargs = {
            k: v
            for k, v in restored.items()
            if k not in ("environment", "disable_progress_bar")
        }
        ll_restored = predict_sorted_spikes_glm_log_likelihood(
            time,
            position,
            spikes,
            y_track,
            **restored_kwargs,
            is_local=is_local,
            time_edges=edges,
            disable_progress_bar=True,
        )
        np.testing.assert_allclose(np.asarray(ll), np.asarray(ll_restored))
    design = np.asarray(
        make_spline_predict_matrix(enc["emission_design_info"], linear[:, None])
    )
    # The predict matrix is float32, like the patsy path's; edges are chosen
    # at full precision.
    np.testing.assert_allclose(
        design, enc["emission_design_info"].design_matrix(linear), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "nodes, edges",
    [
        # A loop attached at a junction (a lollipop).
        (
            {0: (0.0, 0.0), 1: (-30.0, 0.0), 2: (20.0, 20.0), 3: (20.0, -20.0)},
            [(0, 1), (0, 2), (2, 3), (3, 0)],
        ),
        # Two loops through one node (a figure eight).
        (
            {
                0: (0.0, 0.0),
                1: (20.0, 20.0),
                2: (20.0, -20.0),
                3: (-20.0, 20.0),
                4: (-20.0, -20.0),
            },
            [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)],
        ),
        # A short loop at a junction, with fewer than four knots.
        (
            {0: (0.0, 0.0), 1: (-30.0, 0.0), 2: (3.0, 3.0), 3: (3.0, -3.0)},
            [(0, 1), (0, 2), (2, 3), (3, 0)],
        ),
    ],
)
def test_branch_returning_to_its_own_junction(nodes, edges):
    """A branch that leaves a junction and returns to it shares one value at both
    ends: the basis still sums to 1 and is continuous at the junction."""
    env = _track_env(nodes, edges, edges, 10.0, name="loop_at_junction")
    basis = make_track_graph_basis(env, KNOT_SPACING)
    x = np.linspace(0.0, basis.edge_starts[-1] + basis.edge_lengths[-1], 2000)
    np.testing.assert_allclose(
        basis.design_matrix(x)[:, 1:].sum(axis=1), 1.0, atol=1e-10
    )
    # Every edge end at node 0 gives the same row.
    at_node = [
        start if u == 0 else start + length
        for (u, v), start, length in zip(
            edges, basis.edge_starts, basis.edge_lengths, strict=True
        )
        if 0 in (u, v)
    ]
    rows = basis.design_matrix(np.asarray(at_node))
    np.testing.assert_allclose(rows, np.broadcast_to(rows[0], rows.shape), atol=1e-10)


def test_local_rate_at_an_arm_end_that_float32_rounds_onto_the_next_arm():
    """With no gap, the end of arm 1-2 and the junction end of arm 1-3 share a
    linear position. Here that position rounds up in float32, onto arm 1-3, so the
    edge must be chosen at full precision: the local rate at the arm end is the
    field there, not the junction end of the other arm."""
    nodes = {**Y_NODES, 2: (70.0, 31.0)}
    env = _track_env(nodes, Y_EDGES, Y_EDGES, edge_spacing=0.0, name="y-float32")
    linear_end = env._linear_position(np.asarray([nodes[2]]))
    basis = make_track_graph_basis(env, KNOT_SPACING)
    assert basis.edge_index(linear_end)[0] == 1
    assert basis.edge_index(linear_end.astype(np.float32).astype(float))[0] == 2

    rng = np.random.default_rng(0)
    time, position = _walk(env, (0, 1, 2, 1, 3, 1, 0))
    distance = np.linalg.norm(position - np.asarray(nodes[2]), axis=1)
    rate = 0.2 + 20.0 * np.exp(-0.5 * (distance / 4.0) ** 2)
    spikes = time[rng.random(time.size) < rate * 0.01]
    enc = _fit(env, time, position, [spikes])
    coefficients = np.asarray(enc["coefficients"])[0]
    expected = float(np.exp(basis.design_matrix(linear_end) @ coefficients)[0])
    assert expected > 10.0

    dt = 0.01
    decode_time = np.arange(5) * dt
    log_likelihood = predict_sorted_spikes_glm_log_likelihood(
        decode_time,
        np.tile(nodes[2], (decode_time.size, 1)),
        [np.array([])],
        env,
        **{
            k: v
            for k, v in enc.items()
            if k not in ("environment", "disable_progress_bar")
        },
        is_local=True,
        time_edges=time_edges_from_centers(decode_time),
        disable_progress_bar=True,
    )
    # No spikes: each bin's local log likelihood is -rate * dt.
    np.testing.assert_allclose(
        -np.asarray(log_likelihood).ravel() / dt, expected, rtol=1e-3
    )


def test_fit_cells_at_an_arm_end_stay_on_their_edge(monkeypatch):
    """The fit evaluates the basis at each position cell's mean position. Averaging
    the samples at an arm end can land one ulp past the edge's end; with no gap
    that is the junction end of the next arm in linear order, so the cell's
    spikes and exposure must stay on the edge they were recorded on."""
    from non_local_detector.likelihoods import sorted_spikes_glm

    nodes = {**Y_NODES, 2: (70.0, 33.25)}
    env = _track_env(nodes, Y_EDGES, Y_EDGES, edge_spacing=0.0, name="y-ulp")
    time, position = _walk(env, (0, 1, 2, 1, 3, 1, 0), n_laps=40)
    rng = np.random.default_rng(0)
    distance = np.linalg.norm(position - np.asarray(nodes[2]), axis=1)
    rate = 0.2 + 20.0 * np.exp(-0.5 * (distance / 4.0) ** 2)
    spikes = time[rng.random(time.size) < rate * 0.01]

    cells = []
    aggregate = sorted_spikes_glm._aggregate_by_position_cell

    def recording_aggregate(*args, **kwargs):
        cells.append(aggregate(*args, **kwargs)[0])
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(
        sorted_spikes_glm, "_aggregate_by_position_cell", recording_aggregate
    )
    enc = _fit(env, time, position, [spikes])
    basis = enc["emission_design_info"]
    # Guard: here an arm-end cell's mean offset does land past its edge's end.
    cell_edge = np.rint(cells[0][:, 1]).astype(int)
    assert np.any(cells[0][:, 0] > basis.edge_lengths[cell_edge])

    coefficients = np.asarray(enc["coefficients"])[0]
    arm3_junction_end = basis.edge_starts[2] + np.linspace(0.25, 3.0, 12)
    assert np.exp(basis.design_matrix(arm3_junction_end) @ coefficients).max() < 0.5
