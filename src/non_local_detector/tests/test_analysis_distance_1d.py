"""Tests for analysis/distance1D.py — 1D trajectory and ahead/behind distance.

The 1D distance utilities operate on a linearized track graph (nodes carry a
2D ``pos`` attribute, edges carry a ``distance`` attribute). ``get_map_speed``
is exercised separately in ``test_analysis_distance.py`` and is not duplicated
here; this file covers the remaining public surface:

- ``get_trajectory_data``
- ``get_ahead_behind_distance`` (graph-based, signed)

and the ``n_time == 1`` edge case of ``get_map_speed`` (not covered elsewhere).
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.analysis.distance1D import (
    get_ahead_behind_distance,
    get_map_speed,
    get_trajectory_data,
)


def _make_positioned_track(n_nodes: int, edge_length: float = 1.0) -> nx.Graph:
    """Linear chain track graph with 2D node positions along the x-axis.

    Adds both ``distance`` edge attributes and ``pos`` node attributes so the
    distance utilities (which project positions and compute node-to-node
    distances) can operate on it.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (>= 2). Nodes are integer-labeled ``0..n_nodes-1`` and
        placed at ``(i * edge_length, 0)``.
    edge_length : float, optional
        Distance between consecutive nodes, by default 1.0.

    Returns
    -------
    networkx.Graph
        Path graph with ``pos`` node attributes and ``distance`` edge
        attributes.
    """
    graph = nx.path_graph(n_nodes)
    for node in graph.nodes():
        graph.nodes[node]["pos"] = np.array([float(node) * edge_length, 0.0])
    for u, v in graph.edges():
        graph[u][v]["distance"] = edge_length
    return graph


class _FakeEnv:
    """Minimal stand-in for the decoder/environment consumed by the 1D utils.

    ``_get_MAP_estimate_2d_position_edges`` reads only ``is_track_interior_``,
    ``place_bin_center_2D_position_`` and ``place_bin_center_ind_to_edge_id_``
    (and falls back to ``decoder.environment`` / ``decoder`` itself), so a tiny
    object exposing those three attributes drives the real code path without a
    full fitted decoder.
    """

    def __init__(self, place_bin_center_2D_position, edge_ids):
        n_bins = len(place_bin_center_2D_position)
        self.is_track_interior_ = np.ones(n_bins, dtype=bool)
        self.place_bin_center_2D_position_ = np.asarray(place_bin_center_2D_position)
        self.place_bin_center_ind_to_edge_id_ = np.asarray(edge_ids)


def _one_hot_posterior(argmax_bins: np.ndarray, n_bins: int) -> xr.DataArray:
    """One-hot posterior whose argmax over position recovers ``argmax_bins``.

    Parameters
    ----------
    argmax_bins : np.ndarray, shape (n_time,)
        Desired argmax bin index at each time step.
    n_bins : int
        Number of position bins.

    Returns
    -------
    xarray.DataArray, shape (n_time, n_bins)
        Posterior with mass 1.0 on ``argmax_bins[t]`` at each time ``t``.
    """
    n_time = len(argmax_bins)
    data = np.zeros((n_time, n_bins), dtype=float)
    data[np.arange(n_time), argmax_bins] = 1.0
    return xr.DataArray(data, dims=["time", "position"])


@pytest.mark.unit
class TestGetTrajectoryData:
    """``get_trajectory_data`` assembles actual + mental position/edge arrays."""

    def test_output_shapes(self):
        """All five returned arrays have time-aligned leading dimension."""
        track_graph = _make_positioned_track(4, edge_length=1.0)
        # 3 bins, one per edge of the 4-node chain.
        bin_centers_2d = np.array([[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]])
        edge_ids = np.array([0, 1, 2])
        env = _FakeEnv(bin_centers_2d, edge_ids)

        n_time = 3
        argmax_bins = np.array([0, 1, 2])
        posterior = _one_hot_posterior(argmax_bins, n_bins=3)

        actual_projected_position = np.array([[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]])
        track_segment_id = np.array([0, 1, 2])
        actual_orientation = np.array([0.0, 0.0, 0.0])

        (
            out_actual_pos,
            actual_edges,
            out_orientation,
            mental_position_2d,
            mental_position_edges,
        ) = get_trajectory_data(
            posterior=posterior,
            track_graph=track_graph,
            decoder=env,
            actual_projected_position=actual_projected_position,
            track_segment_id=track_segment_id,
            actual_orientation=actual_orientation,
        )

        assert out_actual_pos.shape == (n_time, 2)
        assert actual_edges.shape == (n_time, 2)
        assert out_orientation.shape == (n_time,)
        assert mental_position_2d.shape == (n_time, 2)
        assert mental_position_edges.shape == (n_time, 2)

    def test_forward_motion_is_monotonic(self):
        """A forward-moving MAP sequence yields increasing linear x-position."""
        track_graph = _make_positioned_track(4, edge_length=1.0)
        bin_centers_2d = np.array([[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]])
        edge_ids = np.array([0, 1, 2])
        env = _FakeEnv(bin_centers_2d, edge_ids)

        argmax_bins = np.array([0, 1, 2])  # forward sweep
        posterior = _one_hot_posterior(argmax_bins, n_bins=3)

        actual_projected_position = bin_centers_2d.copy()
        track_segment_id = np.array([0, 1, 2])
        actual_orientation = np.array([0.0, 0.0, 0.0])

        (_, _, _, mental_position_2d, _) = get_trajectory_data(
            posterior=posterior,
            track_graph=track_graph,
            decoder=env,
            actual_projected_position=actual_projected_position,
            track_segment_id=track_segment_id,
            actual_orientation=actual_orientation,
        )

        x = mental_position_2d[:, 0]
        assert np.all(np.diff(x) > 0)
        np.testing.assert_allclose(x, [0.5, 1.5, 2.5])

    def test_mental_edges_match_argmax_bins(self):
        """Mental edges are looked up from the MAP bin's edge_id."""
        track_graph = _make_positioned_track(4, edge_length=1.0)
        bin_centers_2d = np.array([[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]])
        edge_ids = np.array([0, 1, 2])
        env = _FakeEnv(bin_centers_2d, edge_ids)

        # MAP sits in bin 2 -> edge_id 2 -> graph edge (2, 3).
        argmax_bins = np.array([2])
        posterior = _one_hot_posterior(argmax_bins, n_bins=3)

        (_, _, _, _, mental_position_edges) = get_trajectory_data(
            posterior=posterior,
            track_graph=track_graph,
            decoder=env,
            actual_projected_position=np.array([[2.5, 0.0]]),
            track_segment_id=np.array([2]),
            actual_orientation=np.array([0.0]),
        )

        np.testing.assert_array_equal(mental_position_edges[0], [2, 3])

    def test_uses_nodes_df_fallback(self):
        """When the 2D-position attribute is absent, the nodes-df fallback runs.

        ``_get_MAP_estimate_2d_position_edges`` catches ``AttributeError`` and
        reads ``place_bin_centers_nodes_df_`` (x/y position + edge_id) instead.
        """

        class _DfEnv:
            def __init__(self, n_bins, df):
                self.is_track_interior_ = np.ones(n_bins, dtype=bool)
                self.place_bin_centers_nodes_df_ = df

        track_graph = _make_positioned_track(4, edge_length=1.0)
        df = pd.DataFrame(
            {
                "x_position": [0.5, 1.5, 2.5],
                "y_position": [0.0, 0.0, 0.0],
                "edge_id": [0, 1, 2],
            }
        )
        env = _DfEnv(n_bins=3, df=df)

        argmax_bins = np.array([1])
        posterior = _one_hot_posterior(argmax_bins, n_bins=3)

        (_, _, _, mental_position_2d, mental_position_edges) = get_trajectory_data(
            posterior=posterior,
            track_graph=track_graph,
            decoder=env,
            actual_projected_position=np.array([[1.5, 0.0]]),
            track_segment_id=np.array([1]),
            actual_orientation=np.array([0.0]),
        )

        np.testing.assert_allclose(mental_position_2d[0], [1.5, 0.0])
        np.testing.assert_array_equal(mental_position_edges[0], [1, 2])


@pytest.mark.unit
class TestGetAheadBehindDistance:
    """Signed graph distance: positive = ahead of head, negative = behind."""

    def test_ahead_is_positive_same_edge(self):
        """Mental position ahead of the head (in head direction) is positive."""
        track_graph = _make_positioned_track(4, edge_length=1.0)

        # Animal at x=0.5 on edge (0,1), facing +x (head_direction=0 -> node 1).
        actual_projected_position = np.array([[0.5, 0.0]])
        actual_edges = np.array([[0, 1]])
        actual_orientation = np.array([0.0])
        # Mental at x=0.8 on the same edge, i.e., ahead of the animal.
        mental_position_2d = np.array([[0.8, 0.0]])
        mental_position_edges = np.array([[0, 1]])

        dist = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            actual_orientation,
            mental_position_2d,
            mental_position_edges,
        )

        assert dist.shape == (1,)
        assert dist[0] > 0
        np.testing.assert_allclose(dist[0], 0.3, atol=1e-10)

    def test_behind_is_negative_same_edge(self):
        """Mental position behind the head (opposite head direction) is negative."""
        track_graph = _make_positioned_track(4, edge_length=1.0)

        # Animal at x=0.8 on edge (0,1), facing +x. Mental at x=0.3 is behind.
        actual_projected_position = np.array([[0.8, 0.0]])
        actual_edges = np.array([[0, 1]])
        actual_orientation = np.array([0.0])
        mental_position_2d = np.array([[0.3, 0.0]])
        mental_position_edges = np.array([[0, 1]])

        dist = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            actual_orientation,
            mental_position_2d,
            mental_position_edges,
        )

        assert dist[0] < 0
        np.testing.assert_allclose(dist[0], -0.5, atol=1e-10)

    def test_orientation_flips_sign(self):
        """Reversing head direction flips ahead/behind for a fixed geometry."""
        track_graph = _make_positioned_track(4, edge_length=1.0)

        actual_projected_position = np.array([[0.5, 0.0]])
        actual_edges = np.array([[0, 1]])
        mental_position_2d = np.array([[0.8, 0.0]])
        mental_position_edges = np.array([[0, 1]])

        facing_forward = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            np.array([0.0]),  # facing +x toward the mental position
            mental_position_2d,
            mental_position_edges,
        )
        facing_backward = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            np.array([np.pi]),  # facing -x, away from the mental position
            mental_position_2d,
            mental_position_edges,
        )

        assert facing_forward[0] > 0
        assert facing_backward[0] < 0
        # Magnitude (path length) is unchanged; only the sign flips.
        np.testing.assert_allclose(
            abs(facing_forward[0]), abs(facing_backward[0]), atol=1e-10
        )

    def test_different_edges_ahead(self):
        """Mental position one edge ahead returns the multi-segment distance."""
        track_graph = _make_positioned_track(4, edge_length=1.0)

        # Animal at x=0.5 on edge (0,1) facing +x; mental at x=2.5 on edge (2,3).
        actual_projected_position = np.array([[0.5, 0.0]])
        actual_edges = np.array([[0, 1]])
        actual_orientation = np.array([0.0])
        mental_position_2d = np.array([[2.5, 0.0]])
        mental_position_edges = np.array([[2, 3]])

        dist = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            actual_orientation,
            mental_position_2d,
            mental_position_edges,
        )

        assert dist[0] > 0
        # Path: 0.5 -> node 1 (0.5) -> node 2 (1.0) -> mental 2.5 (0.5) = 2.0.
        np.testing.assert_allclose(dist[0], 2.0, atol=1e-10)

    def test_multiple_timesteps(self):
        """Distances are computed independently and isolated per time step."""
        track_graph = _make_positioned_track(4, edge_length=1.0)

        actual_projected_position = np.array([[0.5, 0.0], [0.5, 0.0]])
        actual_edges = np.array([[0, 1], [0, 1]])
        actual_orientation = np.array([0.0, 0.0])
        mental_position_2d = np.array([[0.8, 0.0], [0.2, 0.0]])
        mental_position_edges = np.array([[0, 1], [0, 1]])

        dist = get_ahead_behind_distance(
            track_graph,
            actual_projected_position,
            actual_edges,
            actual_orientation,
            mental_position_2d,
            mental_position_edges,
        )

        assert dist.shape == (2,)
        np.testing.assert_allclose(dist[0], 0.3, atol=1e-10)  # ahead
        np.testing.assert_allclose(dist[1], -0.3, atol=1e-10)  # behind


@pytest.mark.unit
class TestGetMapSpeedSingleSample:
    """The ``n_time == 1`` branch of ``get_map_speed`` (not covered elsewhere)."""

    def test_single_timestep_returns_nan(self):
        """A single time step has no velocity, so speed is NaN."""
        track_graph = _make_positioned_track(4, edge_length=1.0)
        place_bin_center_ind_to_node = np.array([0, 1, 2, 3])
        posterior = np.zeros((1, 4))
        posterior[0, 2] = 1.0

        speed = get_map_speed(
            posterior=posterior,
            track_graph_with_bin_centers_edges=track_graph,
            place_bin_center_ind_to_node=place_bin_center_ind_to_node,
        )

        assert speed.shape == (1,)
        assert np.isnan(speed[0])
