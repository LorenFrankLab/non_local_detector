"""Tests that Environment.get_bin_ind never returns a gap-bin index.

Gap bins arise as structural indexing artifacts when edge_spacing > 0 on a
linearized track, or when track interior is inferred from occupancy with
holes. No valid position belongs in them. Positions that would otherwise
land in a gap bin (e.g. exactly on an arm-boundary edge via
searchsorted(side="right") tie-breaking) must be snapped to the nearest
interior bin by raw position-coordinate distance.
"""

import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment import Environment


def _make_two_arm_track():
    """Two-segment linearized track with a gap (edge_spacing > 0)."""
    g = nx.Graph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(50.0, 0.0))
    g.add_node(2, pos=(60.0, 0.0))
    g.add_node(3, pos=(110.0, 0.0))
    g.add_edge(0, 1, distance=50.0, edge_id=0)
    g.add_edge(2, 3, distance=50.0, edge_id=1)

    env = Environment(
        environment_name="two-arm",
        place_bin_size=5.0,
        track_graph=g,
        edge_order=[(0, 1), (2, 3)],
        edge_spacing=10.0,
    )
    position_1d = np.concatenate(
        [np.linspace(0.0, 50.0, 25), np.linspace(60.0, 110.0, 25)]
    )
    return env.fit_place_grid(position_1d, infer_track_interior=True)


@pytest.mark.unit
class TestGetBinIndSnap:
    """get_bin_ind should never return a gap-bin index when is_track_interior_ is set."""

    def test_snaps_position_at_arm_boundary_edge(self):
        """Position exactly on the arm-A end edge snaps to arm-A's last interior bin.

        This reproduces the real-data bug: get_linearized_position places an
        animal at the arm endpoint, whose linear coordinate equals a gap-bin
        edge. np.searchsorted(side="right") then assigns the position to the
        gap bin. After the fix, it should land on the last arm-A interior bin.
        """
        env = _make_two_arm_track()
        edges = env.edges_[0]
        interior_bin_indices = np.where(env.is_track_interior_.ravel())[0]

        # Find the first gap bin and use its left edge (arm-A end).
        is_interior = env.is_track_interior_.ravel()
        first_gap = int(np.where(~is_interior)[0][0])
        arm_end_edge = edges[first_gap]

        pos = np.array([[arm_end_edge]])
        b = env.get_bin_ind(pos)
        assert b.shape == (1,)
        assert b[0] in interior_bin_indices, (
            f"Position {arm_end_edge} at arm-A-end edge returned bin {b[0]}, "
            f"which is not an interior bin. Interior: {interior_bin_indices.tolist()}"
        )
        # Must be the arm-A side of the gap (one of the interior bins adjacent
        # to the gap on the left).
        assert b[0] == first_gap - 1, (
            f"Expected snap to the interior bin just before the gap "
            f"(bin {first_gap - 1}), got {b[0]}"
        )

    def test_snaps_position_in_middle_of_gap(self):
        """Position in the middle of a gap bin snaps to a flanking interior bin."""
        env = _make_two_arm_track()
        is_interior = env.is_track_interior_.ravel()
        gap_bins = np.where(~is_interior)[0]
        gap_center = env.place_bin_centers_[gap_bins[0], 0]

        b = env.get_bin_ind(np.array([[gap_center]]))
        assert b[0] in np.where(is_interior)[0], (
            f"Mid-gap position {gap_center} returned bin {b[0]}, not interior"
        )

    def test_preserves_interior_bin_assignment(self):
        """Position already on an interior bin is returned unchanged."""
        env = _make_two_arm_track()
        is_interior = env.is_track_interior_.ravel()
        interior_bin_indices = np.where(is_interior)[0]
        # Use the center of the first interior bin.
        pos_on_interior = env.place_bin_centers_[interior_bin_indices[5], 0]

        b = env.get_bin_ind(np.array([[pos_on_interior]]))
        assert b[0] == interior_bin_indices[5], (
            f"Interior-bin position returned bin {b[0]}, expected "
            f"{interior_bin_indices[5]}"
        )

    def test_all_interior_env_preserves_raw_binning(self):
        """With all-interior is_track_interior_, snap is a no-op.

        When ``infer_track_interior=False``, ``fit_place_grid`` sets
        ``is_track_interior_`` to all-True. The snap branch still runs
        but ``needs_snap`` is all-False, so raw ``searchsorted`` results
        pass through unchanged. This guards against breaking simple
        open-field envs with no track graph.
        """
        env = Environment(
            environment_name="open-1d",
            place_bin_size=1.0,
        )
        position_1d = np.linspace(0.0, 10.0, 100)
        env = env.fit_place_grid(position_1d, infer_track_interior=False)
        assert np.all(env.is_track_interior_), (
            "Expected all-True is_track_interior_ when infer=False"
        )
        b = env.get_bin_ind(np.array([[5.0]]))
        assert b.shape == (1,)
        assert 0 <= b[0] < env.place_bin_centers_.shape[0]

    def test_snaps_2d_environment_with_interior_hole(self):
        """2D env with an interior hole: gap-bin positions snap by Euclidean distance.

        Exercises the N-D broadcast path in the snap code where each position
        has shape (2,) and distances are computed across (n_snap, n_interior, 2).
        """
        env = Environment(
            environment_name="open-2d",
            place_bin_size=1.0,
        )
        # L-shaped occupancy: the interior of a 5x5 grid has a hole at (2,2)
        # that never gets visited, which infer_track_interior will mark False.
        xs = np.concatenate(
            [
                np.linspace(0.0, 4.0, 50),
                np.full(50, 0.5),
                np.full(50, 4.0),
                np.linspace(0.0, 4.0, 50),
            ]
        )
        ys = np.concatenate(
            [
                np.full(50, 0.5),
                np.linspace(0.0, 4.0, 50),
                np.linspace(0.0, 4.0, 50),
                np.full(50, 4.0),
            ]
        )
        position_2d = np.stack([xs, ys], axis=-1)
        env = env.fit_place_grid(position_2d, infer_track_interior=True)

        is_interior = env.is_track_interior_.ravel()
        gap_bins = np.where(~is_interior)[0]
        if len(gap_bins) == 0:
            pytest.skip("No gap bins in this 2D env; can't test snap")

        # Use the center of the first gap bin as the query
        gap_center = env.place_bin_centers_[gap_bins[0]]
        b = env.get_bin_ind(gap_center[np.newaxis])
        assert b[0] in np.where(is_interior)[0], (
            f"2D env: gap-bin position {gap_center} returned bin {b[0]}, not interior"
        )

    def test_snaps_vectorized_mixed_input(self):
        """A batch of positions, some on interior bins, some on gap edges, all snap correctly."""
        env = _make_two_arm_track()
        edges = env.edges_[0]
        is_interior = env.is_track_interior_.ravel()
        interior_bin_indices = np.where(is_interior)[0]
        first_gap = int(np.where(~is_interior)[0][0])
        arm_end_edge = edges[first_gap]
        gap_center = env.place_bin_centers_[first_gap, 0]
        interior_center = env.place_bin_centers_[interior_bin_indices[3], 0]

        positions = np.array([[interior_center], [arm_end_edge], [gap_center]])
        bins = env.get_bin_ind(positions)
        assert bins.shape == (3,)
        for i, b in enumerate(bins):
            assert b in interior_bin_indices, (
                f"Row {i}: bin {b} for position {positions[i]} is not interior"
            )
