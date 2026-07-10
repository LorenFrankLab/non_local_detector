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


@pytest.mark.unit
class TestGetBinIndSnapWarning:
    """Large off-grid snaps emit a UserWarning; small snaps stay quiet."""

    def test_off_grid_far_snap_warns(self):
        """A position far from any interior bin warns about the large snap.

        Genuine tracking-glitch / out-of-bounds case: a 2D open field whose
        occupancy is two small clusters far apart leaves a large never-visited
        interior region. A query in the empty middle lands in a non-interior
        bin whose nearest interior bin is many bin-widths away, so the snap
        exceeds the threshold (here ``2 x place_bin_size`` — no track graph, so
        no inter-arm gap term) and warns. This is distinct from a routine
        into-gap snap, which must stay quiet (see
        ``test_into_gap_snap_does_not_warn``).
        """
        env = Environment(environment_name="two-cluster-2d", place_bin_size=1.0)
        rng = np.random.default_rng(0)
        cluster_a = rng.random((300, 2)) * 5.0
        cluster_b = rng.random((300, 2)) * 5.0 + 45.0
        env = env.fit_place_grid(
            np.vstack([cluster_a, cluster_b]), infer_track_interior=True
        )

        is_interior = env.is_track_interior_.ravel()
        assert not np.all(is_interior), (
            "test setup requires a large non-interior region between the clusters"
        )

        # The empty middle (~28 bin-widths from either cluster) far exceeds the
        # 2.0 threshold, so the single queried position warns (count plumbed).
        with pytest.warns(UserWarning, match=r"1 position\(s\) snapped"):
            env.get_bin_ind(np.array([[25.0, 25.0]]))

    def test_nonfinite_position_does_not_suppress_far_snap_warning(self):
        """A NaN position in the batch must not hide a genuine large snap.

        ``np.max`` over snap distances containing a NaN returns NaN, which
        compares ``False`` against the threshold — so without explicit
        non-finite handling a single NaN position would silently suppress the
        warning for a finite position in the same batch that snapped far. The
        max is now taken over finite distances (and non-finite positions are
        surfaced), so the warning still fires.
        """
        env = Environment(environment_name="two-cluster-2d", place_bin_size=1.0)
        rng = np.random.default_rng(0)
        cluster_a = rng.random((300, 2)) * 5.0
        cluster_b = rng.random((300, 2)) * 5.0 + 45.0
        env = env.fit_place_grid(
            np.vstack([cluster_a, cluster_b]), infer_track_interior=True
        )

        # [25, 25] alone snaps ~28 and warns; batching a NaN position with it
        # used to suppress that warning (NaN poisons np.max). It must still warn.
        with pytest.warns(UserWarning, match="snapped to the"):
            env.get_bin_ind(np.array([[25.0, 25.0], [np.nan, np.nan]]))

    def test_into_gap_snap_does_not_warn(self):
        """A routine into-gap snap on a multi-arm track stays quiet.

        A position projected into an inter-arm gap is snapped to the nearest
        on-track bin by up to ~``edge_spacing``. That is expected linearized-
        track structure, not a glitch, so the threshold includes the gap width
        and no warning fires. This pins the false-positive fix: with
        ``place_bin_size=1`` and ``edge_spacing=10`` the gap-center snap is ~5,
        which exceeds the old ``2 x place_bin_size = 2`` threshold (and used to
        warn) but is below the new ``2 + 10 = 12`` threshold.
        """
        import warnings as _warnings

        g = nx.Graph()
        g.add_node(0, pos=(0.0, 0.0))
        g.add_node(1, pos=(50.0, 0.0))
        g.add_node(2, pos=(60.0, 0.0))
        g.add_node(3, pos=(110.0, 0.0))
        g.add_edge(0, 1, distance=50.0, edge_id=0)
        g.add_edge(2, 3, distance=50.0, edge_id=1)
        env = Environment(
            environment_name="two-arm-narrow-bins",
            place_bin_size=1.0,
            track_graph=g,
            edge_order=[(0, 1), (2, 3)],
            edge_spacing=10.0,
        )
        position_1d = np.concatenate(
            [np.linspace(0.0, 50.0, 60), np.linspace(60.0, 110.0, 60)]
        )
        env = env.fit_place_grid(position_1d, infer_track_interior=True)

        is_interior = env.is_track_interior_.ravel()
        gap_center = env.place_bin_centers_[np.where(~is_interior)[0][0], 0]

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            env.get_bin_ind(np.array([[gap_center]]))
        snap_warnings = [w for w in caught if "snap distance" in str(w.message)]
        assert not snap_warnings, (
            "into-gap snap (~5, below the 2 + edge_spacing threshold) should "
            f"not warn; got {[str(w.message) for w in snap_warnings]}"
        )

    def test_close_snap_no_warning(self):
        """A position landing just on an arm-boundary edge snaps without warning."""
        import warnings as _warnings

        g = nx.Graph()
        g.add_node(0, pos=(0.0, 0.0))
        g.add_node(1, pos=(50.0, 0.0))
        g.add_node(2, pos=(52.0, 0.0))
        g.add_node(3, pos=(102.0, 0.0))
        g.add_edge(0, 1, distance=50.0, edge_id=0)
        g.add_edge(2, 3, distance=50.0, edge_id=1)
        # Small gap (edge_spacing=2) with large bins (place_bin_size=5) so any
        # snap distance stays under the 2 x 5 = 10 threshold.
        env = Environment(
            environment_name="small-gap",
            place_bin_size=5.0,
            track_graph=g,
            edge_order=[(0, 1), (2, 3)],
            edge_spacing=2.0,
        )
        position_1d = np.concatenate(
            [np.linspace(0.0, 50.0, 30), np.linspace(52.0, 102.0, 30)]
        )
        env = env.fit_place_grid(position_1d, infer_track_interior=True)

        edges = env.edges_[0]
        is_interior = env.is_track_interior_.ravel()
        first_gap = int(np.where(~is_interior)[0][0])
        arm_end_edge = edges[first_gap]

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            env.get_bin_ind(np.array([[arm_end_edge]]))
        snap_warnings = [w for w in caught if "snap distance" in str(w.message)]
        assert not snap_warnings

    def test_snap_threshold_uses_max_bin_size_for_2d(self):
        """For a 2D env with anisotropic place_bin_size, threshold = 2 x max(bin sizes)."""
        env = Environment(
            environment_name="open-2d-aniso",
            place_bin_size=(1.0, 4.0),
        )
        # Open field; force an interior hole so the snap path can run.
        xs = np.concatenate(
            [np.linspace(0.0, 8.0, 60), np.full(60, 0.5), np.full(60, 8.0)]
        )
        ys = np.concatenate(
            [np.full(60, 0.5), np.linspace(0.0, 8.0, 60), np.linspace(0.0, 8.0, 60)]
        )
        env = env.fit_place_grid(np.stack([xs, ys], axis=-1), infer_track_interior=True)
        # Open field (no track graph) has no inter-arm gap, so the threshold is
        # 2 * max(1.0, 4.0) = 8.0, not 2 * min = 2.0.
        assert env._snap_warn_threshold() == 8.0

    def test_snap_threshold_includes_edge_spacing(self):
        """On a linearized track the threshold adds the largest inter-arm gap.

        A routine snap into an inter-arm gap is up to ~``edge_spacing`` wide and
        must not warn, so the threshold is ``2 x place_bin_size + max(edge_spacing)``.
        No fit is required — the threshold reads only constructor arguments.
        """
        g = nx.Graph()
        g.add_node(0, pos=(0.0, 0.0))
        g.add_node(1, pos=(50.0, 0.0))
        g.add_node(2, pos=(60.0, 0.0))
        g.add_node(3, pos=(110.0, 0.0))
        g.add_edge(0, 1, distance=50.0, edge_id=0)
        g.add_edge(2, 3, distance=50.0, edge_id=1)
        env = Environment(
            environment_name="two-arm-threshold",
            place_bin_size=5.0,
            track_graph=g,
            edge_order=[(0, 1), (2, 3)],
            edge_spacing=10.0,
        )
        # 2 * 5.0 + 10.0 = 20.0, not the open-field 2 * 5.0 = 10.0.
        assert env._snap_warn_threshold() == 20.0

    def test_snap_threshold_includes_max_of_edge_spacing_list(self):
        """A per-gap ``edge_spacing`` list contributes its maximum to the threshold."""
        g = nx.Graph()
        g.add_node(0, pos=(0.0, 0.0))
        g.add_node(1, pos=(50.0, 0.0))
        g.add_node(2, pos=(60.0, 0.0))
        g.add_node(3, pos=(110.0, 0.0))
        g.add_node(4, pos=(130.0, 0.0))
        g.add_node(5, pos=(180.0, 0.0))
        g.add_edge(0, 1, distance=50.0, edge_id=0)
        g.add_edge(2, 3, distance=50.0, edge_id=1)
        g.add_edge(4, 5, distance=50.0, edge_id=2)
        env = Environment(
            environment_name="three-arm-threshold",
            place_bin_size=2.0,
            track_graph=g,
            edge_order=[(0, 1), (2, 3), (4, 5)],
            edge_spacing=[10.0, 20.0],
        )
        # 2 * 2.0 + max(10.0, 20.0) = 24.0.
        assert env._snap_warn_threshold() == 24.0
