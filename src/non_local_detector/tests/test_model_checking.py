"""Tests for model_checking/ — HPD, KL divergence, and overlap."""

import numpy as np
import pytest
import xarray as xr

from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from non_local_detector.model_checking.posterior_consistency import (
    posterior_consistency_hpd_overlap,
    posterior_consistency_kl_divergence,
)

# ---------------------------------------------------------------------------
# HPD threshold tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetHighestPosteriorThreshold:
    """Test HPD threshold computation."""

    def test_delta_function_threshold(self):
        """Delta function: threshold should equal the peak value."""
        probs = np.zeros((1, 10))
        probs[0, 3] = 1.0
        posterior = xr.DataArray(probs, dims=["time", "position"])

        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        assert threshold.shape == (1,)
        assert threshold[0] == pytest.approx(1.0)

    def test_uniform_posterior_low_threshold(self):
        """Uniform posterior: threshold should be ~1/n_bins (all bins needed)."""
        n_bins = 20
        probs = np.ones((1, n_bins)) / n_bins
        posterior = xr.DataArray(probs, dims=["time", "position"])

        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        # All bins are equally likely, so threshold equals any bin value
        assert threshold[0] == pytest.approx(1.0 / n_bins, rel=0.01)

    def test_coverage_monotonicity(self):
        """Lower coverage should yield higher or equal threshold."""
        rng = np.random.default_rng(7)
        probs = rng.dirichlet(np.ones(20), size=1)
        posterior = xr.DataArray(probs, dims=["time", "position"])

        thresh_50 = get_highest_posterior_threshold(posterior, coverage=0.50)
        thresh_95 = get_highest_posterior_threshold(posterior, coverage=0.95)

        assert thresh_50[0] >= thresh_95[0] - 1e-10

    def test_output_shape(self):
        """Output should be (n_time,)."""
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(10), size=5)
        posterior = xr.DataArray(probs, dims=["time", "position"])

        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        assert threshold.shape == (5,)


# ---------------------------------------------------------------------------
# HPD spatial coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetHPDSpatialCoverage:
    """Test HPD spatial coverage computation."""

    def test_delta_function_one_bin(self):
        """Delta function: coverage should equal one bin width."""
        positions = np.arange(0.0, 11.0, 1.0)  # 11 positions, bin width = 1.0
        bin_width = np.diff(positions)[0]
        probs = np.zeros((1, len(positions)))
        probs[0, 5] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert coverage.shape == (1,)
        assert coverage[0] == pytest.approx(bin_width)

    def test_delta_function_wider_bins(self):
        """Delta function with wider bins: coverage should scale with bin width."""
        positions = np.arange(0.0, 22.0, 2.0)  # bin width = 2.0
        bin_width = np.diff(positions)[0]
        probs = np.zeros((1, len(positions)))
        probs[0, 3] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert coverage[0] == pytest.approx(bin_width)

    def test_non_negative(self):
        """Spatial coverage must always be non-negative."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(20), size=3)
        positions = np.linspace(0, 100, 20)
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert np.all(coverage >= 0)

    def test_hpd_spatial_coverage_2d(self):
        """2D posterior coverage equals (bins-in-HPD) * x_width * y_width.

        Constructs a 5x5 grid with bin widths ``(0.1, 0.2)`` (bin area
        ``0.02``). Each frame has a single bin with all the mass, so the
        HPD region contains exactly one bin and the spatial coverage equals
        the bin area.
        """
        n_time = 3
        x_positions = np.arange(0.0, 0.5, 0.1)  # 5 bins, width 0.1
        y_positions = np.arange(0.0, 1.0, 0.2)  # 5 bins, width 0.2
        bin_area = 0.1 * 0.2

        probs = np.zeros((n_time, len(x_positions), len(y_positions)))
        # Deterministically pick a single peak bin per frame.
        peak_indices = [(0, 0), (2, 3), (4, 4)]
        for t, (ix, iy) in enumerate(peak_indices):
            probs[t, ix, iy] = 1.0

        posterior = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position"],
            coords={"x_position": x_positions, "y_position": y_positions},
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert coverage.shape == (n_time,)
        np.testing.assert_allclose(coverage, bin_area, atol=1e-12)

    def test_hpd_spatial_coverage_2d_multibin(self):
        """Coverage scales with the number of bins in the 2D HPD region.

        ``test_hpd_spatial_coverage_2d`` puts all mass on a single bin, so it
        cannot distinguish ``.sum(["x_position", "y_position"])`` from summing
        only one axis. Here each frame spreads its mass uniformly over a known
        number of distinct cells ``k``; with uniform mass the HPD region is all
        ``k`` cells, so coverage must equal ``k * bin_area``.
        """
        x_positions = np.arange(0.0, 0.5, 0.1)  # 5 bins, width 0.1
        y_positions = np.arange(0.0, 1.0, 0.2)  # 5 bins, width 0.2
        bin_area = 0.1 * 0.2

        # Per-frame number of equally-weighted cells and their (ix, iy) coords.
        frames = [
            [(0, 0), (4, 4)],  # k = 2
            [(0, 1), (2, 2), (4, 0)],  # k = 3
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],  # k = 5
        ]
        n_time = len(frames)
        probs = np.zeros((n_time, len(x_positions), len(y_positions)))
        for t, cells in enumerate(frames):
            for ix, iy in cells:
                probs[t, ix, iy] = 1.0 / len(cells)

        posterior = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position"],
            coords={"x_position": x_positions, "y_position": y_positions},
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        expected = np.array([len(cells) for cells in frames]) * bin_area
        assert coverage.shape == (n_time,)
        np.testing.assert_allclose(coverage, expected, atol=1e-12)

    def test_hpd_spatial_coverage_3d(self):
        """3D coverage uses the bin *volume* dx*dy*dz over x/y/z_position dims.

        Exercises the N-D generalization: position dims are detected by name and
        the bin measure is the product of per-axis widths (a volume in 3D).
        """
        x_positions = np.arange(0.0, 0.3, 0.1)  # 3 bins, width 0.1
        y_positions = np.arange(0.0, 0.6, 0.2)  # 3 bins, width 0.2
        z_positions = np.arange(0.0, 0.9, 0.3)  # 3 bins, width 0.3
        bin_volume = 0.1 * 0.2 * 0.3

        n_time = 2
        probs = np.zeros((n_time, len(x_positions), len(y_positions), len(z_positions)))
        peak_indices = [(0, 0, 0), (2, 2, 2)]  # single bin per frame
        for t, (ix, iy, iz) in enumerate(peak_indices):
            probs[t, ix, iy, iz] = 1.0

        posterior = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position", "z_position"],
            coords={
                "x_position": x_positions,
                "y_position": y_positions,
                "z_position": z_positions,
            },
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert coverage.shape == (n_time,)
        np.testing.assert_allclose(coverage, bin_volume, atol=1e-12)

    def test_hpd_spatial_coverage_rejects_unknown_dims(self):
        """Posteriors lacking ``position`` or ``x_position``/``y_position`` dims raise."""
        probs = np.zeros((1, 4))
        probs[0, 1] = 1.0
        posterior = xr.DataArray(probs, dims=["time", "unsupported"])
        with pytest.raises(ValueError, match="_position"):
            get_HPD_spatial_coverage(posterior, np.array([0.5]))

    def test_hpd_spatial_coverage_nonuniform_bin_width(self):
        """``bin_width`` integrates exact per-bin widths on a non-uniform 1D grid.

        A linearized track-graph environment with unequal segments has on-track
        bins of two different widths. The uniform default uses the first bin
        width for all bins (biased when the HPD region spans bins of different
        widths); passing the true per-bin widths (``np.diff(edges)``) gives the
        exact sum of the in-HPD bin widths.
        """
        import networkx as nx

        from non_local_detector.environment import Environment

        track_graph = nx.Graph()
        track_graph.add_node(0, pos=(0.0, 0.0))
        track_graph.add_node(1, pos=(50.0, 0.0))
        track_graph.add_node(2, pos=(60.0, 0.0))
        track_graph.add_node(3, pos=(90.0, 0.0))
        track_graph.add_edge(0, 1, distance=50.0, edge_id=0)
        track_graph.add_edge(2, 3, distance=30.0, edge_id=1)
        env = Environment(
            place_bin_size=8.0,
            track_graph=track_graph,
            edge_order=[(0, 1), (2, 3)],
            edge_spacing=15.0,
        ).fit_place_grid()

        centers = np.asarray(env.place_bin_centers_).ravel()
        bin_width = np.diff(np.asarray(env.edges_[0]).ravel())
        is_interior = np.asarray(env.is_track_interior_).ravel()
        n_bins = centers.size
        # Sanity: the on-track bins are genuinely non-uniform (two arm widths).
        assert len(np.unique(np.round(bin_width[is_interior], 6))) > 1

        # HPD region = two on-track bins from each arm (different widths).
        hpd_bins = np.array([0, 1, n_bins - 2, n_bins - 1])
        assert is_interior[hpd_bins].all()
        probs = np.zeros((1, n_bins))
        probs[0, hpd_bins] = 1.0 / len(hpd_bins)
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": centers}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        exact = get_HPD_spatial_coverage(posterior, threshold, bin_width=bin_width)
        np.testing.assert_allclose(exact, bin_width[hpd_bins].sum(), atol=1e-9)

        # The uniform default differs because the HPD spans two bin widths.
        uniform = get_HPD_spatial_coverage(posterior, threshold)
        assert not np.isclose(exact[0], uniform[0])

    def test_hpd_spatial_coverage_bin_width_rejects_multidim(self):
        """``bin_width`` is only valid for a single position dim."""
        probs = np.zeros((1, 2, 2))
        probs[0, 0, 0] = 1.0
        posterior = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position"],
            coords={"x_position": [0.0, 1.0], "y_position": [0.0, 2.0]},
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
        with pytest.raises(ValueError, match="single"):
            get_HPD_spatial_coverage(
                posterior, threshold, bin_width=np.array([1.0, 1.0])
            )

    def test_hpd_spatial_coverage_rejects_length_one_dim(self):
        """A degenerate single-bin position axis raises a descriptive error.

        ``np.diff`` of a length-1 coordinate is empty, so inferring the bin
        width from coordinates is impossible; the function must raise a clear
        ``ValueError`` rather than a bare numpy ``IndexError``.
        """
        probs = np.ones((2, 1))
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": [0.0]}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
        with pytest.raises(ValueError, match="at least 2"):
            get_HPD_spatial_coverage(posterior, threshold)

    def test_hpd_spatial_coverage_rejects_descending_coords(self):
        """Descending (or unsorted) position coordinates raise instead of
        silently returning a negative/wrong measure from ``np.diff(...)[0]``.
        """
        positions = np.array([4.0, 3.0, 2.0, 1.0, 0.0])  # descending
        probs = np.zeros((1, positions.size))
        probs[0, 2] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
        with pytest.raises(ValueError, match="strictly increasing"):
            get_HPD_spatial_coverage(posterior, threshold)

    def test_hpd_spatial_coverage_bin_width_shape_mismatch(self):
        """A ``bin_width`` whose length does not match the position dim raises."""
        positions = np.arange(0.0, 5.0, 1.0)  # 5 bins
        probs = np.zeros((1, positions.size))
        probs[0, 1] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
        with pytest.raises(ValueError, match="shape"):
            get_HPD_spatial_coverage(
                posterior, threshold, bin_width=np.array([1.0, 1.0, 1.0])
            )


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPosteriorConsistencyKLDivergence:
    """Test KL divergence computation."""

    def test_identical_distributions_zero(self):
        """KL(P || P) should be 0."""
        p = np.array([[0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])

        kl = posterior_consistency_kl_divergence(p, p)

        np.testing.assert_allclose(kl, 0.0, atol=1e-10)

    def test_non_negative(self):
        """KL divergence is always >= 0."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(10), size=5)
        q = rng.dirichlet(np.ones(10), size=5)

        kl = posterior_consistency_kl_divergence(p, q)

        assert np.all(kl >= -1e-10)  # numerical tolerance

    def test_known_analytic_value(self):
        """Verify against hand-computed KL divergence."""
        # KL([0.9, 0.1] || [0.5, 0.5]) = 0.9*ln(0.9/0.5) + 0.1*ln(0.1/0.5)
        p = np.array([[0.9, 0.1]])
        q = np.array([[0.5, 0.5]])
        expected = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)

        kl = posterior_consistency_kl_divergence(p, q)

        assert kl[0] == pytest.approx(expected, rel=1e-6)

    def test_output_shape(self):
        """Output should be (n_time,)."""
        rng = np.random.default_rng(7)
        p = rng.dirichlet(np.ones(10), size=3)
        q = rng.dirichlet(np.ones(10), size=3)

        kl = posterior_consistency_kl_divergence(p, q)

        assert kl.shape == (3,)

    def test_rejects_3d_input(self):
        """Un-flattened (n_time, n_x, n_y) input must raise, not silently reduce.

        ``entropy(..., axis=-1)`` would reduce only the last axis and return a
        wrong-shaped ``(n_time, n_x)`` array; the docstring promises a flattened
        ``(n_time, n_position_bins)`` contract, so 3-D input must raise.
        """
        p = np.ones((2, 3, 3)) / 9.0
        with pytest.raises(ValueError, match="2-D"):
            posterior_consistency_kl_divergence(p, p)

    def test_rejects_shape_mismatch(self):
        """posterior and likelihood must share a shape."""
        p = np.full((2, 3), 1.0 / 3.0)
        q = np.full((2, 4), 1.0 / 4.0)
        with pytest.raises(ValueError, match="same shape"):
            posterior_consistency_kl_divergence(p, q)


# ---------------------------------------------------------------------------
# HPD overlap tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPosteriorConsistencyHPDOverlap:
    """Test HPD overlap computation."""

    def test_identical_distributions_full_overlap(self):
        """Identical distributions should have overlap = 1.0."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(20), size=3)

        overlap = posterior_consistency_hpd_overlap(p, p, coverage=0.95)

        np.testing.assert_allclose(overlap, 1.0)

    def test_disjoint_distributions_zero_overlap(self):
        """Non-overlapping HPD regions should have overlap = 0.0."""
        # p peaked at left, q peaked at right
        p = np.zeros((1, 20))
        p[0, 0] = 1.0
        q = np.zeros((1, 20))
        q[0, 19] = 1.0

        overlap = posterior_consistency_hpd_overlap(p, q, coverage=0.5)

        assert overlap[0] == pytest.approx(0.0)

    def test_overlap_in_unit_range(self):
        """Overlap should be in [0, 1]."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(20), size=5)
        q = rng.dirichlet(np.ones(20), size=5)

        overlap = posterior_consistency_hpd_overlap(p, q, coverage=0.95)

        assert np.all(overlap >= 0.0)
        assert np.all(overlap <= 1.0)

    def test_output_shape(self):
        """Output should be (n_time,)."""
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(10), size=4)
        q = rng.dirichlet(np.ones(10), size=4)

        overlap = posterior_consistency_hpd_overlap(p, q, coverage=0.95)

        assert overlap.shape == (4,)

    def test_rejects_3d_input(self):
        """Un-flattened (n_time, n_x, n_y) input must raise rather than run on
        ``axis=1`` and return a wrong-shaped result.

        Uses ``n_time == n_x`` (3, 3, 3): with this shape the pre-fix code's
        ``posterior >= threshold[:, None]`` broadcast *succeeds* and the
        function silently returns a wrong-shaped ``(3, 3)`` array — the
        genuinely silent failure the guard prevents (a shape like (2, 3, 3)
        would only trip an incidental broadcast error, not the silent path).
        """
        p = np.ones((3, 3, 3)) / 9.0
        with pytest.raises(ValueError, match="2-D"):
            posterior_consistency_hpd_overlap(p, p, coverage=0.95)
