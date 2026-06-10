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

    def test_hpd_spatial_coverage_rejects_unknown_dims(self):
        """Posteriors lacking ``position`` or ``x_position``/``y_position`` dims raise."""
        probs = np.zeros((1, 4))
        probs[0, 1] = 1.0
        posterior = xr.DataArray(probs, dims=["time", "unsupported"])
        with pytest.raises(ValueError, match="x_position"):
            get_HPD_spatial_coverage(posterior, np.array([0.5]))


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
