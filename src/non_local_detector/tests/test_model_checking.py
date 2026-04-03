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
        probs = np.random.dirichlet(np.ones(20), size=1)
        posterior = xr.DataArray(probs, dims=["time", "position"])

        thresh_50 = get_highest_posterior_threshold(posterior, coverage=0.50)
        thresh_95 = get_highest_posterior_threshold(posterior, coverage=0.95)

        assert thresh_50[0] >= thresh_95[0] - 1e-10

    def test_output_shape(self):
        """Output should be (n_time,)."""
        probs = np.random.dirichlet(np.ones(10), size=5)
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
        probs = np.zeros((1, len(positions)))
        probs[0, 5] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert coverage.shape == (1,)
        assert coverage[0] == pytest.approx(1.0)  # one bin width

    def test_non_negative(self):
        """Spatial coverage must always be non-negative."""
        probs = np.random.dirichlet(np.ones(20), size=3)
        positions = np.linspace(0, 100, 20)
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )
        threshold = get_highest_posterior_threshold(posterior, coverage=0.95)

        coverage = get_HPD_spatial_coverage(posterior, threshold)

        assert np.all(coverage >= 0)


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
        np.random.seed(42)
        p = np.random.dirichlet(np.ones(10), size=5)
        q = np.random.dirichlet(np.ones(10), size=5)

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
        p = np.random.dirichlet(np.ones(10), size=3)
        q = np.random.dirichlet(np.ones(10), size=3)

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
        p = np.random.dirichlet(np.ones(20), size=3)

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
        np.random.seed(42)
        p = np.random.dirichlet(np.ones(20), size=5)
        q = np.random.dirichlet(np.ones(20), size=5)

        overlap = posterior_consistency_hpd_overlap(p, q, coverage=0.95)

        assert np.all(overlap >= 0.0)
        assert np.all(overlap <= 1.0)

    def test_output_shape(self):
        """Output should be (n_time,)."""
        p = np.random.dirichlet(np.ones(10), size=4)
        q = np.random.dirichlet(np.ones(10), size=4)

        overlap = posterior_consistency_hpd_overlap(p, q, coverage=0.95)

        assert overlap.shape == (4,)
