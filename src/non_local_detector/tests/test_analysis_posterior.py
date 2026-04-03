"""Tests for analysis/posterior.py — MAP estimate and posterior sampling."""

import numpy as np
import pytest
import xarray as xr

from non_local_detector.analysis.posterior import (
    maximum_a_posteriori_estimate,
    sample_posterior,
)


@pytest.mark.unit
class TestMaximumAPosterioriEstimate:
    """Test MAP estimate correctness."""

    def test_delta_function_1d(self):
        """MAP of a delta function should return the peak position."""
        positions = np.arange(0.0, 10.0, 1.0)
        peak_idx = 3
        probs = np.zeros((1, len(positions)))
        probs[0, peak_idx] = 1.0
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result.shape == (1, 1)
        assert result[0, 0] == positions[peak_idx]

    def test_bimodal_picks_taller_peak(self):
        """MAP should pick the taller peak, not the average."""
        positions = np.arange(0.0, 20.0, 1.0)
        probs = np.zeros((1, len(positions)))
        probs[0, 5] = 0.3  # shorter peak
        probs[0, 15] = 0.7  # taller peak
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result[0, 0] == positions[15]

    def test_close_values_selects_correct_bin(self):
        """With close probabilities, MAP should pick the slightly higher one.

        The implementation uses np.log(posterior).argmax, so this verifies
        the log transform doesn't corrupt the ordering of similar values.
        """
        positions = np.arange(0.0, 5.0, 1.0)
        probs = np.array([[0.10, 0.30, 0.31, 0.20, 0.09]])
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result[0, 0] == positions[2]  # bin with 0.31, not 0.30

    def test_multiple_timesteps(self):
        """Each timestep should get its own MAP estimate."""
        positions = np.arange(0.0, 5.0, 1.0)
        n_time = 5
        probs = np.zeros((n_time, len(positions)))
        for t in range(n_time):
            probs[t, t] = 1.0  # peak moves across bins
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result.shape == (n_time, 1)
        for t in range(n_time):
            assert result[t, 0] == positions[t]

    def test_output_shape_1d(self):
        """1D posterior should return (n_time, 1)."""
        positions = np.linspace(0, 10, 20)
        probs = np.random.dirichlet(np.ones(20), size=3)
        posterior = xr.DataArray(
            probs, dims=["time", "position"], coords={"position": positions}
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result.shape == (3, 1)

    def test_output_shape_2d(self):
        """2D posterior should return (n_time, 2)."""
        x_pos = np.arange(0.0, 3.0)
        y_pos = np.arange(0.0, 4.0)
        probs = np.zeros((2, len(x_pos), len(y_pos)))
        probs[0, 1, 2] = 1.0
        probs[1, 0, 3] = 1.0
        posterior = xr.DataArray(
            probs,
            dims=["time", "x_position", "y_position"],
            coords={"x_position": x_pos, "y_position": y_pos},
        )

        result = maximum_a_posteriori_estimate(posterior)

        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [0.0, 3.0])


@pytest.mark.unit
class TestSamplePosterior:
    """Test posterior sampling correctness."""

    def test_delta_function_samples_in_correct_bin(self):
        """All samples from a delta posterior should be in the peak bin."""
        bin_edges = np.arange(0.0, 6.0, 1.0)  # 5 bins: [0,1), [1,2), ..., [4,5)
        peak_idx = 2  # bin [2, 3)
        probs = np.zeros((1, 5))
        probs[0, peak_idx] = 1.0
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=100)

        assert samples.shape == (1, 100)
        assert np.all(samples >= bin_edges[peak_idx])
        assert np.all(samples <= bin_edges[peak_idx + 1])

    def test_samples_within_domain(self):
        """All samples should be within the bin edge range."""
        bin_edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        probs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=500)

        assert np.all(samples >= bin_edges[0])
        assert np.all(samples <= bin_edges[-1])

    def test_output_shape(self):
        """Output should be (n_time, n_samples)."""
        bin_edges = np.arange(0.0, 4.0)
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=50)

        assert samples.shape == (2, 50)

    def test_mean_converges_to_expected_value(self):
        """Sample mean should approximate the posterior mean for large n."""
        bin_edges = np.arange(0.0, 6.0, 1.0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        probs = np.array([[0.0, 0.1, 0.8, 0.1, 0.0]])  # peaked at bin 2
        posterior = xr.DataArray(probs, dims=["time", "position"])
        expected_mean = np.sum(probs[0] * bin_centers)

        np.random.seed(0)
        samples = sample_posterior(posterior, bin_edges, n_samples=10000)
        sample_mean = np.mean(samples)

        assert abs(sample_mean - expected_mean) < 0.05  # ~1.5 SEM for n=10000
