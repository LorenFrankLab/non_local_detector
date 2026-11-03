"""Tests for analysis/posterior.py — MAP estimate and posterior sampling."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.analysis.posterior import (
    get_spatial_posterior,
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
        rng = np.random.default_rng(42)
        positions = np.linspace(0, 10, 20)
        probs = rng.dirichlet(np.ones(20), size=3)
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

    def test_unequal_bin_widths_preserve_probability_mass(self, monkeypatch):
        """Bin width changes the within-bin density, not the bin's mass."""
        from scipy.stats import rv_histogram  # type: ignore[import-untyped]

        def draw_quantiles(distribution, size, random_state=None):
            return distribution.ppf((np.arange(size) + 0.5) / size)

        monkeypatch.setattr(rv_histogram, "rvs", draw_quantiles)
        posterior = xr.DataArray([[0.5, 0.5], [0.8, 0.2]], dims=["time", "position"])

        samples = sample_posterior(
            posterior, np.array([0.0, 1.0, 10.0]), n_samples=1000
        )

        np.testing.assert_allclose((samples < 1.0).mean(axis=1), [0.5, 0.8])
        np.testing.assert_allclose(samples.mean(axis=1), [3.0, 1.5])

    def test_delta_function_samples_in_correct_bin(self):
        """All samples from a delta posterior should be in the peak bin."""
        bin_edges = np.arange(0.0, 6.0, 1.0)  # 5 bins: [0,1), [1,2), ..., [4,5)
        peak_idx = 2  # bin [2, 3)
        probs = np.zeros((1, 5))
        probs[0, peak_idx] = 1.0
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=100, rng=0)

        assert samples.shape == (1, 100)
        assert np.all(samples >= bin_edges[peak_idx])
        assert np.all(samples <= bin_edges[peak_idx + 1])

    def test_samples_within_domain(self):
        """All samples should be within the bin edge range."""
        bin_edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        probs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=500, rng=0)

        assert np.all(samples >= bin_edges[0])
        assert np.all(samples <= bin_edges[-1])

    def test_output_shape(self):
        """Output should be (n_time, n_samples)."""
        bin_edges = np.arange(0.0, 4.0)
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        posterior = xr.DataArray(probs, dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=50, rng=0)

        assert samples.shape == (2, 50)

    def test_mean_converges_to_expected_value(self):
        """Sample mean should approximate the posterior mean for large n."""
        bin_edges = np.arange(0.0, 6.0, 1.0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        probs = np.array([[0.0, 0.1, 0.8, 0.1, 0.0]])  # peaked at bin 2
        posterior = xr.DataArray(probs, dims=["time", "position"])
        expected_mean = np.sum(probs[0] * bin_centers)

        samples = sample_posterior(posterior, bin_edges, n_samples=10000, rng=0)
        sample_mean = np.mean(samples)

        # Variance = 0.2 (between bins) + 1/12 (uniform within a bin), so the
        # SEM for n=10000 is about 0.0053 and 0.05 is about 9 SEM.
        assert abs(sample_mean - expected_mean) < 0.05

    def test_rng_seed_reproduces_samples(self):
        """The same seed or an equally seeded Generator gives the same samples."""
        bin_edges = np.arange(0.0, 4.0)
        posterior = xr.DataArray(
            [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]], dims=["time", "position"]
        )

        first = sample_posterior(posterior, bin_edges, n_samples=50, rng=3)

        np.testing.assert_array_equal(
            first, sample_posterior(posterior, bin_edges, n_samples=50, rng=3)
        )
        np.testing.assert_array_equal(
            first,
            sample_posterior(
                posterior, bin_edges, n_samples=50, rng=np.random.default_rng(3)
            ),
        )
        assert not np.array_equal(
            first, sample_posterior(posterior, bin_edges, n_samples=50, rng=4)
        )

    def test_int_seed_draws_each_time_point_independently(self):
        """An int seed must not restart the stream at every time point."""
        bin_edges = np.arange(0.0, 4.0)
        posterior = xr.DataArray(np.full((2, 3), 1.0 / 3.0), dims=["time", "position"])

        samples = sample_posterior(posterior, bin_edges, n_samples=50, rng=0)

        assert not np.array_equal(samples[0], samples[1])


def _make_acausal_posterior(position_dims, n_time=3, seed=0):
    """Raw ``acausal_posterior`` layout: Local has NaN position, NL has a grid."""
    rng = np.random.default_rng(seed)
    grids = [np.arange(3.0), np.arange(2.0)][: len(position_dims)]
    grid = np.stack([g.ravel() for g in np.meshgrid(*grids, indexing="ij")], axis=1)
    n_grid = len(grid)
    tuples = [("Local", *([np.nan] * len(position_dims)))]
    tuples += [("NL", *row) for row in grid]
    index = pd.MultiIndex.from_tuples(tuples, names=["state", *position_dims])
    values = rng.random((n_time, n_grid + 1))
    values /= values.sum(axis=1, keepdims=True)
    return xr.DataArray(
        values,
        dims=["time", "state_bins"],
        coords={"time": np.arange(n_time) * 0.1, "state_bins": index},
    )


@pytest.mark.unit
class TestGetSpatialPosterior:
    """Position-less (NaN position) states must not land on a position bin."""

    def test_1d_no_mass_on_last_bin_and_spatial_values_exact(self):
        posterior = _make_acausal_posterior(["position"])
        result = get_spatial_posterior(posterior)

        assert result.dims == ("time", "state", "position")
        assert list(result.state.values) == ["NL"]
        np.testing.assert_array_equal(result.sel(state="NL").values, posterior.values[:, 1:])
        summed = result.sum("state")
        np.testing.assert_array_equal(summed.values, posterior.values[:, 1:])
        # Local's mass is excluded, so total < 1 per time bin
        assert np.all(summed.sum("position").values < 1.0)

    def test_2d_matches_spatial_values(self):
        posterior = _make_acausal_posterior(["x_position", "y_position"])
        result = get_spatial_posterior(posterior)

        assert result.dims == ("time", "state", "x_position", "y_position")
        np.testing.assert_array_equal(
            result.sel(state="NL").values.reshape(3, -1), posterior.values[:, 1:]
        )

    def test_states_argument_selects_states(self):
        posterior = _make_acausal_posterior(["position"])
        result = get_spatial_posterior(posterior, states=["NL"])
        assert list(result.state.values) == ["NL"]
        with pytest.raises(ValueError, match="No state bins"):
            get_spatial_posterior(posterior, states=["Local"])

    def test_requires_state_bins(self):
        with pytest.raises(ValueError, match="state_bins"):
            get_spatial_posterior(
                xr.DataArray(np.ones((2, 3)), dims=["time", "position"])
            )


@pytest.mark.unit
class TestMapFromRawAcausalPosterior:
    """MAP works on the raw ``state_bins`` posterior and ignores position-less states."""

    @pytest.mark.parametrize(
        "position_dims", [["position"], ["x_position", "y_position"]]
    )
    def test_position_less_state_does_not_create_last_bin_map(self, position_dims):
        posterior = _make_acausal_posterior(position_dims, n_time=1)
        values = posterior.values.copy()
        values[0, 0] = 0.9  # large Local mass, no position
        values[0, 1:] = 0.1 / (values.shape[1] - 1)
        values[0, 2] = 0.05  # spatial peak at the second spatial bin
        posterior = posterior.copy(data=values)

        result = maximum_a_posteriori_estimate(posterior)

        grid = np.stack(
            [posterior.indexes["state_bins"].get_level_values(n)[1:] for n in position_dims],
            axis=1,
        )
        np.testing.assert_array_equal(result[0], grid[1])
