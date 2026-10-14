"""Continuous-trajectory exposure regressions with independent integral oracles."""

import numpy as np
import pytest
from scipy.optimize import brentq

from non_local_detector.encoding_time import EncodingSupport
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    clusterless_diffusion,
    clusterless_gmm,
    clusterless_kde,
    clusterless_kde_log,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    fit_sorted_spikes_diffusion_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_poisson_regressions,
    fit_sorted_spikes_glm_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    fit_sorted_spikes_mrf_encoding_model,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "fit",
    [fit_sorted_spikes_kde_encoding_model, fit_sorted_spikes_diffusion_encoding_model],
)
def test_constant_rate_is_invariant_to_tracking_refinement(fit):
    """Adding samples on the same line must not create or remove place fields."""
    time = np.arange(11, dtype=float)
    position = np.array([0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0.0])[:, None]
    fine_time = np.linspace(0, 10, 1001)
    fine_position = np.interp(fine_time, time, position[:, 0])[:, None]
    env = Environment(place_bin_size=0.5, infer_track_interior=False).fit_place_grid(
        np.linspace(0, 10, 101)[:, None]
    )
    kwargs = {
        "spike_times": [np.linspace(0.025, 9.975, 200)],
        "environment": env,
        "position_std": 1.0,
        "encoding_time_range": [0, 10],
        "disable_progress_bar": True,
    }
    coarse = fit(time, position, **kwargs)
    fine = fit(fine_time, fine_position, **kwargs)
    interior = coarse["is_track_interior"]
    rates = np.asarray(coarse["place_fields"])[:, interior]
    assert coarse["encoding_exposure_seconds"] == pytest.approx(10.0)
    np.testing.assert_allclose(rates, 20.0, rtol=0.02)
    np.testing.assert_allclose(
        rates, np.asarray(fine["place_fields"])[:, interior], rtol=0.02
    )


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.1, 2.0])
def test_quadrature_preserves_weight_integral_and_complementary_groups(scale):
    """A clipped linear weight integrates analytically; EM groups partition it."""
    time = np.array([0.0, 1.0, 2.0])
    position = np.array([[0.0], [10.0], [0.0]])
    support = EncodingSupport(time, position, encoding_time_range=[0.25, 1.75])
    quadrature = support.quadrature(position, spatial_scale=scale)
    weights = np.array([0.0, 1.0, 0.0])
    masses = quadrature.weighted_exposure(weights)
    # Integral of the triangular weight on [.25,1.75]: 1 - 2*(.25**2/2).
    assert masses.sum() == pytest.approx(0.9375, abs=1e-12)
    np.testing.assert_allclose(
        masses + quadrature.weighted_exposure(1 - weights),
        quadrature.durations,
        atol=1e-15,
    )
    assert np.all(masses >= 0)
    np.testing.assert_allclose(
        quadrature.positions[:, 0],
        support.interpolate(position, quadrature.times)[:, 0],
    )


@pytest.mark.unit
def test_quadrature_does_not_bridge_missing_tracking_and_holds_endpoints():
    time = np.arange(5, dtype=float)
    position = np.array([[2.0], [2.0], [np.nan], [9.0], [9.0]])
    support = EncodingSupport(time, position, encoding_time_range=[-0.5, 4.5])
    q = support.quadrature(position, spatial_scale=0.5)
    assert np.all((q.times < 1.5) | (q.times > 2.5))
    assert q.durations.sum() == pytest.approx(4.0)
    np.testing.assert_allclose(q.positions[q.times < 1.5], 2.0)
    np.testing.assert_allclose(q.positions[q.times > 2.5], 9.0)
    np.testing.assert_allclose(
        q.weighted_exposure(np.ones(5)).sum(), support.exposure.sum()
    )


@pytest.mark.unit
def test_quadrature_integrates_histogram_bin_crossings_exactly():
    """A diagonal line's residence times follow its analytic crossing times."""
    time = np.array([0.0, 1.0])
    position = np.array([[0.0, 0.0], [10.0, 4.0]])
    support = EncodingSupport(time, position, encoding_time_range=[0.0, 1.0])
    q = support.quadrature(
        position,
        spatial_scale=2.0,
        bin_edges=(np.array([0.0, 2.0, 7.0, 10.0]), np.array([0.0, 3.0, 4.0])),
    )
    bin_x = np.searchsorted([2.0, 7.0], q.positions[:, 0], side="right")
    bin_y = np.searchsorted([3.0], q.positions[:, 1], side="right")
    histogram = np.bincount(
        bin_x * 2 + bin_y,
        weights=q.weighted_exposure(np.array([0.0, 1.0])),
        minlength=6,
    )
    # Integral of w(t)=t between crossings t=.2,.7,.75.
    np.testing.assert_allclose(
        histogram, [0.02, 0.0, 0.225, 0.0, 0.03625, 0.21875], atol=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "fit",
    [
        clusterless_kde.fit_clusterless_kde_encoding_model,
        clusterless_kde_log.fit_clusterless_kde_encoding_model,
        clusterless_diffusion.fit_clusterless_diffusion_encoding_model,
    ],
)
def test_clusterless_constant_ground_rate_uses_trajectory_exposure(fit):
    time = np.arange(11, dtype=float)
    position = np.array([0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0.0])[:, None]
    env = Environment(place_bin_size=0.5, infer_track_interior=False).fit_place_grid(
        np.linspace(0, 10, 101)[:, None]
    )
    encoding = fit(
        time,
        position,
        [np.linspace(0.025, 9.975, 200)],
        [np.zeros((200, 1))],
        env,
        position_std=1.0,
        waveform_std=1.0,
        encoding_time_range=[0, 10],
        disable_progress_bar=True,
    )
    np.testing.assert_allclose(
        encoding["summed_ground_process_intensity"], 20.0, rtol=0.02
    )
    assert encoding["encoding_exposure_seconds"] == pytest.approx(10.0)


@pytest.mark.unit
def test_gmm_occupancy_moments_integrate_continuous_path():
    """Uniform motion has variance L**2/12, not endpoint variance L**2/4."""
    time = np.array([0.0, 1.0])
    position = np.array([[0.0], [10.0]])
    env = Environment(place_bin_size=1.0, infer_track_interior=False).fit_place_grid(
        np.linspace(0, 10, 101)[:, None]
    )
    encoding = clusterless_gmm.fit_clusterless_gmm_encoding_model(
        time,
        position,
        [np.linspace(0.005, 0.995, 100)],
        [np.zeros((100, 1))],
        env,
        encoding_time_range=[0, 1],
        gmm_components_occupancy=1,
        gmm_components_gpi=1,
        gmm_components_joint=1,
        gmm_reg_covar=0.01,
        disable_progress_bar=True,
    )
    occupancy = encoding["occupancy_model"]
    np.testing.assert_allclose(occupancy.means_, [[5.0]], atol=1e-5)
    np.testing.assert_allclose(occupancy.covariances_, [[[100 / 12 + 0.01]]], rtol=1e-5)


@pytest.mark.unit
def test_mrf_rates_are_invariant_to_tracking_refinement():
    time = np.array([0.0, 1.0, 2.0])
    position = np.array([[0.0], [10.0], [0.0]])
    fine_time = np.linspace(0.0, 2.0, 201)
    fine_position = np.interp(fine_time, time, position[:, 0])[:, None]
    env = Environment(place_bin_size=1.0, infer_track_interior=False).fit_place_grid(
        np.linspace(0, 10, 101)[:, None]
    )
    kwargs = {
        "spike_times": [np.linspace(0.025, 1.975, 40)],
        "environment": env,
        "penalty": 1.0,
        "encoding_time_range": [0, 2],
        "disable_progress_bar": True,
    }
    coarse = fit_sorted_spikes_mrf_encoding_model(time, position, **kwargs)
    fine = fit_sorted_spikes_mrf_encoding_model(fine_time, fine_position, **kwargs)
    np.testing.assert_allclose(coarse["place_fields"], fine["place_fields"], rtol=1e-5)


@pytest.mark.unit
def test_continuous_glm_matches_analytic_loglinear_likelihood():
    time = np.array([0.0, 1.0])
    position = time[:, None]
    support = EncodingSupport(time, position, encoding_time_range=[0, 1])
    q = support.quadrature(position, spatial_scale=0.1)
    # Deterministic quantiles of an exponential density; the MLE is calculated
    # independently from its analytic first moment, not the quadrature scheme.
    event_times = np.log1p((np.arange(100) + 0.5) / 100 * np.expm1(2.0)) / 2
    event_stats = np.array([[100.0, event_times.sum()]])
    design = np.column_stack([np.ones(q.times.size), q.times])
    coeffs = np.asarray(
        fit_poisson_regressions(
            design, None, q.durations, l2_penalty=0.0, _event_design_sums=event_stats
        )
    )[0]
    slope = brentq(lambda b: 1 / (-np.expm1(-b)) - 1 / b - event_times.mean(), 0.1, 5.0)
    intercept = np.log(100 * slope / np.expm1(slope))
    np.testing.assert_allclose(coeffs, [intercept, slope], rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_glm_uses_actual_spike_positions_between_tracking_samples():
    time = np.array([0.0, 1.0, 2.0])
    position = np.array([[0.0], [10.0], [0.0]])
    fine_time = np.linspace(0.0, 2.0, 201)
    fine_position = np.interp(fine_time, time, position[:, 0])[:, None]
    env = Environment(place_bin_size=0.5, infer_track_interior=False).fit_place_grid(
        np.linspace(0, 10, 101)[:, None]
    )
    kwargs = {
        "spike_times": [
            np.concatenate([np.linspace(0.45, 0.55, 100), np.linspace(1.45, 1.55, 100)])
        ],
        "environment": env,
        "place_bin_edges": env.place_bin_edges_,
        "edges": env.edges_,
        "is_track_interior": env.is_track_interior_,
        "is_track_boundary": env.is_track_boundary_,
        "emission_knot_spacing": 2.0,
        "l2_penalty": 0.5,
        "encoding_time_range": [0, 2],
        "disable_progress_bar": True,
    }
    coarse = fit_sorted_spikes_glm_encoding_model(time, position, **kwargs)
    fine = fit_sorted_spikes_glm_encoding_model(fine_time, fine_position, **kwargs)
    rates = np.asarray(coarse["place_fields"])[0]
    assert rates[len(rates) // 2] > 3 * rates[0]
    np.testing.assert_allclose(rates, np.asarray(fine["place_fields"])[0], rtol=0.005)


@pytest.mark.unit
def test_get_position_at_time_preserves_already_linearized_track_position():
    """A linear track coordinate must not be projected onto the graph twice."""
    from non_local_detector.likelihoods.common import get_position_at_time
    from non_local_detector.tests.likelihoods.test_sorted_spikes_glm import (
        _corner_track_data,
    )

    env, _, _, _, _ = _corner_track_data()
    position = np.array([[3.0], [14.0]])
    actual = get_position_at_time(np.array([0.0, 1.0]), position, np.array([0.5]), env)
    np.testing.assert_allclose(actual, [[8.5]], atol=1e-12)


@pytest.mark.unit
def test_quadrature_retains_weight_integral_below_unix_timestamp_resolution():
    """Node timestamp rounding must not erase a sub-microsecond ramp's exposure."""
    # The measured interval satisfies the existing timestamp-precision check;
    # its many spatial quadrature subintervals can be smaller than one ULP.
    time = np.array([1.7e9, 1.7e9 + 0.001])
    position = np.array([[0.0], [10.0]])
    support = EncodingSupport(time, position, encoding_time_range=time)
    q = support.quadrature(position, spatial_scale=0.01)
    assert q.weighted_exposure(np.array([0.0, 1.0])).sum() == pytest.approx(
        np.diff(time)[0] / 2, rel=1e-12, abs=0.0
    )
    assert q.positions.min() > 0 and q.positions.max() < 10


@pytest.mark.unit
def test_quadrature_gaussian_occupancy_matches_independent_integral():
    from scipy.integrate import quad

    time = np.array([0.0, 1.7, 3.0])
    position = np.array([[-2.0], [4.0], [1.0]])
    weights = np.array([0.0, 1.0, 0.2])
    support = EncodingSupport(
        time,
        position,
        encoding_time_range=[0.3, 2.8],
        valid_position_intervals=[[0.0, 3.0]],
    )
    q = support.quadrature(position, spatial_scale=0.7)
    masses = q.weighted_exposure(weights)
    for point in [-1.0, 2.0, 5.0]:
        density = np.exp(-0.5 * ((q.positions[:, 0] - point) / 0.7) ** 2) / (
            0.7 * np.sqrt(2 * np.pi)
        )
        actual = np.dot(masses, density)
        expected = quad(
            lambda t, point=point: (
                np.interp(t, time, weights)
                * np.exp(
                    -0.5 * ((np.interp(t, time, position[:, 0]) - point) / 0.7) ** 2
                )
                / (0.7 * np.sqrt(2 * np.pi))
            ),
            0.3,
            2.8,
            points=[1.7],
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.unit
def test_quadrature_cache_reuses_geometry_but_is_bounded(monkeypatch):
    import non_local_detector.encoding_time as encoding_time

    time = np.array([0.0, 1.0])
    position = np.array([[0.0], [1.0]])
    support = EncodingSupport(time, position, encoding_time_range=time)
    q = support.quadrature(position, spatial_scale=0.5)
    assert support.quadrature(position, spatial_scale=0.5) is q
    assert not q.positions.flags.writeable
    q.weighted_exposure(np.array([0.0, 1.0]))
    assert support.quadrature(position, spatial_scale=0.5) is q
    monkeypatch.setattr(encoding_time, "_QUADRATURE_CACHE_BYTES", 1)
    changed = support.quadrature(position * 2, spatial_scale=0.5)
    assert changed is not q
    assert support._quadrature_cache is None


@pytest.mark.unit
def test_quadrature_rejects_resource_explosion_before_allocating(monkeypatch):
    import non_local_detector.encoding_time as encoding_time
    from non_local_detector.exceptions import ValidationError

    time = np.array([0.0, 1.0])
    position = np.array([[0.0], [1.0]])
    support = EncodingSupport(time, position, encoding_time_range=time)
    monkeypatch.setattr(encoding_time, "_QUADRATURE_MAX_BYTES", 1024)
    with pytest.raises(ValidationError, match="geometry budget"):
        support.quadrature(position, spatial_scale=1e-300)
    with pytest.raises(ValidationError, match="geometry budget"):
        support.quadrature(position, spatial_scale=0.01)
    assert support._quadrature_cache is None


@pytest.mark.unit
def test_quadrature_block_boundaries_preserve_the_trajectory():
    """Source and fill block boundaries must not skip or duplicate intervals."""
    time = np.linspace(0, 1, 20001)
    position = (10 * time)[:, None]
    weights = 1 + time
    support = EncodingSupport(time, position, encoding_time_range=[0.1, 0.9])
    q = support.quadrature(
        position, spatial_scale=1.0, bin_edges=(np.array([0.0, 3.0, 6.0, 10.0]),)
    )
    assert np.all(np.diff(q.times) > 0)
    # Integral of 1+t on [.1,.9] is 1.2; integral of t*(1+t) is analytic.
    mass = q.weighted_exposure(weights)
    np.testing.assert_allclose(mass.sum(), 1.2, atol=1e-12)
    expected_first_moment = (0.9**2 - 0.1**2) / 2 + (0.9**3 - 0.1**3) / 3
    np.testing.assert_allclose(
        np.dot(mass, q.positions[:, 0] / 10), expected_first_moment, atol=1e-12
    )


@pytest.mark.unit
def test_quadrature_fill_blocks_preserve_a_large_subdivision():
    """A single long movement spanning fill blocks retains exact linear moments."""
    time = np.array([0.0, 1.0])
    position = time[:, None]
    support = EncodingSupport(time, position, encoding_time_range=time)
    q = support.quadrature(position, spatial_scale=1e-4)
    mass = q.weighted_exposure(np.array([0.0, 1.0]))
    assert len(q.times) == 80000
    assert np.all(q.durations > 0)
    assert np.all(np.diff(q.times) > 0)
    np.testing.assert_allclose(mass.sum(), 0.5, atol=1e-12)
    np.testing.assert_allclose(np.dot(mass, q.positions[:, 0]), 1 / 3, atol=1e-12)
