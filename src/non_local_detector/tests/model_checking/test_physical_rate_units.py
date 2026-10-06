"""Independent physical-time references and explicit legacy unit boundaries."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.stats import expon

from non_local_detector.model_checking.clusterless import (
    _compute_rescaled_isi,
    interval_rescaling_transform,
)
from non_local_detector.model_checking.sorted_spikes import (
    TimeRescaling,
    point_process_residuals,
    uniform_rescaled_ISIs,
)


@pytest.mark.unit
@pytest.mark.parametrize("frequency", [30, 500])
def test_clusterless_hz_rescaling_is_independent_of_tracking_frequency(frequency):
    time = np.arange(frequency + 1) / frequency
    spikes = np.array([0.2, 0.4, 0.6, 0.8])
    rate = np.full(len(time), 5.0)
    np.testing.assert_allclose(_compute_rescaled_isi(rate, spikes, time), 1.0)
    uniform, _ = interval_rescaling_transform(
        time, spikes, np.zeros((4, 1)), rate, np.ones((4, 1))
    )
    np.testing.assert_allclose(uniform, -np.expm1(-1.0))


@pytest.mark.unit
def test_clusterless_irregular_time_integrates_hz_over_seconds():
    time = np.array([0.0, 0.1, 0.4, 0.7, 1.0])
    rate = 2.0 + 4.0 * time
    # The linear rate integrates analytically to 2*t + 2*t**2.
    spikes = np.array([0.05, 0.25, 0.85])
    expected = np.diff(np.r_[0.0, 2 * spikes + 2 * spikes**2])
    np.testing.assert_allclose(_compute_rescaled_isi(rate, spikes, time), expected)


@pytest.mark.unit
@pytest.mark.parametrize("duration", [1e-200, 1e200])
def test_linear_rate_integral_preserves_mass_when_time_units_are_scaled(duration):
    rate = np.array([1.0, 3.0]) / duration
    np.testing.assert_allclose(
        _compute_rescaled_isi(
            rate, np.array([duration / 2]), np.array([0.0, duration])
        ),
        [0.75],
    )


@pytest.mark.unit
@pytest.mark.parametrize("checker", ["clusterless", "sorted"])
@pytest.mark.parametrize(
    "rate",
    [
        np.array([200, 100, 100], dtype=np.uint8),
        np.array([120, 100, 100], dtype=np.int8),
        np.array([2e38, 2e38, 2e38], dtype=np.float32),
    ],
    ids=["unsigned-integer", "signed-integer", "large-float32"],
)
def test_hz_quadrature_promotes_rates_before_addition_or_subtraction(rate, checker):
    first, last = float(rate[0]), float(rate[1])
    first_mass = 0.375 * first + 0.125 * last
    cell_mass = 0.5 * first + 0.5 * last
    expected = np.array([first_mass, cell_mass + 0.5 * last - first_mass])
    time = np.array([0.0, 1.0, 2.0])
    if checker == "clusterless":
        np.testing.assert_allclose(
            _compute_rescaled_isi(rate, np.array([0.5, 1.5]), time), expected
        )
    else:
        np.testing.assert_allclose(
            point_process_residuals(
                rate, np.zeros(3, dtype=bool), rate_units="Hz", time=time
            ),
            [0.0, -cell_mass, -(cell_mass + last)],
        )


@pytest.mark.unit
def test_clusterless_legacy_expected_counts_is_explicit():
    time = np.arange(31) / 30
    spikes = np.array([0.2, 0.4, 0.6])
    counts = np.full(len(time), 5 / 30)
    np.testing.assert_allclose(
        _compute_rescaled_isi(counts, spikes, time, rate_units="expected_counts"),
        1.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize("frequency", [30, 500])
def test_sorted_sampled_hz_rescaling_uses_physical_time(frequency):
    time = np.arange(frequency + 1) / frequency
    is_spike = np.isin(np.arange(len(time)), np.array([1, 2, 3, 4]) * frequency // 5)
    rate = np.full(len(time), 5.0)
    kwargs = {"rate_units": "Hz", "time": time, "adjust_for_short_trials": False}
    np.testing.assert_allclose(
        uniform_rescaled_ISIs(rate, is_spike, **kwargs), -np.expm1(-1.0)
    )
    rescaling = TimeRescaling(rate, is_spike, **kwargs)
    np.testing.assert_allclose(rescaling.uniform_rescaled_ISIs(), -np.expm1(-1.0))
    np.testing.assert_allclose(
        point_process_residuals(rate, is_spike, rate_units="Hz", time=time),
        np.cumsum(is_spike) - 5 * time,
        # Summing 500 floating-point trapezoids accumulates roundoff even
        # when the independent analytic residual is exactly zero.
        atol=1e-12,
    )


@pytest.mark.unit
def test_sorted_irregular_bins_integrate_rate_times_each_duration():
    edges = np.array([1.0, 1.1, 1.4, 1.6, 2.0])
    rate = np.array([2.0, 3.0, 4.0, 5.0])
    is_spike = np.array([False, True, False, True])
    cumulative = np.cumsum(rate * np.diff(edges))
    expected_isi = np.diff(np.r_[0.0, cumulative[is_spike]])
    kwargs = {"rate_units": "Hz", "time_edges": edges, "adjust_for_short_trials": False}
    np.testing.assert_allclose(
        uniform_rescaled_ISIs(rate, is_spike, **kwargs), -np.expm1(-expected_isi)
    )
    np.testing.assert_allclose(
        TimeRescaling(rate, is_spike, **kwargs).uniform_rescaled_ISIs(),
        -np.expm1(-expected_isi),
    )
    np.testing.assert_allclose(
        point_process_residuals(rate, is_spike, rate_units="Hz", time_edges=edges),
        np.cumsum(is_spike) - cumulative,
    )


@pytest.mark.unit
def test_time_rescaling_trials_do_not_integrate_across_other_trials():
    time = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    rate = np.full(6, 5.0)
    is_spike = np.array([False, True, False, True, False, True])
    trial = np.array([0, 0, 1, 1, 0, 0])
    result = TimeRescaling(
        rate,
        is_spike,
        trial_id=trial,
        adjust_for_short_trials=False,
        rate_units="Hz",
        time=time,
    ).uniform_rescaled_ISIs()
    np.testing.assert_allclose(result, -np.expm1(-0.5))


@pytest.mark.unit
def test_legacy_sorted_expected_counts_retains_existing_numerics():
    counts = np.array([0.2, 0.5, 0.3, 0.8])
    spikes = np.array([False, True, False, True])
    cumulative = cumulative_trapezoid(counts, initial=0)
    expected = expon.cdf(np.diff(np.r_[0.0, cumulative[spikes]]))
    np.testing.assert_array_equal(
        uniform_rescaled_ISIs(counts, spikes, adjust_for_short_trials=False), expected
    )
    np.testing.assert_array_equal(
        point_process_residuals(counts, spikes), np.cumsum(spikes - counts)
    )


@pytest.mark.unit
def test_legacy_empty_rescaling_can_still_be_constructed_and_count_spikes():
    rescaling = TimeRescaling(np.array([]), np.array([], dtype=bool))
    assert rescaling.n_spikes == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "checker", [uniform_rescaled_ISIs, point_process_residuals, TimeRescaling]
)
def test_sorted_hz_requires_an_explicit_physical_grid(checker):
    with pytest.raises(ValueError, match="Hz.*time"):
        checker(np.ones(3), np.zeros(3, dtype=bool), rate_units="Hz")


@pytest.mark.unit
@pytest.mark.parametrize(
    "checker", [uniform_rescaled_ISIs, point_process_residuals, TimeRescaling]
)
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rate_units": "wat"}, "rate_units"),
        ({"time": np.arange(3.0)}, "expected_counts"),
        ({"rate_units": "Hz", "time": [0, 1, 1]}, "increasing"),
        ({"rate_units": "Hz", "time_edges": [0, 1, 2]}, "n_time.*1"),
        (
            {"rate_units": "Hz", "time": [0, 1, 2], "time_edges": [0, 1, 2, 3]},
            "exactly one",
        ),
    ],
)
def test_sorted_unit_boundary_rejects_ambiguous_or_invalid_grids(
    checker, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        checker(np.ones(3), np.zeros(3, dtype=bool), **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("bad_spikes", [[-0.1], [1.1], [0.7, 0.2], [np.nan]])
def test_clusterless_rejects_events_outside_or_invalid_for_sampled_support(bad_spikes):
    with pytest.raises(ValueError, match="spike_times"):
        _compute_rescaled_isi(
            np.ones(3), np.asarray(bad_spikes), np.array([0.0, 0.4, 1.0])
        )


@pytest.mark.unit
def test_clusterless_legacy_mode_preserves_index_integral_and_mark_ratio():
    time = np.array([0.0, 0.1, 0.4, 1.0])
    counts = np.array([0.2, 0.7, 0.3, 0.8])
    spikes = np.array([0.05, 0.25, 0.85])
    integrated = cumulative_trapezoid(counts, initial=0)
    expected = np.diff(np.r_[0.0, np.interp(spikes, time, integrated)])
    np.testing.assert_array_equal(
        _compute_rescaled_isi(counts, spikes, time, rate_units="expected_counts"),
        expected,
    )
    joint = np.array([[0.1], [0.2], [0.5]])
    legacy, legacy_marks = interval_rescaling_transform(
        time, spikes, np.zeros((3, 1)), counts, joint, rate_units="expected_counts"
    )
    _, hz_marks = interval_rescaling_transform(
        time, spikes, np.zeros((3, 1)), counts * 30, joint * 30, rate_units="Hz"
    )
    np.testing.assert_array_equal(legacy, expon.cdf(expected))
    np.testing.assert_array_equal(legacy_marks, hz_marks)


@pytest.mark.integration
@pytest.mark.parametrize("frequency", [30, 500])
@pytest.mark.parametrize(
    "algorithm",
    [
        "sorted_spikes_kde",
        "sorted_spikes_glm",
        "sorted_spikes_diffusion",
        "sorted_spikes_mrf",
        "clusterless_kde",
        "clusterless_kde_log",
        "clusterless_gmm",
        "clusterless_diffusion",
    ],
)
def test_fitted_hz_rates_feed_physical_time_checkers(algorithm, frequency):
    from non_local_detector.tests.models.test_rate_units_and_persistence import _decoder

    time = np.arange(frequency) / frequency
    position = np.zeros((frequency, 1))
    encoding_events = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    args = ([encoding_events],)
    if algorithm.startswith("clusterless"):
        args = (*args, [np.zeros((5, 1))])
    detector = _decoder(algorithm, calibration=True).fit(time, position, *args)
    (model,) = detector.encoding_model_.values()
    if "mean_rates" in model:
        rate = np.asarray(model["mean_rates"])[0]
    else:
        environment = detector.environments[0]
        center = np.flatnonzero(
            environment.place_bin_centers_[environment.is_track_interior_.ravel(), 0]
            == 0
        )[0]
        rate = np.asarray(model["place_fields"])[0, model["is_track_interior"]][center]
    assert model["rate_units"] == "Hz"
    # Test actual fitted rates at uniform and irregular checker timestamps.
    check_time = np.array([0.0, 0.05, 0.25, 0.55, 0.7, 1.0])
    events = np.array([0.2, 0.4, 0.6, 0.8])
    np.testing.assert_allclose(
        _compute_rescaled_isi(np.full(len(check_time), rate), events, check_time),
        1.0,
        rtol=1e-5,
    )
    edges = np.arange(6) / 5
    np.testing.assert_allclose(
        uniform_rescaled_ISIs(
            np.full(5, rate),
            np.ones(5, dtype=bool),
            adjust_for_short_trials=False,
            rate_units="Hz",
            time_edges=edges,
        ),
        -np.expm1(-1.0),
        rtol=1e-5,
    )
