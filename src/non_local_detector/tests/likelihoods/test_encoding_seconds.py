"""Independent references for encoding exposure and decode duration."""

import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "bounds, expected",
    [
        ([-0.5, 1.5], [1.0, 1.0]),
        ([0.25, 0.75], [0.25, 0.25]),
        ([0.0, 0.25], [0.21875, 0.03125]),
    ],
)
def test_exposure_integrates_the_original_interpolation_basis(bounds, expected):
    from non_local_detector.encoding_time import EncodingSupport

    support = EncodingSupport(
        np.array([0.0, 1.0]), np.zeros((2, 1)), encoding_time_range=bounds
    )
    np.testing.assert_allclose(support.exposure, expected, rtol=0, atol=1e-15)


@pytest.mark.unit
def test_missing_tracking_excludes_events_and_exposure():
    from non_local_detector.encoding_time import EncodingSupport

    position = np.array([0.0, 1.0, np.nan, 3.0, 4.0])[:, None]
    support = EncodingSupport(np.arange(5.0), position)
    np.testing.assert_allclose(support.exposure, [1.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        support.contains(np.array([1.25, 2.0, 2.75])), [True, False, True]
    )
    np.testing.assert_allclose(
        support.interpolate(position[:, 0], np.array([1.25, 2.75])), [1.0, 3.0]
    )


@pytest.mark.unit
def test_explicit_tracking_segments_do_not_bridge_epochs():
    from non_local_detector.encoding_time import EncodingSupport

    time = np.array([0.0, 1.0, 10.0, 11.0])
    support = EncodingSupport(
        time, time[:, None], valid_position_intervals=[[-0.5, 1.5], [9.5, 11.5]]
    )
    np.testing.assert_allclose(support.exposure, np.ones(4))
    np.testing.assert_allclose(
        support.interpolate(time, np.array([1.25, 9.75])), [1.0, 10.0]
    )
    assert not support.contains(np.array([5.0]))[0]


@pytest.mark.unit
def test_single_sample_requires_explicit_recording_duration():
    from non_local_detector.encoding_time import EncodingSupport
    from non_local_detector.exceptions import ValidationError

    with pytest.raises(ValidationError, match="encoding_time_range"):
        EncodingSupport(np.array([1.0]), np.array([[3.0]]))
    support = EncodingSupport(
        np.array([1.0]), np.array([[3.0]]), encoding_time_range=[0.5, 1.5]
    )
    np.testing.assert_array_equal(support.exposure, [1.0])


@pytest.mark.unit
@pytest.mark.parametrize("sample_kind", ["singleton", "stationary", "missing"])
@pytest.mark.parametrize("one_bin", [False, True])
def test_glm_fits_stationary_tracking_with_known_exposure(sample_kind, one_bin):
    from non_local_detector.environment import Environment
    from non_local_detector.likelihoods.sorted_spikes_glm import (
        fit_sorted_spikes_glm_encoding_model,
    )

    time = np.array([0.0]) if sample_kind == "singleton" else np.arange(3.0)
    position = np.full((len(time), 1), 50.0)
    if sample_kind == "missing":
        position[[0, 2]] = np.nan
    environment = Environment(
        place_bin_size=100 if one_bin else 10, position_range=((0, 100),)
    ).fit_place_grid(np.array([[50.0]]), infer_track_interior=False)
    model = fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        [np.array([0.1 if sample_kind == "singleton" else 1.0])],
        environment,
        environment.place_bin_edges_,
        environment.edges_,
        environment.is_track_interior_,
        environment.is_track_boundary_,
        encoding_time_range=[time[0] - 0.5, time[-1] + 0.5],
        disable_progress_bar=True,
    )
    assert model["encoding_exposure_seconds"] == (
        3.0 if sample_kind == "stationary" else 1.0
    )
    assert np.all(np.isfinite(model["place_fields"]))
    if one_bin:
        np.testing.assert_allclose(
            np.asarray(model["place_fields"])[0, model["is_track_interior"]],
            1 / model["encoding_exposure_seconds"],
            rtol=1e-5,
        )


@pytest.mark.unit
def test_no_spike_likelihood_uses_each_direct_bin_duration():
    from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood

    actual = predict_no_spike_log_likelihood(
        [np.array([])], time_edges=np.array([0.0, 0.002, 0.006]), no_spike_rate=5.0
    )
    np.testing.assert_allclose(np.asarray(actual)[:, 0], [-0.01, -0.02], rtol=1e-6)


@pytest.mark.unit
def test_acquisition_bounds_do_not_authorize_interpolation_across_a_gap():
    from non_local_detector.encoding_time import EncodingSupport
    from non_local_detector.exceptions import DataError

    with pytest.raises(DataError, match="continuous tracking"):
        EncodingSupport(
            np.array([0.0, 1.0, 10.0, 11.0]),
            np.zeros((4, 1)),
            encoding_time_range=[0.0, 12.0],
        )


@pytest.mark.integration
@pytest.mark.parametrize("sampling_frequency", [30, 500])
def test_all_fitted_rates_are_hz_and_empty_bins_use_seconds(sampling_frequency):
    from non_local_detector.environment import Environment
    from non_local_detector.tests.likelihoods.conftest import (
        ALGORITHMS,
        fit_registered_backends,
    )

    time = np.arange(sampling_frequency) / sampling_frequency
    position = (50 + 40 * np.sin(2 * np.pi * time))[:, None]
    environment = Environment(
        place_bin_size=10, position_range=((0, 100),)
    ).fit_place_grid(position, infer_track_interior=False)
    spikes = [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]
    data = {
        "position_time": time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": spikes,
        "encoding_features": [np.zeros((5, 1))],
    }
    params = {
        name: {"encoding_time_range": [0.0, 1.0], "disable_progress_bar": True}
        for name in ALGORITHMS
    }
    for name in ("clusterless_gmm",):
        params[name].update(
            gmm_components_occupancy=1, gmm_components_gpi=1, gmm_components_joint=1
        )
    edges = np.array([0.01, 0.012, 0.016])
    for name, predict, model, clusterless in fit_registered_backends(data, params):
        assert model["rate_units"] == "Hz", name
        assert model["encoding_exposure_seconds"] == pytest.approx(1.0), name
        if "mean_rates" in model:
            np.testing.assert_allclose(
                model["mean_rates"], [5.0], rtol=1e-6, err_msg=name
            )
        args = [time, np.full_like(position, 50.0), [np.array([])]]
        if clusterless:
            args.append([np.empty((0, 1))])
        actual = np.asarray(predict(*args, time_edges=edges, **model))
        ground = model.get("summed_ground_process_intensity")
        if ground is None:
            ground = model["no_spike_part_log_likelihood"][model["is_track_interior"]]
        np.testing.assert_allclose(
            actual,
            -np.diff(edges)[:, None] * np.asarray(ground),
            rtol=1e-6,
            atol=1e-7,
            err_msg=name,
        )
        for local in (False, True):
            empty = np.asarray(
                predict(*args, time_edges=edges, **model, is_local=local)
            )
            event_args = [
                time,
                np.full_like(position, 50.0),
                [np.array([0.011, 0.014])],
            ]
            if clusterless:
                event_args.append([np.zeros((2, 1))])
            event = np.asarray(
                predict(*event_args, time_edges=edges, **model, is_local=local)
            )
            evidence = event - empty
            np.testing.assert_allclose(
                evidence[1] - evidence[0],
                np.log(2.0),
                rtol=1e-6,
                atol=1e-5,
                err_msg=name,
            )
            row = np.asarray(
                predict(
                    *event_args,
                    time_edges=edges,
                    **model,
                    is_local=local,
                    row_slice=slice(1, 2),
                )
            )
            np.testing.assert_allclose(
                row, event[1:2], rtol=1e-6, atol=1e-5, err_msg=name
            )


@pytest.mark.integration
@pytest.mark.parametrize("case", ["clipped", "missing", "disconnected"])
def test_all_backends_fit_the_same_declared_support(case):
    from non_local_detector.environment import Environment
    from non_local_detector.tests.likelihoods.conftest import (
        ALGORITHMS,
        fit_registered_backends,
    )

    if case == "clipped":
        time = np.array([0.0, 1.0])
        position = np.array([25.0, 75.0])[:, None]
        events = np.array([0.1, 0.8])
        support_params = {"encoding_time_range": [0.0, 0.25]}
        expected_exposure, expected_rate = 0.25, 4.0
    elif case == "missing":
        time = np.arange(5.0)
        position = np.array([25.0, 75.0, np.nan, 25.0, 75.0])[:, None]
        events = np.array([-0.25, 1.25, 2.0, 2.75, 4.25])
        support_params = {}
        expected_exposure, expected_rate = 4.0, 1.0
    else:
        time = np.array([0.0, 1.0, 10.0, 11.0])
        position = np.array([25.0, 75.0, 25.0, 75.0])[:, None]
        events = np.array([-0.25, 1.25, 5.0, 9.75, 11.25])
        support_params = {"valid_position_intervals": [[-0.5, 1.5], [9.5, 11.5]]}
        expected_exposure, expected_rate = 4.0, 1.0
    environment = Environment(
        place_bin_size=10, position_range=((0, 100),)
    ).fit_place_grid(np.array([[25.0], [75.0]]), infer_track_interior=False)
    data = {
        "position_time": time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": [events],
        "encoding_features": [np.zeros((len(events), 1))],
    }
    params = {
        name: {**support_params, "disable_progress_bar": True} for name in ALGORITHMS
    }
    params["clusterless_gmm"].update(
        gmm_components_occupancy=1, gmm_components_gpi=1, gmm_components_joint=1
    )
    for name, _, model, _ in fit_registered_backends(data, params):
        assert model["encoding_exposure_seconds"] == pytest.approx(expected_exposure), (
            name
        )
        if "mean_rates" in model:
            np.testing.assert_allclose(
                model["mean_rates"], [expected_rate], rtol=1e-6, err_msg=name
            )
        assert np.isfinite(
            np.asarray(
                model.get("place_fields", model.get("summed_ground_process_intensity"))
            )
        ).all(), name


@pytest.mark.unit
def test_adjacent_tracking_segments_keep_their_own_endpoint_basis():
    from non_local_detector.encoding_time import EncodingSupport

    time = np.arange(4.0)
    support = EncodingSupport(
        time, time[:, None], valid_position_intervals=[[-0.5, 1.1], [1.1, 3.5]]
    )
    np.testing.assert_allclose(support.exposure, [1.0, 0.6, 1.4, 1.0])
    np.testing.assert_allclose(
        support.interpolate(time, np.array([1.05, 1.1])), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        support.event_counts(np.array([1.1]), np.ones(4)), [0.0, 0.0, 1.0, 0.0]
    )
