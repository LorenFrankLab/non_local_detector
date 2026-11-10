"""Independent references for encoding exposure and decode duration."""

import numpy as np
import pytest

from tests.conftest import tiny_fixture_warnings

pytestmark = tiny_fixture_warnings


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


# synthetic tracking jumps are incidental to this test.
@pytest.mark.filterwarnings("ignore:Tracking moves up to:UserWarning")
@pytest.mark.integration
@pytest.mark.parametrize("sampling_frequency", [30, 500])
def test_all_fitted_rates_are_hz_and_empty_bins_use_seconds(sampling_frequency):
    from non_local_detector.environment import Environment
    from tests.likelihoods.conftest import (
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
        params[name].update(gmm_components_occupancy=1, gmm_components_joint=1)
    edges = np.array([0.01, 0.012, 0.016])
    for name, predict, model, clusterless in fit_registered_backends(data, params):
        args = [time, np.full_like(position, 50.0), [np.array([])]]
        if clusterless:
            args.append([np.empty((0, 1))])
        actual = np.asarray(predict(*args, time_edges=edges, **model))
        # Sorted backends: the per-neuron place fields, not the fit's own sum.
        ground = model.get("summed_ground_process_intensity")
        if ground is None:
            ground = np.asarray(model["place_fields"])[
                :, model["is_track_interior"]
            ].sum(axis=0)
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


# synthetic tracking jumps are incidental to this test.
@pytest.mark.filterwarnings("ignore:Tracking moves up to:UserWarning")
@pytest.mark.integration
@pytest.mark.parametrize("case", ["clipped", "missing", "disconnected"])
def test_all_backends_fit_the_same_declared_support(case):
    from non_local_detector.environment import Environment
    from tests.likelihoods.conftest import (
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
    params["clusterless_gmm"].update(gmm_components_occupancy=1, gmm_components_joint=1)
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
