"""Declared tracking endpoints are independent of samples outside the segment."""

import numpy as np
import pytest

from non_local_detector.encoding_time import EncodingSupport


@pytest.mark.unit
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_outside_nan_does_not_shorten_declared_tracking_support(side):
    time = np.arange(-1.0, 4.0)
    position = np.zeros((5, 1))
    if side in ("left", "both"):
        position[0] = np.nan
    if side in ("right", "both"):
        position[-1] = np.nan
    support = EncodingSupport(time, position, valid_position_intervals=[[-0.75, 2.75]])
    np.testing.assert_allclose(support.exposure, [0.0, 1.25, 1.0, 1.25, 0.0])
    events = np.array([-0.6, 0.0, 1.0, 2.0, 2.6])
    np.testing.assert_array_equal(support.contains(events), np.ones(5, dtype=bool))
    assert support.event_counts(events, np.ones(5)).sum() == 5.0


@pytest.mark.unit
def test_declared_interval_still_splits_at_internal_nan_samples():
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    position[2] = np.nan
    support = EncodingSupport(time, position, valid_position_intervals=[[-0.5, 4.5]])
    np.testing.assert_allclose(support.exposure, [1.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        support.contains(np.array([1.25, 1.6, 2.4, 2.75])),
        [True, False, False, True],
    )


@pytest.mark.unit
def test_acquisition_clip_keeps_original_basis_inside_declared_interval():
    support = EncodingSupport(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0], [1.0], [np.nan]]),
        valid_position_intervals=[[-0.5, 1.5]],
        encoding_time_range=[0.0, 0.25],
    )
    np.testing.assert_allclose(support.exposure, [0.21875, 0.03125, 0.0])
    np.testing.assert_allclose(
        support.interpolate(np.array([0.0, 1.0, np.nan]), np.array([0.125])),
        [0.125],
    )


@pytest.mark.integration
def test_all_backends_recover_declared_rate_with_nan_outside_segment():
    from non_local_detector.environment import Environment
    from non_local_detector.tests.likelihoods.conftest import (
        ALGORITHMS,
        fit_registered_backends,
    )

    time = np.arange(4.0)
    position = np.zeros((4, 1))
    position[-1] = np.nan
    environment = Environment(
        place_bin_size=1, position_range=((-1.5, 1.5),)
    ).fit_place_grid(np.array([[0.0]]), infer_track_interior=False)
    spikes = np.array([0.0, 1.0, 2.0, 2.6])
    data = {
        "position_time": time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": [spikes],
        "encoding_features": [np.zeros((4, 1))],
    }
    params = {
        name: {
            "valid_position_intervals": [[-0.5, 2.75]],
            "disable_progress_bar": True,
        }
        for name in ALGORITHMS
    }
    params["sorted_spikes_glm"]["emission_knot_spacing"] = 1.0
    params["sorted_spikes_mrf"]["penalty"] = 1.0
    params["clusterless_gmm"].update(
        gmm_components_occupancy=1, gmm_components_gpi=1, gmm_components_joint=1
    )
    edges = np.array([0.01, 0.012, 0.016])
    center = np.flatnonzero(
        environment.place_bin_centers_[environment.is_track_interior_.ravel(), 0] == 0.0
    )[0]
    for name, predict, model, clusterless in fit_registered_backends(data, params):
        assert model["encoding_exposure_seconds"] == pytest.approx(3.25), name
        if "mean_rates" in model:
            np.testing.assert_allclose(model["mean_rates"], [4 / 3.25], err_msg=name)
        args = [time, np.zeros_like(position), [np.array([])]]
        if clusterless:
            args.append([np.empty((0, 1))])
        likelihood = np.asarray(predict(*args, time_edges=edges, **model))
        np.testing.assert_allclose(
            -likelihood[:, center],
            (4 / 3.25) * np.diff(edges),
            rtol=1e-5,
            atol=0,
            err_msg=name,
        )
