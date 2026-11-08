"""Every backend recovers the declared rate when tracking is NaN outside the
declared segment.

The ``EncodingSupport`` cases are in ``tests/test_encoding_time.py``.
"""

import numpy as np
import pytest

pytestmark = [
    # Tiny synthetic fixtures: a small position_std and zero-rate electrodes are
    # properties of the fixture, not under test here.
    pytest.mark.filterwarnings("ignore:position_std \\(:UserWarning"),
    pytest.mark.filterwarnings("ignore:Clusterless GMM:UserWarning"),
    pytest.mark.filterwarnings("ignore:Clusterless mark GLM:UserWarning"),
    # k-means seeding finds fewer distinct points than components on the tiny fixtures.
    pytest.mark.filterwarnings(
        "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:electrode \\d+ has zero total encoding weight:UserWarning"
    ),
]


@pytest.mark.integration
def test_all_backends_recover_declared_rate_with_nan_outside_segment():
    from non_local_detector.environment import Environment
    from tests.likelihoods.conftest import (
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
    params["clusterless_gmm"].update(gmm_components_occupancy=1, gmm_components_joint=1)
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
