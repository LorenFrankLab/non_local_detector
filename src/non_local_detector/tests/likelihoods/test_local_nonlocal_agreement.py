"""Local and non-local likelihoods agree where the animal sits at a bin centre.

The local path scores spikes at the animal's position; the non-local path
scores them at every interior bin. Where the animal holds still at a bin
centre, the local row is the non-local row's entry for that bin, for every
registered backend.
"""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    FLOAT32_ROUNDING,
    fit_registered_backends,
)
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = pytest.mark.integration

DECODE_DT = 0.02
N_DECODE_BINS = 10
FIELD_CENTERS = (25.0, 75.0)
# Smooth across the 10 cm bins (the default position_std is below one bin).
FIT_PARAMS = {
    "sorted_spikes_diffusion": {"position_std": 10.0},
    "clusterless_diffusion": {"position_std": 10.0},
}


@pytest.fixture(scope="module")
def encoding_data():
    """Two units with fields at 25 and 75 cm on a 0-100 cm, 10 cm-bin track."""
    rng = np.random.default_rng(3)
    position_time = np.arange(0.0, 20.0, 0.01)
    position = (50.0 + 45.0 * np.sin(2 * np.pi * position_time / 5.0))[:, None]
    environment = Environment(
        environment_name="line", place_bin_size=10.0, position_range=((0.0, 100.0),)
    ).fit_place_grid(position=position, infer_track_interior=False)
    spike_times, features = [], []
    for center in FIELD_CENTERS:
        near = np.abs(position[:, 0] - center) < 10.0
        times = np.sort(rng.choice(position_time[near], 300, replace=False))
        spike_times.append(times + rng.uniform(-0.005, 0.005, times.size))
        features.append(rng.standard_normal((300, 2)) * 3.0 + center / 5.0)
    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "encoding_spike_times": spike_times,
        "encoding_features": features,
    }


@pytest.fixture(scope="module")
def fitted(encoding_data):
    return {
        name: (predict, model, is_clusterless)
        for name, predict, model, is_clusterless in fit_registered_backends(
            encoding_data, FIT_PARAMS
        )
    }


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("bin_center", [5.0, 25.0, 75.0])
def test_local_equals_nonlocal_at_the_animals_bin(
    fitted, encoding_data, name, bin_center
):
    """``local[:, 0] == nonlocal[:, j]`` for the bin ``j`` the animal sits in.

    Tolerance: ``FLOAT32_ROUNDING`` (rtol 1e-6, atol 1e-5). The two paths
    evaluate the same rates and marks through different float32 reductions
    (per-row matmuls or scatter-adds): on this fixture they differ by at most
    3.1e-5 absolute on log likelihoods of order 200, 1.7e-7 relative. A
    misread position or a lost spike moves a log likelihood by O(1); every
    non-local row here spans more than 4 across bins.
    """
    predict, model, is_clusterless = fitted[name]
    environment = encoding_data["environment"]
    rng = np.random.default_rng(1)

    decode_time = 100.0 + np.arange(N_DECODE_BINS) * DECODE_DT
    time_edges = time_edges_from_centers(decode_time)
    # Hold the animal at the bin centre across the whole stretch, so the local
    # paths read the same position at bin centres and at spike times.
    position_time = np.linspace(time_edges[0] - 1.0, time_edges[-1] + 1.0, 50)
    position = np.full((position_time.size, 1), bin_center)
    # Two to four spikes per bin per unit, jittered within the bin.
    spike_times = [
        np.sort(
            np.concatenate(
                [
                    rng.uniform(start, stop, rng.integers(2, 5))
                    for start, stop in zip(time_edges[:-1], time_edges[1:], strict=True)
                ]
            )
        )
        for _ in FIELD_CENTERS
    ]
    args = [position_time, position, spike_times]
    if is_clusterless:
        args.append(
            [
                rng.standard_normal((times.size, 2)) * 3.0 + center / 5.0
                for times, center in zip(spike_times, FIELD_CENTERS, strict=True)
            ]
        )

    local = np.asarray(predict(*args, time_edges=time_edges, is_local=True, **model))
    nonlocal_ = np.asarray(
        predict(*args, time_edges=time_edges, is_local=False, **model)
    )

    interior = environment.is_track_interior_.ravel()
    (interior_bin,) = np.flatnonzero(
        np.isclose(environment.place_bin_centers_[interior, 0], bin_center)
    )
    assert local.shape == (N_DECODE_BINS, 1)
    assert nonlocal_.shape == (N_DECODE_BINS, interior.sum())
    # The non-local row must discriminate bins, or the identity is vacuous.
    assert np.all(np.ptp(nonlocal_, axis=1) > 0.5)
    np.testing.assert_allclose(
        local[:, 0], nonlocal_[:, interior_bin], **FLOAT32_ROUNDING
    )
