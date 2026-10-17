"""The GLM's spline basis is defined on the environment, not on the tracking.

A basis fitted to the sampled positions puts its natural-spline boundaries at
the edges of the visited range, so log-rate extrapolates linearly into bins the
animal never visited. Anchored to the environment, those bins are governed by
the penalty instead and stay near the neuron's baseline.
"""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
)

pytestmark = pytest.mark.unit


def _fit(env, time, position, spikes, **kwargs):
    return fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        spikes,
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
        disable_progress_bar=True,
        **kwargs,
    )


def _spline_knots(model):
    return [
        transform._all_knots
        for info in model["emission_design_info"].factor_infos.values()
        for transform in info.state["transforms"].values()
        if hasattr(transform, "_all_knots")
    ]


def _partial_track(sampling_frequency=30.0, duration=300.0, seed=0):
    """An animal running over about 0-140 cm of a 0-170 cm track, with a place
    cell whose field is still rising at the edge of the visited range."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, duration, 1 / sampling_frequency)
    position = 70.0 + 70.0 * np.sin(2 * np.pi * time / 19.3)
    rate = 20.0 * np.exp(-0.5 * ((position - 150.0) / 10.0) ** 2)
    spikes = np.sort(
        time[rng.random(time.size) < rate / sampling_frequency]
        + rng.uniform(0, 1 / sampling_frequency)
    )
    env = Environment(place_bin_size=2.0, position_range=((0.0, 170.0),))
    env = env.fit_place_grid(position[:, None], infer_track_interior=False)
    return env, time, position[:, None], [spikes[spikes < time[-1]]]


def test_unvisited_bins_do_not_extrapolate_the_rate():
    env, time, position, spikes = _partial_track()
    model = _fit(env, time, position, spikes)
    centers = env.place_bin_centers_[env.is_track_interior_.ravel()].ravel()
    field = np.asarray(model["place_fields"])[0]
    visited_peak = field[centers <= 140.0].max()
    assert field[centers > 145.0].max() <= visited_peak


def test_basis_does_not_depend_on_the_tracking_rate():
    coarse = _partial_track(sampling_frequency=30.0)
    fine = _partial_track(sampling_frequency=120.0)
    knots = [_spline_knots(_fit(*track)) for track in (coarse, fine)]
    np.testing.assert_array_equal(knots[0][0], knots[1][0])


@pytest.mark.parametrize("bin_size", [5.0, 10.0])
def test_natural_spline_boundaries_are_the_physical_track_ends(bin_size):
    time = np.linspace(0, 1, 201)
    position = (100 * time)[:, None]
    env = Environment(
        place_bin_size=bin_size, position_range=((0.0, 100.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    model = _fit(
        env,
        time,
        position,
        [np.linspace(0.55, 0.95, 20)],
        encoding_time_range=[time[0], time[-1]],
    )
    (knots,) = _spline_knots(model)
    np.testing.assert_array_equal(knots[[0, -1]], [0.0, 100.0])


def test_one_center_grid_is_intercept_only():
    time = np.array([0.0, 1.0])
    position = time[:, None]
    env = Environment(place_bin_size=2.0, position_range=((0.0, 1.0),)).fit_place_grid(
        position, infer_track_interior=False
    )
    model = _fit(
        env,
        time,
        position,
        [np.array([0.25, 0.75])],
        encoding_time_range=[time[0], time[-1]],
    )
    assert model["coefficients"].shape == (1, 1)
    np.testing.assert_allclose(model["place_fields"], 2.0, rtol=1e-6)


def test_unvisited_bins_do_not_extrapolate_in_two_dimensions():
    """The tensor-product basis spans the whole environment in each dimension."""
    rng = np.random.default_rng(0)
    sampling_frequency = 30.0
    time = np.arange(0.0, 300.0, 1 / sampling_frequency)
    # The animal covers x in about [0, 70] of a 100 x 100 box; the field is
    # still rising at x = 70.
    x = 35.0 + 35.0 * np.sin(2 * np.pi * time / 23.7)
    y = 50.0 + 50.0 * np.sin(2 * np.pi * time / 31.3 + 1.0)
    rate = 20.0 * np.exp(-0.5 * (((x - 80.0) / 10.0) ** 2 + ((y - 50.0) / 10.0) ** 2))
    spikes = [time[rng.random(time.size) < rate / sampling_frequency]]
    position = np.c_[x, y]
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 100.0), (0.0, 100.0))
    ).fit_place_grid(position, infer_track_interior=False)
    model = _fit(env, time, position, spikes)
    edges = env.place_bin_edges_
    for knots, low, high in zip(
        _spline_knots(model), edges.min(axis=0), edges.max(axis=0), strict=True
    ):
        np.testing.assert_array_equal(knots[[0, -1]], [low, high])
    centers = env.place_bin_centers_[env.is_track_interior_.ravel()]
    field = np.asarray(model["place_fields"])[0]
    assert field[centers[:, 0] > 75.0].max() <= field[centers[:, 0] <= 70.0].max()
