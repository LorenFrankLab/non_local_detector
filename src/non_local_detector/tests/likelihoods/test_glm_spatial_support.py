"""The spline's physical domain must include the full environment support."""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
)


def _fit(env, time, position, spikes):
    return fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        [spikes],
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
        encoding_time_range=[time[0], time[-1]],
        disable_progress_bar=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize("bin_size", [5.0, 10.0])
def test_public_glm_natural_spline_boundaries_match_physical_environment(bin_size):
    time = np.linspace(0, 1, 201)
    position = (100 * time)[:, None]
    env = Environment(
        place_bin_size=bin_size, position_range=((0.0, 100.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    model = _fit(env, time, position, np.linspace(0.55, 0.95, 20))
    splines = [
        transform
        for info in model["emission_design_info"].factor_infos.values()
        for transform in info.state["transforms"].values()
        if hasattr(transform, "_all_knots")
    ]
    assert len(splines) == 1
    # Natural spline boundary conditions belong at the physical track ends.
    # Imposing them at interior bin centers changes both the spline and its
    # coefficient prior, and makes the half-bin inside each end extrapolation.
    np.testing.assert_array_equal(splines[0]._all_knots[[0, -1]], [0.0, 100.0])


@pytest.mark.unit
def test_public_glm_one_center_grid_retains_intercept_only_model():
    time = np.array([0.0, 1.0])
    position = time[:, None]
    env = Environment(place_bin_size=2.0, position_range=((0.0, 1.0),)).fit_place_grid(
        position, infer_track_interior=False
    )
    model = _fit(env, time, position, np.array([0.25, 0.75]))
    assert model["coefficients"].shape == (1, 1)
    np.testing.assert_allclose(model["place_fields"], 2.0, rtol=1e-6)
