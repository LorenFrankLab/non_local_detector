"""Numerical integration nodes must not add statistical mixture components."""

import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "weights,expected",
    [(None, 3), ([1.0, 0.0, 0.25], 2), ([0.0, 0.0, 0.0], 3)],
    ids=["unweighted", "weighted", "zero-weight-fallback"],
)
def test_occupancy_component_cap_is_independent_of_trajectory_quadrature(
    weights, expected
):
    time = np.arange(3.0)
    position = time[:, None]
    fits = []
    for grid_spacing in [2.0, 0.25]:
        environment = Environment(
            place_bin_size=grid_spacing, position_range=((0.0, 2.0),)
        ).fit_place_grid(position, infer_track_interior=False)
        with pytest.warns(
            UserWarning,
            match=f"reduced occupancy components from 10 to {expected}",
        ):
            fits.append(
                fit_clusterless_gmm_encoding_model(
                    time,
                    position,
                    [np.empty(0)],
                    [np.empty((0, 1))],
                    environment,
                    weights=None if weights is None else np.asarray(weights),
                    gmm_components_occupancy=10,
                    disable_progress_bar=True,
                )
            )
    assert [fit["occupancy_model"].n_components for fit in fits] == [expected] * 2
    assert [fit["gmm_effective_components"]["occupancy"] for fit in fits] == [
        expected
    ] * 2


@pytest.mark.unit
def test_clipped_support_counts_original_interpolation_anchors_for_gmm_cap():
    time = np.arange(5.0)
    position = time[:, None]
    environment = Environment(
        place_bin_size=0.25, position_range=((0.0, 4.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    # Only the basis functions at original rows 1, 2, and 3 carry exposure.
    # The two outer anchors remain statistical observations despite their
    # timestamps lying outside the acquisition clip.
    with pytest.warns(
        UserWarning, match="reduced occupancy components from 10 to 3"
    ):
        fit = fit_clusterless_gmm_encoding_model(
            time,
            position,
            [np.empty(0)],
            [np.empty((0, 1))],
            environment,
            encoding_time_range=np.array([1.4, 2.6]),
            gmm_components_occupancy=10,
            disable_progress_bar=True,
        )
    assert fit["occupancy_model"].n_components == 3
    assert fit["gmm_effective_components"]["occupancy"] == 3
