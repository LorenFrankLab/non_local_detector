"""Detectors fit their own copies of the ``Environment`` objects they are given.

Fitting never changes the caller's environment, so detectors that share one
keep their own grids: fitting a second detector on other positions leaves the
first detector's grid, and its decodes, unchanged.
"""

import numpy as np
import pytest

from non_local_detector import Environment, SortedSpikesDecoder, Uniform


def _decoder(environment):
    return SortedSpikesDecoder(
        environments=environment,
        continuous_transition_types=[[Uniform()]],
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


@pytest.mark.integration
def test_detectors_sharing_an_environment_keep_their_own_grids():
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    edges = np.arange(81) * 0.002
    environment = Environment(place_bin_size=5)

    first = _decoder(environment).fit(time, position, spikes)
    expected = first.predict(spikes, time_edges=edges, return_outputs="log_likelihood")
    first_centers = first.environments[0].place_bin_centers_.copy()

    # 37 cm further along the track: a different grid.
    second = _decoder(environment).fit(time, position + 37.0, spikes)
    second_result = second.predict(
        spikes, time_edges=edges, return_outputs="log_likelihood"
    )
    assert np.isfinite(second_result.log_likelihood.values).all()
    np.testing.assert_allclose(
        second.environments[0].place_bin_centers_, first_centers + 37.0
    )

    # The first detector's grid and decodes are unchanged.
    np.testing.assert_array_equal(
        first.environments[0].place_bin_centers_, first_centers
    )
    actual = first.predict(spikes, time_edges=edges, return_outputs="log_likelihood")
    np.testing.assert_array_equal(actual.log_likelihood, expected.log_likelihood)

    # Each detector holds its own fitted copy; the caller's object stays unfitted.
    assert first.environments[0] is not environment
    assert second.environments[0] is not first.environments[0]
    assert not environment._is_fitted
    assert environment.place_bin_centers_ is None
    assert environment.get_cached("linear_position") is None
    assert environment.get_params() == Environment(place_bin_size=5).get_params()
