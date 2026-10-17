"""A detector rejects decoding after its shared Environment is refit elsewhere.

Detectors keep the ``Environment`` objects they are given, and ``fit`` refits
their grids in place. Fitting a second detector that shares the object on other
positions therefore changes the first detector's grid under its fitted encoding
model.
"""

import numpy as np
import pytest

from non_local_detector import Environment, SortedSpikesDecoder, Uniform
from non_local_detector.exceptions import ValidationError


def _decoder(environment):
    return SortedSpikesDecoder(
        environments=environment,
        continuous_transition_types=[[Uniform()]],
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


@pytest.mark.integration
def test_refitting_a_shared_environment_invalidates_the_other_detector():
    time = np.arange(100) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spikes = [np.array([0.021, 0.067, 0.121, 0.167])]
    edges = np.arange(81) * 0.002
    environment = Environment(place_bin_size=5)

    first = _decoder(environment).fit(time, position, spikes)
    expected = first.predict(spikes, time_edges=edges, return_outputs="log_likelihood")
    n_interior = int(environment.is_track_interior_.sum())

    # Same number of bins, 37 cm further along the track.
    _decoder(environment).fit(time, position + 37.0, spikes)
    assert int(environment.is_track_interior_.sum()) == n_interior
    with pytest.raises(ValidationError, match="different environment grid"):
        first.predict(spikes, time_edges=edges)

    # Refitting the first detector restores a consistent model.
    first.fit(time, position, spikes)
    actual = first.predict(spikes, time_edges=edges, return_outputs="log_likelihood")
    np.testing.assert_array_equal(actual.log_likelihood, expected.log_likelihood)
