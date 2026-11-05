"""Changing a detector's parameters after fitting leaves the fit as it was.

``set_params`` changes the configuration, not the fit: decoding uses the state
layout and continuous transition the fit recorded (``state_space_``) until the
next fit, and a detector whose parameters no longer describe its fit is not
saved.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.exceptions import ValidationError
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fitted(sorted_sim):
    """A fitted detector, its recording, and what it decodes."""
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    detector.fit(time, position, spike_times)
    decode = {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:200]),
        "position_time": time,
        "position": position,
    }
    return detector, decode, detector.predict(**decode, return_outputs="all")


def _renamed_observation_models(detector):
    # A new encoding group for the Non-Local states, which the fit has no
    # encoding model of.
    return [
        obs
        if obs.is_local or obs.is_no_spike
        else ObservationModel(obs.environment_name, encoding_group=1)
        for obs in detector.observation_models
    ]


CHANGES = {
    "local_position_std": lambda detector: {"local_position_std": 0.0},
    "state_names": lambda detector: {
        "state_names": [f"state {i}" for i in range(len(detector.state_names))]
    },
    "observation_models": lambda detector: {
        "observation_models": _renamed_observation_models(detector)
    },
}


@pytest.mark.parametrize("change", CHANGES)
def test_decoding_after_set_params_uses_the_fit(fitted, change):
    detector, decode, expected = fitted
    changed = detector.copy().set_params(**CHANGES[change](detector))

    actual = changed.predict(**decode, return_outputs="all")

    for name in expected.data_vars:
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
    for name in ("state_ind", "states", "state", "environments", "encoding_groups"):
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
    np.testing.assert_array_equal(
        changed.most_likely_sequence(**decode).state,
        detector.most_likely_sequence(**decode).state,
    )


@pytest.mark.parametrize("change", CHANGES)
def test_saving_after_set_params_is_refused(fitted, change, tmp_path):
    detector, _, _ = fitted
    changed = detector.copy().set_params(**CHANGES[change](detector))

    with pytest.raises(ValidationError, match=change):
        changed.save_model(tmp_path / "model.nld")

    # Setting the parameters back saves the fit again.
    changed.set_params(**{change: detector.get_params(deep=False)[change]}).save_model(
        tmp_path / "model.nld"
    )
