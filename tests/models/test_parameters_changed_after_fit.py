"""Changing a detector's parameters after fitting leaves the fit as it was.

``set_params`` changes the configuration, not the fit: decoding uses the state
layout and continuous transition the fit recorded (``state_space_``) until the
next fit, and a detector whose parameters no longer describe its fit is not
saved.
"""

import copy

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    RandomWalk,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import SORTED_SIM_RATE

pytestmark = pytest.mark.integration

# The parameters a fit records in ``state_space_``.
RECORDED = (
    "state_names",
    "observation_models",
    "continuous_transition_types",
    "local_position_std",
)


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


@pytest.mark.parametrize("name", RECORDED)
def test_saving_after_setting_an_equal_copy_works(fitted, name, tmp_path):
    """An equal copy of a recorded parameter is not a change, whatever the
    equality of its objects (``Discrete`` compares by identity)."""
    detector, _, _ = fitted
    value = copy.deepcopy(detector.get_params(deep=False)[name])
    detector.copy().set_params(**{name: value}).save_model(tmp_path / "model.nld")


def test_saving_after_copying_a_transition_with_an_array_field_works(
    sorted_sim, tmp_path
):
    """A 2-D random walk's covariance is an array, which ``==`` cannot reduce."""
    time, linear, spike_times = sorted_sim
    position = np.column_stack([linear, 14.0 + 10.0 * np.sin(time)])
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=4.0),
        continuous_transition_types=[[RandomWalk(movement_var=6.0 * np.eye(2))]],
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    detector.fit(time, position, spike_times)
    detector.set_params(
        continuous_transition_types=copy.deepcopy(detector.continuous_transition_types)
    ).save_model(tmp_path / "model.nld")


def test_saving_after_changing_a_transition_in_place_is_refused(fitted, tmp_path):
    """The fit records copies, so an in-place change is a change too."""
    detector, _, _ = fitted
    changed = detector.copy()
    walk = next(
        transition
        for row in changed.continuous_transition_types
        for transition in row
        if isinstance(transition, RandomWalk)
    )
    walk.movement_var = 2 * walk.movement_var

    with pytest.raises(ValidationError, match="continuous_transition_types"):
        changed.save_model(tmp_path / "model.nld")
