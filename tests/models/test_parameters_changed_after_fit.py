"""Changing a detector's parameters after fitting leaves the fit as it was.

``set_params`` changes the configuration, not the fit: decoding uses the state
layout and continuous transition the fit recorded (``state_space_``) until the
next fit, and a detector whose parameters no longer describe its fit is not
saved.
"""

import copy

import matplotlib as mpl
import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    RandomWalk,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ParameterWarning, ValidationError
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import SORTED_SIM_RATE

mpl.use("Agg")
import matplotlib.pyplot as plt

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


def _changed_parameter_warnings(record, change):
    return [
        warning
        for warning in record
        if issubclass(warning.category, ParameterWarning)
        and change in str(warning.message)
    ]


@pytest.mark.parametrize("change", CHANGES)
def test_decoding_after_set_params_uses_the_fit(fitted, change):
    detector, decode, expected = fitted
    changed = detector.copy().set_params(**CHANGES[change](detector))

    with pytest.warns(ParameterWarning, match="decoding uses the values"):
        actual = changed.predict(**decode, return_outputs="all")
    with pytest.warns(ParameterWarning, match="decoding uses the values"):
        sequence = changed.most_likely_sequence(**decode)

    for name in expected.data_vars:
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
    for name in ("state_ind", "states", "state", "environments", "encoding_groups"):
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
    np.testing.assert_array_equal(
        sequence.state, detector.most_likely_sequence(**decode).state
    )


DECODES = {
    "predict": lambda detector, decode: detector.predict(**decode),
    "checkpointed predict": lambda detector, decode: detector.predict(
        **decode, inference_mode="checkpointed", output_mode="compact"
    ),
    "most_likely_sequence": lambda detector, decode: detector.most_likely_sequence(
        **decode
    ),
    "compute_log_likelihood": lambda detector, decode: detector.compute_log_likelihood(
        **decode
    ),
}


@pytest.mark.parametrize("decoding", DECODES)
def test_decoding_after_set_params_warns_once(fitted, decoding):
    detector, decode, _ = fitted
    changed = detector.copy().set_params(local_position_std=2.0)

    with pytest.warns(ParameterWarning) as record:
        DECODES[decoding](changed, decode)

    (warning,) = _changed_parameter_warnings(record, "local_position_std")
    assert "Refit" in str(warning.message)


@pytest.mark.parametrize("name", RECORDED)
def test_decoding_after_setting_an_equal_copy_does_not_warn(fitted, name):
    detector, decode, expected = fitted
    value = copy.deepcopy(detector.get_params(deep=False)[name])
    unchanged = detector.copy().set_params(**{name: value})

    # The suite turns warnings into errors.
    actual = unchanged.predict(**decode, return_outputs="all")

    np.testing.assert_array_equal(
        actual.acausal_posterior.values, expected.acausal_posterior.values
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


def _open_field_decoder(sorted_sim, movement_var):
    """A decoder fitted in 2-D, whose random walk has covariance ``movement_var``."""
    time, linear, spike_times = sorted_sim
    position = np.column_stack([linear, 14.0 + 10.0 * np.sin(time)])
    detector = SortedSpikesDecoder(
        environments=Environment(place_bin_size=4.0),
        continuous_transition_types=[[RandomWalk(movement_var=movement_var)]],
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    detector.fit(time, position, spike_times)
    decode = {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:50]),
        "position_time": time,
        "position": position,
    }
    return detector, decode


def test_saving_after_copying_a_transition_with_an_array_field_works(
    sorted_sim, tmp_path
):
    """A 2-D random walk's covariance is an array, which ``==`` cannot reduce."""
    detector, _ = _open_field_decoder(sorted_sim, 6.0 * np.eye(2))
    detector.set_params(
        continuous_transition_types=copy.deepcopy(detector.continuous_transition_types)
    ).save_model(tmp_path / "model.nld")


def test_object_array_parameter_decodes_but_is_not_saved(sorted_sim, tmp_path):
    """The saved format cannot hold it, so it cannot be compared with its
    fitted value: decoding does not warn (warnings are errors here), and
    ``save_model`` refuses it as it refuses any object array."""
    movement_var = np.array([[6.0, 0.0], [0.0, 6.0]], dtype=object)
    detector, decode = _open_field_decoder(sorted_sim, movement_var)

    results = detector.predict(**decode)

    assert np.isfinite(results.acausal_state_probabilities.values).all()
    with pytest.raises(ValidationError, match="Cannot save the object array"):
        detector.save_model(tmp_path / "model.nld")


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


STATE_CHANGES = {
    "state_names": CHANGES["state_names"],
    # Every state non-local, so the Local and No-Spike states would join the
    # conditional posterior.
    "observation_models": lambda detector: {
        "observation_models": [
            ObservationModel(obs.environment_name)
            for obs in detector.observation_models
        ]
    },
}


@pytest.mark.parametrize("change", STATE_CHANGES)
def test_reading_results_after_set_params_uses_the_fitted_states(fitted, change):
    detector, _, results = fitted
    changed = detector.copy().set_params(**STATE_CHANGES[change](detector))

    xr.testing.assert_identical(
        changed.get_conditional_non_local_posterior(results),
        detector.get_conditional_non_local_posterior(results),
    )
    fig, ax = plt.subplots()
    try:
        changed.plot_discrete_state_transition(ax=ax)
        labels = [label.get_text() for label in ax.get_xticklabels()]
    finally:
        plt.close(fig)
    assert labels == list(detector.state_names)
