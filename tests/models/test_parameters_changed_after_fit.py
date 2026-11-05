"""Changing a detector's parameters after fitting leaves the fit as it was.

``set_params`` changes the configuration, not the fit: decoding uses the state
layout and continuous transition the fit recorded (``state_space_``) until the
next fit, and a detector whose parameters no longer describe its fit is not
saved.
"""

import copy
import pickle

import matplotlib as mpl
import networkx as nx
import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryDiagonal,
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    RandomWalk,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import ParameterWarning, ValidationError
from non_local_detector.observation_models import ObservationModel
from tests.models._helpers import CLUSTERLESS_SIM_RATE, SORTED_SIM_RATE

mpl.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.integration

# The parameters a fit records in ``state_space_``.
RECORDED = (
    "state_names",
    "observation_models",
    "continuous_transition_types",
    "local_position_std",
    "discrete_transition_type",
    "discrete_initial_conditions",
    "environments",
)


@pytest.fixture(scope="module")
def recordings(sorted_sim, clusterless_sim):
    """Each family's fitted detector and what it decodes."""
    time, position, spike_times = sorted_sim
    sorted_detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times)
    clusterless_detector = NonLocalClusterlessDetector(
        sampling_frequency=CLUSTERLESS_SIM_RATE,
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        clusterless_sim.position_time,
        clusterless_sim.position,
        clusterless_sim.spike_times,
        clusterless_sim.spike_waveform_features,
    )
    return {
        "sorted": (
            sorted_detector,
            {
                "spike_times": spike_times,
                "time_edges": time_edges_from_centers(time[:200]),
                "position_time": time,
                "position": position,
            },
        ),
        "clusterless": (
            clusterless_detector,
            {
                "spike_times": clusterless_sim.spike_times,
                "spike_waveform_features": clusterless_sim.spike_waveform_features,
                "time_edges": time_edges_from_centers(
                    clusterless_sim.position_time[:200]
                ),
                "position_time": clusterless_sim.position_time,
                "position": clusterless_sim.position,
            },
        ),
    }


def _saved_and_loaded(detector, tmp_path_factory):
    path = tmp_path_factory.mktemp("saved") / "model.nld"
    detector.save_model(path)
    return type(detector).load_model(path)


def _pickled(detector, tmp_path_factory):
    return pickle.loads(pickle.dumps(detector))


def _fresh(detector, tmp_path_factory):
    return detector


ROUTES = {"fresh": _fresh, "saved": _saved_and_loaded, "pickled": _pickled}


@pytest.fixture(
    scope="module",
    params=[
        (family, route) for family in ("sorted", "clusterless") for route in ROUTES
    ],
    ids=lambda param: "-".join(param),
)
def fitted(request, recordings, tmp_path_factory):
    """A fitted detector, by each route, its inputs, and what it decodes."""
    family, route = request.param
    detector, decode = recordings[family]
    detector = ROUTES[route](detector, tmp_path_factory)
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
    "discrete_transition_type": lambda detector: {
        "discrete_transition_type": DiscreteStationaryDiagonal(
            diagonal_values=np.full(len(detector.state_names), 0.9)
        )
    },
    "discrete_initial_conditions": lambda detector: {
        "discrete_initial_conditions": np.roll(detector.discrete_initial_conditions, 1)
    },
    "environments": lambda detector: {
        "environments": tuple(
            Environment(
                **{**env.get_params(), "place_bin_size": 2 * env.place_bin_size}
            )
            for env in detector.environments
        )
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


def test_a_checkpointed_store_after_set_params_uses_the_fit(sorted_sim, tmp_path):
    """A store written after ``set_params`` holds the fit's state labels and
    posterior, not the new parameters'."""
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times, transition_representation="structured")
    decode = {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:200]),
        "position_time": time,
        "position": position,
        "inference_mode": "checkpointed",
        "output_mode": "spatial",
        "chunk_size": 50,
    }
    expected = detector.predict(**decode, result_path=tmp_path / "before")
    changed = detector.copy().set_params(
        state_names=[f"state {i}" for i in range(len(detector.state_names))]
    )

    with pytest.warns(ParameterWarning, match="decoding uses the values"):
        actual = changed.predict(**decode, result_path=tmp_path / "after")
    loaded = changed.load_results(tmp_path / "after")

    for result in (actual, loaded):
        np.testing.assert_array_equal(result["state"].values, expected["state"].values)
        np.testing.assert_array_equal(
            result["state_ind"].values, expected["state_ind"].values
        )
        np.testing.assert_array_equal(
            result.acausal_posterior.values, expected.acausal_posterior.values
        )


@pytest.mark.parametrize("decoding", DECODES)
def test_decoding_after_set_params_warns_once(fitted, decoding):
    detector, decode, _ = fitted
    changed = detector.copy().set_params(local_position_std=2.0)

    with pytest.warns(ParameterWarning) as record:
        DECODES[decoding](changed, decode)

    (warning,) = _changed_parameter_warnings(record, "local_position_std")
    assert "Refit" in str(warning.message)


def test_a_parameter_set_to_none_after_a_fit_is_a_change(sorted_sim):
    """None is a parameter's saved form, not the lack of one."""
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        local_position_std=2.0,
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times)
    detector.set_params(local_position_std=None)

    with pytest.warns(ParameterWarning) as record:
        detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time[:50]),
            position_time=time,
            position=position,
        )

    assert _changed_parameter_warnings(record, "local_position_std")


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


class _CopyRefusingName(str):
    """A state name that fails if it is deep-copied."""

    __slots__ = ()

    def __deepcopy__(self, memo):
        msg = "the live parameters were deep-copied"
        raise AssertionError(msg)


def test_decoding_does_not_copy_the_parameters(fitted):
    """The change check runs on every decode, so it reads them in place."""
    detector, decode, expected = fitted
    names = [_CopyRefusingName(name) for name in detector.state_names]
    unchanged = detector.copy().set_params(state_names=names)

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


def test_saving_after_changing_an_environment_in_place_is_refused(fitted, tmp_path):
    detector, _, _ = fitted
    changed = detector.copy()
    changed.environments[0].place_bin_size *= 2

    with pytest.raises(ValidationError, match="environments"):
        changed.save_model(tmp_path / "model.nld")


def test_saving_a_covariate_fit_given_a_stationary_transition_is_refused(
    sorted_sim, tmp_path
):
    """The file could not be read back: its covariates need a design."""
    time, position, spike_times = sorted_sim
    n_states = len(NonLocalSortedSpikesDetector().state_names)
    detector = NonLocalSortedSpikesDetector(
        discrete_transition_type=DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(n_states, 0.98), formula="1 + speed"
        ),
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data={"speed": np.linspace(0.0, 5.0, len(time))},
    )
    detector.set_params(
        discrete_transition_type=DiscreteStationaryDiagonal(
            diagonal_values=np.full(n_states, 0.98)
        )
    )

    with pytest.raises(ValidationError, match="discrete_transition_type"):
        detector.save_model(tmp_path / "model.nld")


def _track_graph_environment(position):
    length = position.max() - position.min()
    graph = nx.Graph()
    graph.add_node(0, pos=(position.min(), 0.0))
    graph.add_node(1, pos=(position.max(), 0.0))
    graph.add_edge(0, 1, distance=length, edge_id=0)
    environment = Environment(
        track_graph=graph,
        edge_order=[(0, 1)],
        edge_spacing=0.0,
        place_bin_size=4.0,
    )
    return environment, np.column_stack([position, np.zeros_like(position)])


def _environment_with_interior(position):
    is_interior = np.ones(50, dtype=bool)
    is_interior[20:25] = False
    environment = Environment(
        place_bin_size=(position.max() - position.min()) / 50,
        position_range=[(position.min(), position.max())],
        is_track_interior=is_interior,
    )
    return environment, position


@pytest.mark.parametrize(
    "make_environment", [_track_graph_environment, _environment_with_interior]
)
def test_a_configured_environment_is_unchanged_after_fitting(
    sorted_sim, make_environment, tmp_path
):
    time, position, spike_times = sorted_sim
    environment, position = make_environment(position)
    detector = NonLocalSortedSpikesDetector(
        environments=environment,
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times)

    # The suite turns warnings into errors.
    detector.predict(
        spike_times=spike_times,
        time_edges=time_edges_from_centers(time[:50]),
        position_time=time,
        position=position,
    )
    detector.save_model(tmp_path / "model.nld")
    copied = copy.deepcopy(detector.environments)
    detector.copy().set_params(environments=copied).save_model(tmp_path / "copy.nld")


def test_the_fit_records_no_fitted_grid_of_a_fitted_environment(sorted_sim):
    time, position, spike_times = sorted_sim
    environment = Environment(place_bin_size=1.0)
    environment.fit_place_grid(position[:, np.newaxis])
    detector = NonLocalSortedSpikesDetector(
        environments=environment,
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times)

    recorded = len(pickle.dumps(detector.state_space_.environments))

    assert recorded < len(pickle.dumps(detector.environments_)) / 10


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


def test_a_parameter_that_cannot_be_compared_does_not_hide_a_changed_one(sorted_sim):
    class UnregisteredWalk(RandomWalk):
        """Not registered for saving, so its parameter cannot be compared."""

    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(
        continuous_transition_types=[
            [
                UnregisteredWalk(**vars(transition))
                if isinstance(transition, RandomWalk)
                else transition
                for transition in row
            ]
            for row in NonLocalSortedSpikesDetector().continuous_transition_types
        ],
        sampling_frequency=SORTED_SIM_RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    ).fit(time, position, spike_times)
    detector.set_params(local_position_std=2.0)

    with pytest.warns(ParameterWarning) as record:
        detector.predict(
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time[:50]),
            position_time=time,
            position=position,
        )

    (warning,) = _changed_parameter_warnings(record, "local_position_std")
    assert "continuous_transition_types" not in str(warning.message)


def test_refusing_an_unsaveable_recorded_value_names_its_parameter(
    sorted_sim, tmp_path
):
    """The fit recorded an object array, which has no saved form, so the new
    value cannot equal it: the error names the parameter."""
    movement_var = np.array([[6.0, 0.0], [0.0, 6.0]], dtype=object)
    detector, _ = _open_field_decoder(sorted_sim, movement_var)
    detector.set_params(
        continuous_transition_types=[[RandomWalk(movement_var=6.0 * np.eye(2))]]
    )

    with pytest.raises(ValidationError, match="continuous_transition_types"):
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
