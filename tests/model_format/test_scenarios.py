"""Saved models of the configurations whose state is rebuilt differently.

Each case fits a detector, saves and reloads it, and requires the same fitted
state and exactly the same posterior: covariate-dependent discrete
transitions (rebuilt patsy design), a track graph (grid rebuilt from the
graph), a 2-D open field (grid rebuilt from its edges and mask), an
``EmpiricalMovement`` block (stored, as the fit positions are not), the
structured and auto transition representations, and models after EM.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import non_local_detector as nld
from non_local_detector import Environment, time_edges_from_centers
from non_local_detector.models.base import SortedSpikesDetector
from non_local_detector.observation_models import ObservationModel
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import N_DECODE, algorithm_params, decode

pytestmark = pytest.mark.integration


def _speed(time, position):
    return np.abs(np.gradient(position, time, axis=0)).reshape(len(time), -1).sum(1)


def _round_trip(detector, tmp_path):
    path = tmp_path / "model.nld"
    detector.save_model(path)
    return type(detector).load_model(path)


def _assert_round_trip(detector, tmp_path, time, position, spikes, **decode_kwargs):
    loaded = _round_trip(detector, tmp_path)
    assert_same_fitted_state(detector, loaded)
    expected = decode(detector, time, position, spikes, **decode_kwargs)
    actual = decode(loaded, time, position, spikes, **decode_kwargs)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    return loaded


def _sorted_detector(**kwargs):
    return nld.NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"), **kwargs
    )


@pytest.mark.parametrize("container", [dict, pd.DataFrame])
def test_covariate_dependent_discrete_transitions(
    container, sorted_recording, tmp_path
):
    time, position, spike_times = sorted_recording
    speed = _speed(time, position)
    detector = _sorted_detector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.array([0.98, 0.98, 0.98, 0.98])
        )
    )
    detector.fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data=container({"speed": speed}),
    )
    covariates = container({"speed": speed[:N_DECODE]})

    loaded = _assert_round_trip(
        detector,
        tmp_path,
        time,
        position,
        {"spike_times": spike_times},
        discrete_transition_covariate_data=covariates,
    )
    assert loaded.discrete_transition_design_matrix_ is not None


@pytest.mark.filterwarnings("ignore:EM did not converge:UserWarning")  # 1-2 iterations
def test_covariate_transitions_after_em(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    speed = _speed(time, position)
    detector = _sorted_detector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.array([0.98, 0.98, 0.98, 0.98])
        )
    )
    detector.estimate_parameters(
        time,
        position,
        spike_times,
        time_edges=time_edges_from_centers(time),
        discrete_transition_covariate_data={"speed": speed},
        max_iter=2,
        store_log_likelihood=True,
    )

    _assert_round_trip(
        detector,
        tmp_path,
        time,
        position,
        {"spike_times": spike_times},
        discrete_transition_covariate_data={"speed": speed[:N_DECODE]},
    )


def _track_graph():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(15.0, 0.0))
    graph.add_node(2, pos=(15.0, 15.0))
    graph.add_edge(0, 1, distance=15.0, edge_id=0)
    graph.add_edge(1, 2, distance=15.0, edge_id=1)
    return graph


def _on_track(linear):
    """2-D positions along the L-shaped track of ``_track_graph``."""
    return np.column_stack(
        [np.minimum(linear, 15.0), np.clip(linear - 15.0, 0.0, None)]
    )


@pytest.mark.parametrize("representation", ["dense", "auto"])
def test_track_graph_environment(representation, sorted_recording, tmp_path):
    time, linear, spike_times = sorted_recording
    position = _on_track(linear)
    environment = Environment(
        place_bin_size=2.0,
        track_graph=_track_graph(),
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
    )
    detector = _sorted_detector(environments=environment)
    detector.fit(time, position, spike_times, transition_representation=representation)

    _assert_round_trip(detector, tmp_path, time, position, {"spike_times": spike_times})


@pytest.mark.parametrize("representation", ["dense", "structured", "auto"])
@pytest.mark.parametrize("algorithm", ["sorted_spikes_kde", "sorted_spikes_glm"])
def test_two_dimensional_open_field(
    representation, algorithm, sorted_recording, tmp_path
):
    time, linear, spike_times = sorted_recording
    position = np.column_stack([linear, 14.0 + 10.0 * np.sin(time)])
    detector = nld.NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=4.0),
        sorted_spikes_algorithm=algorithm,
        sorted_spikes_algorithm_params=algorithm_params(algorithm),
    )
    detector.fit(time, position, spike_times, transition_representation=representation)

    _assert_round_trip(detector, tmp_path, time, position, {"spike_times": spike_times})


def test_empirical_movement_block_is_stored(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    detector = nld.SortedSpikesDecoder(
        continuous_transition_types=[[nld.EmpiricalMovement()]],
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )
    detector.fit(time, position, spike_times)

    loaded = _assert_round_trip(
        detector, tmp_path, time, position, {"spike_times": spike_times}
    )
    assert loaded._rebuild_check_.stored_blocks == ((0, 0),)


@pytest.mark.filterwarnings("ignore:EM did not converge:UserWarning")  # 1-2 iterations
def test_em_after_an_auto_fit_rebuilds_the_dense_matrix(
    clusterless_recording, tmp_path
):
    """Dense EM after an "auto" fit converts its operator to a dense matrix
    while the environments keep deferred distances."""
    time, position, spike_times, features = clusterless_recording
    detector = nld.NonLocalClusterlessDetector(
        clusterless_algorithm_params=algorithm_params("clusterless_kde"),
    )
    spikes = {"spike_times": spike_times, "spike_waveform_features": features}
    detector.estimate_parameters(
        time,
        position,
        **spikes,
        time_edges=time_edges_from_centers(time[:2000]),
        max_iter=1,
    )
    assert detector._transition_representation_ == "dense"

    _assert_round_trip(detector, tmp_path, time, position, spikes)


def test_multi_bin_local_state(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    detector = _sorted_detector(local_position_std=3.0)
    detector.fit(time, position, spike_times)

    _assert_round_trip(detector, tmp_path, time, position, {"spike_times": spike_times})


def test_encoding_groups(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    detector = SortedSpikesDetector(
        environments=(Environment(),),
        discrete_transition_concentration=1.1,
        discrete_transition_stickiness=np.zeros(2),
        discrete_transition_regularization=0.1,
        observation_models=[
            ObservationModel(encoding_group="first"),
            ObservationModel(encoding_group="second"),
        ],
        continuous_transition_types=[
            [nld.RandomWalk(), nld.Uniform()],
            [nld.Uniform(), nld.RandomWalk()],
        ],
        discrete_initial_conditions=np.array([0.5, 0.5]),
        continuous_initial_conditions_types=[
            nld.UniformInitialConditions(),
            nld.UniformInitialConditions(),
        ],
        discrete_transition_type=nld.DiscreteStationaryDiagonal(
            diagonal_values=np.array([0.98, 0.98])
        ),
        state_names=["first", "second"],
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )
    half = len(time) // 2
    labels = np.array(["first"] * half + ["second"] * (len(time) - half))
    detector.fit(time, position, spike_times, encoding_group_labels=labels)

    loaded = _assert_round_trip(
        detector, tmp_path, time, position, {"spike_times": spike_times}
    )
    assert set(loaded.encoding_model_) == {("", "first"), ("", "second")}
