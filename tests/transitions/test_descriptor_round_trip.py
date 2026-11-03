"""Transition and initial-condition objects hold only their configuration.

A fit must not add attributes to them (for example the environment they were
built on), so ``type(obj)(**vars(obj))`` rebuilds each one after a fit, as
Spyglass does with stored parameters.
"""

import pickle
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from non_local_detector import (
    Discrete,
    DiscreteNonStationaryCustom,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
    EmpiricalMovement,
    Environment,
    Identity,
    NonLocalSortedSpikesDetector,
    RandomWalk,
    RandomWalkDirection1,
    RandomWalkDirection2,
    SortedSpikesDecoder,
    Uniform,
    UniformInitialConditions,
)
from non_local_detector.observation_models import ObservationModel

pytestmark = pytest.mark.integration

SAMPLING_FREQUENCY = 500.0


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    time = np.arange(2000) / SAMPLING_FREQUENCY
    position = 50 + 40 * np.sin(2 * np.pi * time / 2.0)
    spike_times = []
    for center in (20.0, 50.0, 80.0):
        rate = 30 * np.exp(-((position - center) ** 2) / (2 * 8.0**2)) + 1.0
        counts = rng.poisson(rate / SAMPLING_FREQUENCY)
        jitter = rng.uniform(-0.5, 0.5, counts.sum()) / SAMPLING_FREQUENCY
        spike_times.append(np.sort(np.repeat(time, counts) + jitter))
    speed = np.abs(np.gradient(position, time))
    return time, position[:, None], spike_times, speed


def _assert_configuration_only(obj):
    if is_dataclass(obj):
        assert set(vars(obj)) == {item.name for item in fields(obj)}, type(obj)
    else:
        assert vars(obj) == {}, type(obj)
    rebuilt = type(obj)(**vars(obj))
    assert set(vars(rebuilt)) == set(vars(obj))


def _fitted_objects(detector):
    transitions = [t for row in detector.continuous_transition_types for t in row]
    return [
        *transitions,
        detector.discrete_transition_type,
        *detector.continuous_initial_conditions_types,
    ]


@pytest.mark.parametrize(
    ("discrete_transition_type", "covariates"),
    [
        (DiscreteStationaryDiagonal(np.full(4, 0.98)), False),
        (DiscreteStationaryCustom(np.full((4, 4), 0.25)), False),
        (DiscreteNonStationaryDiagonal(np.full(4, 0.98), formula="1 + speed"), True),
        (
            DiscreteNonStationaryCustom(np.full((4, 4), 0.25), formula="1 + speed"),
            True,
        ),
    ],
)
def test_detector_fit_leaves_transitions_and_initial_conditions_rebuildable(
    data, discrete_transition_type, covariates
):
    time, position, spike_times, speed = data
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=5.0),
        discrete_transition_type=discrete_transition_type,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )
    detector.fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data={"speed": speed} if covariates else None,
    )
    objects = _fitted_objects(detector)
    assert {type(obj) for obj in objects} >= {
        Discrete,
        Uniform,
        RandomWalk,
        UniformInitialConditions,
        type(discrete_transition_type),
    }
    for obj in objects:
        _assert_configuration_only(obj)


def _spatial_decoder():
    transitions = [
        [RandomWalk(use_manifold_distance=True), Identity(), Uniform()],
        [EmpiricalMovement(), RandomWalkDirection1(), Uniform(environment2_name="")],
        [Uniform(), RandomWalkDirection2(), RandomWalk()],
    ]
    return SortedSpikesDecoder(
        environments=Environment(place_bin_size=5.0),
        discrete_initial_conditions=np.full(3, 1 / 3),
        continuous_initial_conditions_types=[UniformInitialConditions()] * 3,
        continuous_transition_types=transitions,
        discrete_transition_type=DiscreteStationaryDiagonal(np.full(3, 0.9)),
        observation_models=[ObservationModel()] * 3,
        discrete_transition_stickiness=0.0,
        state_names=["a", "b", "c"],
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


def test_spatial_transitions_gain_no_attributes_in_a_fit(data):
    time, position, spike_times, _ = data
    decoder = _spatial_decoder().fit(time, position, spike_times)
    for obj in _fitted_objects(decoder):
        _assert_configuration_only(obj)


def test_pickles_from_earlier_releases_drop_stored_environments(data):
    """Unpickling drops the environments earlier releases stored on transitions."""
    time, position, spike_times, _ = data
    decoder = _spatial_decoder().fit(time, position, spike_times)
    time_edges = np.linspace(time[0], time[-1], 101)
    expected = decoder.predict(spike_times, time_edges=time_edges).acausal_posterior

    # The state earlier releases pickled: make_state_transition stored the
    # environment it was built on.
    environment = decoder.environments[0]
    for row in decoder.continuous_transition_types:
        for transition in row:
            if isinstance(transition, Uniform):
                vars(transition).update(
                    environment1=environment, environment2=environment
                )
            else:
                vars(transition)["environment"] = environment

    loaded = pickle.loads(pickle.dumps(decoder))
    for obj in _fitted_objects(loaded):
        _assert_configuration_only(obj)
    np.testing.assert_array_equal(
        loaded.predict(spike_times, time_edges=time_edges).acausal_posterior,
        expected,
    )

    loaded.fit(time, position, spike_times)
    for obj in _fitted_objects(loaded):
        _assert_configuration_only(obj)
    np.testing.assert_array_equal(
        loaded.predict(spike_times, time_edges=time_edges).acausal_posterior,
        expected,
    )
