"""Default discrete initial conditions give every state a positive probability.

A state that starts at exactly zero stays at zero through EM (the update sets
the initial conditions to the posterior at the first bin, which is zero
wherever the prior is), and its conditional posterior is undefined at the
first bin. Models with a nominal start state start one transition after it.
"""

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

ALL_MODEL_CLASSES = [
    SortedSpikesDecoder,
    ClusterlessDecoder,
    ContFragSortedSpikesClassifier,
    ContFragClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
    MultiEnvironmentClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    NoSpikeContFragClusterlessClassifier,
    NonLocalSortedSpikesDetector,
    NonLocalClusterlessDetector,
]


def _default_transition_row(model, state):
    """Row ``state`` of the model's default stationary transition matrix."""
    matrix, _, _ = model.discrete_transition_type.make_state_transition(None)
    return matrix[state]


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES, ids=lambda c: c.__name__)
def test_default_initial_conditions_are_positive_distributions(model_cls):
    initial_conditions = model_cls().discrete_initial_conditions
    assert np.all(initial_conditions > 0)
    np.testing.assert_allclose(initial_conditions.sum(), 1.0, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls, start_state",
    [
        (NonLocalSortedSpikesDetector, 0),
        (NonLocalClusterlessDetector, 0),
        (MultiEnvironmentSortedSpikesClassifier, 0),
        (MultiEnvironmentClusterlessClassifier, 0),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_default_starts_one_transition_after_nominal_start(model_cls, start_state):
    """Non-local models start one step after Local, multi-environment models
    one step after the first environment."""
    model = model_cls()
    np.testing.assert_array_equal(
        model.discrete_initial_conditions,
        _default_transition_row(model, start_state),
    )


@pytest.fixture(scope="module")
def sorted_recording():
    """Two seconds of 25 simulated place cells at 1500 Hz."""
    _, position, spike_times, time, *_ = make_simulated_data(seed=0, n_neurons=25)
    n = 3_000
    return time[:n], position[:n], [st[st <= time[n - 1]] for st in spike_times]


@pytest.mark.integration
def test_default_conditional_non_local_posterior_is_finite(sorted_recording):
    """With no zero-probability start state, the posterior given a non-local
    state is defined at the first bin too."""
    time, position, spike_times = sorted_recording
    detector = NonLocalSortedSpikesDetector(sampling_frequency=1500)
    detector.fit(time, position, spike_times)
    results = detector.predict(
        spike_times,
        time_edges=time_edges_from_centers(time),
        position=position,
        position_time=time,
    )
    posterior = detector.get_conditional_non_local_posterior(results).values
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.integration
def test_em_can_move_default_start_off_local(sorted_recording):
    """EM re-estimates every state's initial probability, which stays positive."""
    time, position, spike_times = sorted_recording
    detector = NonLocalSortedSpikesDetector(sampling_frequency=1500)
    initial = detector.discrete_initial_conditions.copy()
    detector.estimate_parameters(
        time,
        position,
        spike_times,
        time_edges=time_edges_from_centers(time),
        max_iter=2,
        estimate_encoding_model=False,
        estimate_initial_conditions=True,
    )
    learned = detector.discrete_initial_conditions_
    assert np.all(learned > 0)
    assert not np.allclose(learned, initial, rtol=1e-3, atol=0)
