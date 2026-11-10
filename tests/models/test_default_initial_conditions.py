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
from tests.conftest import em_stops_at_max_iter
from tests.models._helpers import SORTED_SIM_RATE

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


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES, ids=lambda c: c.__name__)
def test_default_initial_conditions_are_positive_distributions(model_cls):
    initial_conditions = model_cls().discrete_initial_conditions
    assert np.all(initial_conditions > 0)
    np.testing.assert_allclose(initial_conditions.sum(), 1.0, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls",
    [
        NonLocalSortedSpikesDetector,
        NonLocalClusterlessDetector,
        MultiEnvironmentSortedSpikesClassifier,
        MultiEnvironmentClusterlessClassifier,
    ],
    ids=lambda c: c.__name__,
)
def test_default_starts_one_transition_after_nominal_start(model_cls):
    """Non-local models start one step after Local, multi-environment models
    one step after the first environment (state 0 in both)."""
    model = model_cls()
    transition_matrix, _, _ = model.discrete_transition_type.make_state_transition(None)
    np.testing.assert_array_equal(
        model.discrete_initial_conditions, transition_matrix[0]
    )


@pytest.mark.integration
def test_default_conditional_non_local_posterior_is_finite(sorted_sim):
    """With no zero-probability start state, the posterior given a non-local
    state is defined at the first bin too."""
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
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
@em_stops_at_max_iter
def test_em_can_move_default_start_off_local(sorted_sim):
    """EM re-estimates every state's initial probability, which stays positive."""
    time, position, spike_times = sorted_sim
    detector = NonLocalSortedSpikesDetector(sampling_frequency=SORTED_SIM_RATE)
    initial = detector.discrete_initial_conditions.copy()
    detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time_edges=time_edges_from_centers(time),
        max_iter=2,
        estimate_encoding_model=False,
        estimate_initial_conditions=True,
    )
    learned = detector.discrete_initial_conditions_
    assert np.all(learned > 0)
    assert not np.allclose(learned, initial, rtol=1e-3, atol=0)
