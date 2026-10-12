"""Detectors with covariate-dependent transitions can be saved and copied.

patsy design matrices cannot be pickled, so the fitted covariates are kept
and the design matrix is rebuilt on load (issue #50).
"""

import copy
import pickle

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

pytestmark = pytest.mark.integration

N_PREDICT = 2_001


@pytest.fixture(scope="module")
def fitted():
    speed, position, spike_times, time, _, _, is_event, _ = make_simulated_data(
        seed=42, n_neurons=5
    )
    detector = NonLocalSortedSpikesDetector(
        discrete_transition_type=DiscreteNonStationaryDiagonal(
            np.array([0.99, 0.98, 0.96, 0.96])
        ),
    )
    detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time_edges=time_edges_from_centers(time),
        is_training=~is_event,
        max_iter=2,
        estimate_encoding_model=False,
        discrete_transition_covariate_data={"speed": speed},
    )
    predict_data = {
        "position_time": time[:N_PREDICT],
        "position": position[:N_PREDICT],
        "spike_times": [times[times < time[N_PREDICT - 1]] for times in spike_times],
        "time_edges": time_edges_from_centers(time[:N_PREDICT]),
        # Speeds beyond the fitted range exercise the rebuilt spline state.
        "discrete_transition_covariate_data": {
            "speed": np.linspace(0, 2 * speed.max(), N_PREDICT)
        },
    }
    return detector, predict_data


def _state_probabilities(detector, predict_data):
    return detector.predict(**predict_data).acausal_state_probabilities.values


def test_pickle_round_trip_predicts_identically(fitted):
    detector, predict_data = fitted
    restored = pickle.loads(pickle.dumps(detector))

    np.testing.assert_array_equal(
        np.asarray(restored.discrete_transition_design_matrix_),
        np.asarray(detector.discrete_transition_design_matrix_),
    )
    np.testing.assert_array_equal(
        _state_probabilities(restored, predict_data),
        _state_probabilities(detector, predict_data),
    )


def test_save_and_load_model(fitted, tmp_path):
    detector, predict_data = fitted
    path = tmp_path / "covariate_model.pkl"
    detector.save_model(str(path))
    restored = NonLocalSortedSpikesDetector.load_model(str(path))

    np.testing.assert_array_equal(
        _state_probabilities(restored, predict_data),
        _state_probabilities(detector, predict_data),
    )


def test_deepcopy(fitted):
    detector, predict_data = fitted
    np.testing.assert_array_equal(
        _state_probabilities(copy.deepcopy(detector), predict_data),
        _state_probabilities(detector, predict_data),
    )


def test_covariates_stay_out_of_params(fitted):
    """Spyglass rebuilds detectors from their parameters; the stored
    covariates are fitted state, not a parameter."""
    detector, _ = fitted
    assert not any("covariate" in name for name in detector.get_params())
