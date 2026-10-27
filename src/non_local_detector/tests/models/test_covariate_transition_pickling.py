"""Detectors with covariate-dependent transitions can be saved and copied.

patsy design matrices cannot be pickled, so the fitted covariates are kept
and the design matrix is rebuilt on load.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.discrete_state_transitions import (
    predict_discrete_state_transitions,
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


@pytest.mark.parametrize("container", [dict, pd.DataFrame])
@pytest.mark.parametrize(
    "round_trip",
    [lambda detector: pickle.loads(pickle.dumps(detector)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_round_trip_ignores_later_changes_to_the_covariates(container, round_trip):
    """The kept covariates are a snapshot: changing the caller's data after
    the fit does not move the knots a round trip rebuilds."""
    covariates = container({"speed": np.linspace(0.0, 50.0, 201)})
    detector = NonLocalSortedSpikesDetector(
        discrete_transition_type=DiscreteNonStationaryDiagonal(np.full(4, 0.9))
    )
    detector.initialize_discrete_state_transition(covariates)
    # Speed-dependent coefficients, so the knots change the predictions.
    n_basis = detector.discrete_transition_coefficients_.shape[0] - 1
    detector.discrete_transition_coefficients_[1:, 0, 0] = np.linspace(
        -3.0, 3.0, n_basis
    )

    def predict(model):
        return predict_discrete_state_transitions(
            model.discrete_transition_design_matrix_,
            model.discrete_transition_coefficients_,
            {"speed": np.linspace(0.0, 50.0, 101)},
        )

    expected = predict(detector)
    covariates["speed"] *= 2.0

    np.testing.assert_array_equal(predict(round_trip(detector)), expected)
