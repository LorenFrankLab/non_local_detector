"""Tests for predict-time covariate transition handling in base.py.

Verifies:
1. predict does not mutate self.discrete_state_transitions_ (state isolation)
2. Mismatched covariate/decode time lengths raise ValueError
"""

import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    DiscreteNonStationaryDiagonal,
    predict_discrete_state_transitions,
)


@pytest.fixture
def nonstationary_transition_artifacts():
    """Create fitted non-stationary transition artifacts (coefficients + design matrix)."""
    diag = np.array([0.8, 0.85, 0.9])
    model = DiscreteNonStationaryDiagonal(
        diagonal_values=diag,
        formula="1 + speed",
    )
    covariate_data = {"speed": np.linspace(0, 10, 20)}
    transition, coefficients, design_matrix = model.make_state_transition(
        covariate_data
    )
    return transition, coefficients, design_matrix


@pytest.mark.unit
class TestPredictDoesNotMutateFittedState:
    """Verify that calling predict with covariate data does not overwrite fitted transitions."""

    def test_predict_discrete_state_transitions_returns_new_array(
        self, nonstationary_transition_artifacts
    ):
        """predict_discrete_state_transitions returns a new array, not modifying inputs."""
        _, coefficients, design_matrix = nonstationary_transition_artifacts
        original_coefficients = coefficients.copy()

        new_covariate_data = {"speed": np.linspace(0, 10, 10)}
        predict_discrete_state_transitions(
            design_matrix, coefficients, new_covariate_data
        )

        # Coefficients should be unchanged
        np.testing.assert_array_equal(coefficients, original_coefficients)


@pytest.mark.unit
@pytest.mark.parametrize("family", ["sorted", "clusterless"])
def test_predict_with_new_covariates_keeps_fitted_transitions(
    checkpoint_recording, family
):
    """``predict`` decodes with transitions predicted from its own covariates
    and leaves the fitted ``discrete_state_transitions_`` untouched."""
    model, fit, predict = checkpoint_recording(family)
    model.discrete_transition_type = DiscreteNonStationaryDiagonal(
        np.array([0.75, 0.85, 0.9, 0.95]), formula="1 + speed"
    )
    fit["discrete_transition_covariate_data"] = {"speed": np.linspace(0, 3, 61)}
    model.fit(**fit)
    # Give speed a non-zero effect so different covariates give different
    # transitions.
    model.discrete_transition_coefficients_[1] = np.arange(12).reshape(4, 3) / 10
    fitted_transitions = model.discrete_state_transitions_.copy()
    n_decode = len(predict["time_edges"]) - 1
    assert fitted_transitions.shape[0] != n_decode

    moving = model.predict(
        **predict,
        discrete_transition_covariate_data={"speed": np.sin(np.arange(n_decode) / 5)},
    )
    still = model.predict(
        **predict,
        discrete_transition_covariate_data={"speed": np.zeros(n_decode)},
    )

    np.testing.assert_array_equal(model.discrete_state_transitions_, fitted_transitions)
    assert not np.allclose(moving.acausal_posterior, still.acausal_posterior)


@pytest.mark.unit
class TestCovariateTimeLengthValidation:
    """Verify that mismatched covariate/decode time lengths are caught.

    Tests exercise the _validate_covariate_time_length static method
    extracted from the predict path in base.py.
    """

    def test_mismatched_covariate_length_raises_valueerror(
        self, nonstationary_transition_artifacts
    ):
        """When covariate data produces different time steps than decode time, raise ValueError."""
        from non_local_detector.models.base import _validate_covariate_time_length

        _, coefficients, design_matrix = nonstationary_transition_artifacts

        new_covariate_data = {"speed": np.linspace(0, 10, 10)}
        predicted = predict_discrete_state_transitions(
            design_matrix, coefficients, new_covariate_data
        )

        decode_time = np.arange(15)  # 15 != 10

        with pytest.raises(ValueError, match="time steps"):
            _validate_covariate_time_length(predicted, decode_time)

    def test_matching_covariate_length_passes(self, nonstationary_transition_artifacts):
        """When covariate and decode time match, no error is raised."""
        from non_local_detector.models.base import _validate_covariate_time_length

        _, coefficients, design_matrix = nonstationary_transition_artifacts

        new_covariate_data = {"speed": np.linspace(0, 10, 10)}
        predicted = predict_discrete_state_transitions(
            design_matrix, coefficients, new_covariate_data
        )

        decode_time = np.arange(10)  # matches
        _validate_covariate_time_length(predicted, decode_time)  # should not raise
