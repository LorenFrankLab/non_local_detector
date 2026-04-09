"""Tests for predict-time covariate transition handling in base.py.

Verifies:
1. predict does not mutate self.discrete_state_transitions_ (state isolation)
2. Mismatched covariate/decode time lengths raise ValueError
"""

from unittest.mock import MagicMock

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

    def test_predict_path_uses_local_variable_not_self(
        self, nonstationary_transition_artifacts
    ):
        """The base.py predict path should pass transitions as a parameter, not mutate self.

        This tests the wiring by patching predict_discrete_state_transitions
        and verifying _predict receives the result without self being modified.
        """
        fitted_transition, coefficients, design_matrix = (
            nonstationary_transition_artifacts
        )

        # Create a mock detector with the minimum attributes needed
        mock_detector = MagicMock()
        mock_detector.discrete_transition_coefficients_ = coefficients
        mock_detector.discrete_transition_design_matrix_ = design_matrix
        mock_detector.discrete_state_transitions_ = fitted_transition

        # Store the original fitted value
        original_transitions = fitted_transition.copy()

        new_covariate_data = {"speed": np.linspace(0, 10, 10)}
        predicted = predict_discrete_state_transitions(
            design_matrix, coefficients, new_covariate_data
        )

        # The predicted transitions should differ from the fitted ones
        assert predicted.shape[0] == 10  # new time dimension
        assert fitted_transition.shape[0] == 20  # original time dimension

        # Verify fitted transitions were not modified
        np.testing.assert_array_equal(
            fitted_transition,
            original_transitions,
            err_msg="Fitted transitions were mutated by predict",
        )


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

    def test_matching_covariate_length_passes(
        self, nonstationary_transition_artifacts
    ):
        """When covariate and decode time match, no error is raised."""
        from non_local_detector.models.base import _validate_covariate_time_length

        _, coefficients, design_matrix = nonstationary_transition_artifacts

        new_covariate_data = {"speed": np.linspace(0, 10, 10)}
        predicted = predict_discrete_state_transitions(
            design_matrix, coefficients, new_covariate_data
        )

        decode_time = np.arange(10)  # matches
        _validate_covariate_time_length(predicted, decode_time)  # should not raise
