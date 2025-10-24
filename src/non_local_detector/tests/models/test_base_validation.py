"""Tests for validation methods in models/base.py.

These tests cover the validation methods that ensure proper model configuration:
- _validate_initial_conditions
- _validate_probability_distributions
- _validate_numerical_parameters
- _validate_discrete_transition_type
"""

import numpy as np
import pytest

from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.models import SortedSpikesDecoder


@pytest.fixture
def two_state_environment(simple_1d_environment):
    """Create an environment list with 2 states for testing."""
    return [simple_1d_environment, simple_1d_environment]


class TestValidateInitialConditions:
    """Test _validate_initial_conditions method."""

    def test_mismatched_discrete_and_continuous_initial_conditions(self):
        """Test that mismatch between discrete and continuous initial conditions raises ValidationError."""
        # Arrange: 2 continuous initial conditions but 3 discrete probabilities
        with pytest.raises(
            ValidationError,
            match="Mismatch between discrete initial conditions and continuous initial conditions",
        ):
            # Act: Try to create decoder with mismatched initial conditions
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.3, 0.2]),  # 3 values
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],  # 2 types
                continuous_transition_types=["random_walk", "random_walk"],  # 2 types
            )

    def test_mismatched_discrete_conditions_and_transition_types(self):
        """Test that mismatch between discrete initial conditions and transition types raises ValidationError."""
        # Arrange: 3 transition types but 2 discrete probabilities, 3 continuous initial conditions
        with pytest.raises(
            ValidationError,
            match="Mismatch",  # Can be either "discrete initial conditions and continuous" or "continuous transition"
        ):
            # Act: Try to create decoder with mismatched counts
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),  # 2 values
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                    "uniform_on_track",
                ],  # 3 types
                continuous_transition_types=[
                    "random_walk",
                    "random_walk",
                    "random_walk",
                ],  # 3 types
            )

    def test_mismatched_stickiness_array_length(self):
        """Test that stickiness array length must match number of states."""
        # Arrange: 2 states but 3 stickiness values
        with pytest.raises(
            ValidationError,
            match="Discrete transition stickiness must be set for all 2 states",
        ):
            # Act: Try to create decoder with wrong stickiness array length
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),  # 2 states
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=np.array(
                    [0.0, 0.5, 1.0]
                ),  # 3 values - WRONG!
            )

    def test_valid_configuration_passes(self, two_state_environment):
        """Test that correctly matched configuration passes validation."""
        # Arrange & Act: Create decoder with correct configuration
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,  # Float is always valid
            state_names=["state1", "state2"],
            environments=two_state_environment,
        )

        # Assert: Model was created successfully
        assert decoder is not None
        assert len(decoder.discrete_initial_conditions) == 2


class TestValidateProbabilityDistributions:
    """Test _validate_probability_distributions method."""

    def test_non_1d_array_raises_error(self, two_state_environment):
        """Test that 2D array for initial conditions raises ValidationError."""
        # Arrange & Act: Try to create decoder with 2D initial conditions
        with pytest.raises(
            ValidationError
        ):  # Will fail with Mismatch error due to shape
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([[0.5, 0.5]]),  # 2D array
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
                state_names=["state1", "state2"],
                environments=two_state_environment,
            )

    def test_nan_values_raise_data_error(self, two_state_environment):
        """Test that NaN values in initial conditions raise DataError."""
        # Arrange & Act: Try to create decoder with NaN
        with pytest.raises(DataError, match="non-finite"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, np.nan]),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
                environments=two_state_environment,
            )

    def test_inf_values_raise_data_error(self, two_state_environment):
        """Test that Inf values in initial conditions raise DataError."""
        # Arrange & Act: Try to create decoder with Inf
        with pytest.raises(DataError, match="non-finite"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, np.inf]),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
                environments=two_state_environment,
            )

    def test_negative_values_raise_error(self, two_state_environment):
        """Test that negative probabilities raise ValidationError."""
        # Arrange & Act: Try to create decoder with negative probability
        with pytest.raises(ValidationError, match="non-negative"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([1.5, -0.5]),  # Negative!
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
                environments=two_state_environment,
            )

    def test_probabilities_not_summing_to_one_raise_error(self, two_state_environment):
        """Test that probabilities not summing to 1 raise ValidationError."""
        # Arrange & Act: Try to create decoder with probabilities summing to 0.8
        with pytest.raises(ValidationError, match="probability distribution"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.3, 0.5]),  # Sums to 0.8
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
                state_names=["state1", "state2"],
                environments=two_state_environment,
            )

    def test_valid_probability_distribution_passes(self, two_state_environment):
        """Test that valid probability distribution passes."""
        # Arrange & Act: Create decoder with valid probabilities
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array(
                [0.3, 0.7]
            ),  # Valid: sum to 1, all positive
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_stickiness=0.0,  # Float to avoid stickiness validation error
            state_names=["state1", "state2"],
            environments=two_state_environment,
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert np.allclose(decoder.discrete_initial_conditions.sum(), 1.0)


class TestValidateNumericalParameters:
    """Test _validate_numerical_parameters method."""

    def test_negative_concentration_raises_error(self):
        """Test that negative concentration parameter raises ValidationError."""
        # Arrange & Act: Try to create decoder with negative concentration
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                discrete_transition_concentration=-1.0,  # Negative!
            )

    def test_zero_concentration_raises_error(self):
        """Test that zero concentration parameter raises ValidationError."""
        # Arrange & Act: Try to create decoder with zero concentration (must be strictly positive)
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                discrete_transition_concentration=0.0,  # Zero not allowed for concentration
            )

    def test_negative_regularization_raises_error(self):
        """Test that negative regularization parameter raises ValidationError."""
        # Arrange & Act: Try to create decoder with negative regularization
        with pytest.raises(ValidationError, match=">= 0"):
            SortedSpikesDecoder(
                discrete_transition_regularization=-0.1,  # Negative!
            )

    def test_zero_regularization_is_valid(self):
        """Test that zero regularization is allowed (non-strict inequality)."""
        # Arrange & Act: Create decoder with zero regularization
        decoder = SortedSpikesDecoder(
            discrete_transition_regularization=0.0,  # Zero is OK for regularization
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert decoder.discrete_transition_regularization == 0.0

    def test_negative_sampling_frequency_raises_error(self):
        """Test that negative sampling frequency raises ValidationError."""
        # Arrange & Act: Try to create decoder with negative sampling frequency
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                sampling_frequency=-500.0,  # Negative!
            )

    def test_zero_sampling_frequency_raises_error(self):
        """Test that zero sampling frequency raises ValidationError."""
        # Arrange & Act: Try to create decoder with zero sampling frequency
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                sampling_frequency=0.0,  # Zero not allowed
            )

    def test_negative_no_spike_rate_raises_error(self):
        """Test that negative no_spike_rate raises ValidationError."""
        # Arrange & Act: Try to create decoder with negative no_spike_rate
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                no_spike_rate=-1e-10,  # Negative!
            )

    def test_zero_no_spike_rate_raises_error(self):
        """Test that zero no_spike_rate raises ValidationError."""
        # Arrange & Act: Try to create decoder with zero no_spike_rate
        with pytest.raises(ValidationError, match="> 0"):
            SortedSpikesDecoder(
                no_spike_rate=0.0,  # Zero not allowed
            )

    def test_valid_numerical_parameters_pass(self):
        """Test that valid numerical parameters pass validation."""
        # Arrange & Act: Create decoder with valid numerical parameters
        decoder = SortedSpikesDecoder(
            discrete_transition_concentration=2.0,  # Positive
            discrete_transition_regularization=1e-5,  # Non-negative
            sampling_frequency=500.0,  # Positive
            no_spike_rate=1e-10,  # Positive
        )

        # Assert: Model created successfully with correct values
        assert decoder is not None
        assert decoder.discrete_transition_concentration == 2.0
        assert decoder.discrete_transition_regularization == 1e-5
        assert decoder.sampling_frequency == 500.0
        assert decoder.no_spike_rate == 1e-10


class TestValidateDiscreteTransitionType:
    """Test _validate_discrete_transition_type method."""

    def test_custom_transition_with_nan_raises_error(self, two_state_environment):
        """Test that custom transition matrix with NaN raises DataError."""
        # Arrange: Create transition matrix with NaN
        transition_matrix = np.array(
            [
                [0.7, 0.3],
                [np.nan, 0.5],  # NaN value
            ]
        )

        # Act & Assert: Try to create decoder with NaN in transition matrix
        with pytest.raises(DataError, match="non-finite"):
            SortedSpikesDecoder(
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                environments=two_state_environment,
            )

    def test_custom_transition_with_inf_raises_error(self, two_state_environment):
        """Test that custom transition matrix with Inf raises DataError."""
        # Arrange: Create transition matrix with Inf
        transition_matrix = np.array(
            [
                [0.7, 0.3],
                [np.inf, 0.0],  # Inf value
            ]
        )

        # Act & Assert: Try to create decoder with Inf in transition matrix
        with pytest.raises(DataError, match="non-finite"):
            SortedSpikesDecoder(
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                environments=two_state_environment,
            )

    def test_custom_transition_non_square_raises_error(self, two_state_environment):
        """Test that non-square transition matrix raises ValidationError."""
        # Arrange: Create non-square matrix
        transition_matrix = np.array(
            [
                [0.5, 0.5],
                [0.3, 0.7],
                [0.2, 0.8],  # 3x2 matrix
            ]
        )

        # Act & Assert: Try to create decoder with non-square matrix
        with pytest.raises(ValidationError, match="square"):
            SortedSpikesDecoder(
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                environments=two_state_environment,
            )

    def test_custom_transition_wrong_size_raises_error(self, two_state_environment):
        """Test that transition matrix with wrong size raises ValidationError."""
        # Arrange: Create 3x3 matrix but model expects 2x2 (2 states)
        transition_matrix = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        # Act & Assert: Try to create decoder with wrong-sized matrix
        with pytest.raises(ValidationError, match="size must match"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),  # 2 states
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],  # 2 states
                continuous_transition_types=["random_walk", "random_walk"],  # 2 states
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),  # 3x3 matrix
                discrete_transition_stickiness=0.0,
                state_names=["state1", "state2"],  # Match 2 states
                environments=two_state_environment,
            )

    def test_custom_transition_negative_values_raise_error(self, two_state_environment):
        """Test that negative values in transition matrix raise ValidationError."""
        # Arrange: Create matrix with negative value
        transition_matrix = np.array(
            [
                [1.5, -0.5],  # Negative value
                [0.2, 0.8],
            ]
        )

        # Act & Assert: Try to create decoder with negative values
        with pytest.raises(ValidationError, match="non-negative"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                discrete_transition_stickiness=0.0,
                state_names=["state1", "state2"],
                environments=two_state_environment,
            )

    def test_custom_transition_values_outside_range_raise_error(
        self, two_state_environment
    ):
        """Test that values > 1 in transition matrix raise ValidationError."""
        # Arrange: Create matrix with value > 1
        transition_matrix = np.array(
            [
                [0.5, 0.5],
                [1.2, -0.2],  # 1.2 > 1.0 AND -0.2 < 0
            ]
        )

        # Act & Assert: Try to create decoder with out-of-range values (will catch negative first)
        with pytest.raises(ValidationError):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                discrete_transition_stickiness=0.0,
                state_names=["state1", "state2"],
                environments=two_state_environment,
            )

    def test_custom_transition_rows_not_summing_to_one_raise_error(
        self, two_state_environment
    ):
        """Test that non-stochastic transition matrix raises ValidationError."""
        # Arrange: Create matrix where rows don't sum to 1
        transition_matrix = np.array(
            [
                [0.5, 0.3],  # Sums to 0.8
                [0.2, 0.8],
            ]
        )

        # Act & Assert: Try to create decoder with non-stochastic matrix
        with pytest.raises(ValidationError, match="stochastic"):
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],
                continuous_transition_types=["random_walk", "random_walk"],
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                discrete_transition_stickiness=0.0,
                state_names=["state1", "state2"],
                environments=two_state_environment,
            )

    def test_valid_custom_transition_passes(self, two_state_environment):
        """Test that valid custom transition matrix passes validation."""
        # Arrange: Create valid stochastic matrix
        transition_matrix = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        )

        # Act: Create decoder with valid custom transition
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_type=DiscreteStationaryCustom(values=transition_matrix),
            discrete_transition_stickiness=0.0,
            state_names=["state1", "state2"],
            environments=two_state_environment,
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert isinstance(decoder.discrete_transition_type, DiscreteStationaryCustom)

    def test_diagonal_transition_type_passes(self, two_state_environment):
        """Test that DiscreteStationaryDiagonal type passes validation."""
        # Arrange & Act: Create decoder with diagonal transition type
        decoder = SortedSpikesDecoder(
            discrete_initial_conditions=np.array([0.5, 0.5]),
            continuous_initial_conditions_types=[
                "uniform_on_track",
                "uniform_on_track",
            ],
            continuous_transition_types=["random_walk", "random_walk"],
            discrete_transition_type=DiscreteStationaryDiagonal(
                diagonal_values=np.array([0.9, 0.8])
            ),
            discrete_transition_stickiness=0.0,
            state_names=["state1", "state2"],
            environments=two_state_environment,
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert isinstance(decoder.discrete_transition_type, DiscreteStationaryDiagonal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
