"""Tests for validation methods in models/base.py.

These tests cover the validation methods that ensure proper model configuration:
- _validate_initial_conditions
- _validate_probability_distributions
- _validate_numerical_parameters
- _validate_discrete_transition_type
"""

import warnings

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


def _two_state_decoder(environments, **overrides):
    """Build a valid 2-state SortedSpikesDecoder with keyword overrides."""
    kwargs = {
        "discrete_initial_conditions": np.array([0.5, 0.5]),
        "continuous_initial_conditions_types": [
            "uniform_on_track",
            "uniform_on_track",
        ],
        "continuous_transition_types": ["random_walk", "random_walk"],
        # Float is always valid, so stickiness validation never fires
        "discrete_transition_stickiness": 0.0,
        "state_names": ["state1", "state2"],
        "environments": environments,
    }
    kwargs.update(overrides)
    return SortedSpikesDecoder(**kwargs)


@pytest.mark.unit
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
        # Arrange: 2 discrete probabilities and 2 continuous initial conditions
        # (so the initial-condition check passes) but 3 transition types
        with pytest.raises(
            ValidationError,
            match="Mismatch between discrete initial conditions and continuous transition types",
        ):
            # Act: Try to create decoder with mismatched counts
            SortedSpikesDecoder(
                discrete_initial_conditions=np.array([0.5, 0.5]),  # 2 values
                continuous_initial_conditions_types=[
                    "uniform_on_track",
                    "uniform_on_track",
                ],  # 2 types
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
        decoder = _two_state_decoder(two_state_environment)

        # Assert: Model was created successfully
        assert decoder is not None
        assert len(decoder.discrete_initial_conditions) == 2


@pytest.mark.unit
class TestValidateProbabilityDistributions:
    """Test _validate_probability_distributions method."""

    @pytest.mark.parametrize(
        "initial_conditions, exc, match",
        [
            # Shape (2, 1) has length 2, so it passes the count checks and
            # reaches the dimensionality check.
            (
                np.array([[0.5], [0.5]]),
                ValidationError,
                "discrete_initial_conditions must be a 1-dimensional array",
            ),
            (np.array([0.5, np.nan]), DataError, "non-finite"),
            (np.array([0.5, np.inf]), DataError, "non-finite"),
            (np.array([1.5, -0.5]), ValidationError, "non-negative"),
            # Sums to 0.8
            (np.array([0.3, 0.5]), ValidationError, "probability distribution"),
        ],
        ids=["2d-array", "nan", "inf", "negative", "not-summing-to-one"],
    )
    def test_invalid_initial_conditions_raise(
        self, two_state_environment, initial_conditions, exc, match
    ):
        """Test that malformed discrete initial conditions are rejected."""
        with pytest.raises(exc, match=match):
            _two_state_decoder(
                two_state_environment, discrete_initial_conditions=initial_conditions
            )

    def test_valid_probability_distribution_passes(self, two_state_environment):
        """Test that valid probability distribution passes."""
        # Arrange & Act: Create decoder with valid probabilities
        decoder = _two_state_decoder(
            two_state_environment,
            discrete_initial_conditions=np.array([0.3, 0.7]),  # sum to 1, positive
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert np.allclose(decoder.discrete_initial_conditions.sum(), 1.0)


@pytest.mark.unit
class TestValidateNumericalParameters:
    """Test _validate_numerical_parameters method."""

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"discrete_transition_concentration": -1.0}, ">= 1.0"),
            ({"discrete_transition_concentration": 0.0}, ">= 1.0"),
            # Sparse Dirichlet concentration
            ({"discrete_transition_concentration": 0.5}, ">= 1.0"),
            ({"discrete_transition_regularization": -0.1}, ">= 0"),
            ({"sampling_frequency": -500.0}, "> 0"),
            ({"sampling_frequency": 0.0}, "> 0"),
            ({"no_spike_rate": -1e-10}, "> 0"),
            ({"no_spike_rate": 0.0}, "> 0"),
        ],
        ids=[
            "negative-concentration",
            "zero-concentration",
            "concentration-below-one",
            "negative-regularization",
            "negative-sampling-frequency",
            "zero-sampling-frequency",
            "negative-no-spike-rate",
            "zero-no-spike-rate",
        ],
    )
    def test_invalid_numerical_parameter_raises(self, kwargs, match):
        """Test that out-of-range numerical parameters raise ValidationError."""
        with pytest.raises(ValidationError, match=match):
            SortedSpikesDecoder(**kwargs)

    def test_zero_regularization_is_valid(self):
        """Test that zero regularization is allowed (non-strict inequality)."""
        # Arrange & Act: Create decoder with zero regularization
        decoder = SortedSpikesDecoder(
            discrete_transition_regularization=0.0,  # Zero is OK for regularization
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert decoder.discrete_transition_regularization == 0.0

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


@pytest.mark.unit
class TestValidateDiscreteTransitionType:
    """Test _validate_discrete_transition_type method."""

    @pytest.mark.parametrize(
        "transition_matrix, exc, match",
        [
            (np.array([[0.7, 0.3], [np.nan, 0.5]]), DataError, "non-finite"),
            (np.array([[0.7, 0.3], [np.inf, 0.0]]), DataError, "non-finite"),
            # 3x2 matrix
            (
                np.array([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]]),
                ValidationError,
                "square",
            ),
        ],
        ids=["nan", "inf", "non-square"],
    )
    def test_malformed_custom_transition_raises(
        self, two_state_environment, transition_matrix, exc, match
    ):
        """Test that a non-finite or non-square custom matrix is rejected.

        Uses the decoder's default (single-state) configuration.
        """
        with pytest.raises(exc, match=match):
            SortedSpikesDecoder(
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
                environments=two_state_environment,
            )

    @pytest.mark.parametrize(
        "transition_matrix, match",
        [
            # 3x3 matrix but the model has 2 states
            (
                np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                "size must match",
            ),
            (np.array([[1.5, -0.5], [0.2, 0.8]]), "non-negative"),
            # Non-negative with a value > 1, so the range check (not the
            # non-negativity check) rejects it
            (
                np.array([[1.2, 0.0], [0.2, 0.8]]),
                r"discrete_transition_type\.values values must be in range",
            ),
            # First row sums to 0.8
            (np.array([[0.5, 0.3], [0.2, 0.8]]), "stochastic"),
        ],
        ids=["wrong-size", "negative", "above-one", "rows-not-summing-to-one"],
    )
    def test_invalid_custom_transition_raises(
        self, two_state_environment, transition_matrix, match
    ):
        """Test that an invalid custom matrix for a 2-state model is rejected."""
        with pytest.raises(ValidationError, match=match):
            _two_state_decoder(
                two_state_environment,
                discrete_transition_type=DiscreteStationaryCustom(
                    values=transition_matrix
                ),
            )

    def test_valid_custom_transition_passes(self, two_state_environment):
        """Test that valid custom transition matrix passes validation."""
        # Arrange & Act: Create decoder with a valid stochastic matrix
        transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])
        decoder = _two_state_decoder(
            two_state_environment,
            discrete_transition_type=DiscreteStationaryCustom(values=transition_matrix),
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert isinstance(decoder.discrete_transition_type, DiscreteStationaryCustom)

    def test_diagonal_transition_type_passes(self, two_state_environment):
        """Test that DiscreteStationaryDiagonal type passes validation."""
        # Arrange & Act: Create decoder with diagonal transition type
        decoder = _two_state_decoder(
            two_state_environment,
            discrete_transition_type=DiscreteStationaryDiagonal(
                diagonal_values=np.array([0.9, 0.8])
            ),
        )

        # Assert: Model created successfully
        assert decoder is not None
        assert isinstance(decoder.discrete_transition_type, DiscreteStationaryDiagonal)


@pytest.mark.unit
class TestPositionDimensionalityWarning:
    """Test that 1D position without track_graph triggers a warning."""

    def _fit_and_collect_warnings(self, decoder, position):
        """Call _fit and return the 1-D-position warnings it emitted.

        The fit must complete, so a negative case cannot pass by raising
        before the warning check is reached.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decoder._fit(position=position)
        return [
            x for x in w if "1D" in str(x.message) and "track_graph" in str(x.message)
        ]

    def test_1d_position_no_track_graph_warns(self):
        """1D position without track_graph should warn the user."""
        from non_local_detector.environment import Environment

        env = Environment(
            place_bin_size=5.0,
            position_range=((0.0, 100.0),),
        )
        decoder = SortedSpikesDecoder(environments=[env])
        position_1d = np.linspace(0, 100, 50)

        position_warnings = self._fit_and_collect_warnings(decoder, position_1d)
        assert len(position_warnings) == 1
        assert position_warnings[0].category is UserWarning

    def test_2d_position_no_warning(self):
        """2D position should not trigger the warning."""
        from non_local_detector.environment import Environment

        env = Environment(
            place_bin_size=5.0,
            position_range=((0.0, 100.0), (0.0, 100.0)),
        )
        decoder = SortedSpikesDecoder(environments=[env])
        position_2d = np.column_stack(
            [np.linspace(0, 100, 50), np.linspace(0, 100, 50)]
        )

        position_warnings = self._fit_and_collect_warnings(decoder, position_2d)
        assert len(position_warnings) == 0

    def test_1d_column_position_no_track_graph_warns(self):
        """1D column position (n, 1) without track_graph should warn."""
        from non_local_detector.environment import Environment

        env = Environment(
            place_bin_size=5.0,
            position_range=((0.0, 100.0),),
        )
        decoder = SortedSpikesDecoder(environments=[env])
        position_col = np.linspace(0, 100, 50)[:, np.newaxis]

        position_warnings = self._fit_and_collect_warnings(decoder, position_col)
        assert len(position_warnings) == 1
        assert position_warnings[0].category is UserWarning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
