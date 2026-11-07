"""Tests for custom exceptions module."""

import numpy as np
import pytest

from non_local_detector.exceptions import (
    ConfigurationError,
    DataError,
    NonLocalDetectorError,
    ValidationError,
)
from non_local_detector.graph_distances import GraphDistanceBudgetError
from non_local_detector.transition_operators import (
    DenseTransitionBudgetError,
    UnsupportedTransitionError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test that exception inheritance works correctly."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from NonLocalDetectorError."""
        assert issubclass(ValidationError, NonLocalDetectorError)
        assert issubclass(ConfigurationError, NonLocalDetectorError)
        assert issubclass(DataError, NonLocalDetectorError)

    @pytest.mark.parametrize(
        "exc_cls", [ValidationError, ConfigurationError, DataError]
    )
    def test_input_errors_are_value_errors(self, exc_cls):
        """Input errors are catchable as ValueError and as the package base."""
        assert issubclass(exc_cls, ValueError)
        error = exc_cls("boom")
        with pytest.raises(ValueError, match="boom"):
            raise error
        with pytest.raises(NonLocalDetectorError, match="boom"):
            raise error

    @pytest.mark.parametrize(
        ("exc_cls", "builtin"),
        [
            (UnsupportedTransitionError, ValueError),
            (DenseTransitionBudgetError, MemoryError),
            (GraphDistanceBudgetError, MemoryError),
        ],
    )
    def test_transition_errors_are_package_errors(self, exc_cls, builtin):
        """Transition errors are caught by the package catch-all and keep
        their builtin base."""
        assert issubclass(exc_cls, NonLocalDetectorError)
        assert issubclass(exc_cls, builtin)


@pytest.mark.unit
@pytest.mark.parametrize("exc_cls", [ValidationError, ConfigurationError, DataError])
def test_constructors_take_expected_got_hint_as_keywords(exc_cls):
    """Every structured exception formats expected, got and hint the same way,
    and takes them only as keywords."""
    message = str(exc_cls("Problem", expected="a", got="b", hint="c"))
    assert message == "Problem\n\nExpected: a\nGot: b\n\nHint: c"
    with pytest.raises(TypeError):
        exc_cls("Problem", "a")


@pytest.mark.unit
class TestCommonMessageFormatting:
    """Test message formatting shared by every exception class."""

    @pytest.mark.parametrize(
        ("exc_cls", "message"),
        [
            (ValidationError, "Something went wrong"),
            (ConfigurationError, "Invalid configuration"),
            (DataError, "Found NaN values"),
        ],
        ids=[
            "ValidationError",
            "ConfigurationError",
            "DataError",
        ],
    )
    def test_formats_simple_message_correctly(self, exc_cls, message):
        """Test that an exception with just a message produces clean output."""
        error = exc_cls(message)
        assert str(error) == message

    @pytest.mark.parametrize(
        ("exc_cls", "message", "kwargs", "expected"),
        [
            (
                ValidationError,
                "Invalid value",
                {"expected": "x > 0", "got": "x = -1", "hint": "Use positive values"},
                ["Hint: Use positive values"],
            ),
            (
                ConfigurationError,
                "Incompatible parameters",
                {"hint": "Use either A or B, not both"},
                ["Hint: Use either A or B, not both"],
            ),
            (
                DataError,
                "Found NaN values",
                {
                    "data_name": "spike_times",
                    "hint": "Use np.nan_to_num() to replace NaN values",
                },
                [
                    "data: spike_times",
                    "Hint: Use np.nan_to_num() to replace NaN values",
                ],
            ),
        ],
        ids=[
            "ValidationError",
            "ConfigurationError",
            "DataError",
        ],
    )
    def test_message_with_hint(self, exc_cls, message, kwargs, expected):
        """Test that the hint (and any other field) appears in the message."""
        msg = str(exc_cls(message, **kwargs))
        assert message in msg
        for line in expected:
            assert line in msg


@pytest.mark.unit
class TestExceptionIntegration:
    """Test that exceptions work correctly in actual validation scenarios."""

    def test_validation_error_in_model_initialization(self):
        """Test ValidationError raised from actual model code."""
        from non_local_detector import NonLocalClusterlessDetector

        with pytest.raises(ValidationError) as exc_info:
            # Try to create detector with mismatched states
            NonLocalClusterlessDetector(
                discrete_initial_conditions=np.array([1.0, 0.0])  # Only 2 states!
                # Default is 4 states
            )

        error_msg = str(exc_info.value)
        assert "Mismatch" in error_msg
        assert "Expected:" in error_msg
        assert "Got:" in error_msg
        assert "Hint:" in error_msg
        assert "Example:" in error_msg

    def test_validation_error_in_gmm(self):
        """Test ValidationError raised from GMM validation."""
        from non_local_detector.likelihoods.gmm import GaussianMixtureModel

        with pytest.raises(ValidationError) as exc_info:
            GaussianMixtureModel(n_components=0, covariance_type="full")

        error_msg = str(exc_info.value)
        assert "GMM" in error_msg or "components" in error_msg
        assert "Expected:" in error_msg
        assert "Got:" in error_msg

    def test_package_validation_error_is_catchable_as_value_error(self):
        """Errors raised by package code can be caught with ``except ValueError``."""
        from non_local_detector.likelihoods.gmm import GaussianMixtureModel

        with pytest.raises(ValueError, match="components"):
            GaussianMixtureModel(n_components=0, covariance_type="full")

    def test_validation_error_in_gmm_covariance_type(self):
        """Test ValidationError for invalid GMM covariance type."""
        from non_local_detector.likelihoods.gmm import GaussianMixtureModel

        with pytest.raises(ValidationError) as exc_info:
            GaussianMixtureModel(n_components=3, covariance_type="invalid_type")

        error_msg = str(exc_info.value)
        assert "covariance" in error_msg.lower()
        assert "full" in error_msg  # Should list valid options
        assert "tied" in error_msg
        assert "diag" in error_msg
        assert "spherical" in error_msg
