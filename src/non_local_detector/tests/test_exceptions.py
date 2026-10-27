"""Tests for custom exceptions module."""

import numpy as np
import pytest

from non_local_detector.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DataError,
    FittingError,
    NonLocalDetectorError,
    ValidationError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test that exception inheritance works correctly."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from NonLocalDetectorError."""
        assert issubclass(ValidationError, NonLocalDetectorError)
        assert issubclass(FittingError, NonLocalDetectorError)
        assert issubclass(ConfigurationError, NonLocalDetectorError)
        assert issubclass(ConvergenceError, NonLocalDetectorError)
        assert issubclass(DataError, NonLocalDetectorError)

    def test_convergence_error_inherits_from_fitting_error(self):
        """ConvergenceError should inherit from FittingError."""
        assert issubclass(ConvergenceError, FittingError)
        assert issubclass(ConvergenceError, NonLocalDetectorError)


@pytest.mark.unit
class TestCommonMessageFormatting:
    """Test message formatting shared by every exception class."""

    @pytest.mark.parametrize(
        "exc_cls, message",
        [
            (ValidationError, "Something went wrong"),
            (FittingError, "Model fitting failed"),
            (ConfigurationError, "Invalid configuration"),
            (ConvergenceError, "Failed to converge"),
            (DataError, "Found NaN values"),
        ],
        ids=[
            "ValidationError",
            "FittingError",
            "ConfigurationError",
            "ConvergenceError",
            "DataError",
        ],
    )
    def test_formats_simple_message_correctly(self, exc_cls, message):
        """Test that an exception with just a message produces clean output."""
        error = exc_cls(message)
        assert str(error) == message

    @pytest.mark.parametrize(
        "exc_cls, message, kwargs, expected",
        [
            (
                ValidationError,
                "Invalid value",
                {"expected": "x > 0", "got": "x = -1", "hint": "Use positive values"},
                ["Hint: Use positive values"],
            ),
            (
                FittingError,
                "Model fitting failed",
                {"hint": "Try increasing max_iterations"},
                ["Hint: Try increasing max_iterations"],
            ),
            (
                ConfigurationError,
                "Incompatible parameters",
                {"hint": "Use either A or B, not both"},
                ["Hint: Use either A or B, not both"],
            ),
            (
                ConvergenceError,
                "Failed to converge",
                {
                    "iterations": 1000,
                    "hint": "Try increasing max_iterations or relaxing tolerance",
                },
                ["Hint: Try increasing max_iterations or relaxing tolerance"],
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
            "FittingError",
            "ConfigurationError",
            "ConvergenceError",
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
class TestValidationError:
    """Test ValidationError message formatting."""

    def test_message_with_expected_and_got(self):
        """Test ValidationError with expected and got parameters."""
        error = ValidationError("Invalid value", expected="x > 0", got="x = -1")
        msg = str(error)
        assert "Invalid value" in msg
        assert "Expected: x > 0" in msg
        assert "Got: x = -1" in msg

    def test_message_with_example(self):
        """Test ValidationError with example parameter."""
        error = ValidationError(
            "Invalid value",
            expected="x > 0",
            got="x = -1",
            hint="Use positive values",
            example="    x = 10",
        )
        msg = str(error)
        assert "Example:" in msg
        assert "x = 10" in msg

    def test_validation_error_formats_complete_structured_message_with_all_fields(self):
        """Test ValidationError with all parameters produces comprehensive formatted output."""
        error = ValidationError(
            "Array shape mismatch",
            expected="shape (100, 2)",
            got="shape (100, 3)",
            hint="Position array should have exactly 2 columns (x, y)",
            example="    position = np.array([[1, 2], [3, 4]])",
        )
        msg = str(error)
        assert "Array shape mismatch" in msg
        assert "Expected: shape (100, 2)" in msg
        assert "Got: shape (100, 3)" in msg
        assert "Hint: Position array should have exactly 2 columns (x, y)" in msg
        assert "Example:" in msg
        assert "position = np.array([[1, 2], [3, 4]])" in msg

    def test_message_formatting_structure(self):
        """Test that message components are properly separated."""
        error = ValidationError(
            "Problem",
            expected="A",
            got="B",
            hint="Fix it",
            example="code",
        )
        msg = str(error)
        # Should have blank line before Hint
        assert "\n\nHint:" in msg
        # Should have blank line before Example
        assert "\n\nExample:" in msg


@pytest.mark.unit
class TestConvergenceError:
    """Test ConvergenceError message formatting."""

    def test_message_with_iterations(self):
        """Test ConvergenceError with iterations parameter."""
        error = ConvergenceError("Failed to converge", iterations=1000)
        msg = str(error)
        assert "Failed to converge" in msg
        assert "iterations: 1000" in msg

    def test_message_with_tolerance(self):
        """Test ConvergenceError with tolerance parameter."""
        error = ConvergenceError("Failed to converge", tolerance=1e-4)
        msg = str(error)
        assert "Failed to converge" in msg
        assert "tolerance: 0.0001" in msg

    def test_message_with_iterations_and_tolerance(self):
        """Test ConvergenceError with both iterations and tolerance."""
        error = ConvergenceError("Failed to converge", iterations=1000, tolerance=1e-4)
        msg = str(error)
        assert "Failed to converge" in msg
        assert "iterations: 1000" in msg
        assert "tolerance: 0.0001" in msg


@pytest.mark.unit
class TestDataError:
    """Test DataError message formatting."""

    def test_message_with_data_name(self):
        """Test DataError with data_name parameter."""
        error = DataError("Found NaN values", data_name="spike_times")
        msg = str(error)
        assert "Found NaN values" in msg
        assert "data: spike_times" in msg


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


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_none_parameters_are_skipped(self):
        """Test that None parameters don't appear in message."""
        error = ValidationError(
            "Problem", expected=None, got=None, hint=None, example=None
        )
        msg = str(error)
        assert msg == "Problem"
        assert "Expected:" not in msg
        assert "Got:" not in msg
        assert "Hint:" not in msg
        assert "Example:" not in msg

    def test_exception_handles_empty_string_parameters_gracefully(self):
        """Test that empty string parameters are displayed without error."""
        error = ValidationError("Problem", expected="", got="", hint="")
        msg = str(error)
        # Empty strings should still appear (they're not None)
        assert "Expected:" in msg
        assert "Got:" in msg
        assert "Hint:" in msg

    def test_special_characters_in_messages(self):
        """Test that special characters don't break formatting."""
        error = ValidationError(
            "Problem with special chars: \n\t{}[]",
            expected="x > 0",
            got="x = -1",
            hint="Use positive\nvalues",
            example="    x = 10\n    y = 20",
        )
        msg = str(error)
        # Should not raise any exceptions and should contain the content
        assert "Problem with special chars" in msg
        assert "x > 0" in msg

    def test_exception_handles_very_long_messages_without_truncation(self):
        """Test that very long messages are preserved completely."""
        long_message = "A" * 1000
        error = ValidationError(long_message)
        msg = str(error)
        assert len(msg) >= 1000
        assert long_message in msg

    def test_exception_preserves_unicode_characters_in_messages(self):
        """Test that unicode characters are handled correctly without corruption."""
        error = ValidationError(
            "Problem with unicode: α β γ δ",
            hint="Use ASCII characters: alpha beta gamma delta",
        )
        msg = str(error)
        assert "α β γ δ" in msg
        assert "alpha beta gamma delta" in msg
