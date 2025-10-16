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

    def test_can_catch_with_base_exception(self):
        """Should be able to catch any package exception with base class."""
        with pytest.raises(NonLocalDetectorError):
            raise ValidationError("test")

        with pytest.raises(NonLocalDetectorError):
            raise FittingError("test")

        with pytest.raises(NonLocalDetectorError):
            raise ConvergenceError("test")


class TestValidationError:
    """Test ValidationError message formatting."""

    def test_basic_message(self):
        """Test ValidationError with just a message."""
        error = ValidationError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_message_with_expected_and_got(self):
        """Test ValidationError with expected and got parameters."""
        error = ValidationError("Invalid value", expected="x > 0", got="x = -1")
        msg = str(error)
        assert "Invalid value" in msg
        assert "Expected: x > 0" in msg
        assert "Got: x = -1" in msg

    def test_message_with_hint(self):
        """Test ValidationError with hint parameter."""
        error = ValidationError(
            "Invalid value",
            expected="x > 0",
            got="x = -1",
            hint="Use positive values",
        )
        msg = str(error)
        assert "Hint: Use positive values" in msg

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

    def test_full_structured_message(self):
        """Test ValidationError with all parameters."""
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


class TestFittingError:
    """Test FittingError message formatting."""

    def test_basic_message(self):
        """Test FittingError with just a message."""
        error = FittingError("Model fitting failed")
        assert str(error) == "Model fitting failed"

    def test_message_with_hint(self):
        """Test FittingError with hint parameter."""
        error = FittingError(
            "Model fitting failed", hint="Try increasing max_iterations"
        )
        msg = str(error)
        assert "Model fitting failed" in msg
        assert "Hint: Try increasing max_iterations" in msg


class TestConfigurationError:
    """Test ConfigurationError message formatting."""

    def test_basic_message(self):
        """Test ConfigurationError with just a message."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"

    def test_message_with_hint(self):
        """Test ConfigurationError with hint parameter."""
        error = ConfigurationError(
            "Incompatible parameters",
            hint="Use either A or B, not both",
        )
        msg = str(error)
        assert "Incompatible parameters" in msg
        assert "Hint: Use either A or B, not both" in msg


class TestConvergenceError:
    """Test ConvergenceError message formatting."""

    def test_basic_message(self):
        """Test ConvergenceError with just a message."""
        error = ConvergenceError("Failed to converge")
        assert str(error) == "Failed to converge"

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

    def test_message_with_hint(self):
        """Test ConvergenceError with hint parameter."""
        error = ConvergenceError(
            "Failed to converge",
            iterations=1000,
            hint="Try increasing max_iterations or relaxing tolerance",
        )
        msg = str(error)
        assert "Hint: Try increasing max_iterations or relaxing tolerance" in msg


class TestDataError:
    """Test DataError message formatting."""

    def test_basic_message(self):
        """Test DataError with just a message."""
        error = DataError("Found NaN values")
        assert str(error) == "Found NaN values"

    def test_message_with_data_name(self):
        """Test DataError with data_name parameter."""
        error = DataError("Found NaN values", data_name="spike_times")
        msg = str(error)
        assert "Found NaN values" in msg
        assert "data: spike_times" in msg

    def test_message_with_hint(self):
        """Test DataError with hint parameter."""
        error = DataError(
            "Found NaN values",
            data_name="spike_times",
            hint="Use np.nan_to_num() to replace NaN values",
        )
        msg = str(error)
        assert "Found NaN values" in msg
        assert "data: spike_times" in msg
        assert "Hint: Use np.nan_to_num() to replace NaN values" in msg


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

    def test_empty_string_parameters(self):
        """Test that empty string parameters are handled."""
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

    def test_very_long_messages(self):
        """Test that very long messages are handled."""
        long_message = "A" * 1000
        error = ValidationError(long_message)
        msg = str(error)
        assert len(msg) >= 1000
        assert long_message in msg

    def test_unicode_in_messages(self):
        """Test that unicode characters are handled correctly."""
        error = ValidationError(
            "Problem with unicode: α β γ δ",
            hint="Use ASCII characters: alpha beta gamma delta",
        )
        msg = str(error)
        assert "α β γ δ" in msg
        assert "alpha beta gamma delta" in msg


class TestExceptionReRaise:
    """Test that exceptions can be caught and re-raised."""

    def test_catch_and_reraise_validation_error(self):
        """Test catching and re-raising ValidationError."""
        with pytest.raises(ValidationError):
            try:
                raise ValidationError("Original error")
            except ValidationError:
                raise  # Re-raise the same exception

    def test_catch_specific_and_raise_general(self):
        """Test catching specific exception and raising more general one."""
        with pytest.raises(NonLocalDetectorError):
            try:
                raise ValidationError("Specific error")
            except ValidationError as e:
                raise NonLocalDetectorError(f"Wrapped: {e}") from e

    def test_exception_chaining(self):
        """Test that exception chaining works correctly."""
        with pytest.raises(FittingError) as exc_info:
            try:
                raise ValidationError("Input problem")
            except ValidationError as e:
                raise FittingError("Fitting failed due to invalid input") from e

        # Check that the original exception is preserved
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValidationError)
