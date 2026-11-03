"""Custom exceptions for non_local_detector.

This module provides specialized exceptions that give helpful, actionable error
messages to guide users toward correct usage. Following Raymond Hettinger's
principle: "Error messages should be your best documentation."

Usage Guidelines
----------------
Choose the appropriate exception type based on when and why the error occurs:

- **ValidationError**: Input parameters don't meet requirements (raised during
  initialization or before model fitting). Use when user-provided values are
  invalid, have wrong shapes, or violate constraints.

- **ConfigurationError**: Incompatible parameter combinations or missing
  required configuration. Use when parameters are individually valid but
  incompatible with each other.

- **DataError**: Data quality issues like NaN, Inf, or wrong structure. Use
  when the data values (not just shapes) are problematic.

All exceptions inherit from **NonLocalDetectorError**, allowing users to catch
any package-specific error with a single except clause. ValidationError,
ConfigurationError and DataError also inherit from ``ValueError``, so
``except ValueError`` catches them too. The same holds for the transition
errors: ``transition_operators.UnsupportedTransitionError`` (a
``ValueError``), ``transition_operators.DenseTransitionBudgetError`` and
``graph_distances.GraphDistanceBudgetError`` (both ``MemoryError``).

The constructors take the optional message parts as keywords:
``expected``, ``got`` and ``hint`` on all three, ``example`` on
ValidationError and ``data_name`` on DataError.

Examples
--------
>>> from non_local_detector.exceptions import ValidationError, NonLocalDetectorError
>>>
>>> # Raising a validation error with structured message:
>>> try:
...     raise ValidationError(
...         "Array shape mismatch",
...         expected="shape (100, 2)",
...         got="shape (100, 3)",
...         hint="Position array should have exactly 2 columns (x, y)",
...         example="position = np.array([[1, 2], [3, 4]])"
...     )
... except ValidationError as e:
...     print(e)
...
>>> # Catching all package errors:
>>> try:
...     # Some operation that might fail
...     pass
... except NonLocalDetectorError as e:
...     # Handle any package error
...     print(f"Package error: {e}")
"""


class NonLocalDetectorError(Exception):
    """Base exception for all non_local_detector errors.

    All custom exceptions in this package inherit from this base class,
    making it easy to catch any package-specific error.

    Examples
    --------
    >>> try:
    ...     # Any non_local_detector operation
    ...     pass
    ... except NonLocalDetectorError:
    ...     # Handle any package error
    ...     pass
    """


def _structured_message(
    message: str,
    *,
    expected: str | None = None,
    got: str | None = None,
    hint: str | None = None,
    example: str | None = None,
) -> str:
    """Join a message with its optional expected/got/hint/example parts."""
    parts = [message]
    if expected is not None:
        parts.append(f"\nExpected: {expected}")
    if got is not None:
        parts.append(f"Got: {got}")
    if hint is not None:
        parts.append(f"\nHint: {hint}")
    if example is not None:
        parts.append(f"\nExample:\n{example}")
    return "\n".join(parts)


class ValidationError(NonLocalDetectorError, ValueError):
    """Raised when input validation fails.

    This exception indicates that user-provided data doesn't meet the
    required format, shape, or constraints. The error message should
    explain what was expected, what was received, and how to fix it.

    Parameters
    ----------
    message : str
        Description of what went wrong.
    expected : str, optional
        What was expected (for structured error messages).
    got : str, optional
        What was actually received.
    hint : str, optional
        Actionable suggestion for fixing the error.
    example : str, optional
        Code snippet showing correct usage.

    Examples
    --------
    >>> raise ValidationError(
    ...     "Array dimensions don't match",
    ...     expected="shape (100, 2)",
    ...     got="shape (100, 3)",
    ...     hint="Check that position array has exactly 2 columns (x, y)"
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        got: str | None = None,
        hint: str | None = None,
        example: str | None = None,
    ) -> None:
        """Initialize ValidationError with structured message components."""
        super().__init__(
            _structured_message(
                message, expected=expected, got=got, hint=hint, example=example
            )
        )


class ConfigurationError(NonLocalDetectorError, ValueError):
    """Raised when configuration is invalid or inconsistent.

    This exception indicates that model parameters are incompatible
    or that required configuration is missing.

    Parameters
    ----------
    message : str
        Description of the configuration problem.
    expected : str, optional
        The configuration that was expected.
    got : str, optional
        The configuration that was received.
    hint : str, optional
        Actionable suggestion for fixing the configuration.

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Cannot use 'uniform' initial conditions with 'custom' transitions",
    ...     hint="Either use UniformInitialConditions() or provide compatible transitions"
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        expected: str | None = None,
        got: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize ConfigurationError with structured message components."""
        super().__init__(
            _structured_message(message, expected=expected, got=got, hint=hint)
        )


class DataError(NonLocalDetectorError, ValueError):
    """Raised when input data has problems (NaN, Inf, wrong structure).

    This exception indicates issues with the actual data values rather
    than structural validation issues.

    Parameters
    ----------
    message : str
        Description of the data problem.
    data_name : str, optional
        Name of the problematic data variable.
    expected : str, optional
        What the data was expected to contain.
    got : str, optional
        What the data contains.
    hint : str, optional
        Actionable suggestion for fixing the data issue.

    Examples
    --------
    >>> raise DataError(
    ...     "Found NaN values in spike data",
    ...     data_name="multiunit_firing_rates",
    ...     hint="Check for missing data or use np.nan_to_num() to replace NaN values"
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        data_name: str | None = None,
        expected: str | None = None,
        got: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize DataError with structured message components."""
        if data_name is not None:
            message = f"{message} (data: {data_name})"
        super().__init__(
            _structured_message(message, expected=expected, got=got, hint=hint)
        )
