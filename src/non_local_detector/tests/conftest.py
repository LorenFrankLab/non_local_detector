"""Shared test fixtures and utilities for non_local_detector tests.

This module provides:
1. A builder for synthetic decoder results and a JAX precision context manager
2. Environment fixtures for common test scenarios
3. Assertion helpers for validating probability distributions and stochastic
   matrices
4. Posterior and design-matrix fixtures for transition-estimation tests

Usage:
    pytest automatically discovers fixtures in conftest.py. Import assertion
    helpers explicitly if needed:

    from conftest import assert_probability_distribution
"""

from contextlib import contextmanager

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.environment import Environment

# ==============================================================================
# SYNTHETIC RESULTS BUILDERS
# ==============================================================================


@contextmanager
def precision_mode(x64: bool):
    """Set JAX's global x64 flag for the block, then restore the previous one."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def make_state_bins_results(
    state_names,
    position_columns,
    n_time=10,
    seed=0,
):
    """Build a synthetic results ``xr.Dataset`` matching ``predict()`` output.

    ``acausal_posterior`` has dims ``(time, state_bins)`` where ``state_bins`` is
    a MultiIndex over ``("state", *position_columns)`` — the same layout the
    detector classes produce. Each state is given the same per-state position
    grid, and the posterior mass is random but normalized over ``state_bins`` at
    each time step.

    Parameters
    ----------
    state_names : sequence of str
        Discrete state labels, one block of position bins per state.
    position_columns : dict[str, np.ndarray]
        Mapping from position-dim name (e.g. ``"position"`` for 1D or
        ``"x_position"``/``"y_position"`` for 2D) to the 1-D array of position
        values for a single state. All arrays must have the same length
        (the number of position bins per state); they are interpreted as
        paired columns, not a Cartesian grid.
    n_time : int, optional
        Number of time steps, by default 10.
    seed : int, optional
        RNG seed for reproducibility, by default 0.

    Returns
    -------
    xr.Dataset
        Dataset with ``acausal_posterior`` (time, state_bins).
    """
    rng = np.random.default_rng(seed)
    state_names = np.asarray(state_names)
    n_states = len(state_names)
    n_bins = len(next(iter(position_columns.values())))

    arrays = [np.repeat(state_names, n_bins)]
    names = ["state"]
    for name, values in position_columns.items():
        arrays.append(np.tile(np.asarray(values, dtype=float), n_states))
        names.append(name)
    mindex = pd.MultiIndex.from_arrays(arrays, names=names)

    raw = rng.random((n_time, len(mindex)))
    raw = raw / raw.sum(axis=1, keepdims=True)

    coords = xr.Coordinates.from_pandas_multiindex(mindex, "state_bins")
    return xr.Dataset(
        {"acausal_posterior": (("time", "state_bins"), raw)},
        coords={**coords, "time": np.arange(n_time, dtype=float)},
    )


# ==============================================================================
# ENVIRONMENT FIXTURES
# ==============================================================================


@pytest.fixture
def simple_1d_environment():
    """Simple 1D linear environment with 1.0 bin size, range [0, 10].

    Returns:
        Environment: Fitted environment with 10 position bins.
    """
    env = Environment(
        environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),)
    )
    # Provide position array for fit_place_grid to determine dimensions
    dummy_pos = np.linspace(0.0, 10.0, 11)[:, None]
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================


def assert_probability_distribution(arr, axis=-1, rtol=1e-5, atol=1e-6):
    """Assert array is a valid probability distribution.

    Checks that all values are non-negative, <= 1, and sum to 1.0
    along the given axis.

    Parameters
    ----------
    arr : array-like
        Array to validate (numpy or jax array).
    axis : int, optional
        Axis along which to sum (default: -1).
    rtol : float, optional
        Relative tolerance for sum check.
    atol : float, optional
        Absolute tolerance for sum check.

    Raises
    ------
    AssertionError
        If any validation fails.
    """
    assert np.all(arr >= 0), "Probabilities must be non-negative"
    assert np.all(arr <= 1), "Probabilities must be <= 1"
    sums = arr.sum(axis=axis)
    assert np.allclose(sums, 1.0, rtol=rtol, atol=atol), (
        f"Probabilities must sum to 1 along axis {axis}, got sums: {sums}"
    )


def assert_stochastic_matrix(matrix, rtol=1e-5, atol=1e-6):
    """Assert matrix is row-stochastic (each row sums to 1).

    Parameters
    ----------
    matrix : array-like
        2D array to validate.
    rtol : float, optional
        Relative tolerance for row sum check.
    atol : float, optional
        Absolute tolerance for row sum check.

    Raises
    ------
    AssertionError
        If validation fails.
    """
    assert matrix.ndim == 2, "Stochastic matrix must be 2D"
    assert matrix.shape[0] == matrix.shape[1], "Must be square matrix"
    assert np.all(matrix >= 0), "Elements must be non-negative"

    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, rtol=rtol, atol=atol), (
        f"Rows must sum to 1, got row sums: {row_sums}"
    )


# ==============================================================================
# DISCRETE STATE TRANSITION TEST FIXTURES
# ==============================================================================


@pytest.fixture
def posterior_data():
    """Generate realistic posterior distributions for EM algorithm testing.

    Returns
    -------
    dict
        Contains causal_posterior, acausal_posterior, predictive_distribution,
        transition_matrix for a 4-state, 50-timestep HMM.
    """
    rng = np.random.default_rng(42)
    n_time, n_states = 50, 4

    # Generate smooth posteriors that look realistic
    causal_posterior = rng.dirichlet(np.ones(n_states) * 2, n_time)
    acausal_posterior = rng.dirichlet(np.ones(n_states) * 2, n_time)

    # Create a reasonable transition matrix
    transition_matrix = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )

    # Compute predictive distribution
    predictive_distribution = np.zeros((n_time, n_states))
    for t in range(n_time):
        predictive_distribution[t] = causal_posterior[t] @ transition_matrix

    return {
        "causal_posterior": causal_posterior,
        "acausal_posterior": acausal_posterior,
        "predictive_distribution": predictive_distribution,
        "transition_matrix": transition_matrix,
        "n_time": n_time,
        "n_states": n_states,
    }


@pytest.fixture
def design_matrix_data():
    """Generate design matrix for non-stationary transition testing.

    Returns
    -------
    dict
        Contains design_matrix, transition_coefficients for testing.
    """
    rng = np.random.default_rng(43)
    n_time, n_coefficients, n_states = 50, 3, 4

    # Create design matrix with intercept and two covariates
    design_matrix = np.column_stack(
        [
            np.ones(n_time),
            np.linspace(0, 1, n_time),
            np.sin(np.linspace(0, 2 * np.pi, n_time)),
        ]
    )

    # Initialize coefficients (small values to avoid extreme probabilities)
    transition_coefficients = (
        rng.standard_normal((n_coefficients, n_states, n_states - 1)) * 0.1
    )

    return {
        "design_matrix": design_matrix,
        "transition_coefficients": transition_coefficients,
        "n_time": n_time,
        "n_coefficients": n_coefficients,
        "n_states": n_states,
    }
