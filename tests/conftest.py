"""Shared test fixtures and utilities for non_local_detector tests.

This module provides:
1. A builder for synthetic decoder results and a JAX precision context manager
2. Environment fixtures for common test scenarios
3. Assertion helpers for validating probability distributions and stochastic
   matrices

Usage:
    pytest automatically discovers fixtures in conftest.py. Import assertion
    helpers explicitly if needed:

    from tests.conftest import assert_probability_distribution
"""

import dataclasses
import os
from contextlib import contextmanager
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from non_local_detector.environment import Environment, PlaceGrid

# Persistent JAX compilation cache. Many tests clear JAX's caches or rebuild
# identical functions, and the suite is mostly XLA compile time: on the six
# heaviest test files (469 tests) a filled cache cut 182-194 s to 60-77 s.
# Set JAX_COMPILATION_CACHE_DIR to use another directory, or
# NLD_TEST_JAX_CACHE=0 to turn the cache off (e.g. to time cold compiles), also
# when JAX_COMPILATION_CACHE_DIR is set. A cache entry JAX cannot read or write
# is reported as a warning, not a test failure (pyproject.toml filterwarnings).
if os.environ.get("NLD_TEST_JAX_CACHE", "1") != "0":
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        jax.config.update(
            "jax_compilation_cache_dir",
            str(Path(__file__).resolve().parents[1] / ".cache" / "jax-compilation"),
        )
    # Cache every program: most test programs compile in well under JAX's
    # default 1 s threshold but there are thousands of them.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
else:
    jax.config.update("jax_enable_compilation_cache", False)

TEST_CATEGORIES = {"unit", "integration", "property"}


def pytest_collection_modifyitems(items):
    """Mark tests without a category marker as unit tests.

    ``-m unit`` / ``-m integration`` would otherwise silently skip them.
    """
    for item in items:
        if not TEST_CATEGORIES & {marker.name for marker in item.iter_markers()}:
            item.add_marker(pytest.mark.unit)


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


@pytest.fixture
def x64():
    """Enable float64 for the rest of the test."""
    with precision_mode(True):
        yield


def set_place_grid(environment, **fields):
    """Give ``environment`` a hand-built grid holding ``fields``.

    For tests of code that reads a few fitted attributes: the given
    ``PlaceGrid`` fields replace those of the current grid (if any), other
    fields of a new grid are None, and the grid counts as fitted with the
    environment's current configuration. The cache is cleared, as by a fit.

    Returns
    -------
    environment : Environment
    """
    grid = environment._place_grid
    values = (
        dict.fromkeys(item.name for item in dataclasses.fields(PlaceGrid))
        if grid is None
        else {item.name: getattr(grid, item.name) for item in dataclasses.fields(grid)}
    )
    values.update(fields, configuration=environment._grid_configuration())
    environment._place_grid = PlaceGrid(**values)
    environment.clear_cache()
    return environment


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
