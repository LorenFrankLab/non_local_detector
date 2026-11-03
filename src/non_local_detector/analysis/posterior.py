"""Point estimates and samples from decoded posterior distributions."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram

from non_local_detector._position_dims import _is_position_dim


def get_spatial_posterior(
    posterior: xr.DataArray, states: Sequence[str] | None = None
) -> xr.DataArray:
    """Unstack a ``state_bins`` posterior into per-state position grids.

    States without a position (for example Local and No-Spike) are stored in
    ``state_bins`` with NaN position values. ``DataArray.unstack("state_bins")``
    mishandles those NaN index levels and puts the position-less states'
    probability on the *last* position bin. This function drops the position-less
    state bins before unstacking, so every returned value sits on its true bin.

    The probability of the position-less states is not in the result; it is in
    ``results.acausal_state_probabilities``. Consequently
    ``result.sum("state")`` is the posterior over position *jointly with being in
    a spatial state* and does **not** sum to 1 over position when position-less
    states have mass.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_state_bins)
        A posterior such as ``results.acausal_posterior`` whose ``state_bins``
        MultiIndex has the level ``state`` followed by the position levels
        (``position`` for 1D, ``x_position``/``y_position`` for 2D, and so on).
    states : sequence of str, optional
        Keep only these states. By default all states with a position are kept.

    Returns
    -------
    spatial_posterior : xr.DataArray, shape (n_time, n_state, *n_position_bins)
        Dimensions are ``(time, state, position)`` for 1D and
        ``(time, state, x_position, y_position)`` for 2D. Position bins a state
        does not have (for example off-track bins) are NaN.

    Raises
    ------
    ValueError
        If ``posterior`` has no ``state_bins`` MultiIndex with a ``state`` level,
        or no state bin with a position remains after selection.
    """
    if "state_bins" not in posterior.dims:
        msg = f"posterior needs a 'state_bins' dimension; got dims {posterior.dims}."
        raise ValueError(msg)
    index = posterior.indexes["state_bins"]
    if index.nlevels < 2 or "state" not in index.names:
        msg = "posterior's 'state_bins' must be a MultiIndex with a 'state' level."
        raise ValueError(msg)

    position_levels = [n for n in index.names if _is_position_dim(n)]
    keep = np.ones(len(index), dtype=bool)
    for level in position_levels:
        keep &= ~np.isnan(np.asarray(index.get_level_values(level), dtype=float))
    if states is not None:
        keep &= index.get_level_values("state").isin(list(states))
    if not keep.any():
        msg = f"No state bins with a position remain (states={states!r})."
        raise ValueError(msg)
    return posterior.isel(state_bins=np.flatnonzero(keep)).unstack("state_bins")


def maximum_a_posteriori_estimate(posterior: xr.DataArray) -> np.ndarray:
    """Find the most likely position from the posterior distribution.

    Computes the maximum a posteriori (MAP) estimate by finding the position
    bin with the highest probability at each time point. Handles both 1D
    and 2D position representations.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or (n_time, n_x_bins, n_y_bins)
        Posterior probability distribution over position bins. For 1D tracks,
        dimensions are (time, position). For 2D environments, dimensions are
        (time, x_position, y_position). A raw ``results.acausal_posterior``
        with a ``state_bins`` dimension is also accepted: it is passed through
        :func:`get_spatial_posterior` and summed over ``state``, so
        position-less states (Local, No-Spike) are ignored.

    Returns
    -------
    map_estimate : np.ndarray, shape (n_time, n_spatial_dims)
        Most likely position coordinates at each time point. For 1D tracks,
        shape is (n_time, 1). For 2D environments, shape is (n_time, 2).
    """
    if "state_bins" in posterior.dims:
        posterior = get_spatial_posterior(posterior).sum("state")
    try:
        stacked_posterior = posterior.stack(z=["x_position", "y_position"])
        map_estimate = stacked_posterior.z[stacked_posterior.argmax("z")]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior.position[posterior.argmax("position")]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def sample_posterior(
    posterior: xr.DataArray | np.ndarray,
    place_bin_edges: np.ndarray,
    n_samples: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate random samples from the posterior distribution.

    Treats the posterior as a probability mass function and generates random
    samples from it at each time point using scipy's rv_histogram. Useful
    for uncertainty quantification and Bayesian inference.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, n_position_bins) or (n_time, n_x_bins, n_y_bins)
        Posterior probability distribution over position bins. For 2D environments,
        the spatial dimensions are automatically flattened.
    place_bin_edges : np.ndarray, shape (n_position_bins + 1,)
        Bin edges defining the boundaries of the position bins. Must have one
        more element than the number of position bins.
    n_samples : int, optional
        Number of random samples to generate at each time point, by default 1000.
    rng : numpy.random.Generator or int, optional
        Generator, or seed for one, that draws all time points' samples. By
        default a fresh, unseeded generator.

    Returns
    -------
    posterior_samples : np.ndarray, shape (n_time, n_samples)
        Random samples drawn from the posterior distribution at each time point.
        Each row contains samples for one time point.
    """
    # Stack 2D positions into one dimension
    try:
        posterior = (
            cast("xr.DataArray", posterior).stack(z=["x_position", "y_position"]).values
        )
    except (KeyError, AttributeError):
        posterior = np.asarray(posterior)

    place_bin_edges = place_bin_edges.squeeze()
    n_time = posterior.shape[0]
    # One generator for every time point: reseeding each draw with the same int
    # would give every time point the same uniform variates.
    rng = np.random.default_rng(rng)

    posterior_samples = [
        rv_histogram((posterior[time_ind], place_bin_edges), density=False).rvs(
            size=n_samples, random_state=rng
        )
        for time_ind in range(n_time)
    ]

    return np.asarray(posterior_samples)
