"""Point estimates and samples from decoded posterior distributions."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram

from non_local_detector._position_dims import (
    _is_position_dim,
    get_position_dim_names,
    get_position_dims,
)

__all__ = [
    "get_position_posterior",
    "get_spatial_posterior",
    "maximum_a_posteriori_estimate",
    "sample_posterior",
]


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
        does not have (for example off-track bins) are NaN. Results' posteriors
        also give each state's ``environment``, a coordinate along ``state``.

    Raises
    ------
    ValueError
        If ``posterior`` has no ``state_bins`` MultiIndex with a ``state`` level
        and a position level, if ``states`` names a state the posterior does not
        have, or if no state bin with a position remains after selection.
    """
    if "state_bins" not in posterior.dims:
        msg = f"posterior needs a 'state_bins' dimension; got dims {posterior.dims}."
        raise ValueError(msg)
    index = posterior.indexes["state_bins"]
    if index.nlevels < 2 or "state" not in index.names:
        msg = "posterior's 'state_bins' must be a MultiIndex with a 'state' level."
        raise ValueError(msg)

    position_levels = [n for n in index.names if _is_position_dim(n)]
    if not position_levels:
        msg = (
            "posterior's 'state_bins' has no position level (such as 'position' "
            f"or 'x_position'); got levels {list(index.names)}."
        )
        raise ValueError(msg)
    state_names = index.get_level_values("state")
    if states is not None:
        available = list(dict.fromkeys(state_names))
        unknown = [state for state in states if state not in available]
        if unknown:
            msg = f"Unknown states {unknown}; the posterior has states {available}."
            raise ValueError(msg)
    keep = np.ones(len(index), dtype=bool)
    for level in position_levels:
        keep &= ~np.isnan(np.asarray(index.get_level_values(level), dtype=float))
    if states is not None:
        keep &= state_names.isin(list(states))
    if not keep.any():
        msg = f"No state bins with a position remain (states={states!r})."
        raise ValueError(msg)
    # state_ind and environment would unstack into NaN-padded coordinates that
    # only repeat the ``state`` dimension.
    spatial = (
        posterior.isel(state_bins=np.flatnonzero(keep))
        .drop_vars(["state_ind", "environment"], errors="ignore")
        .unstack("state_bins")
    )
    if "environment" in posterior.coords:
        environments = dict(
            zip(state_names[keep], posterior["environment"].values[keep], strict=True)
        )
        spatial = spatial.assign_coords(
            environment=(
                "state",
                [environments[state] for state in spatial["state"].values],
            )
        )
    return spatial


def get_position_posterior(posterior: xr.DataArray) -> xr.DataArray:
    """Reduce a posterior to position alone, whatever its layout.

    Parameters
    ----------
    posterior : xr.DataArray
        A raw ``results.acausal_posterior`` with a ``state_bins`` dimension,
        the per-state grids of :func:`get_spatial_posterior`, or a posterior
        that already has only position dimensions.

    Returns
    -------
    position_posterior : xr.DataArray, shape (n_time, *n_position_bins)
        Summed over the states with a position, with the position-less states
        (Local, No-Spike) left out, so it sums to 1 over position only when
        those states have no mass. Off-track bins are 0.

    Raises
    ------
    ValueError
        If ``posterior`` has neither a ``state_bins`` nor a position dimension,
        if ``get_spatial_posterior`` cannot read its ``state_bins``, or if its
        states are in more than one environment. A position means nothing in
        another environment, so select one first, with
        ``posterior.isel(state_bins=posterior.environment == name)``.
    """
    if "state_bins" in posterior.dims:
        posterior = get_spatial_posterior(posterior)
    if "state" in posterior.dims:
        if "environment" in posterior.coords:
            environments = sorted(set(posterior["environment"].values.tolist()))
            if len(environments) > 1:
                msg = (
                    f"posterior's states are in environments {environments}, and "
                    "positions in different environments cannot be summed. Select "
                    "one first: posterior.isel(state_bins=posterior.environment == "
                    f"{environments[0]!r}), or .sel(state=[...]) on per-state grids."
                )
                raise ValueError(msg)
        posterior = posterior.sum("state")
    if not get_position_dims(posterior):
        msg = f"posterior has no position dimension; got dims {posterior.dims}."
        raise ValueError(msg)
    return posterior


def _state_probabilities(results: xr.Dataset) -> dict[str, xr.DataArray]:
    """Acausal probability of each state, by name.

    A one-state result (a decoder's) holds ``states`` as a scalar coordinate
    and its probabilities without a ``states`` dimension, which neither
    iterating ``states`` nor ``sel`` accepts.

    Parameters
    ----------
    results : xr.Dataset
        Output of a detector's ``predict``.

    Returns
    -------
    state_probabilities : dict of str to xr.DataArray, shape (n_time,)
    """
    probabilities = results["acausal_state_probabilities"]
    states = results["states"]
    if "states" not in probabilities.dims:
        return {str(states.item()): probabilities}
    return {str(state): probabilities.sel(states=state) for state in states.values}


def maximum_a_posteriori_estimate(posterior: xr.DataArray) -> np.ndarray:
    """Find the most likely position from the posterior distribution.

    Computes the maximum a posteriori (MAP) estimate by finding the position
    bin with the highest probability at each time point, in any number of
    position dimensions. NaN bins (for example off-track bins) are ignored.

    Parameters
    ----------
    posterior : xr.DataArray, shape (n_time, *n_position_bins)
        Posterior probability distribution over position bins. For 1D tracks,
        dimensions are (time, position). For 2D environments, dimensions are
        (time, x_position, y_position), and so on. A raw
        ``results.acausal_posterior`` with a ``state_bins`` dimension is also
        accepted: it is passed through :func:`get_position_posterior`, so
        position-less states (Local, No-Spike) are ignored.

    Returns
    -------
    map_estimate : np.ndarray, shape (n_time, n_position_dims)
        Most likely position coordinates at each time point, with columns in
        the order of the position names (x, y, z, ...). Time points with no
        probability on any position bin (zero or non-finite total, e.g. all
        mass in Local or No-Spike, or every bin NaN) are NaN rather than the
        first bin.
    """
    posterior = get_position_posterior(posterior)
    position_dims = get_position_dim_names(len(get_position_dims(posterior)))
    probability = posterior.transpose(..., *position_dims).to_numpy()
    probability = probability.reshape(probability.shape[0], -1)
    bin_coordinates = np.stack(
        [
            grid.ravel()
            for grid in np.meshgrid(
                *(posterior[dim].to_numpy() for dim in position_dims), indexing="ij"
            )
        ],
        axis=1,
    ).astype(float)

    total = np.nansum(probability, axis=1)
    has_mass = np.isfinite(total) & (total > 0)
    map_index = np.argmax(np.where(np.isnan(probability), -np.inf, probability), axis=1)
    map_estimate: np.ndarray = bin_coordinates[map_index]
    map_estimate[~has_mass] = np.nan
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
