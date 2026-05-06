"""Posterior reduction and collapse helpers.

This module hosts:

- ``maximum_a_posteriori_estimate`` / ``sample_posterior`` — pre-existing
  helpers for working with smoothed-posterior arrays.
- ``conditional_non_local_posterior`` — dataset-level helper extracted
  from the inline algorithm in
  ``non_local_detector.visualization.static.plot_non_local_model``.
- ``collapse_log_likelihood_to_position`` /
  ``collapse_posterior_to_position`` / ``select_reduction`` /
  ``PosteriorReduction`` — schema-aware row-level collapse helpers
  consumed by the interactive viewer (and by anyone who wants to project
  a single time bin's per-state slices down onto the position axis).

Three reductions:

- ``MARGINAL`` — column-sum spatial-state slices, no renormalization.
- ``CONDITIONAL_ON_SPATIAL`` — column-sum spatial-state slices, then
  divide by their total mass (used by NoSpikeContFrag).
- ``CONDITIONAL_NON_LOCAL`` — column-sum slices for states whose name
  contains ``"Non-Local"``, then divide by their total mass (the
  statespacecheck-style paper view; default for
  ``NonLocalSortedSpikesDetector``).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.stats import rv_histogram  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


class PosteriorReduction(Enum):
    """Strategy for collapsing a per-state posterior row down to position.

    See module docstring for the full algorithm sketches.
    """

    MARGINAL = "marginal"
    CONDITIONAL_ON_SPATIAL = "conditional_on_spatial"
    CONDITIONAL_NON_LOCAL = "conditional_non_local"


def select_reduction(
    state_names: Sequence[str], bin_sizes_: np.ndarray
) -> PosteriorReduction:
    """Pick a sensible default reduction for the given detector schema.

    Auto-detect rules, in priority order:

    1. Any state name contains ``"Non-Local"`` → ``CONDITIONAL_NON_LOCAL``.
    2. Schema has singleton states but no ``"Non-Local"`` name →
       ``CONDITIONAL_ON_SPATIAL`` (e.g. NoSpikeContFrag).
    3. Fully-spatial schema → ``MARGINAL`` (Decoder, ContFrag).

    Parameters
    ----------
    state_names : sequence of str
    bin_sizes_ : np.ndarray, shape (n_discrete_states,)

    Returns
    -------
    PosteriorReduction
    """
    if any("Non-Local" in s for s in state_names):
        return PosteriorReduction.CONDITIONAL_NON_LOCAL
    if (np.asarray(bin_sizes_) == 1).any():
        return PosteriorReduction.CONDITIONAL_ON_SPATIAL
    return PosteriorReduction.MARGINAL


def _spatial_state_ids(detector: _DetectorBase) -> np.ndarray:
    """Discrete-state ids whose ``bin_sizes_[s] > 1`` (spatial states)."""
    return np.flatnonzero(np.asarray(detector.bin_sizes_) > 1)


def _non_local_state_ids(detector: _DetectorBase) -> np.ndarray:
    """Discrete-state ids whose name contains ``"Non-Local"``."""
    return np.flatnonzero(["Non-Local" in s for s in detector.state_names])


def _conditional_row(
    post_row: np.ndarray,
    detector: _DetectorBase,
    selected_state_ids: Sequence[int] | np.ndarray,
    zero_mass_fill: float = np.nan,
) -> np.ndarray:
    """Sum selected state-bin slices, then divide by their total mass.

    Shared private implementation backing both ``CONDITIONAL_*``
    strategies.

    Parameters
    ----------
    post_row : np.ndarray, shape (n_state_bins,)
        One row of ``acausal_posterior`` over ``state_bins``. May
        contain NaN at non-interior positions (the
        ``_create_masked_posterior`` padding written when results
        are converted to xarray).
    detector : _DetectorBase
    selected_state_ids : sequence of int
        Discrete-state ids (indices into ``detector.state_names``).
        Internally expanded to a column mask via
        ``np.isin(detector.state_ind_, selected_state_ids)``.
    zero_mass_fill : float, optional
        Value to write on rows where the selected mass is zero. Default
        is ``np.nan``.

    Returns
    -------
    np.ndarray, shape (n_position_bins,)
        Conditional position curve. Sums to 1.0 (over interior bins) on
        rows with positive selected mass; otherwise filled with
        ``zero_mass_fill``.
    """
    state_ind = np.asarray(detector.state_ind_)
    bin_sizes = np.asarray(detector.bin_sizes_)
    selected_state_ids = np.asarray(list(selected_state_ids), dtype=int)

    if selected_state_ids.size == 0:
        raise ValueError(
            "_conditional_row requires at least one selected state id; "
            "got an empty sequence."
        )
    spatial_mask = bin_sizes > 1
    invalid_singletons = [int(s) for s in selected_state_ids if not spatial_mask[s]]
    if invalid_singletons:
        raise ValueError(
            "_conditional_row received singleton state ids "
            f"{invalid_singletons} (their bin_sizes_ entries are 1). "
            "Singleton states have no position axis to project onto."
        )

    spatial_state_ids = np.flatnonzero(spatial_mask)
    n_pos = int(bin_sizes[spatial_state_ids[0]])
    if not np.all(bin_sizes[spatial_state_ids] == n_pos):
        raise ValueError(
            "_conditional_row requires all spatial states to share the "
            f"same n_position_bins. Got bin_sizes_={bin_sizes.tolist()}."
        )

    selected_mask = np.isin(state_ind, selected_state_ids)
    # Cast to float64 so accumulation matches the static-plot inline
    # algorithm at base.py-style float64 accumulator precision (the
    # in-results arrays are float32; summing into a float32 accumulator
    # would diverge by ~1e-7 per row).
    selected = (
        post_row[selected_mask]
        .reshape(selected_state_ids.size, n_pos)
        .astype(np.float64)
    )

    column_sum = selected.sum(axis=0)
    mass = np.nansum(column_sum)
    if not (mass > 0):
        return np.full(n_pos, zero_mass_fill, dtype=np.float64)
    return column_sum / mass


def conditional_non_local_posterior(
    results: xr.Dataset,
    detector: _DetectorBase,
    zero_mass_fill: float = np.nan,
) -> xr.DataArray:
    """Compute ``p(x_t | non-local_t)`` over the full session.

    Dataset-level helper that allocates a full-session
    ``(n_time, n_position_bins)`` array. Used by
    ``plot_non_local_model`` (and any other caller that genuinely needs
    the entire session reduced at once). Per-row callers should prefer
    ``collapse_posterior_to_position(post_row, detector,
    PosteriorReduction.CONDITIONAL_NON_LOCAL)``, which delegates to the
    same private ``_conditional_row`` helper.

    Parameters
    ----------
    results : xr.Dataset
        Decoder output dataset with ``acausal_posterior``.
    detector : _DetectorBase
    zero_mass_fill : float, optional
        Value to write on time bins where the non-local mass is zero.
        Default is ``np.nan``.

    Returns
    -------
    xr.DataArray, dims ("time", "position")
        Conditional non-local posterior. Each row sums to 1.0 (over
        interior bins) on rows with positive non-local mass; otherwise
        filled with ``zero_mass_fill``. Non-interior position bins are
        re-padded to NaN to mirror the static-plot behavior.
    """
    nl_state_ids = _non_local_state_ids(detector)
    if nl_state_ids.size == 0:
        raise ValueError(
            "conditional_non_local_posterior requires at least one state "
            "whose name contains 'Non-Local'. Got "
            f"state_names={list(detector.state_names)!r}."
        )

    post = results["acausal_posterior"].values
    env = detector.environments[0]
    n_pos = env.place_bin_centers_.shape[0]
    n_time = post.shape[0]

    out = np.empty((n_time, n_pos), dtype=np.float64)
    for t in range(n_time):
        out[t] = _conditional_row(
            post[t],
            detector,
            selected_state_ids=nl_state_ids,
            zero_mass_fill=zero_mass_fill,
        )
    out[:, ~env.is_track_interior_] = np.nan
    return xr.DataArray(
        out,
        dims=("time", "position"),
        coords={
            "time": results["time"].values,
            "position": env.place_bin_centers_.squeeze(),
        },
        name="conditional_non_local_posterior",
    )


def collapse_log_likelihood_to_position(
    log_lik_row: np.ndarray, detector: _DetectorBase
) -> np.ndarray:
    """Collapse one row of ``log_likelihood`` over ``state_bins`` to position.

    Pure log-likelihood handling: NaN/-inf cleaning, max-subtracted
    exponentiation (statespacecheck-style float32 overflow guard), sum
    across spatial states, peak-normalize for display. **No probability
    normalization** — this is a likelihood curve, not a posterior.

    Parameters
    ----------
    log_lik_row : np.ndarray, shape (n_state_bins,)
        One row of ``log_likelihood`` over ``state_bins``. May contain
        NaN at non-interior positions and ``-inf`` elsewhere.
    detector : _DetectorBase

    Returns
    -------
    np.ndarray, shape (n_position_bins,)
        Peak-normalized likelihood curve over position. All-zero on
        rows where every spatial entry is non-finite (matches the
        post-exp semantics of ``-inf`` inputs).

    Raises
    ------
    ValueError
        If the detector has no spatial states (every ``bin_sizes_`` is 1).
    """
    spatial_state_ids = _spatial_state_ids(detector)
    if spatial_state_ids.size == 0:
        raise ValueError(
            "Detector has no spatial states. "
            f"bin_sizes_={np.asarray(detector.bin_sizes_).tolist()}"
        )
    bin_sizes = np.asarray(detector.bin_sizes_)
    n_pos = int(bin_sizes[spatial_state_ids[0]])
    if not np.all(bin_sizes[spatial_state_ids] == n_pos):
        raise ValueError(
            "collapse_log_likelihood_to_position requires all spatial "
            "states to share the same n_position_bins. Got "
            f"bin_sizes_={bin_sizes.tolist()}."
        )

    state_ind = np.asarray(detector.state_ind_)
    log_per_state = np.stack([log_lik_row[state_ind == s] for s in spatial_state_ids])

    # Map non-finite values (NaN and any -inf already present) to -inf.
    log_per_state = np.where(np.isfinite(log_per_state), log_per_state, -np.inf)

    if not np.isfinite(log_per_state).any():
        return np.zeros(n_pos, dtype=log_per_state.dtype)

    finite_max = log_per_state[np.isfinite(log_per_state)].max()
    log_per_state = log_per_state - finite_max
    lik_per_state = np.exp(log_per_state)
    lik_curve = lik_per_state.sum(axis=0)
    peak = lik_curve.max()
    return lik_curve / peak if peak > 0 else lik_curve


def collapse_posterior_to_position(
    post_row: np.ndarray,
    detector: _DetectorBase,
    reduction: PosteriorReduction,
    zero_mass_fill: float = np.nan,
) -> np.ndarray:
    """Collapse one row of a posterior over ``state_bins`` to position.

    Probability-domain analogue of
    ``collapse_log_likelihood_to_position``. Selects state-bin slices
    and column-sums them; renormalizes only when ``reduction`` is one
    of the ``CONDITIONAL_*`` variants.

    Parameters
    ----------
    post_row : np.ndarray, shape (n_state_bins,)
        One row of ``acausal_posterior`` (or ``predictive_posterior``)
        over ``state_bins``.
    detector : _DetectorBase
    reduction : PosteriorReduction
        Strategy for collapsing the per-state slices to position.
    zero_mass_fill : float, optional
        Value to write on zero-mass rows when ``reduction`` is one of
        the ``CONDITIONAL_*`` variants. Default is ``np.nan``. The
        ``MARGINAL`` strategy ignores this parameter (it never divides).

    Returns
    -------
    np.ndarray, shape (n_position_bins,)
    """
    if reduction is PosteriorReduction.MARGINAL:
        spatial_state_ids = _spatial_state_ids(detector)
        if spatial_state_ids.size == 0:
            raise ValueError(
                "MARGINAL reduction requires at least one spatial state. "
                f"bin_sizes_={np.asarray(detector.bin_sizes_).tolist()}"
            )
        bin_sizes = np.asarray(detector.bin_sizes_)
        n_pos = int(bin_sizes[spatial_state_ids[0]])
        if not np.all(bin_sizes[spatial_state_ids] == n_pos):
            raise ValueError(
                "MARGINAL reduction requires all spatial states to share "
                f"the same n_position_bins. Got bin_sizes_={bin_sizes.tolist()}."
            )
        state_ind = np.asarray(detector.state_ind_)
        per_state = np.stack([post_row[state_ind == s] for s in spatial_state_ids])
        # Column-sum: keeps NaN at non-interior columns (each spatial
        # slice carries the same NaN pattern there).
        return per_state.sum(axis=0)

    if reduction is PosteriorReduction.CONDITIONAL_ON_SPATIAL:
        return _conditional_row(
            post_row,
            detector,
            selected_state_ids=_spatial_state_ids(detector),
            zero_mass_fill=zero_mass_fill,
        )

    if reduction is PosteriorReduction.CONDITIONAL_NON_LOCAL:
        return _conditional_row(
            post_row,
            detector,
            selected_state_ids=_non_local_state_ids(detector),
            zero_mass_fill=zero_mass_fill,
        )

    raise ValueError(f"Unknown reduction strategy: {reduction!r}")


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
        (time, x_position, y_position).

    Returns
    -------
    map_estimate : np.ndarray, shape (n_time, n_spatial_dims)
        Most likely position coordinates at each time point. For 1D tracks,
        shape is (n_time, 1). For 2D environments, shape is (n_time, 2).

    """
    try:
        stacked_posterior = posterior.stack(z=["x_position", "y_position"])
        map_estimate = stacked_posterior.z[stacked_posterior.argmax("z")]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior.position[np.log(posterior).argmax("position")]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def sample_posterior(
    posterior: xr.DataArray, place_bin_edges: np.ndarray, n_samples: int = 1000
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

    Returns
    -------
    posterior_samples : np.ndarray, shape (n_time, n_samples)
        Random samples drawn from the posterior distribution at each time point.
        Each row contains samples for one time point.

    """
    # Stack 2D positions into one dimension
    try:
        posterior = posterior.stack(z=["x_position", "y_position"]).values
    except (KeyError, AttributeError):
        posterior = np.asarray(posterior)

    place_bin_edges = place_bin_edges.squeeze()
    n_time = posterior.shape[0]

    posterior_samples = [
        rv_histogram((posterior[time_ind], place_bin_edges)).rvs(size=n_samples)
        for time_ind in range(n_time)
    ]

    return np.asarray(posterior_samples)
