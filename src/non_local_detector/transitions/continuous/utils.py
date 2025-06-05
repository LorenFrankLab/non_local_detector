from typing import Optional

import numpy as np

from ...environment import Environment
from .base import Array


def _normalize_row_probability(x: np.ndarray) -> np.ndarray:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_rows, n_cols)

    Returns
    -------
    normalized_x : np.ndarray, shape (n_rows, n_cols)

    """
    # Handle cases where the sum is zero to avoid division by zero -> NaN
    row_sums = x.sum(axis=1, keepdims=True)
    # Use np.errstate to temporarily ignore invalid division warnings
    with np.errstate(invalid="ignore"):
        normalized_x = np.where(row_sums > 0, x / row_sums, 0.0)
    # Ensure any remaining NaNs (though unlikely with the above) are zero
    normalized_x[np.isnan(normalized_x)] = 0.0
    return normalized_x


def _atomic_matrix(src_bins: int, dst_bins: int) -> Array:
    """
    Return an identity (if src == dst == 1) or uniform row-stochastic matrix
    for transitions involving at least one atomic state.
    """
    if src_bins == 1 and dst_bins == 1:
        return np.ones((1, 1))  # stay put
    return np.full((src_bins, dst_bins), 1.0 / dst_bins)


def get_n_bins(env: Optional[Environment], default_n_bins: int = 1) -> int:
    """Gets the number of bins for an environment, defaulting if None."""
    return default_n_bins if env is None else env.n_bins


def _handle_intra_env_kernel_edges(
    src_env: Optional[Environment],
    dst_env: Optional[Environment],
) -> Optional[Array]:
    """
    Handles atomic cases or cross-environment transitions for kernels
    that primarily operate within a single environment.

    Parameters
    ----------
    src_env : Optional[Environment]
        Source environment.
    dst_env : Optional[Environment]
        Destination environment.

    Returns
    -------
    Optional[Array]
        A transition matrix if the condition is met (atomic or cross-env),
        otherwise None.
    """
    n_src_bins = get_n_bins(src_env)
    n_dst_bins = get_n_bins(dst_env)

    if src_env is None or dst_env is None:
        return _atomic_matrix(n_src_bins, n_dst_bins)

    if src_env is not dst_env:
        if n_dst_bins == 0:
            return np.zeros((n_src_bins, 0))
        return np.full((n_src_bins, n_dst_bins), 1.0 / n_dst_bins)  # uniform entry

    return None  # Indicates main kernel logic should proceed


def estimate_movement_var(position: np.ndarray) -> np.ndarray:
    """Estimates the movement variance based on position differences.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, n_position_dim)
        Position of the animal

    Returns
    -------
    movement_var : np.ndarray, shape (n_position_dim,)
        Variance of the movement per time bin

    """
    position = position if position.ndim > 1 else position[:, np.newaxis]
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    if position.shape[0] < 2:
        raise ValueError("At least two time points are required to estimate variance.")
    return np.cov(np.diff(position, axis=0), rowvar=False)
