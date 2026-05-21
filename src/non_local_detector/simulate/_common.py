"""Shared simulator utilities.

Single source of truth for utility functions used by both clusterless and
sorted-spikes simulators. Module-specific helpers (e.g., variants of
``simulate_position`` with different parameter conventions) stay in their
respective simulator modules.

Conventions
-----------
- All times are in seconds.
- ``simulate_poisson_spikes`` returns raw Poisson counts (not binary
  indicators). Callers that want a boolean spike indicator should apply
  ``counts > 0`` or ``counts.astype(bool)``.
- ``get_trajectory_direction`` returns a ``(direction_label, is_inbound)``
  tuple. Callers that only need one of these can discard the other with
  tuple unpacking.
"""

import numpy as np
from scipy.stats import multivariate_normal  # type: ignore[import-untyped]


def simulate_time(n_samples: int, sampling_frequency: float) -> np.ndarray:
    """Generate a time vector for the given number of samples and sampling rate.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    sampling_frequency : float
        Samples per second.

    Returns
    -------
    time : np.ndarray, shape (n_samples,)
        Time in seconds.
    """
    return np.arange(n_samples) / sampling_frequency


def simulate_poisson_spikes(
    rate: np.ndarray,
    sampling_frequency: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Given a rate, return a time series of Poisson spike counts.

    Parameters
    ----------
    rate : np.ndarray, shape (n_time,)
        Instantaneous firing rate in Hz.
    sampling_frequency : float
        Samples per second.
    seed : int | None, optional
        Random seed for reproducibility. Ignored if ``rng`` is provided.
    rng : np.random.Generator or None, optional
        Random number generator. If None, creates one from ``seed``.

    Returns
    -------
    counts : np.ndarray, shape (n_time,)
        Poisson-sampled spike counts per time bin. Use ``counts > 0`` to
        obtain a boolean spike indicator if needed.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.poisson(rate / sampling_frequency)


def simulate_place_field_firing_rate(
    means: np.ndarray,
    position: np.ndarray,
    max_rate: float = 15.0,
    variance: float = 12.5,
    is_condition: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate the firing rate of a neuron with a place field at ``means``.

    Parameters
    ----------
    means : np.ndarray, shape (n_position_dims,)
        Place-field center.
    position : np.ndarray, shape (n_time, n_position_dims)
        Position trajectory.
    max_rate : float, optional
        Peak firing rate at the field center.
    variance : float, optional
        Place-field variance.
    is_condition : None or np.ndarray, shape (n_time,)
        Boolean mask of times when the place field is active. Default: always.

    Returns
    -------
    firing_rate : np.ndarray, shape (n_time,)
        Firing rate at each time point.
    """
    if is_condition is None:
        is_condition = np.ones(position.shape[0], dtype=bool)
    position = position if position.ndim > 1 else position[:, np.newaxis]
    firing_rate = np.asarray(multivariate_normal(means, variance).pdf(position))
    firing_rate /= firing_rate.max()
    firing_rate *= max_rate
    firing_rate[~is_condition] = 0.0

    return firing_rate


def simulate_neuron_with_place_field(
    means: np.ndarray,
    position: np.ndarray,
    max_rate: float = 15.0,
    variance: float = 12.5,
    sampling_frequency: float = 500,
    is_condition: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate the spiking of a neuron with a place field at ``means``.

    Parameters
    ----------
    means : np.ndarray, shape (n_position_dims,)
        Place-field center.
    position : np.ndarray, shape (n_time, n_position_dims)
        Position trajectory.
    max_rate : float, optional
        Peak firing rate at the field center.
    variance : float, optional
        Place-field variance.
    sampling_frequency : float, optional
        Samples per second.
    is_condition : None or np.ndarray, shape (n_time,)
        Boolean mask of times when the place field is active. Default: always.
    rng : np.random.Generator or None, optional
        Random number generator.

    Returns
    -------
    spikes : np.ndarray, shape (n_time,)
        Poisson-sampled spike counts per time bin.
    """
    firing_rate = simulate_place_field_firing_rate(
        means, position, max_rate, variance, is_condition
    )
    return simulate_poisson_spikes(firing_rate, sampling_frequency, rng=rng)


def get_trajectory_direction(
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each time step of a 1D trajectory as inbound or outbound.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,)
        1D position trajectory.

    Returns
    -------
    direction : np.ndarray of str, shape (n_time,)
        ``"Inbound"`` where position is decreasing, ``"Outbound"`` otherwise.
    is_inbound : np.ndarray of bool, shape (n_time,)
        True where position is decreasing.
    """
    is_inbound = np.insert(np.diff(position) < 0, 0, False)
    return np.where(is_inbound, "Inbound", "Outbound"), is_inbound
