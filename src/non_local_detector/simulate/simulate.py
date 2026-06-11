"""Main code for simulating position and sorted spikes or clusterless spikes and waveforms."""

import numpy as np
from scipy.stats import multivariate_normal  # type: ignore[import-untyped]

from non_local_detector.simulate._common import (
    get_trajectory_direction,
    simulate_neuron_with_place_field,
    simulate_place_field_firing_rate,
    simulate_poisson_spikes,
    simulate_time,
)

__all__ = [
    "get_trajectory_direction",
    "simulate_multiunit_with_place_fields",
    "simulate_neuron_with_place_field",
    "simulate_place_field_firing_rate",
    "simulate_poisson_spikes",
    "simulate_position",
    "simulate_position_with_pauses",
    "simulate_time",
]


def simulate_position(
    time: np.ndarray, track_height: float, running_speed: float = 15
) -> np.ndarray:
    """Simulate an animal moving sinusoidally along a linear track.

    ``running_speed`` is interpreted as a velocity (distance per second);
    the back-and-forth period is ``2 * track_height / running_speed``.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time in seconds.
    track_height : float
        The height of the simulated track.
    running_speed : float, optional
        Running velocity (default 15).

    Returns
    -------
    position : np.ndarray, shape (n_time,)
        The simulated position of the animal.
    """
    half_height = track_height / 2
    freq = 1 / (2 * track_height / running_speed)
    return half_height * np.sin(freq * 2 * np.pi * time - np.pi / 2) + half_height


def simulate_position_with_pauses(
    time: np.ndarray,
    track_height: float,
    running_speed: float = 15,
    pause: float = 0.5,
    sampling_frequency: float = 1,
) -> np.ndarray:
    """Simulate an animal moving with pauses at track endpoints.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        The time vector.
    track_height : float
        The height of the track.
    running_speed : float, optional
        Running velocity (default 15).
    pause : float, optional
        Pause duration in seconds (default 0.5).
    sampling_frequency : float, optional
        The sampling frequency (default 1).

    Returns
    -------
    position : np.ndarray, shape (n_time,)
        Position trajectory including pauses at endpoints.
    """
    position = simulate_position(time, track_height, running_speed)
    peaks = np.nonzero(np.isclose(position, track_height))[0]
    n_pause_samples = int(pause * sampling_frequency)
    pause_position = np.zeros((time.size + n_pause_samples * peaks.size,))
    pause_ind = peaks[:, np.newaxis] + np.arange(n_pause_samples)
    pause_ind += np.arange(peaks.size)[:, np.newaxis] * n_pause_samples

    pause_position[pause_ind.ravel()] = track_height
    pause_position[pause_position == 0] = position

    return pause_position[: time.size]


def simulate_multiunit_with_place_fields(
    place_means: np.ndarray,
    position: np.ndarray,
    mark_spacing: int = 5,
    n_mark_dims: int = 4,
    place_variance: float = 12.5,
    mark_variance: float = 1.0,
    max_rate: float = 100.0,
    sampling_frequency: int = 1000,
    is_condition: np.ndarray | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a multiunit with neurons at ``place_means``.

    Parameters
    ----------
    place_means : np.ndarray, shape (n_neurons, n_position_dims)
    position : np.ndarray, shape (n_time, n_position_dims)
    mark_spacing : int, optional
    n_mark_dims : int, optional
    place_variance : float
    max_rate : float
    sampling_frequency : int
    is_condition : np.ndarray or None
    seed : int | None, optional
        Random seed for reproducibility. Ignored if rng is provided.
    rng : np.random.Generator or None, optional
        Random number generator. If None, creates one from seed.

    Returns
    -------
    multiunit : np.ndarray, shape (n_time, n_mark_dims)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n_neurons = place_means.shape[0]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    n_time = position.shape[0]
    marks = np.full((n_time, n_mark_dims), np.nan)
    for mean, mark_center in zip(place_means, mark_centers, strict=False):
        is_spike = (
            simulate_neuron_with_place_field(
                mean,
                position,
                max_rate=max_rate,
                variance=place_variance,
                sampling_frequency=sampling_frequency,
                is_condition=is_condition,
                rng=rng,
            )
            > 0
        )
        n_spikes = int(is_spike.sum())
        marks[is_spike] = multivariate_normal(
            mean=[mark_center] * n_mark_dims, cov=mark_variance
        ).rvs(size=n_spikes, random_state=rng)
    return marks
