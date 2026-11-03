"""Main code for simulating position and sorted spikes or clusterless spikes and waveforms."""

import numpy as np
from scipy.stats import multivariate_normal

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
    """Simulate an animal moving with pauses at the far end of the track.

    The animal pauses for ``pause`` seconds once per arrival at ``track_height``.

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
    # Samples within tolerance of the track end form one run per arrival; pause
    # once at the first sample of each run, not at every sample of it.
    at_end = np.isclose(position, track_height)
    arrivals = np.nonzero(at_end & ~np.concatenate(([False], at_end[:-1])))[0]
    n_pause_samples = int(pause * sampling_frequency)
    paused = np.insert(position, np.repeat(arrivals, n_pause_samples), track_height)

    return paused[: time.size]


def simulate_multiunit_with_place_fields(
    place_means: np.ndarray,
    position: np.ndarray,
    mark_spacing: float = 5,
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
        Place-field centers, one row per neuron.
    position : np.ndarray, shape (n_time, n_position_dims)
        Position trajectory.
    mark_spacing : float, optional
        Spacing between the mark centers of successive neurons, by default 5.
        Neuron ``i`` emits marks centered at ``i * mark_spacing`` in every
        mark dimension.
    n_mark_dims : int, optional
        Number of mark (waveform feature) dimensions, by default 4.
    place_variance : float, optional
        Variance of each neuron's Gaussian place field, by default 12.5.
    mark_variance : float, optional
        Variance of the isotropic Gaussian from which each spike's marks are
        drawn, by default 1.0.
    max_rate : float, optional
        Peak firing rate of each neuron at its field center, in Hz, by
        default 100.0.
    sampling_frequency : int, optional
        Samples per second of ``position``, by default 1000.
    is_condition : np.ndarray, shape (n_time,) or None, optional
        Boolean mask of times when the place fields are active, by default
        None (always active).
    seed : int | None, optional
        Random seed for reproducibility. Ignored if rng is provided.
    rng : np.random.Generator or None, optional
        Random number generator. If None, creates one from seed.

    Returns
    -------
    multiunit : np.ndarray, shape (n_time, n_mark_dims)
        Marks at each time sample; NaN where no neuron spiked.
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
