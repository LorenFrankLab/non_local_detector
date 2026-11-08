"""Non-local detector EM recovers a known discrete transition matrix.

Spikes are simulated from the four-state model the detector assumes (Local,
No-Spike, Non-Local Continuous, Non-Local Fragmented) with a known transition
matrix. EM then estimates the transitions from the spikes alone, with the
place fields learned from the true Local periods.
"""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteStationaryDiagonal,
    Environment,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from tests.conftest import spread_over_sample_bins

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SAMPLING_FREQUENCY = 500.0
TRACK_LENGTH = 180.0
# Rows/columns: Local, No-Spike, Non-Local Continuous, Non-Local Fragmented.
TRUE_TRANSITION = np.array(
    [
        [0.994, 0.002, 0.002, 0.002],
        [0.010, 0.970, 0.010, 0.010],
        [0.010, 0.005, 0.980, 0.005],
        [0.020, 0.010, 0.020, 0.950],
    ]
)


def simulate_non_local_recording(
    seed: int,
    duration_s: float,
    n_neurons: int = 40,
    peak_rate: float = 25.0,
    field_width: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Simulate sorted spikes from the four-state non-local model.

    Returns
    -------
    time : np.ndarray, shape (n_time,)
    position : np.ndarray, shape (n_time,)
        Animal position, running back and forth along a linear track.
    spike_times : list of np.ndarray
        One array per neuron.
    states : np.ndarray, shape (n_time,)
        True discrete state of each time bin.
    """
    rng = np.random.default_rng(seed)
    n_time = int(duration_s * SAMPLING_FREQUENCY)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    position = TRACK_LENGTH / 2 * (1 - np.cos(2 * np.pi * time / 14.0))

    states = np.zeros(n_time, dtype=int)
    represented = position.copy()
    cumulative = np.cumsum(TRUE_TRANSITION, axis=1)
    draws = rng.uniform(size=n_time)
    for t in range(1, n_time):
        states[t] = min(np.searchsorted(cumulative[states[t - 1]], draws[t]), 3)
        if states[t] == 2 and states[t - 1] == 2:
            # The detector's default random walk: variance 6 cm^2 per bin.
            represented[t] = np.clip(
                represented[t - 1] + rng.normal(0.0, np.sqrt(6.0)), 0, TRACK_LENGTH
            )
        elif states[t] in (2, 3):
            represented[t] = rng.uniform(0, TRACK_LENGTH)

    centers = np.linspace(0, TRACK_LENGTH, n_neurons)
    rates = peak_rate * np.exp(
        -0.5 * ((represented[:, None] - centers) / field_width) ** 2
    )
    rates[states == 1] = 0.0
    counts = rng.poisson(rates / SAMPLING_FREQUENCY)
    spike_times = [
        np.sort(
            spread_over_sample_bins(rng, time, counts[:, neuron], SAMPLING_FREQUENCY)
        )
        for neuron in range(n_neurons)
    ]
    return time, position, spike_times, states


def empirical_transition(states: np.ndarray) -> np.ndarray:
    """Transition frequencies of the true state sequence (the oracle MLE)."""
    counts = np.zeros((4, 4))
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def binomial_standard_error(states: np.ndarray) -> np.ndarray:
    """Sampling standard error of each transition frequency, shape (4, 4).

    Entry ``(i, j)`` is ``sqrt(p_ij (1 - p_ij) / n_i)`` for the true
    probability ``p_ij`` and the ``n_i`` transitions out of state ``i``, so
    the Local row's 0.002 off-diagonals get a tolerance their size allows.
    """
    n_from = np.bincount(states[:-1], minlength=4)
    return np.sqrt(TRUE_TRANSITION * (1 - TRUE_TRANSITION) / n_from[:, None])


def fit_transitions(
    recording: tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray],
    transition_representation: str = "dense",
    tolerance: float = 1e-8,
    **model_kwargs,
) -> NonLocalSortedSpikesDetector:
    """Estimate the discrete transitions of a simulated recording by EM.

    The place fields are fit on the true Local bins and held fixed, so EM
    updates only the transitions and initial conditions; it runs until the
    penalized log-likelihood changes by less than ``tolerance`` relative.

    Parameters
    ----------
    recording : tuple
        ``(time, position, spike_times, states)`` from
        :func:`simulate_non_local_recording`.
    transition_representation : {"dense", "structured"}, optional
        EM transition path, by default "dense".
    tolerance : float, optional
        EM convergence tolerance, by default 1e-8.
    **model_kwargs
        Passed to :class:`NonLocalSortedSpikesDetector`.

    Returns
    -------
    detector : NonLocalSortedSpikesDetector
        The fitted detector.
    """
    time, position, spike_times, states = recording
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(
            place_bin_size=2.0, position_range=[(0, TRACK_LENGTH)]
        ),
        **model_kwargs,
    )
    detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time_edges=time_edges_from_centers(time),
        is_training=states == 0,
        estimate_encoding_model=False,
        max_iter=500,
        tolerance=tolerance,
        transition_representation=transition_representation,
    )
    return detector


MAXIMUM_LIKELIHOOD = {
    "discrete_transition_stickiness": np.zeros(4),
    "discrete_transition_concentration": 1.0,
    "frozen_discrete_transition_rows": None,
    # Far from the truth (every stay probability 0.9, up to 0.094 off), so
    # the tolerances below fail unless EM actually learns the transitions.
    "discrete_transition_type": DiscreteStationaryDiagonal(
        diagonal_values=np.full(4, 0.9)
    ),
}

# Entry errors in binomial standard errors (see binomial_standard_error). The
# test recording (seed 0) reaches 1.4 against the oracle and 2.4 against the
# truth; over 12 simulated 120 s recordings (seeds 0-11, dense) the worst were
# 3.6 and 4.8. An estimate stuck in Local (row [1, 0, 0, 0]) is 16 and 15 off,
# and EM's starting matrix more still.
MAX_STANDARD_ERRORS = 6.0


@pytest.fixture(scope="module")
def recording():
    return simulate_non_local_recording(seed=0, duration_s=120.0)


@pytest.fixture(scope="module")
def maximum_likelihood_fit(recording):
    """Maximum-likelihood EM fit with dense transitions.

    Structured EM's agreement with dense EM is tested in test_checkpointed_em.py.
    """
    return fit_transitions(recording, **MAXIMUM_LIKELIHOOD)


def test_maximum_likelihood_em_recovers_true_transitions(
    recording, maximum_likelihood_fit
):
    """Without a prior, EM's estimate matches the transition frequencies of
    the hidden state sequence (and so the true matrix)."""
    detector = maximum_likelihood_fit
    estimate = np.asarray(detector.discrete_state_transitions_)
    states = recording[3]
    tolerance = MAX_STANDARD_ERRORS * binomial_standard_error(states)

    assert detector.converged_
    assert detector.em_monotonicity_violations_ == []
    np.testing.assert_array_less(
        np.abs(estimate - empirical_transition(states)), tolerance
    )
    np.testing.assert_array_less(np.abs(estimate - TRUE_TRANSITION), tolerance)


def test_default_prior_shrinks_toward_its_stickiness(recording, maximum_likelihood_fit):
    """The default prior moves the Local stay probability away from the
    maximum-likelihood estimate in the documented direction.

    Only the direction is checked, so EM stops at a 1e-4 tolerance (the
    Local stay probability is then within 4e-7 of the 1e-8 fit's 0.9986,
    against 0.9937 without the prior). The frozen No-Spike row is covered in
    test_frozen_transition_rows.py.
    """
    default = np.asarray(
        fit_transitions(recording, tolerance=1e-4).discrete_state_transitions_
    )
    maximum_likelihood = np.asarray(maximum_likelihood_fit.discrete_state_transitions_)

    # 1e5 pseudo-counts on the Local diagonal pull its stay probability up.
    assert default[0, 0] > maximum_likelihood[0, 0]
