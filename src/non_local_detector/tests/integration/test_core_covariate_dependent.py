"""Covariate-dependent Viterbi on a sequence too long to enumerate.

Short sequences are checked against exhaustive path scoring in
``core/test_conditioning_reference.py``; this checks a 30-step, 7-state chain
whose transitions change at every step against a float64 NumPy dynamic
program.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import viterbi_covariate_dependent

pytestmark = pytest.mark.unit


def make_time_varying_transitions(n_time: int, n_states: int) -> np.ndarray:
    """Create a smooth, time-varying row-stochastic transition matrix sequence."""
    mats = []
    for t in range(n_time):
        # base: near-identity
        tm = np.eye(n_states) * 0.85
        # small drift to the right that oscillates over time
        drift = 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * t / max(2, n_time)))
        for i in range(n_states):
            j = min(n_states - 1, i + 1)
            tm[i, j] += drift
        # renormalize rows
        tm = tm / tm.sum(axis=1, keepdims=True)
        mats.append(tm)
    return np.stack(mats, axis=0)


def viterbi_reference(
    initial: np.ndarray, transitions: np.ndarray, log_likelihoods: np.ndarray
) -> np.ndarray:
    """Most likely path by float64 dynamic programming.

    Parameters
    ----------
    initial : np.ndarray, shape (n_states,)
    transitions : np.ndarray, shape (n_time, n_states, n_states)
        ``transitions[t]`` moves ``s_t -> s_{t+1}``; the last entry is unused.
    log_likelihoods : np.ndarray, shape (n_time, n_states)

    Returns
    -------
    path : np.ndarray, shape (n_time,)
    """
    with np.errstate(divide="ignore"):
        log_initial = np.log(initial)
        log_transitions = np.log(transitions)
    score = log_initial + log_likelihoods[0]
    back_pointers = []
    for t in range(1, len(log_likelihoods)):
        candidates = score[:, None] + log_transitions[t - 1]
        back_pointers.append(candidates.argmax(axis=0))
        score = candidates.max(axis=0) + log_likelihoods[t]
    path = [int(score.argmax())]
    for back in reversed(back_pointers):
        path.append(int(back[path[-1]]))
    return np.array(path[::-1])


def test_viterbi_covariate_dependent_matches_dynamic_programming():
    n_time = 30
    n_states = 7
    # One bin per discrete state, and an all-ones continuous transition, so
    # the expanded per-step transition (continuous * discrete) is the
    # time-varying discrete matrix itself.
    state_ind = np.arange(n_states)
    d_tm = make_time_varying_transitions(n_time, n_states)
    c_tm = np.ones((n_states, n_states))

    # synthetic log-likelihoods with a drifting peak
    centers = np.linspace(0, n_states - 1, n_time)
    ll = -0.5 * (np.arange(n_states)[None, :] - centers[:, None]) ** 2

    init = np.ones((n_states,)) / n_states
    expected = viterbi_reference(init, d_tm, ll)
    # The drift has to matter: freezing the first step's transition for the
    # whole sequence gives a different optimum.
    frozen = viterbi_reference(init, np.broadcast_to(d_tm[0], d_tm.shape), ll)
    assert not np.array_equal(expected, frozen)

    path = viterbi_covariate_dependent(
        jnp.asarray(init),
        jnp.asarray(d_tm),
        jnp.asarray(c_tm),
        jnp.asarray(state_ind),
        jnp.asarray(ll),
    )

    np.testing.assert_array_equal(np.asarray(path), expected)
