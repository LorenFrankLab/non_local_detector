import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother_covariate_dependent,
    filter_covariate_dependent,
    smoother_covariate_dependent,
    viterbi_covariate_dependent,
)


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


@pytest.mark.parametrize("n_chunks", [2, 3, 5])
@pytest.mark.integration
def test_covariate_dependent_chunked_equals_nonchunked(n_chunks):
    n_time = 30
    n_states = 7
    # state_ind: identity mapping of bins to states
    state_ind = np.arange(n_states)

    # time-varying transitions and identity continuous transitions
    d_tm = make_time_varying_transitions(n_time, n_states)
    c_tm = np.eye(n_states)

    # synthetic log-likelihoods with a drifting peak
    centers = np.linspace(0, n_states - 1, n_time)
    ll = np.zeros((n_time, n_states))
    for t in range(n_time):
        diffs = np.arange(n_states) - centers[t]
        ll[t] = -0.5 * (diffs / 1.0) ** 2  # unnormalized log-Gaussian over states

    init = np.ones((n_states,)) / n_states

    # Non-chunked
    (marg_like, pred_next), (filtered, predicted) = filter_covariate_dependent(
        jnp.asarray(init),
        jnp.asarray(d_tm),
        jnp.asarray(c_tm),
        jnp.asarray(state_ind),
        jnp.asarray(ll),
    )
    smoothed = smoother_covariate_dependent(
        jnp.asarray(d_tm), jnp.asarray(c_tm), jnp.asarray(state_ind), filtered
    )

    # Chunked with precomputed ll to ensure parity
    (
        acausal_post_chunked,
        _,
        _,
        _,
        _,
        _,
        causal_post_chunked,
    ) = chunked_filter_smoother_covariate_dependent(
        time=np.arange(n_time),
        state_ind=state_ind,
        initial_distribution=init,
        discrete_transition_matrix=d_tm,
        continuous_transition_matrix=c_tm,
        log_likelihood_func=lambda *args, **kwargs: None,
        log_likelihood_args=(),
        is_missing=None,
        n_chunks=n_chunks,
        log_likelihoods=ll,
        cache_log_likelihoods=False,
    )

    assert np.allclose(acausal_post_chunked, np.asarray(smoothed), rtol=1e-5, atol=1e-6)
    assert np.allclose(causal_post_chunked, np.asarray(filtered), rtol=1e-5, atol=1e-6)

    # Viterbi covariate-dependent path validity
    path = viterbi_covariate_dependent(
        jnp.asarray(init),
        jnp.asarray(d_tm),
        jnp.asarray(c_tm),
        jnp.asarray(state_ind),
        jnp.asarray(ll),
    )
    assert path.shape == (n_time,)
    assert int(path.min()) >= 0 and int(path.max()) < n_states
