"""The public HMM drivers reject mismatched shapes and dtypes at entry."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
    row_slice_aware,
)

pytestmark = pytest.mark.unit

N_TIME, N_STATES = 6, 2
STATE_IND = np.array([0, 0, 1])
N_BINS = STATE_IND.size


def _inputs():
    rng = np.random.default_rng(0)
    continuous = np.full((N_BINS, N_BINS), 1 / N_BINS)
    return {
        "time": np.arange(N_TIME, dtype=float),
        "state_ind": STATE_IND,
        "initial_distribution": np.full(N_BINS, 1 / N_BINS),
        "transition_matrix": continuous,
        "continuous_transition_matrix": continuous,
        "discrete_transition_matrix": np.full((N_TIME, N_STATES, N_STATES), 0.5),
        "log_likelihoods": rng.normal(size=(N_TIME, N_BINS)),
        "is_missing": np.zeros(N_TIME, dtype=bool),
    }


DRIVERS = {
    "chunked": (
        chunked_filter_smoother,
        ("time", "state_ind", "initial_distribution", "transition_matrix"),
    ),
    "chunked_covariate": (
        chunked_filter_smoother_covariate_dependent,
        (
            "time",
            "state_ind",
            "initial_distribution",
            "discrete_transition_matrix",
            "continuous_transition_matrix",
        ),
    ),
    "viterbi": (
        most_likely_sequence,
        ("time", "initial_distribution", "transition_matrix"),
    ),
    "viterbi_covariate": (
        most_likely_sequence_covariate_dependent,
        (
            "time",
            "state_ind",
            "initial_distribution",
            "discrete_transition_matrix",
            "continuous_transition_matrix",
        ),
    ),
}


def _call(name, inputs, **overrides):
    driver, keys = DRIVERS[name]
    kwargs = {key: inputs[key] for key in keys}
    kwargs.update(
        log_likelihood_func=None,
        log_likelihood_args=(),
        log_likelihoods=inputs["log_likelihoods"],
        is_missing=inputs["is_missing"],
    )
    kwargs.update(overrides)
    return driver(**kwargs)


@pytest.mark.parametrize("name", DRIVERS)
def test_valid_inputs_pass(name):
    _call(name, _inputs())


BAD = {
    "short log_likelihoods": ("log_likelihoods", np.zeros((N_TIME - 1, N_BINS))),
    "wrong-bin log_likelihoods": ("log_likelihoods", np.zeros((N_TIME, N_BINS + 1))),
    "short is_missing": ("is_missing", np.zeros(N_TIME - 1, dtype=bool)),
    "2-D time": ("time", np.zeros((N_TIME, 1))),
    "empty time": ("time", np.zeros(0)),
    "integer dtype": ("dtype", jnp.int32),
}


@pytest.mark.parametrize("name", DRIVERS)
@pytest.mark.parametrize("case", BAD)
def test_mismatched_inputs_raise(name, case):
    key, value = BAD[case]
    with pytest.raises(ValueError, match=key):
        _call(name, _inputs(), **{key: value})


@pytest.mark.parametrize("name", DRIVERS)
def test_mismatched_transition_raises(name):
    key = DRIVERS[name][1][-1]
    with pytest.raises(ValueError, match=key):
        _call(name, _inputs(), **{key: np.eye(N_BINS + 1)})


@pytest.mark.parametrize("name", ["chunked_covariate", "viterbi_covariate"])
def test_discrete_transitions_must_match_time_and_states(name):
    inputs = _inputs()
    with pytest.raises(ValueError, match="discrete_transition_matrix"):
        _call(
            name,
            inputs,
            discrete_transition_matrix=inputs["discrete_transition_matrix"][1:],
        )
    with pytest.raises(ValueError, match="discrete state 1"):
        _call(
            name,
            inputs,
            discrete_transition_matrix=np.ones((N_TIME, 1, 1)),
        )


@pytest.mark.parametrize("name", ["chunked", "chunked_covariate"])
def test_state_ind_must_be_integer_per_bin(name):
    with pytest.raises(ValueError, match="state_ind"):
        _call(name, _inputs(), state_ind=STATE_IND[:-1])
    with pytest.raises(ValueError, match="integer"):
        _call(name, _inputs(), state_ind=STATE_IND.astype(float))


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("name", ["chunked", "chunked_covariate"])
def test_callback_rows_must_match_the_request(name, cache):
    """A callback returning one row too few would silently truncate outputs."""
    inputs = _inputs()

    @row_slice_aware
    def short(time, is_missing=None, row_slice=None):
        rows = inputs["log_likelihoods"][row_slice or slice(None)]
        return rows[:-1]

    with pytest.raises(ValueError, match="log_likelihood_func output"):
        _call(
            name,
            inputs,
            log_likelihoods=None,
            log_likelihood_func=short,
            cache_log_likelihoods=cache,
            n_chunks=2,
        )
