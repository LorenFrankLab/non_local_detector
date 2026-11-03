"""Posteriors left on the device match the host outputs exactly."""

import jax
import numpy as np
import pytest

from non_local_detector.core import (
    FilterSmootherResult,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
)
from non_local_detector.discrete_state_transitions import (
    estimate_discrete_transition_counts_from_factorized_posteriors,
    estimate_discrete_transition_responses_from_factorized_posteriors,
)

pytestmark = pytest.mark.unit

N_TIME, STATE_IND = 40, np.array([0, 0, 0, 1, 1, 2])
SPATIAL = ("acausal_posterior", "causal_posterior", "predictive_posterior")
STATE_PROBABILITIES = (
    "acausal_state_probabilities",
    "causal_state_probabilities",
    "predictive_state_probabilities",
)


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n_bins = STATE_IND.size
    continuous = rng.random((n_bins, n_bins)).astype(np.float32)
    continuous /= continuous.sum(axis=1, keepdims=True)
    discrete = rng.random((N_TIME, 3, 3)).astype(np.float32)
    discrete /= discrete.sum(axis=2, keepdims=True)
    return {
        "continuous": continuous,
        "discrete": discrete,
        "log_likelihoods": rng.normal(size=(N_TIME, n_bins)).astype(np.float32),
        "initial": np.full(n_bins, 1.0 / n_bins, np.float32),
    }


def _decode(model, covariate, n_chunks, host_outputs):
    common = {
        "time": np.arange(N_TIME, dtype=float),
        "state_ind": STATE_IND,
        "initial_distribution": model["initial"],
        "log_likelihood_func": None,
        "log_likelihood_args": (),
        "n_chunks": n_chunks,
        "log_likelihoods": model["log_likelihoods"],
        "host_outputs": host_outputs,
    }
    if covariate:
        return chunked_filter_smoother_covariate_dependent(
            discrete_transition_matrix=model["discrete"],
            continuous_transition_matrix=model["continuous"],
            **common,
        )
    transition = (
        model["continuous"] * model["discrete"][0][np.ix_(STATE_IND, STATE_IND)]
    )
    return chunked_filter_smoother(transition_matrix=transition, **common)


@pytest.mark.parametrize("n_chunks", [1, 3])
@pytest.mark.parametrize("covariate", [False, True])
def test_device_posteriors_equal_host_posteriors(model, covariate, n_chunks):
    host = _decode(model, covariate, n_chunks, host_outputs=True)
    device = _decode(model, covariate, n_chunks, host_outputs=False)
    for name in SPATIAL:
        assert isinstance(getattr(host, name), np.ndarray)
        assert isinstance(getattr(device, name), jax.Array)
        np.testing.assert_array_equal(
            np.asarray(getattr(device, name)), getattr(host, name)
        )
    for name in STATE_PROBABILITIES:  # stay on the host either way
        assert isinstance(getattr(device, name), np.ndarray)
        np.testing.assert_array_equal(getattr(device, name), getattr(host, name))


@pytest.mark.parametrize("covariate", [False, True])
def test_result_is_a_named_tuple_in_the_historical_order(model, covariate):
    """Positional unpacking keeps working: fields are in the old tuple order."""
    result = _decode(model, covariate, n_chunks=2, host_outputs=True)
    assert isinstance(result, FilterSmootherResult)
    assert FilterSmootherResult._fields == (
        "acausal_posterior",
        "acausal_state_probabilities",
        "marginal_log_likelihood",
        "causal_state_probabilities",
        "predictive_state_probabilities",
        "log_likelihoods",
        "causal_posterior",
        "predictive_posterior",
    )
    (acausal, _, marginal, _, _, log_likelihoods, causal, _) = result
    assert acausal is result.acausal_posterior
    assert causal is result.causal_posterior
    assert marginal == result.marginal_log_likelihood
    assert isinstance(marginal, float)
    np.testing.assert_array_equal(log_likelihoods, model["log_likelihoods"])


@pytest.mark.parametrize(
    "estimator",
    [
        estimate_discrete_transition_counts_from_factorized_posteriors,
        estimate_discrete_transition_responses_from_factorized_posteriors,
    ],
)
def test_transition_statistics_from_device_posteriors(model, estimator):
    host = _decode(model, covariate=False, n_chunks=1, host_outputs=True)
    device = _decode(model, covariate=False, n_chunks=1, host_outputs=False)
    discrete = (
        model["discrete"][0]
        if estimator is estimate_discrete_transition_counts_from_factorized_posteriors
        else model["discrete"]
    )

    def statistics(outputs):
        return estimator(
            outputs[6], outputs[7], outputs[0], model["continuous"], discrete, STATE_IND
        )

    np.testing.assert_array_equal(statistics(device), statistics(host))
