"""HMM products keep full float32 precision on every backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import core
from non_local_detector.core import (
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
)
from tests.likelihoods._sorted_spikes_helpers import dot_general_precisions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_state_aggregation_matches_float64_sum_without_tf32(covariate, n_chunks):
    """State probabilities equal the float64 sum of their state bins.

    The aggregation is a matmul, and CUDA's default TF32 precision visibly
    changes these sums, so it must run at ``Precision.HIGHEST``. This can only
    catch a lost precision flag on a GPU: CPU float32 matmuls are full
    precision either way, so on CPU the test passes without the flag.
    """
    n_rows, n_bins = 100, 100
    labels = np.r_[0, np.ones(n_bins - 1, dtype=int)]
    prior = np.r_[0.99, np.full(n_bins - 1, 0.01 / (n_bins - 1))].astype(np.float32)
    likelihood = np.zeros((n_rows, n_bins), dtype=np.float32)
    arguments = {
        "time": np.arange(n_rows),
        "state_ind": labels,
        "initial_distribution": prior,
        "log_likelihood_func": lambda time, *args: likelihood[time],
        "log_likelihood_args": (),
        "log_likelihoods": likelihood,
        "n_chunks": n_chunks,
    }
    if covariate:
        result = chunked_filter_smoother_covariate_dependent(
            **arguments,
            continuous_transition_matrix=np.eye(n_bins),
            discrete_transition_matrix=np.broadcast_to(np.eye(2), (n_rows, 2, 2)),
        )
    else:
        result = chunked_filter_smoother(**arguments, transition_matrix=np.eye(n_bins))
    for posterior, state_probabilities in [
        (result[0], result[1]),
        (result[6], result[3]),
        (result[7], result[4]),
    ]:
        expected = np.column_stack(
            [
                np.asarray(posterior)[:, labels == state].sum(axis=1, dtype=np.float64)
                for state in range(2)
            ]
        )
        np.testing.assert_allclose(state_probabilities, expected, rtol=1e-6, atol=1e-7)


def _transition_passes():
    """Each HMM pass of ``core`` as a function of small float32 inputs."""
    n_time, n_bins = 4, 3
    state_ind = jnp.array([0, 1, 1])
    initial = jnp.full(n_bins, 1 / n_bins, jnp.float32)
    transition = jnp.full((n_bins, n_bins), 1 / n_bins, jnp.float32)
    discrete = jnp.full((n_time, 2, 2), 0.5, jnp.float32)
    log_likelihoods = jnp.zeros((n_time, n_bins), jnp.float32)
    probabilities = jnp.full((n_time, n_bins), 1 / n_bins, jnp.float32)
    return {
        "filter": (core.filter, (initial, transition, log_likelihoods)),
        "smoother": (core.smoother, (transition, probabilities)),
        "pair_products": (
            lambda f, r: core._StationaryTransition(
                transition, state_ind, 2
            ).pair_products(f, r),
            (initial, initial),
        ),
        "covariate_filter": (
            core.filter_covariate_dependent,
            (initial, discrete, transition, state_ind, log_likelihoods),
        ),
        "covariate_smoother": (
            core.smoother_covariate_dependent,
            (discrete, transition, state_ind, probabilities),
        ),
    }


@pytest.mark.parametrize(
    "name",
    ["filter", "smoother", "pair_products", "covariate_filter", "covariate_smoother"],
)
def test_transition_products_request_highest_precision(name):
    """Every transition product of the HMM passes requests HIGHEST precision.

    Without it, CUDA may evaluate a product in TF32 (XLA currently lowers
    the vector-matrix products to full-float32 reductions, but need not), and
    the stationary and covariate paths would then disagree on GPU. CPU
    float32 is full precision either way, so this checks the request in the
    traced program, under the default matmul precision so an ambient setting
    cannot satisfy it.
    """
    function, arguments = _transition_passes()[name]
    with jax.default_matmul_precision("default"):
        traced = jax.make_jaxpr(function)(*arguments)
    precisions = dot_general_precisions(traced.jaxpr)
    assert precisions
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    assert all(precision == highest for precision in precisions), precisions
