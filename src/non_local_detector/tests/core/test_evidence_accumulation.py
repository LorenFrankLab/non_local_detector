"""The dense chunked drivers sum per-row evidence increments in float64.

The reference is a model whose evidence is known exactly: two bins with a
uniform transition keep every prediction at exactly [0.5, 0.5], and equal
log likelihoods in both bins make each row's normalizer exactly that row's log
likelihood. The total is then the float64 sum of the inputs, which a float32
running total over these 200,000 rows misses by about 13 units (713 at a
million rows). Fewer rows would still fail the 1e-12 tolerance but no longer
show a drift of the size that misleads EM convergence checks.
"""

import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
)

pytestmark = pytest.mark.unit

N_TIME = 200_000


@pytest.fixture(scope="module")
def exact_evidence_model():
    rng = np.random.default_rng(20261011)
    row_values = rng.uniform(-30.0, -5.0, N_TIME).astype(np.float32)
    log_likelihoods = np.repeat(row_values[:, None], 2, axis=1)
    expected = float(np.sum(row_values, dtype=np.float64))
    return log_likelihoods, expected


def _arguments():
    initial = np.array([0.5, 0.5], np.float32)
    uniform = np.full((2, 2), 0.5, np.float32)
    state_ind = np.array([0, 1])
    # Expanded transition = continuous * discrete[state_ind][:, state_ind]:
    # all-ones discrete weights keep it uniform.
    discrete = np.ones((N_TIME, 2, 2), np.float32)
    return initial, uniform, state_ind, discrete


@pytest.mark.parametrize("n_chunks", [1, 7])
@pytest.mark.parametrize("covariate", [False, True])
def test_chunked_evidence_is_the_float64_sum(exact_evidence_model, n_chunks, covariate):
    log_likelihoods, expected = exact_evidence_model
    initial, uniform, state_ind, discrete = _arguments()
    common = {
        "time": np.arange(N_TIME, dtype=float),
        "state_ind": state_ind,
        "initial_distribution": initial,
        "log_likelihood_func": None,
        "log_likelihood_args": (),
        "n_chunks": n_chunks,
        "log_likelihoods": log_likelihoods,
    }
    if covariate:
        result = chunked_filter_smoother_covariate_dependent(
            discrete_transition_matrix=discrete,
            continuous_transition_matrix=uniform,
            **common,
        )
    else:
        result = chunked_filter_smoother(transition_matrix=uniform, **common)
    np.testing.assert_allclose(result[2], expected, rtol=1e-12)


def test_checkpointed_evidence_is_the_float64_sum(exact_evidence_model):
    from non_local_detector.checkpointed_inference import (
        checkpointed_forward_backward,
    )

    log_likelihoods, expected = exact_evidence_model
    initial, uniform, state_ind, _ = _arguments()
    result = checkpointed_forward_backward(
        np.arange(N_TIME + 1, dtype=float) * 0.002,
        initial,
        lambda edges, **kwargs: log_likelihoods[kwargs["row_slice"]],
        transition_matrix=uniform,
        state_ind=state_ind,
        chunk_size=N_TIME // 4,
    )
    np.testing.assert_allclose(result.marginal_log_likelihood, expected, rtol=1e-12)
