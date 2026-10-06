"""State probabilities preserve the precision of their spatial posterior."""

import numpy as np
import pytest

from non_local_detector.core import (
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
)


@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("n_chunks", [1, 3])
def test_state_aggregation_matches_float64_sum_without_tf32(covariate, n_chunks):
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
