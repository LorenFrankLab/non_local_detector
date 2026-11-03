"""Per-discrete-state covariate transition products match the dense transition."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.core import (
    _covariate_backward_product,
    _covariate_forward_product,
    _state_one_hot,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_states", [1, 4, 16])
@pytest.mark.parametrize("permuted", [False, True])
def test_products_match_the_dense_transition(n_states, permuted):
    rng = np.random.default_rng(n_states)
    n_bins = 48
    state_ind = np.sort(rng.integers(0, n_states, n_bins))
    if permuted:
        state_ind = rng.permutation(state_ind)
    continuous = rng.random((n_bins, n_bins))
    continuous[rng.random((n_bins, n_bins)) < 0.2] = 0.0
    discrete = rng.random((n_states, n_states))
    discrete[0, -1] = 0.0
    dense = continuous * discrete[np.ix_(state_ind, state_ind)]
    vector = rng.random(n_bins)
    args = (
        jnp.asarray(discrete),
        jnp.asarray(continuous),
        jnp.asarray(state_ind),
        _state_one_hot(jnp.asarray(state_ind), n_states, jnp.float32),
    )

    forward = _covariate_forward_product(jnp.asarray(vector), *args)
    backward = _covariate_backward_product(jnp.asarray(vector), *args)

    np.testing.assert_allclose(np.asarray(forward), vector @ dense, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(backward), dense @ vector, rtol=1e-5)
