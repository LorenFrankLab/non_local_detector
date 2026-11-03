"""Every transition operator the HMM passes accept satisfies one protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.continuous_state_transitions import Uniform
from non_local_detector.core import (
    TransitionOperator,
    TransitionProducts,
    _CovariateTransition,
    _StationaryTransition,
)
from non_local_detector.environment import Environment
from non_local_detector.transition_operators import build_transition_operator

pytestmark = pytest.mark.unit

STATE_IND = np.array([0, 0, 1, 1])
N_BINS = STATE_IND.size


def _block_operator():
    env = Environment()
    env.place_bin_centers_ = np.arange(N_BINS, dtype=float)[:, None]
    env.centers_shape_ = (N_BINS,)
    env.is_track_interior_ = np.ones((N_BINS,), dtype=bool)
    return build_transition_operator([[Uniform()]], [env], (N_BINS,))


def _matrix():
    rng = np.random.default_rng(3)
    matrix = rng.random((N_BINS, N_BINS), dtype=np.float32)
    return jnp.asarray(matrix / matrix.sum(1, keepdims=True))


OPERATORS = {
    "stationary": lambda: _StationaryTransition(_matrix(), jnp.asarray(STATE_IND), 2),
    "covariate": lambda: _CovariateTransition.create(_matrix(), STATE_IND, 2),
    "block": _block_operator,
    "fused": lambda: _block_operator().fused(),
}


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_satisfy_the_protocol(name):
    operator = OPERATORS[name]()
    assert isinstance(operator, TransitionOperator)
    assert isinstance(operator, TransitionProducts)


@pytest.mark.parametrize("name", ["stationary", "covariate", "block"])
def test_default_backward_and_pairs_is_backward_and_pair_products(name):
    """Operators without a fused form inherit the unfused pair of products."""
    operator = OPERATORS[name]()
    weight = None if name == "stationary" else jnp.full((2, 2), 0.5)
    if name == "block":
        weight = jnp.ones((1, 1))
    n_bins = int(operator.forward(jnp.ones(N_BINS), weight).shape[0])
    rng = np.random.default_rng(4)
    values = jnp.asarray(rng.random(n_bins, dtype=np.float32))
    filtered = jnp.asarray(rng.random(n_bins, dtype=np.float32))

    backward, pairs = operator.backward_and_pairs(values, filtered, weight)

    np.testing.assert_array_equal(backward, operator.backward(values, weight))
    np.testing.assert_array_equal(
        pairs, operator.pair_products(filtered, values, weight)
    )


@jax.tree_util.register_pytree_node_class
class _ForwardBackwardOnly:
    """Registered pytree with the products but no transition statistics."""

    def __init__(self, matrix):
        self.matrix = matrix

    def forward(self, p, weight=None):
        return p @ self.matrix

    def backward(self, v, weight=None):
        return self.matrix @ v

    def tree_flatten(self):
        return (self.matrix,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _checkpointed(operator, likelihood, **kwargs):
    return checkpointed_forward_backward(
        np.arange(5) * 0.002,
        np.full(N_BINS, 1 / N_BINS, dtype=np.float32),
        likelihood,
        transition_operator=operator,
        state_ind=STATE_IND,
        **kwargs,
    )


def test_checkpointed_prediction_needs_only_the_transition_products():
    """A forward/backward-only operator decodes like the dense matrix."""
    matrix = _matrix()
    log_likelihoods = np.random.default_rng(5).normal(size=(4, N_BINS))

    def likelihood(edges, **kw):
        return log_likelihoods[kw["row_slice"]]

    products = _checkpointed(_ForwardBackwardOnly(matrix), likelihood)
    dense = checkpointed_forward_backward(
        np.arange(5) * 0.002,
        np.full(N_BINS, 1 / N_BINS, dtype=np.float32),
        likelihood,
        transition_matrix=np.asarray(matrix),
        state_ind=STATE_IND,
    )
    np.testing.assert_allclose(
        products.dataset.acausal_state_probabilities,
        dense.dataset.acausal_state_probabilities,
        rtol=1e-5,
    )


def test_checkpointed_statistics_reject_products_only_before_any_callback():
    calls = []
    with pytest.raises(TypeError, match="pair_products") as caught:
        _checkpointed(
            _ForwardBackwardOnly(_matrix()),
            lambda edges, **kw: calls.append(1) or np.zeros((4, N_BINS)),
            transition_statistics="counts",
        )
    message = str(caught.value)
    assert "TransitionOperator for transition_statistics" in message
    assert "backward_and_pairs" in message
    assert calls == []


def test_checkpointed_rejects_an_operator_without_products():
    @jax.tree_util.register_pytree_node_class
    class ForwardOnly(_ForwardBackwardOnly):
        backward = None

    with pytest.raises(
        TypeError, match="TransitionProducts; ForwardOnly lacks backward"
    ):
        _checkpointed(ForwardOnly(_matrix()), lambda edges, **kw: None)
