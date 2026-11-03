"""Every transition operator the HMM passes accept satisfies one protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.continuous_state_transitions import Uniform
from non_local_detector.core import (
    TransitionOperator,
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
    assert isinstance(OPERATORS[name](), TransitionOperator)


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


def test_checkpointed_rejects_a_nonconforming_operator_before_any_callback():
    calls = []
    with pytest.raises(TypeError, match="pair_products") as caught:
        checkpointed_forward_backward(
            np.arange(5) * 0.002,
            np.full(N_BINS, 1 / N_BINS, dtype=np.float32),
            lambda edges, **kw: calls.append(1) or np.zeros((4, N_BINS)),
            transition_operator=_ForwardBackwardOnly(_matrix()),
            state_ind=STATE_IND,
        )
    assert "TransitionOperator" in str(caught.value)
    assert calls == []
