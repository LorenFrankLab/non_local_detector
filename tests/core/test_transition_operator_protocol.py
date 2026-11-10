"""Every transition operator the HMM passes accept satisfies one protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.continuous_state_transitions import RandomWalk, Uniform
from non_local_detector.core import (
    TransitionOperator,
    TransitionProducts,
    _check_operator_rows,
    _CovariateTransition,
    _StationaryTransition,
    row_slice_aware,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.models._defaults import _ModelDefaults
from non_local_detector.transition_operators import (
    BlockTransitionOperator,
    DenseBlock,
    build_transition_operator,
)
from tests.conftest import precision_mode, set_place_grid

pytestmark = pytest.mark.unit

STATE_IND = np.array([0, 0, 1, 1])
N_BINS = STATE_IND.size


def _block_operator():
    env = set_place_grid(
        Environment(),
        place_bin_centers=np.arange(N_BINS, dtype=float)[:, None],
        centers_shape=(N_BINS,),
        is_track_interior=np.ones((N_BINS,), dtype=bool),
    )
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

    @row_slice_aware
    def likelihood(**kw):
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
            row_slice_aware(lambda **kw: calls.append(1) or np.zeros((4, N_BINS))),
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
        _checkpointed(ForwardOnly(_matrix()), row_slice_aware(lambda **kw: None))


def _bad_block_operator(matrix):
    """One-state block operator over ``matrix``, bound to a unit discrete weight."""
    return BlockTransitionOperator(
        ((DenseBlock(np.asarray(matrix)),),), (N_BINS,)
    ).bind_discrete(np.ones((1, 1)))


# Operators whose joint rows all sum to 1.5: each would inflate the evidence.
INFLATING_OPERATORS = {
    "stationary": lambda: _StationaryTransition(1.5 * _matrix()),
    "products_only": lambda: _ForwardBackwardOnly(1.5 * _matrix()),
    "block": lambda: _bad_block_operator(1.5 * _matrix()),
    "fused": lambda: _bad_block_operator(1.5 * _matrix()).fused(),
}


def _rejects_before_any_callback(operator, *pieces, **kwargs):
    calls = []
    with pytest.raises(ValidationError) as caught:
        _checkpointed(
            operator,
            row_slice_aware(lambda **kw: calls.append(1) or np.zeros((4, N_BINS))),
            **kwargs,
        )
    message = str(caught.value)
    for piece in pieces:
        assert piece in message, message
    assert "Hint:" in message, message
    assert calls == []
    return message


@pytest.mark.parametrize("name", INFLATING_OPERATORS)
def test_checkpointed_rejects_operator_rows_above_one(name):
    """An operator's rows ``T @ 1`` are checked like a dense matrix's rows.

    ``_StationaryTransition(1.5 * A)`` used to be accepted (raising the
    marginal log likelihood) while ``transition_matrix=1.5 * A`` raised.
    """
    _rejects_before_any_callback(
        INFLATING_OPERATORS[name](),
        "transition_operator rows must be probability distributions",
        "4 of 4 rows sum above 1 (row 0 sums to 1.5)",
    )


def test_checkpointed_rejects_nonfinite_operator_rows():
    matrix = np.array(_matrix())
    matrix[2, 1] = np.nan
    _rejects_before_any_callback(
        _StationaryTransition(jnp.asarray(matrix)),
        "transition_operator rows must be probability distributions",
        "non-finite row sums",
    )


def test_checkpointed_rejects_operator_of_another_size():
    rng = np.random.default_rng(6)
    square = rng.random((N_BINS + 1, N_BINS + 1), dtype=np.float32)
    # A square operator of another size cannot multiply n_bins values.
    _rejects_before_any_callback(
        _ForwardBackwardOnly(jnp.asarray(square / square.sum(1, keepdims=True))),
        "transition_operator could not be applied to an all-ones vector",
        "TypeError",
    )
    # One with n_bins columns but more rows returns too many row sums.
    rows = square[:, :N_BINS]
    _rejects_before_any_callback(
        _ForwardBackwardOnly(jnp.asarray(rows / rows.sum(1, keepdims=True))),
        "transition_operator gives row sums of the wrong shape",
        f"Expected: shape ({N_BINS},)",
        f"Got: shape ({N_BINS + 1},)",
    )


def test_checkpointed_accepts_substochastic_operator_rows():
    """Rows below 1, such as bins nothing can leave, are valid as for a matrix."""
    matrix = np.array(_matrix())
    matrix[1] = 0
    _checkpointed(
        _StationaryTransition(jnp.asarray(matrix)),
        row_slice_aware(lambda **kw: np.zeros((4, N_BINS))[kw["row_slice"]]),
    )


def _covariate_block_sums_above_one():
    """Continuous transition whose bin 2 sums to 1.2 over discrete state 0's bins.

    Its full rows sum to at most 1.2 * 0.5 + 0.8 * 0.5 = 1 under the
    discrete rows of 0.5 used below, but a discrete row (1, 0) would make bin
    2's joint row sum to 1.2, so the dense covariate drivers reject it.
    """
    continuous = np.full((N_BINS, N_BINS), 0.25, dtype=np.float32)
    continuous[2] = [0.6, 0.6, 0.4, 0.4]
    return continuous


@pytest.mark.parametrize("name", ["covariate", "block", "fused"])
def test_checkpointed_rejects_covariate_operator_block_rows_above_one(name):
    """With per-row weights, each bin's row is checked per target state."""
    continuous = _covariate_block_sums_above_one()
    if name == "covariate":
        operator = _CovariateTransition.create(jnp.asarray(continuous), STATE_IND, 2)
    else:
        blocks = tuple(
            tuple(
                DenseBlock(
                    continuous[np.ix_(np.equal(STATE_IND, i), np.equal(STATE_IND, j))]
                )
                for j in range(2)
            )
            for i in range(2)
        )
        operator = BlockTransitionOperator(blocks, (2, 2))
        operator = operator.fused() if name == "fused" else operator
    _rejects_before_any_callback(
        operator,
        "transition_operator rows must be probability distributions",
        "bin 2 to discrete state 0 sums to 1.2",
        discrete_transition_matrix=np.full((4, 2, 2), 0.5),
    )


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("statistics", [None, "counts"])
def test_checkpointed_accepts_valid_operators(name, statistics):
    """The package's operators pass the check on both transition paths."""
    operator = OPERATORS[name]()
    if name in ("block", "fused"):
        operator = _block_operator().bind_discrete(np.ones((1, 1)))
        operator = operator.fused() if name == "fused" else operator
        state_ind = np.zeros(N_BINS, dtype=int)
    else:
        state_ind = STATE_IND
    weights = (
        {}
        if name == "stationary"
        else {
            "discrete_transition_matrix": np.full(
                (4, state_ind.max() + 1, state_ind.max() + 1), 1 / (state_ind.max() + 1)
            )
        }
    )
    checkpointed_forward_backward(
        np.arange(5) * 0.002,
        np.full(N_BINS, 1 / N_BINS, dtype=np.float32),
        row_slice_aware(lambda **kw: np.zeros((4, N_BINS))[kw["row_slice"]]),
        transition_operator=operator,
        state_ind=state_ind,
        transition_statistics=statistics,
        **weights,
    )


def test_checkpointed_reports_an_operator_that_cannot_be_applied():
    """An operator that raises on the all-ones product is reported as such.

    A covariate operator passed without ``discrete_transition_matrix`` gets
    no per-row weights; its own error is kept, not reported as a shape
    mismatch.
    """
    message = _rejects_before_any_callback(
        _CovariateTransition.create(_matrix(), STATE_IND, 2),
        "transition_operator could not be applied to an all-ones vector",
        "needs each row's discrete weights",
    )
    assert "wrong shape" not in message, message


@jax.tree_util.register_pytree_node_class
class _MismatchedProducts(TransitionOperator):
    """``forward`` with one matrix, ``backward`` and pairs with another.

    Both are row-stochastic, so the row sums ``backward(1)`` are all 1.
    """

    def __init__(self, forward_matrix, backward_matrix, pair_scale=1.0):
        self.forward_matrix, self.backward_matrix = forward_matrix, backward_matrix
        self.pair_scale = pair_scale

    def forward(self, p, weight=None):
        return p @ self.forward_matrix

    def backward(self, v, weight=None):
        return self.backward_matrix @ v

    def pair_products(self, filtered, ratio, weight=None):
        states = jax.nn.one_hot(STATE_IND, 2, dtype=filtered.dtype)
        terms = filtered[:, None] * self.backward_matrix * ratio[None, :]
        return self.pair_scale * states.T @ terms @ states

    def tree_flatten(self):
        return (self.forward_matrix, self.backward_matrix), self.pair_scale

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, aux)


def test_checkpointed_rejects_operator_whose_products_disagree():
    """Forward and backward with different matrices predict and smooth with
    different transitions; equal row sums do not show it."""
    matrix = _matrix()
    _rejects_before_any_callback(
        _MismatchedProducts(jnp.roll(matrix, 1, axis=0), matrix),
        "transition_operator's products do not apply one transition",
        "forward(u) @ v",
    )


def test_checkpointed_rejects_pair_products_that_disagree_with_backward():
    matrix = _matrix()
    _rejects_before_any_callback(
        _MismatchedProducts(matrix, matrix, pair_scale=2.0),
        "transition_operator's products do not apply one transition",
        "pair_products(u, v).sum()",
        transition_statistics="counts",
    )


def test_checkpointed_rejects_pair_products_of_another_shape():
    """Statistics are accumulated per discrete state pair of ``state_ind``."""
    operator = _StationaryTransition(_matrix(), jnp.asarray(STATE_IND), 3)
    _rejects_before_any_callback(
        operator,
        "transition_operator.pair_products has the wrong shape",
        "Expected: shape (2, 2)",
        "Got: shape (3, 3)",
        transition_statistics="counts",
    )
    _rejects_before_any_callback(
        _StationaryTransition(_matrix()),
        "transition_operator could not be applied to probe vectors",
        "Expected: forward, backward and pair_products of two (4,) vectors",
        "needs state_ind and n_states",
        transition_statistics="counts",
    )


def _default_detector_operator(shape, local_position_std):
    """Structured operator of the default non-local detector's transitions."""
    axes = [np.arange(n, dtype=float) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    is_track_interior = np.ones(shape, dtype=bool)
    is_track_interior.ravel()[[0, 3, -2]] = False
    env = set_place_grid(
        Environment(),
        place_bin_centers=np.stack([x.ravel() for x in mesh], axis=1),
        centers_shape=shape,
        is_track_interior=is_track_interior,
    )
    defaults = _ModelDefaults.non_local_defaults()
    count = len(env.place_bin_centers_)
    sizes = (1 if local_position_std is None else count, 1, count, count)
    operator = build_transition_operator(
        defaults["continuous_transition_types"](),
        [env],
        sizes,
        defaults["observation_models"](),
        local_position_std,
    )
    interior = np.concatenate(
        [np.ones(1, dtype=bool) if n == 1 else is_track_interior.ravel() for n in sizes]
    )
    return operator, interior


@pytest.mark.parametrize("local_position_std", [None, 2.0])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_detector_operators_pass_the_product_checks(local_position_std, fused, dtype):
    """The default detector's structured operators pass every check.

    Per-row weights (as for covariate transitions) and bound weights on the
    interior bins (as for stationary prediction), with the pair terms of the
    transition statistics.
    """
    with precision_mode(dtype == np.float64):
        operator, interior = _default_detector_operator((5, 7), local_position_std)
        weights = np.full((4, 4), 0.25)
        per_row = operator.fused() if fused else operator
        _check_operator_rows(
            jax.device_put(per_row), len(interior), dtype, 4, pair_states=4
        )
        bound = operator.restricted(interior).bind_discrete(weights)
        bound = bound.fused() if fused else bound
        _check_operator_rows(
            jax.device_put(bound), int(interior.sum()), dtype, pair_states=4
        )


@pytest.mark.parametrize("fused", [False, True])
def test_gaussian_grid_operator_passes_the_product_checks(fused):
    """A 2-D random walk on a grid with holes, the structured Gaussian block."""
    shape = (30, 40)
    axes = [np.arange(n, dtype=float) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    is_track_interior = np.ones(shape, dtype=bool)
    is_track_interior[10:14, 5:30] = False
    env = set_place_grid(
        Environment(),
        place_bin_centers=np.stack([x.ravel() for x in mesh], axis=1),
        centers_shape=shape,
        is_track_interior=is_track_interior,
    )
    n_bins = int(np.prod(shape))
    operator = build_transition_operator([[RandomWalk()]], [env], (n_bins,))
    operator = operator.bind_discrete(np.ones((1, 1)))
    operator = operator.fused() if fused else operator
    _check_operator_rows(jax.device_put(operator), n_bins, np.float32, pair_states=1)
