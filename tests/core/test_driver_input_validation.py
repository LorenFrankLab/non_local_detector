"""The public HMM drivers enforce one input contract at entry.

The dense drivers (``chunked_filter_smoother``, ``most_likely_sequence`` and
their covariate-dependent variants) and ``checkpointed_forward_backward`` share
one validator: the same shapes, dtype rule and probability constraints, raised
as ``ValidationError`` before any likelihood is evaluated. The jit-composable
passes (``filter``, ``smoother``, ``viterbi`` and their covariate-dependent
variants) apply its shape and probability checks to concrete inputs.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import core
from non_local_detector.checkpointed_inference import checkpointed_forward_backward
from non_local_detector.core import (
    PROBABILITY_SUM_TOLERANCE,
    chunked_filter_smoother,
    chunked_filter_smoother_covariate_dependent,
    most_likely_sequence,
    most_likely_sequence_covariate_dependent,
    row_slice_aware,
)
from non_local_detector.exceptions import ValidationError
from tests.conftest import precision_mode

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


DENSE_DRIVERS = {
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
CHECKPOINTED_DRIVERS = {
    "checkpointed": ("transition_matrix",),
    "checkpointed_covariate": (
        "discrete_transition_matrix",
        "continuous_transition_matrix",
    ),
}
ALL_DRIVERS = [*DENSE_DRIVERS, *CHECKPOINTED_DRIVERS]
COVARIATE_DRIVERS = [name for name in ALL_DRIVERS if name.endswith("covariate")]
STATIONARY_DRIVERS = [name for name in ALL_DRIVERS if name not in COVARIATE_DRIVERS]


def _call(name, inputs, **overrides):
    if name in CHECKPOINTED_DRIVERS:
        return _call_checkpointed(name, inputs, **overrides)
    driver, keys = DENSE_DRIVERS[name]
    kwargs = {key: inputs[key] for key in keys}
    kwargs.update(
        log_likelihood_func=None,
        log_likelihood_args=(),
        log_likelihoods=inputs["log_likelihoods"],
        is_missing=inputs["is_missing"],
    )
    kwargs.update(overrides)
    return driver(**kwargs)


def _call_checkpointed(name, inputs, **overrides):
    calls = []

    def likelihood(time_edges, *, row_slice, is_missing):
        calls.append(row_slice)
        return inputs["log_likelihoods"][row_slice]

    kwargs = {key: inputs[key] for key in CHECKPOINTED_DRIVERS[name]}
    kwargs.update(
        state_ind=inputs["state_ind"],
        is_missing=inputs["is_missing"],
        chunk_size=4,
    )
    kwargs.update(overrides)
    try:
        return checkpointed_forward_backward(
            np.arange(N_TIME + 1) * 0.002,
            kwargs.pop("initial_distribution", inputs["initial_distribution"]),
            likelihood,
            **kwargs,
        )
    finally:
        # Validation happens before the likelihood is ever evaluated.
        _call_checkpointed.last_calls = calls


@pytest.mark.parametrize("name", ALL_DRIVERS)
def test_valid_inputs_pass(name):
    _call(name, _inputs())


def _raises(name, inputs, *pieces, **overrides):
    """Assert the driver raises a ValidationError whose message has ``pieces``."""
    with pytest.raises(ValidationError) as caught:
        _call(name, inputs, **overrides)
    message = str(caught.value)
    for piece in pieces:
        assert piece in message, message
    assert "Hint:" in message, message
    if name in CHECKPOINTED_DRIVERS:
        assert _call_checkpointed.last_calls == []
    return message


# (argument, bad value, the message's first line, its expected and got parts)
BAD_SHAPES = {
    "short log_likelihoods": (
        "log_likelihoods",
        np.zeros((N_TIME - 1, N_BINS)),
        "log_likelihoods has the wrong shape",
        f"shape ({N_TIME}, {N_BINS})",
        f"shape ({N_TIME - 1}, {N_BINS})",
    ),
    "wrong-bin log_likelihoods": (
        "log_likelihoods",
        np.zeros((N_TIME, N_BINS + 1)),
        "log_likelihoods has the wrong shape",
        f"shape ({N_TIME}, {N_BINS})",
        f"shape ({N_TIME}, {N_BINS + 1})",
    ),
    "short is_missing": (
        "is_missing",
        np.zeros(N_TIME - 1, dtype=bool),
        "is_missing has the wrong shape",
        f"shape ({N_TIME},)",
        f"shape ({N_TIME - 1},)",
    ),
    "2-D time": (
        "time",
        np.zeros((N_TIME, 1)),
        "time must be a nonempty vector",
        "shape (n_time,) with n_time >= 1",
        f"shape ({N_TIME}, 1)",
    ),
    "empty time": (
        "time",
        np.zeros(0),
        "time must be a nonempty vector",
        "shape (n_time,) with n_time >= 1",
        "shape (0,)",
    ),
}


@pytest.mark.parametrize("name", DENSE_DRIVERS)
@pytest.mark.parametrize("case", BAD_SHAPES)
def test_mismatched_shapes_raise(name, case):
    key, value, first_line, expected, got = BAD_SHAPES[case]
    message = _raises(
        name,
        _inputs(),
        f"Expected: {expected}",
        f"Got: {got}",
        **{key: value},
    )
    assert message.startswith(first_line + "\n"), message


@pytest.mark.parametrize("name", CHECKPOINTED_DRIVERS)
def test_checkpointed_short_is_missing_raises(name):
    _raises(
        name,
        _inputs(),
        "is_missing has the wrong shape",
        f"Expected: shape ({N_TIME},)",
        is_missing=np.zeros(N_TIME - 1, dtype=bool),
    )


@pytest.mark.parametrize("name", ALL_DRIVERS)
@pytest.mark.parametrize(
    ("dtype", "got"),
    [(jnp.int32, "int32"), (jnp.float16, "float16"), (jnp.bfloat16, "bfloat16")],
)
def test_unsupported_dtypes_raise(name, dtype, got):
    """Floating dtypes other than float32/float64 are rejected, not downcast."""
    _raises(
        name,
        _inputs(),
        "dtype must be float32 or float64",
        f"Got: {got}",
        dtype=dtype,
    )


@pytest.mark.parametrize("name", ALL_DRIVERS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_supported_dtypes_pass(name, dtype):
    with precision_mode(dtype == jnp.float64):
        _call(name, _inputs(), dtype=dtype)


@pytest.mark.parametrize("name", ALL_DRIVERS)
def test_float64_without_x64_raises(name):
    """Every driver refuses float64 with x64 off instead of computing in float32."""
    with precision_mode(False):
        _raises(
            name,
            _inputs(),
            "float64 requires JAX's 64-bit mode",
            "JAX_ENABLE_X64",
            dtype=jnp.float64,
        )


@pytest.mark.parametrize("name", ALL_DRIVERS)
@pytest.mark.parametrize(
    ("initial", "got"),
    [
        # Each entry 5: the unnormalized distribution of the original report,
        # which the drivers accepted and turned into a +log(15) evidence gain.
        (np.full(N_BINS, 5.0), "sums to 15"),
        (np.array([0.5, 0.6, -0.1]), "minimum entry -0.1"),
        (np.array([0.5, np.nan, 0.5]), "nan"),
        (np.array([0.5, 0.2, 0.2]), "sums to 0.9"),
    ],
)
def test_initial_distribution_must_be_a_distribution(name, initial, got):
    _raises(
        name,
        _inputs(),
        "initial_distribution must be a probability distribution",
        got,
        initial_distribution=initial,
    )


@pytest.mark.parametrize("name", STATIONARY_DRIVERS)
@pytest.mark.parametrize(
    ("transition", "got"),
    [
        # Rows summing to 7: the original report's marginal of +10.49.
        (np.full((N_BINS, N_BINS), 7 / N_BINS), "row 0 sums to 7"),
        (
            np.array([[1.1, -0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "minimum entry -0.1",
        ),
        (
            np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]),
            "nan",
        ),
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
            "1 of 3 rows",
        ),
    ],
)
def test_transition_rows_must_not_create_mass(name, transition, got):
    _raises(
        name,
        _inputs(),
        "transition_matrix rows must be probability distributions",
        got,
        transition_matrix=transition,
    )


@pytest.mark.parametrize("name", STATIONARY_DRIVERS)
def test_substochastic_transition_rows_pass(name):
    """Rows below 1 are bins with no (or partial) outgoing transitions.

    The detectors build them: a continuous transition row is all zero for a
    bin nothing leaves (e.g. an ``EmpiricalMovement`` bin never visited), so
    the joint row keeps only its other discrete states' mass.
    """
    transition = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.25, 0.0]])
    _call(name, _inputs(), transition_matrix=transition)


@pytest.mark.parametrize("name", ALL_DRIVERS)
def test_float32_rounded_distributions_pass(name):
    """Float32 normalization error is far inside the tolerance."""
    rng = np.random.default_rng(1)
    inputs = _inputs()
    initial = rng.random(N_BINS, dtype=np.float32)
    inputs["initial_distribution"] = initial / initial.sum()
    discrete = rng.random((N_TIME, N_STATES, N_STATES), dtype=np.float32)
    inputs["discrete_transition_matrix"] = discrete / discrete.sum(-1, keepdims=True)
    transition = rng.random((N_BINS, N_BINS), dtype=np.float32)
    inputs["transition_matrix"] = transition / transition.sum(1, keepdims=True)
    _call(name, inputs)


@pytest.mark.parametrize("name", STATIONARY_DRIVERS)
def test_row_sum_tolerance_boundary(name):
    within = np.eye(N_BINS)
    within[0, 0] += 0.5 * PROBABILITY_SUM_TOLERANCE
    _call(name, _inputs(), transition_matrix=within)
    beyond = np.eye(N_BINS)
    beyond[0, 0] += 2 * PROBABILITY_SUM_TOLERANCE
    _raises(
        name,
        _inputs(),
        "transition_matrix rows must be probability distributions",
        transition_matrix=beyond,
    )


@pytest.mark.parametrize("name", COVARIATE_DRIVERS)
@pytest.mark.parametrize(
    ("scale", "shift", "got"),
    [(2.0, 0.0, "sums to 2"), (1.0, -0.6, "minimum entry -0.1")],
)
def test_discrete_transition_rows_must_sum_to_one(name, scale, shift, got):
    inputs = _inputs()
    discrete = inputs["discrete_transition_matrix"].copy()
    discrete[3, 1] = discrete[3, 1] * scale + np.array([shift, -shift])
    _raises(
        name,
        inputs,
        "discrete_transition_matrix rows must be probability distributions",
        got,
        "time 3",
        discrete_transition_matrix=discrete,
    )


@pytest.mark.parametrize("name", COVARIATE_DRIVERS)
def test_continuous_block_rows_must_not_create_mass(name):
    """Each bin's row within one target state's bins sums to at most 1.

    The joint transition ``C[i, j] * D[s_i, s_j]`` then has rows summing to at
    most 1 for every discrete transition ``D``.
    """
    continuous = np.full((N_BINS, N_BINS), 0.5)  # bins 0, 1 -> state 0 sum to 1
    continuous[2, :2] = 0.6  # bin 2 -> state 0 sums to 1.2
    _raises(
        name,
        _inputs(),
        "continuous_transition_matrix rows must be probability distributions",
        "bin 2 to discrete state 0 sums to 1.2",
        continuous_transition_matrix=continuous,
    )
    _raises(
        name,
        _inputs(),
        "continuous_transition_matrix rows must be probability distributions",
        "minimum entry -0.5",
        continuous_transition_matrix=np.where(np.eye(N_BINS) > 0, -0.5, 0.5),
    )


@pytest.mark.parametrize("name", DENSE_DRIVERS)
def test_mismatched_transition_raises(name):
    key = DENSE_DRIVERS[name][1][-1]
    _raises(
        name,
        _inputs(),
        f"{key} has the wrong shape",
        f"Expected: shape ({N_BINS}, {N_BINS})",
        f"Got: shape ({N_BINS + 1}, {N_BINS + 1})",
        **{key: np.eye(N_BINS + 1)},
    )


@pytest.mark.parametrize(
    "name", ["chunked_covariate", "viterbi_covariate", "checkpointed_covariate"]
)
def test_discrete_transitions_must_match_time_and_states(name):
    inputs = _inputs()
    _raises(
        name,
        inputs,
        "discrete_transition_matrix has the wrong shape",
        f"Expected: shape ({N_TIME}, {N_STATES}, {N_STATES})",
        discrete_transition_matrix=inputs["discrete_transition_matrix"][1:],
    )


@pytest.mark.parametrize("name", ["chunked_covariate", "viterbi_covariate"])
def test_state_ind_must_name_a_discrete_state(name):
    _raises(
        name,
        _inputs(),
        "state_ind refers to a discrete state the transitions do not have",
        "Got: discrete state 1 with discrete_transition_matrix of 1 states",
        discrete_transition_matrix=np.ones((N_TIME, 1, 1)),
    )


@pytest.mark.parametrize("name", ["chunked", "chunked_covariate", "checkpointed"])
def test_state_ind_must_be_integer_per_bin(name):
    _raises(
        name,
        _inputs(),
        "state_ind has the wrong shape",
        f"Expected: shape ({N_BINS},)",
        "Got: shape (2,)",
        state_ind=STATE_IND[:-1],
    )
    _raises(
        name,
        _inputs(),
        "state_ind must hold nonnegative integers",
        "Got: dtype float64",
        state_ind=STATE_IND.astype(float),
    )
    _raises(
        name,
        _inputs(),
        "state_ind must hold nonnegative integers",
        "Got: minimum -1",
        state_ind=np.array([-1, 0, 1]),
    )


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("name", ["chunked", "chunked_covariate"])
def test_callback_rows_must_match_the_request(name, cache):
    """A callback returning one row too few would silently truncate outputs."""
    inputs = _inputs()

    @row_slice_aware
    def short(time, is_missing=None, row_slice=None):
        rows = inputs["log_likelihoods"][row_slice or slice(None)]
        return rows[:-1]

    with pytest.raises(
        ValidationError, match=r"^log_likelihood_func output has the wrong shape"
    ):
        _call(
            name,
            inputs,
            log_likelihoods=None,
            log_likelihood_func=short,
            cache_log_likelihoods=cache,
            n_chunks=2,
        )


@pytest.mark.parametrize("name", ["chunked", "chunked_covariate"])
def test_too_many_chunks_raise(name):
    _raises(
        name,
        _inputs(),
        f"n_chunks ({N_TIME + 1}) cannot exceed n_time ({N_TIME})",
        n_chunks=N_TIME + 1,
    )


@pytest.mark.parametrize("name", CHECKPOINTED_DRIVERS)
def test_checkpointed_output_mode_raises_validation_error(name):
    _raises(name, _inputs(), "output_mode must be", output_mode="dense")


def test_checkpointed_missing_transition_raises():
    _raises(
        "checkpointed",
        _inputs(),
        "checkpointed_forward_backward needs a transition",
        "Got: no transition_matrix",
        transition_matrix=None,
    )


def test_error_lists_offending_row_count():
    inputs = _inputs()
    transition = np.full((N_BINS, N_BINS), 2 / N_BINS)
    message = _raises("chunked", inputs, transition_matrix=transition)
    assert re.search(r"3 of 3 rows sum above 1", message), message


@pytest.mark.parametrize("name", ["chunked", "chunked_covariate"])
def test_result_reports_no_degenerate_steps_as_an_empty_integer_array(name):
    result = _call(name, _inputs())
    assert result.degenerate_indices.shape == (0,)
    assert result.degenerate_indices.dtype.kind == "i"


def test_checkpointed_spatial_without_state_ind_names_the_discrete_matrix():
    """Without ``state_ind`` every bin is its own state; the error says so
    instead of blaming a ``state_ind`` the caller never passed."""
    inputs = _inputs()
    message = _raises(
        "checkpointed_covariate",
        inputs,
        "discrete_transition_matrix needs one discrete state per hidden bin "
        "without state_ind",
        f"Expected: shape ({N_TIME}, {N_BINS}, {N_BINS})",
        f"Got: shape ({N_TIME}, {N_STATES}, {N_STATES})",
        state_ind=None,
        output_mode="spatial",
    )
    assert "Pass state_ind" in message


def _filtered():
    return np.full((N_TIME, N_BINS), 1 / N_BINS)


# Each public pass with its inputs, in its positional order.
PUBLIC_PASSES = {
    "filter": (
        core.filter,
        ("initial_distribution", "transition_matrix", "log_likelihoods"),
    ),
    "smoother": (core.smoother, ("transition_matrix", "filtered_probs")),
    "viterbi": (
        core.viterbi,
        ("initial_distribution", "transition_matrix", "log_likelihoods"),
    ),
    "filter_covariate": (
        core.filter_covariate_dependent,
        (
            "initial_distribution",
            "discrete_transition_matrix",
            "continuous_transition_matrix",
            "state_ind",
            "log_likelihoods",
        ),
    ),
    "smoother_covariate": (
        core.smoother_covariate_dependent,
        (
            "discrete_transition_matrix",
            "continuous_transition_matrix",
            "state_ind",
            "filtered_probs",
        ),
    ),
    "viterbi_covariate": (
        core.viterbi_covariate_dependent,
        (
            "initial_distribution",
            "discrete_transition_matrix",
            "continuous_transition_matrix",
            "state_ind",
            "log_likelihoods",
        ),
    ),
}


def _public_args(name, **overrides):
    inputs = {**_inputs(), "filtered_probs": _filtered(), **overrides}
    return [
        jnp.asarray(inputs[key], jnp.int32 if key == "state_ind" else jnp.float32)
        for key in PUBLIC_PASSES[name][1]
    ]


def _public_raises(name, *pieces, **overrides):
    with pytest.raises(ValidationError) as caught:
        PUBLIC_PASSES[name][0](*_public_args(name, **overrides))
    message = str(caught.value)
    for piece in pieces:
        assert piece in message, message
    assert "Hint:" in message, message


@pytest.mark.parametrize("name", PUBLIC_PASSES)
def test_public_passes_accept_valid_inputs(name):
    PUBLIC_PASSES[name][0](*_public_args(name))


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in PUBLIC_PASSES
        if "initial_distribution" in PUBLIC_PASSES[name][1]
    ],
)
def test_public_passes_check_initial_distribution(name):
    """``filter(3 * p0, 1.5 * A, ll)`` returned a marginal of 17.82, not 9.02."""
    _public_raises(
        name,
        "initial_distribution must be a probability distribution",
        "sums to 3",
        initial_distribution=np.full(N_BINS, 1.0),
    )


@pytest.mark.parametrize("covariate", [False, True])
def test_public_filter_chains_through_a_lossy_prediction(covariate):
    """A filter's final prediction is a valid next initial distribution.

    With a bin nothing can leave (an all-zero continuous row), the
    prediction sums below 1. Passing it on to the next call must give the
    marginal of one call over the whole recording, as the chunked drivers
    do by carrying it across chunks.
    """
    inputs = _inputs()
    continuous = inputs["continuous_transition_matrix"].copy()
    continuous[2] = 0.0
    if covariate:
        pass_ = core.filter_covariate_dependent
        transitions = [
            inputs["discrete_transition_matrix"],
            continuous,
            inputs["state_ind"],
        ]
    else:
        pass_ = core.filter
        transitions = [continuous]
    ll = inputs["log_likelihoods"]
    split = N_TIME // 2

    def run(initial, rows):
        sliced = [t[rows] if t.ndim == 3 else t for t in map(np.asarray, transitions)]
        return pass_(initial, *sliced, ll[rows])

    (whole, _), _ = run(inputs["initial_distribution"], slice(None))
    (first, prediction), _ = run(inputs["initial_distribution"], slice(split))
    assert float(np.sum(prediction)) < 1 - 1e-3
    (second, _), _ = run(prediction, slice(split, None))
    np.testing.assert_allclose(float(first) + float(second), float(whole), rtol=1e-5)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in PUBLIC_PASSES
        if "initial_distribution" in PUBLIC_PASSES[name][1]
    ],
)
def test_public_passes_reject_initial_negative_entries(name):
    _public_raises(
        name,
        "initial_distribution must be a probability distribution",
        "minimum entry -0.1",
        initial_distribution=np.array([0.5, 0.6, -0.1]),
    )


@pytest.mark.parametrize("name", ["filter", "smoother", "viterbi"])
def test_public_stationary_passes_check_transition_rows(name):
    _public_raises(
        name,
        "transition_matrix rows must be probability distributions",
        "row 0 sums to 1.5",
        transition_matrix=np.full((N_BINS, N_BINS), 1.5 / N_BINS),
    )


@pytest.mark.parametrize(
    "name", ["filter_covariate", "smoother_covariate", "viterbi_covariate"]
)
def test_public_covariate_passes_check_transitions(name):
    discrete = np.full((N_TIME, N_STATES, N_STATES), 0.5)
    discrete[3, 1] = 1.0
    _public_raises(
        name,
        "discrete_transition_matrix rows must be probability distributions",
        "time 3",
        discrete_transition_matrix=discrete,
    )
    continuous = np.full((N_BINS, N_BINS), 0.5)
    continuous[2, :2] = 0.6
    _public_raises(
        name,
        "continuous_transition_matrix rows must be probability distributions",
        "bin 2 to discrete state 0 sums to 1.2",
        continuous_transition_matrix=continuous,
    )


@pytest.mark.parametrize("name", PUBLIC_PASSES)
def test_public_passes_check_transition_shape(name):
    key = PUBLIC_PASSES[name][1][1 if name != "smoother" else 0]
    if key == "discrete_transition_matrix":
        key = "continuous_transition_matrix"
    _public_raises(
        name,
        f"{key} has the wrong shape",
        f"Expected: shape ({N_BINS}, {N_BINS})",
        **{key: np.eye(N_BINS + 1)},
    )


@pytest.mark.parametrize("name", PUBLIC_PASSES)
def test_public_passes_skip_checks_under_jit(name):
    """Traced inputs are not checked, so the passes compose with ``jax.jit``.

    The jitted pass of valid inputs equals the checked call, and the jitted
    pass of invalid inputs runs without a host transfer of its traced
    arguments.
    """
    function = PUBLIC_PASSES[name][0]
    args = _public_args(name)
    jitted = jax.jit(function)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(function(*args)),
        jax.tree_util.tree_leaves(jitted(*args)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    bad = _public_args(
        name,
        initial_distribution=np.full(N_BINS, 1.0),
        transition_matrix=np.full((N_BINS, N_BINS), 1.5 / N_BINS),
        discrete_transition_matrix=np.ones((N_TIME, N_STATES, N_STATES)),
    )
    with jax.transfer_guard_device_to_host("disallow"):
        jax.block_until_ready(jitted(*bad))
