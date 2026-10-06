"""Spike reductions preserve ownership and exact repeated CUDA observations."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.common import SpikeSelection, sum_spikes_into_rows
from non_local_detector.tests.likelihoods.conftest import FLOAT32_ROUNDING

pytestmark = pytest.mark.unit


@contextmanager
def precision_mode(dtype):
    previous = jax.config.x64_enabled
    jax.config.update(
        "jax_enable_x64", np.dtype(dtype).itemsize > 4 and dtype != np.complex64
    )
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def selection(ids, n_rows):
    ids = np.asarray(ids)
    return SpikeSelection(
        slice(0, len(ids)), ids, bool(np.all(ids[1:] >= ids[:-1])), len(ids), n_rows
    )


def ordered_reference(values, ids, n_rows):
    values = np.asarray(values)
    output = np.zeros((n_rows, *values.shape[1:]), dtype=values.dtype)
    for row, value in zip(ids, values, strict=True):
        if 0 <= row < n_rows:
            output[row] += value
    return output


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ],
)
@pytest.mark.parametrize("tail", [(), (5,), (2, 3)])
def test_fixed_input_order_and_invalid_row_drop(dtype, tail):
    ids = np.array([2, -1, 2, 0, 99, 1, 2])
    values = (
        np.arange(7 * int(np.prod(tail) if tail else 1)).reshape(7, *tail) % 13 - 6
    ).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += (values * 0.25j).astype(dtype)
    with precision_mode(dtype):
        actual = sum_spikes_into_rows(jnp.asarray(values), selection(ids, 4))
        assert actual.dtype == jnp.asarray(values).dtype
        np.testing.assert_array_equal(actual, ordered_reference(values, ids, 4))


@pytest.mark.parametrize(
    "n_values,n_rows,tail", [(0, 0, (3,)), (0, 4, (3,)), (7, 0, (3,)), (7, 4, (0,))]
)
def test_empty_shapes_keep_the_exact_requested_rows(n_values, n_rows, tail):
    values = jnp.zeros((n_values, *tail), dtype=jnp.float32)
    actual = sum_spikes_into_rows(values, selection(np.zeros(n_values, int), n_rows))
    assert actual.shape == (n_rows, *tail)
    np.testing.assert_array_equal(actual, np.zeros(actual.shape, np.float32))


def test_boolean_values_retain_the_unsupported_reduction_error():
    with pytest.raises(TypeError):
        sum_spikes_into_rows(jnp.array([True, False]), selection([0, 0], 1))


def test_cpu_invalid_id_normalization_preserves_the_verified_sorted_hint(monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU segment primitive uses the verified sorted-index hint")
    original = jax.ops.segment_sum

    def checked(values, ids, **kwargs):
        assert kwargs["indices_are_sorted"]
        assert np.all(np.diff(np.asarray(ids)) >= 0)
        return original(values, ids, **kwargs)

    monkeypatch.setattr(jax.ops, "segment_sum", checked)
    ids = np.array([-100, -1, 0, 1, 99, 101])
    actual = sum_spikes_into_rows(jnp.ones(len(ids)), selection(ids, 3))
    np.testing.assert_array_equal(actual, [1.0, 1.0, 0.0])


def test_nan_and_infinity_stay_in_their_owned_rows():
    ids = np.array([0, 0, 2, 3, 3, -1, 99])
    values = np.array(
        [
            [1, 2],
            [np.nan, 3],
            [np.inf, 4],
            [np.inf, 5],
            [-np.inf, 6],
            [np.nan, np.inf],
            [np.inf, np.nan],
        ],
        np.float32,
    )
    with np.errstate(invalid="ignore"):
        expected = ordered_reference(values, ids, 5)
    actual = sum_spikes_into_rows(jnp.asarray(values), selection(ids, 5))
    np.testing.assert_array_equal(actual, expected)


def test_values_gradient_gathers_owned_rows_and_drops_invalid_entries():
    ids = np.array([2, -1, 2, 0, 9, 1, 2])
    values = jnp.arange(21, dtype=jnp.float32).reshape(7, 3)
    weights = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    selected = selection(ids, 4)
    gradient = jax.grad(lambda v: (sum_spikes_into_rows(v, selected) * weights).sum())(
        values
    )
    expected = np.zeros((7, 3), np.float32)
    valid = (ids >= 0) & (ids < 4)
    expected[valid] = np.asarray(weights)[ids[valid]]
    np.testing.assert_array_equal(gradient, expected)


def test_repeated_unequal_float_rows_are_bitwise_identical():
    rng = np.random.default_rng(7612)
    ids = np.repeat(np.arange(10), 7)
    order = rng.permutation(len(ids))
    ids = ids[order]
    values = rng.normal(-13, 7, (len(ids), 254)).astype(np.float32)[order]
    selected = selection(ids, 12)
    values = jnp.asarray(values)
    reference = np.asarray(sum_spikes_into_rows(values, selected)).copy()
    np.testing.assert_array_equal(reference, ordered_reference(values, ids, 12))
    for _ in range(50):
        actual = np.asarray(sum_spikes_into_rows(values, selected))
        np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))


def test_large_component_mass_matches_float64_without_sequential_drift():
    from non_local_detector.likelihoods.diffusion import _component_mass_sum

    n_nodes = 66_250
    labels = np.repeat([0, 1], [33_124, n_nodes - 33_124]).astype(np.int32)
    fields = np.column_stack(
        (np.full(n_nodes, 1 / n_nodes, np.float32), np.full(n_nodes, 0.3, np.float32))
    )
    expected = np.stack(
        [fields[labels == c].sum(axis=0, dtype=np.float64) for c in range(2)]
    )
    actual = _component_mass_sum(jnp.asarray(fields), jnp.asarray(labels), 2)
    np.testing.assert_allclose(actual, expected, **FLOAT32_ROUNDING)
    reference = np.asarray(actual).copy()
    for _ in range(20):
        np.testing.assert_array_equal(
            _component_mass_sum(jnp.asarray(fields), jnp.asarray(labels), 2), reference
        )


def test_component_mass_drops_invalid_labels_without_poisoning_other_components():
    from non_local_detector.likelihoods.diffusion import _component_mass_sum

    fields = jnp.array([[1.0, 2.0], [jnp.inf, 3.0], [jnp.nan, jnp.inf]])
    actual = _component_mass_sum(fields, jnp.array([0, 1, -1]), 2)
    np.testing.assert_array_equal(actual, [[1.0, 2.0], [np.inf, 3.0]])


def test_gmm_tile_updates_keep_initial_term_and_each_spike_rounding_order():
    from non_local_detector.likelihoods.clusterless_gmm import (
        _accumulate_log_likelihood_block,
    )

    initial = np.zeros((3, 4), np.float32)
    initial[1, 1:3] = 2**24
    contribution = np.array([[-(2**24), -(2**24)], [1.0, 2.0], [2.0, 3.0]], np.float32)
    ids = np.array([1, 1, 1], np.int32)
    columns = np.arange(1, 3, dtype=np.int32)
    expected = initial.copy()
    for value in contribution:
        expected[1, columns] += value
    actual = _accumulate_log_likelihood_block(
        jnp.asarray(initial),
        jnp.asarray(contribution),
        jnp.asarray(ids),
        jnp.asarray(columns),
        jnp.array(0.0),
        jnp.zeros(2),
    )
    np.testing.assert_array_equal(actual, expected)
