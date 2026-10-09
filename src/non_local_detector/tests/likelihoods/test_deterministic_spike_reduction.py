"""Spike reductions preserve ownership and are bitwise repeatable on every backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import non_local_detector.likelihoods.common as common
from non_local_detector.likelihoods.common import (
    SpikeSelection,
    deterministic_segment_sum,
    sum_spikes_into_rows,
)
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import FLOAT32_ROUNDING

pytestmark = pytest.mark.unit


def dtype_precision(dtype):
    """``precision_mode`` with x64 enabled for 64-bit dtypes."""
    return precision_mode(np.dtype(dtype).itemsize > 4 and dtype != np.complex64)


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


def assert_within_pairwise_bound(actual, values, ids, n_rows):
    """``actual`` is within the pairwise-summation bound of the float64 row sums.

    The bound is ``(ceil(log2 n) + 2)`` float32 roundings of each row's summed
    magnitude, which any fixed summation tree over ``n`` values satisfies.
    """
    values = np.asarray(values, np.float64)
    exact = np.zeros((n_rows, *values.shape[1:]))
    magnitude = np.zeros((n_rows, *values.shape[1:]))
    for row, value in zip(ids, values, strict=True):
        if 0 <= row < n_rows:
            exact[row] += value
            magnitude[row] += np.abs(value)
    n_values = len(ids)
    bound = (np.ceil(np.log2(n_values)) + 2) * np.finfo(np.float32).eps * magnitude
    assert np.all(np.abs(np.asarray(actual, np.float64) - exact) <= bound)


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
def test_fixed_input_order_and_invalid_row_drop(dtype, tail, reduction_path):
    ids = np.array([2, -1, 2, 0, 99, 1, 2])
    values = (
        np.arange(7 * int(np.prod(tail) if tail else 1)).reshape(7, *tail) % 13 - 6
    ).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += (values * 0.25j).astype(dtype)
    with dtype_precision(dtype):
        actual = sum_spikes_into_rows(jnp.asarray(values), selection(ids, 4))
        assert actual.dtype == jnp.asarray(values).dtype
        np.testing.assert_array_equal(actual, ordered_reference(values, ids, 4))


@pytest.mark.parametrize(
    "n_values,n_rows,tail", [(0, 0, (3,)), (0, 4, (3,)), (7, 0, (3,)), (7, 4, (0,))]
)
def test_empty_shapes_keep_the_exact_requested_rows(
    n_values, n_rows, tail, reduction_path
):
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


def test_nan_and_infinity_stay_in_their_owned_rows(reduction_path):
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


def test_values_gradient_gathers_owned_rows_and_drops_invalid_entries(
    reduction_path,
):
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


def test_repeated_unequal_float_rows_are_bitwise_identical(reduction_path):
    rng = np.random.default_rng(7612)
    ids = np.repeat(np.arange(10), 7)
    order = rng.permutation(len(ids))
    ids = ids[order]
    values = rng.normal(-13, 7, (len(ids), 254)).astype(np.float32)[order]
    selected = selection(ids, 12)
    values = jnp.asarray(values)
    reference = np.asarray(sum_spikes_into_rows(values, selected)).copy()
    if common._reduces_sequentially(values):
        # XLA:CPU scatter adds rows in input order.
        np.testing.assert_array_equal(reference, ordered_reference(values, ids, 12))
    else:
        # The segmented scan sums in a fixed tree, not input order.
        assert_within_pairwise_bound(reference, values, ids, 12)
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


def test_gmm_tile_updates_keep_initial_term_without_cancellation_loss(
    reduction_path,
):
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


@pytest.mark.parametrize("is_sorted", [True, False])
@pytest.mark.parametrize("tail", [(), (5,), (2, 3)])
def test_deterministic_segment_sum_matches_float64_reference(is_sorted, tail):
    rng = np.random.default_rng(7613)
    n_values, n_segments = 300, 17
    ids = rng.integers(-2, n_segments + 2, n_values)
    if is_sorted:
        ids = np.sort(ids)
    values = rng.normal(size=(n_values, *tail)).astype(np.float32)
    actual = deterministic_segment_sum(
        jnp.asarray(values), jnp.asarray(ids, jnp.int32), n_segments, is_sorted
    )
    assert_within_pairwise_bound(actual, values, ids, n_segments)


def test_deterministic_segment_sum_is_bitwise_repeatable():
    rng = np.random.default_rng(7614)
    values = jnp.asarray(rng.normal(-13, 7, (5000, 254)).astype(np.float32))
    ids = jnp.asarray(np.sort(rng.integers(0, 256, 5000)).astype(np.int32))
    reference = np.asarray(deterministic_segment_sum(values, ids, 256, True))
    for _ in range(20):
        actual = np.asarray(deterministic_segment_sum(values, ids, 256, True))
        np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))


def clusterless_log_likelihood(algorithm):
    """A small two-electrode detector; returns a function evaluating rows 2-12."""
    from non_local_detector import Environment, NonLocalClusterlessDetector

    rng = np.random.default_rng(7412)
    position_time = np.linspace(0, 2, 201)
    position = (4 + 3 * np.sin(position_time * 3))[:, None]
    spikes = [np.sort(rng.uniform(0, 2, 72)) for _ in range(2)]
    marks = [rng.normal(size=(72, 2)) for _ in range(2)]
    model = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=0.5, position_range=((0, 8),)),
        infer_track_interior=False,
        clusterless_algorithm=algorithm,
    )
    with pytest.warns(UserWarning, match="1D"):
        model.fit(position_time, position, spikes, marks)
    decode_spikes = [np.sort(rng.uniform(0, 1.6, 51)) for _ in range(2)]
    decode_marks = [rng.normal(size=(51, 2)) for _ in range(2)]
    # Three same-bin spikes with unequal marks on the first electrode.
    decode_spikes[0][:3] = [0.41, 0.42, 0.43]
    edges = np.linspace(0, 1.6, 17)

    def evaluate():
        return np.asarray(
            model.compute_log_likelihood(
                position_time,
                position,
                decode_spikes,
                decode_marks,
                row_slice=slice(2, 12),
                time_edges=edges,
            )
        )

    return evaluate


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm",
    [
        "clusterless_kde",
        "clusterless_kde_log",
        "clusterless_gmm",
        "clusterless_diffusion",
    ],
)
def test_clusterless_likelihoods_are_bitwise_repeatable(algorithm, monkeypatch):
    evaluate = clusterless_log_likelihood(algorithm)
    results = {}
    for path in ("sequential", "segmented_scan"):
        if path == "segmented_scan":
            monkeypatch.setattr(common, "_reduces_sequentially", lambda values: False)
            jax.clear_caches()
        repeats = [evaluate() for _ in range(10)]
        for repeat in repeats[1:]:
            np.testing.assert_array_equal(
                repeat.view(np.uint8), repeats[0].view(np.uint8)
            )
        results[path] = repeats[0]
    jax.clear_caches()
    np.testing.assert_allclose(
        results["segmented_scan"], results["sequential"], **FLOAT32_ROUNDING
    )


@pytest.mark.parametrize("id_dtype", [np.uint8, np.uint32, np.int8, np.int16])
def test_deterministic_segment_sum_handles_unsigned_and_narrow_ids(id_dtype):
    ids = np.array([0, 1, 1, 5, 22, 23, 100, 120], dtype=id_dtype)
    values = np.arange(1, 9, dtype=np.float32)[:, None]
    n_segments = 300
    actual = deterministic_segment_sum(
        jnp.asarray(values), jnp.asarray(ids), n_segments, False
    )
    np.testing.assert_array_equal(actual, ordered_reference(values, ids, n_segments))


def test_concrete_off_cpu_calls_share_bucketed_executables(monkeypatch):
    monkeypatch.setattr(common, "_reduces_sequentially", lambda values: False)
    jax.clear_caches()
    rng = np.random.default_rng(7615)
    before = deterministic_segment_sum._cache_size()
    for n_spikes in range(1, 65):
        ids = np.sort(rng.integers(0, 12, n_spikes))
        values = rng.normal(size=(n_spikes, 3)).astype(np.float32)
        actual = common.deterministic_row_sum(
            jnp.asarray(values), ids, 12, indices_are_sorted=True
        )
        np.testing.assert_allclose(
            actual, ordered_reference(values, ids, 12), rtol=1e-6, atol=1e-6
        )
    # Spike counts 1..64 pad to the seven powers of two 1, 2, 4, ..., 64.
    assert deterministic_segment_sum._cache_size() - before <= 7
    jax.clear_caches()


@pytest.mark.parametrize("id_dtype", [np.uint8, np.int8])
@pytest.mark.parametrize("is_sorted", [True, False])
def test_bucketed_row_sum_keeps_narrow_ids_valid(id_dtype, is_sorted, reduction_path):
    # Seven ids pad to eight on the scan path; the 300-row padding id must not
    # wrap to a valid narrow id.
    ids = np.array([0, 1, 1, 5, 22, 100, 120], dtype=id_dtype)
    if not is_sorted:
        ids = ids[::-1].copy()
    values = np.arange(1, 8, dtype=np.float32)[:, None]
    actual = common.deterministic_row_sum(
        jnp.asarray(values), ids, 300, indices_are_sorted=is_sorted
    )
    np.testing.assert_array_equal(actual, ordered_reference(values, ids, 300))


def test_row_sum_rejects_mismatched_values_and_ids():
    with pytest.raises(ValueError, match="one row per row id"):
        common.deterministic_row_sum(jnp.ones((5, 2)), np.zeros(4, np.int32), 3)
