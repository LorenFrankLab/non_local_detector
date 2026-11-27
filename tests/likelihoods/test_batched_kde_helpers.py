"""Contracts of the helpers the batched clusterless KDE passes build on."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.likelihoods.common import (
    _log_kernel_matrix,
    compute_log_mark_kernel_gemm,
    deterministic_row_accumulate,
    padded_block_count,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("columns", [None, 3], ids=["rows", "row-tiles"])
def test_row_accumulate_adds_each_spike_to_its_row_and_drops_n_rows(
    columns, reduction_path
):
    """Spikes add to their rows of the running output; id ``n_rows`` is dropped."""
    rng = np.random.default_rng(0)
    n_rows = 5
    shape = (n_rows,) if columns is None else (n_rows, columns)
    output = rng.normal(size=shape).astype(np.float32)
    row_ids = np.array([0, 0, 2, 4, 4, 4, n_rows, n_rows])
    values = rng.normal(size=(row_ids.size, *shape[1:])).astype(np.float32)
    expected = output.astype(float).copy()
    np.add.at(expected, row_ids[row_ids < n_rows], values[row_ids < n_rows])
    actual = deterministic_row_accumulate(
        jnp.asarray(output),
        jnp.asarray(values),
        jnp.asarray(row_ids),
        indices_are_sorted=True,
    )
    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_padded_block_count_is_the_next_power_of_four():
    counts = [1, 2, 4, 5, 16, 17, 64, 65, 256, 257]
    assert [padded_block_count(n) for n in counts] == [
        1,
        4,
        4,
        16,
        16,
        64,
        64,
        256,
        256,
        1024,
    ]
    assert padded_block_count(0) == 1


def test_masked_kernels_equal_the_kernels_of_the_unpadded_features():
    """Zero features padded onto both operands, masked out, change nothing:
    neither the squared distance nor the normalizer counts them."""
    rng = np.random.default_rng(1)
    queries = rng.normal(0, 30, (6, 2)).astype(np.float32)
    samples = rng.normal(0, 30, (9, 2)).astype(np.float32)
    stds = np.array([20.0, 35.0], np.float32)
    pad = ((0, 0), (0, 3))
    padded_stds = np.concatenate([stds, np.ones(3, np.float32)])
    mask = np.arange(5) < 2
    np.testing.assert_array_equal(
        compute_log_mark_kernel_gemm(
            jnp.asarray(np.pad(queries, pad)),
            jnp.asarray(np.pad(samples, pad)),
            jnp.asarray(padded_stds),
            jnp.asarray(mask),
        ),
        compute_log_mark_kernel_gemm(
            jnp.asarray(queries), jnp.asarray(samples), jnp.asarray(stds)
        ),
    )
    np.testing.assert_array_equal(
        _log_kernel_matrix(
            jnp.asarray(np.pad(queries, pad)),
            jnp.asarray(np.pad(samples, pad)),
            jnp.asarray(padded_stds),
            jnp.asarray(mask),
        ),
        _log_kernel_matrix(
            jnp.asarray(queries), jnp.asarray(samples), jnp.asarray(stds)
        ),
    )
