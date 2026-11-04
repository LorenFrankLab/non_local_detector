"""Deterministic segment and row sums of per-spike values."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np


def _segmented_inclusive_sum(starts: jax.Array, values: jax.Array) -> jax.Array:
    """Return the running sum of ``values`` that restarts at each ``starts`` position.

    A Hillis-Steele scan: step ``k`` adds the partial sum ``2**k`` rows back
    unless a segment start lies between. Its ``ceil(log2 n)`` steps run in a
    loop, so the compiled program holds one step however many run, and
    reverse-mode differentiation works. The summation tree has depth
    ``ceil(log2 n)`` and depends only on the segment layout.

    Parameters
    ----------
    starts : jax.Array, shape (n,), bool
        True at the first row of each segment; ``starts[0]`` must be True.
    values : jax.Array, shape (n, ...)

    Returns
    -------
    sums : jax.Array, shape (n, ...)
        Each row holds the sum of its segment's rows up to and including it.
    """
    n = values.shape[0]
    position = jnp.arange(n)
    broadcast = (n,) + (1,) * (values.ndim - 1)

    def step(
        k: jax.Array, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        reaches_start, sums = carry
        shift = jnp.left_shift(1, k)
        # Rows that already reach a start (all rows below ``shift`` do) keep
        # their sum; the others add the partial sum that ends ``shift`` back.
        keep = (reaches_start | (position < shift)).reshape(broadcast)
        sums = jnp.where(keep, sums, jnp.roll(sums, shift, axis=0) + sums)
        return reaches_start | jnp.roll(reaches_start, shift), sums

    return cast(
        jax.Array,
        jax.lax.fori_loop(0, max(n - 1, 0).bit_length(), step, (starts, values))[1],
    )


def _segment_totals(
    values: jax.Array,
    segment_ids: jax.Array | np.ndarray,
    num_segments: int,
    indices_are_sorted: bool,
) -> tuple[jax.Array, jax.Array]:
    """Each segment's total, held at its last row, and the row it belongs to.

    Parameters
    ----------
    values : jax.Array, shape (n, ...), n > 0
    segment_ids : jax.Array, shape (n,), integer
    num_segments : int
    indices_are_sorted : bool

    Returns
    -------
    sums : jax.Array, shape (n, ...)
        Running segment sums, in sorted id order.
    targets : jax.Array, shape (n,)
        The segment at each segment's last row; a distinct negative number
        elsewhere and for ids outside ``[0, num_segments)``, so the valid
        targets are unique and every other row is out of bounds.
    """
    n = values.shape[0]
    # A canonical signed index dtype keeps -1 representable for unsigned and
    # narrow ids; values that wrap negative are invalid and dropped. Clipping
    # keeps sorted ids sorted: low invalid ids first, high ones last.
    index_dtype = jax.dtypes.canonicalize_dtype(jnp.int64)
    ids = jnp.clip(jnp.asarray(segment_ids).astype(index_dtype), -1, num_segments)
    if not indices_are_sorted:
        order = jnp.argsort(ids, stable=True)
        ids, values = ids[order], values[order]
    changes = ids[1:] != ids[:-1]
    sums = _segmented_inclusive_sum(
        jnp.concatenate([jnp.ones(1, bool), changes]), values
    )
    is_total = jnp.concatenate([changes, jnp.ones(1, bool)])
    is_total &= (ids >= 0) & (ids < num_segments)
    targets = jnp.where(is_total, ids, -1 - jnp.arange(n, dtype=ids.dtype))
    return sums, targets


def _scatter_add_segment_totals(
    output: jax.Array,
    sums: jax.Array,
    targets: jax.Array,
    column_start: int | jax.Array | None = None,
) -> jax.Array:
    """Add each segment's total to its row of ``output``; drop every other row.

    Each output element receives at most one update, so the result does not
    depend on the order in which the backend applies them.

    Parameters
    ----------
    output : jax.Array, shape (num_segments, ...)
    sums, targets : jax.Array
        From ``_segment_totals``.
    column_start : int or scalar array, optional
        With a 2-D ``output``, the first column of the tile ``sums`` covers.
        As with ``dynamic_slice``, a negative start counts from the last
        column and a tile that would run past the last column is shifted
        left. None adds whole rows.

    Returns
    -------
    output : jax.Array, same shape as ``output``
    """
    if column_start is None:
        indices = targets[:, None]
        scatter_dims: tuple[int, ...] = (0,)
    else:
        n_columns = output.shape[1]
        column = jnp.asarray(column_start).astype(targets.dtype)
        column = jnp.where(column < 0, column + n_columns, column)
        column = jnp.clip(column, 0, n_columns - sums.shape[1])
        indices = jnp.stack([targets, jnp.broadcast_to(column, targets.shape)], 1)
        scatter_dims = (0, 1)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=tuple(range(1, sums.ndim)),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=scatter_dims,
    )
    return jax.lax.scatter_add(
        output,
        indices,
        sums,
        dimension_numbers,
        indices_are_sorted=False,
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
    )


@partial(jax.jit, static_argnames=("num_segments", "indices_are_sorted"))
def deterministic_segment_sum(
    values: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    indices_are_sorted: bool = False,
) -> jax.Array:
    """Sum rows into segments with a fixed reduction tree.

    Parameters
    ----------
    values : jax.Array, shape (n, ...)
        Values to sum, one row per segment id.
    segment_ids : jax.Array, shape (n,), int
        Segment of each row. Ids outside ``[0, num_segments)`` are dropped. With x64 disabled,
        64-bit ids wrap to 32 bits first; normalize them on the host (see
        ``spike_row_ids``).
    num_segments : int
        Must be below 2**31.
    indices_are_sorted : bool
        True only when ``segment_ids`` is verified nondecreasing.

    Returns
    -------
    sums : jax.Array, shape (num_segments, ...)
        Repeated evaluation of the same shapes is bitwise identical on every
        backend; a different segment layout changes the tree and can change
        rounding.

    Notes
    -----
    A segmented scan forms each segment's total; a scatter whose indices are
    unique then places it, so no output element accumulates floating-point
    updates in an order the backend chooses.
    """
    n = values.shape[0]
    shape = (num_segments, *values.shape[1:])
    if n == 0 or num_segments == 0:
        return jnp.zeros(shape, values.dtype)
    sums, targets = _segment_totals(
        values, segment_ids, num_segments, indices_are_sorted
    )
    return _scatter_add_segment_totals(jnp.zeros(shape, values.dtype), sums, targets)


def _reduces_sequentially(values: jax.Array) -> bool:
    """Whether ``values`` live on CPU, where XLA scatter adds updates in order.

    Traced values carry no device, so the default backend decides; a CPU
    default with work explicitly placed on an accelerator would therefore use
    the scatter. The package never places work that way.
    """
    if isinstance(values, jax.core.Tracer):
        return jax.default_backend() == "cpu"
    return all(device.platform == "cpu" for device in values.devices())


def _bucketed_size(n: int) -> int:
    """Next power of two, so varying spike counts share few executables."""
    return 1 << max(n - 1, 0).bit_length()


def deterministic_row_sum(
    values: jax.Array | np.ndarray,
    row_ids: jax.Array | np.ndarray,
    n_rows: int,
    *,
    indices_are_sorted: bool = False,
) -> jax.Array:
    """Deterministic row sums: sequential scatter on CPU, segmented scan elsewhere.

    Parameters
    ----------
    values : jax.Array, shape (n, ...)
        Values to sum, one row per row id.
    row_ids : array-like, shape (n,), int
        Output row of each value. Ids outside ``[0, n_rows)`` are dropped. With
        x64 disabled, 64-bit ids wrap to 32 bits first; normalize them on the
        host (see ``spike_row_ids``).
    n_rows : int
        Number of output rows.
    indices_are_sorted : bool
        True only when ``row_ids`` is verified nondecreasing.

    Returns
    -------
    sums : jax.Array, shape (n_rows, ...)
        Sum of the values assigned to each row; rows with no values are zero.

    Notes
    -----
    XLA:CPU scatter applies updates sequentially, so it is deterministic and
    fastest there. Other backends avoid floating-point scatter atomics, which
    change results between otherwise identical evaluations. Concrete
    off-CPU calls pad the spike axis to a power of two with dropped ids, so
    chunks with varying spike counts reuse a bounded set of executables.
    """
    if not isinstance(values, jax.core.Tracer):
        values = jnp.asarray(values)
    if values.ndim == 0 or values.shape[0] != np.shape(row_ids)[0]:
        msg = "values must contain one row per row id"
        raise ValueError(msg)
    if _reduces_sequentially(values):
        return jax.ops.segment_sum(
            values,
            row_ids,
            num_segments=n_rows,
            indices_are_sorted=indices_are_sorted,
        )
    # Widen before padding so the n_rows sentinel cannot wrap in narrow dtypes.
    row_ids = jnp.asarray(row_ids).astype(jax.dtypes.canonicalize_dtype(jnp.int64))
    if not isinstance(values, jax.core.Tracer):
        padding = _bucketed_size(values.shape[0]) - values.shape[0]
        if padding:
            # Trailing n_rows ids are dropped and keep sorted ids sorted.
            values = jnp.concatenate(
                [values, jnp.zeros((padding, *values.shape[1:]), values.dtype)]
            )
            row_ids = jnp.concatenate(
                [row_ids, jnp.full(padding, n_rows, dtype=row_ids.dtype)]
            )
    return cast(
        jax.Array,
        deterministic_segment_sum(values, row_ids, n_rows, indices_are_sorted),
    )


def deterministic_row_add(
    output: jax.Array,
    values: jax.Array | np.ndarray,
    row_ids: jax.Array | np.ndarray,
    column_start: int | jax.Array = 0,
    *,
    indices_are_sorted: bool = False,
) -> jax.Array:
    """Add per-spike column-tile vectors into ``output[:, column_start:...]``.

    Parameters
    ----------
    output : jax.Array, shape (n_rows, n_columns)
        Array that receives the sums of the tile.
    values : jax.Array, shape (n, tile_columns)
        Per-spike vectors for the output columns of the tile.
    row_ids : array-like, shape (n,), int
        Row of ``output`` that each value is added to.
    column_start : int or scalar array
        First output column of the tile.
    indices_are_sorted : bool
        True only when ``row_ids`` is verified nondecreasing.

    Returns
    -------
    output : jax.Array, shape (n_rows, n_columns)
        The tile holds ``initial + sum(contributions)``. On CPU each call
        builds an ``(n_rows, tile_columns)`` row-sum temporary. Elsewhere a
        segmented scan sums each row's contributions, keeping temporaries
        about twice the size of ``values`` (measured on XLA:CPU), and only
        the rows they own are updated. As with ``dynamic_slice``, a negative
        ``column_start`` counts from the last column and a tile that would
        run past the last column is shifted left, so callers must pass tiles
        that fit.
    """
    values = jnp.asarray(values).astype(output.dtype)
    if output.shape[0] == 0 or values.shape[0] == 0 or values.size == 0:
        return output
    if values.shape[0] != np.shape(row_ids)[0]:
        msg = "values must contain one row per row id"
        raise ValueError(msg)
    if not _reduces_sequentially(values):
        sums, targets = _segment_totals(
            values, row_ids, output.shape[0], indices_are_sorted
        )
        return _scatter_add_segment_totals(output, sums, targets, column_start)
    tile = deterministic_row_sum(
        values, row_ids, output.shape[0], indices_are_sorted=indices_are_sorted
    )
    # dynamic_slice needs one index dtype; x64 makes a literal 0 int64.
    column_start = jnp.asarray(column_start)
    start = (jnp.zeros((), column_start.dtype), column_start)
    current = jax.lax.dynamic_slice(output, start, tile.shape)
    return jax.lax.dynamic_update_slice(output, current + tile, start)
