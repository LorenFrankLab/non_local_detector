# Designs

[← back to PLAN.md](PLAN.md)

- [D1. Deterministic spike-row reduction](#d1-deterministic-spike-row-reduction) — phase 2
- [D2. Graph-distance row cache and caller-owned outputs](#d2-graph-distance-row-cache-and-caller-owned-outputs) — phase 4
- [D3. Result-store chunk lookup](#d3-result-store-chunk-lookup) — phase 3
- [D4. Accuracy-relative sorted-emission test](#d4-accuracy-relative-sorted-emission-test) — phase 1a

## D1. Deterministic spike-row reduction

**Requirement.** Checkpointed replay needs the same likelihood bits on the
forward pass and on replay for the *same chunk shape*
([appendix E3](appendix.md#e3-serial-spike-row-loop-structure)). It does not
need the sequential addition order, and it does not need bitwise agreement
across tile shapes or platforms.

**Approach.** A segmented inclusive scan with a fixed, shape-determined
combination tree, followed by a gather of each segment's last prefix. There are
no scatters, so no atomics. When ids are not already nondecreasing, a stable
`argsort` sorts them first. It is pairwise-style summation, so it is at least
as accurate as sequential addition. On 300 random float32 rows it measured
6e-7 to 1.1e-6 max error against float64, versus 9e-7 to 1.4e-6 for sequential
`segment_sum` ([evidence/det_segsum.py](evidence/det_segsum.py)).

```python
# src/non_local_detector/likelihoods/common.py

def _segmented_add(left, right):
    """Associative segmented sum: a segment start in ``right`` resets the total."""
    left_start, left_value = left
    right_start, right_value = right
    keep = right_start.reshape(
        right_start.shape + (1,) * (right_value.ndim - right_start.ndim)
    )
    return left_start | right_start, jnp.where(keep, right_value, left_value + right_value)


@partial(jax.jit, static_argnames=("num_segments", "indices_are_sorted"))
def deterministic_segment_sum(
    values: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    indices_are_sorted: bool = False,
) -> jnp.ndarray:
    """Sum rows into segments with a fixed reduction tree and no scatters.

    Parameters
    ----------
    values : jnp.ndarray, shape (n, ...)
    segment_ids : jnp.ndarray, shape (n,), integer
        Ids outside ``[0, num_segments)`` are dropped.
    num_segments : int
    indices_are_sorted : bool
        True only when ``segment_ids`` is verified nondecreasing.

    Returns
    -------
    sums : jnp.ndarray, shape (num_segments, ...)
        Repeated evaluation of the same shapes is bitwise identical on every
        backend. Different ``n`` changes the tree and can change rounding.
    """
    n = values.shape[0]
    shape = (num_segments, *values.shape[1:])
    if n == 0 or num_segments == 0:
        return jnp.zeros(shape, values.dtype)
    # Clip keeps sorted input sorted: low invalid ids stay first, high ones last.
    ids = jnp.clip(segment_ids, -1, num_segments)
    if not indices_are_sorted:
        order = jnp.argsort(ids, stable=True)
        ids, values = ids[order], values[order]
    starts = jnp.concatenate([jnp.ones(1, bool), ids[1:] != ids[:-1]])
    _, prefix = jax.lax.associative_scan(_segmented_add, (starts, values))
    segments = jnp.arange(num_segments, dtype=ids.dtype)
    last = jnp.clip(jnp.searchsorted(ids, segments, side="right") - 1, 0, n - 1)
    present = (ids[last] == segments).reshape((num_segments,) + (1,) * (values.ndim - 1))
    return jnp.where(present, prefix[last], jnp.zeros((), values.dtype))
```

Measured on the prototype: invalid ids are dropped, a NaN stays in its own row,
the gradient with respect to `values` gathers the cotangent at each owned row
(0 for invalid), the empty and all-invalid cases return zeros, and 20 repeats
are bitwise identical.

**Backend dispatch.** On CPU the scan was 3–6× slower than both
`jax.ops.segment_sum` and the current serial loop
([evidence/det_segsum.py](evidence/det_segsum.py): for example 9.8 ms versus
1.4 ms and 2.1 ms at 20000 × 500). XLA:CPU scatter is sequential, so it is
deterministic. One dispatcher replaces every `_ordered_spike_row_*` caller:

```python
# Set from the phase-2 A100 baseline (task 1). None means "always scan off CPU".
_SMALL_SERIAL_SPIKES: int | None = None


def deterministic_row_sum(values, row_ids, n_rows, *, indices_are_sorted=False):
    """Deterministic row sums: sequential scatter on CPU, segmented scan elsewhere."""
    on_cpu = (
        all(device.platform == "cpu" for device in values.devices())
        if not isinstance(values, jax.core.Tracer)
        else jax.default_backend() == "cpu"
    )
    if on_cpu:
        return jax.ops.segment_sum(
            values, row_ids, num_segments=n_rows, indices_are_sorted=indices_are_sorted
        )
    if _SMALL_SERIAL_SPIKES is not None and values.shape[0] <= _SMALL_SERIAL_SPIKES:
        return _serial_row_sum(values, row_ids, n_rows)  # today's loop, renamed
    return deterministic_segment_sum(values, row_ids, n_rows, indices_are_sorted)
```

`values.shape[0]` is static under `jit`, so the small-`n` branch is decided at
trace time and costs nothing at run time. Keep `_serial_row_sum` only if the
A100 baseline shows the serial loop is faster for checkpoint-sized chunks
(about 10–50 spikes); otherwise delete it and the constant.

**Column-tile variant** (used by GMM and streamed row-returning calls, which
today pass `column_start` to `_ordered_spike_row_add`):

```python
def deterministic_row_add(output, values, row_ids, column_start, *, indices_are_sorted=False):
    """Add per-spike column-tile vectors into ``output[:, column_start:...]``."""
    tile = deterministic_row_sum(
        values, row_ids, output.shape[0], indices_are_sorted=indices_are_sorted
    ).astype(output.dtype)
    current = jax.lax.dynamic_slice(output, (0, column_start), tile.shape)
    return jax.lax.dynamic_update_slice(output, current + tile, (0, column_start))
```

The tile workspace is `(n_rows, tile_columns)`, the same size as the output
tile being updated. Rounding changes from `((initial + c1) + c2) + c3` to
`initial + (c1 + c2 + c3)`.

**Encoding-spike reductions (clusterless diffusion).** Encoding bins are fixed
per electrode, so sort each electrode's encoding spikes by bin once on the
host, before `jnp.asarray` at `clusterless_diffusion.py:699-701` and
`826-828`:

```python
order = np.argsort(np.asarray(electrode_bins), kind="stable")
electrode_bins = np.asarray(electrode_bins)[order]
electrode_marks = np.asarray(electrode_marks)[order]
electrode_weights = np.asarray(electrode_weights)[order]
```

Then call `deterministic_row_sum(weighted_kernel, enc_bins, n_bins,
indices_are_sorted=True)`. `weighted_kernel` rows follow the sorted encoding
order, so no per-block `argsort` is needed.

**Alternatives rejected.**

- `XLA_FLAGS=--xla_gpu_deterministic_ops=true`: a process-global setting a
  library should not impose, and its scatter behavior was not verified here.
- One-hot matmul `onehot(ids).T @ values`: deterministic, but its
  `(n_bins, n_enc)` operand is 1.35 GB at 16,930 bins × 20,000 spikes.
- Padded `(n_segments, max_per_segment)` gather: the memory grows with the
  busiest bin. With encoding spikes, that is reward-well occupancy.

## D2. Graph-distance row cache and caller-owned outputs

**Problem** ([appendix E6](appendix.md#e6-review-findings-confirmed-by-code-reading)):
`LazyGraphDistances._entries` (`graph_distances.py:107-137`) budgets the
*output* (`rows.size * 8`) against `max_dense_bytes` and recomputes Dijkstra for
every source on every call. `Environment.get_distances_to_interior_bins`
(`environment.py:1031-1038`) asks for the whole `(n_positions, n_interior)`
block. With the dense representation, the same output is allocated without
any budget.

**Design.**

1. The output of a cross-distance query belongs to the caller, so it is not
   budgeted. Only the Dijkstra workspace and the cache are budgeted.
2. A bounded LRU cache of full source rows lives on the instance. It is never
   pickled.

```python
# src/non_local_detector/graph_distances.py
from collections import OrderedDict

DEFAULT_DISTANCE_CACHE_BYTES = 64 * 1024**2


class LazyGraphDistances:
    def __init__(self, graph, *, directed=False,
                 max_dense_bytes=DEFAULT_MAX_DENSE_DISTANCE_BYTES,
                 max_workspace_bytes=DEFAULT_DISTANCE_WORKSPACE_BYTES,
                 max_cache_bytes=DEFAULT_DISTANCE_CACHE_BYTES):
        ...  # existing validation
        _budget(0, max_cache_bytes, "Graph distance row cache")
        self.max_cache_bytes = int(max_cache_bytes)
        self._row_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_row_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        state.setdefault("max_cache_bytes", DEFAULT_DISTANCE_CACHE_BYTES)
        state.setdefault("_row_cache", OrderedDict())
        self.__dict__.update(state)

    def _rows_for(self, sources):
        """Full float64 rows for ``sources`` (1-D, unique), filling the LRU cache."""
        row_bytes = self.shape[1] * 8
        result = np.empty((len(sources), self.shape[1]))
        missing = []
        for position, source in enumerate(sources):
            row = self._row_cache.get(int(source))
            if row is None:
                missing.append(position)
            else:
                self._row_cache.move_to_end(int(source))
                result[position] = row
        batch_rows = max(1, self.max_workspace_bytes // row_bytes)
        for start in range(0, len(missing), batch_rows):
            batch = missing[start:start + batch_rows]
            result[batch] = dijkstra(self.graph, directed=self.directed,
                                     indices=sources[batch])
            if row_bytes <= self.max_cache_bytes:
                for position in batch:
                    self._row_cache[int(sources[position])] = result[position].copy()
                while len(self._row_cache) * row_bytes > self.max_cache_bytes:
                    self._row_cache.popitem(last=False)
        return result

    def cross_distances(self, rows, columns):
        """Exact ``(len(rows), len(columns))`` distances; the output is caller-owned.

        Only the Dijkstra workspace and the row cache are budgeted.
        """
        rows = np.asarray(rows, dtype=np.intp)
        columns = np.asarray(columns, dtype=np.intp)
        _budget(self.shape[1] * 8, self.max_workspace_bytes,
                "One source-row graph distance workspace")
        out = np.empty((len(rows), len(columns)))
        sources, inverse = np.unique(rows, return_inverse=True)
        batch_rows = max(1, self.max_workspace_bytes // (self.shape[1] * 8))
        for start in range(0, len(sources), batch_rows):
            stop = min(start + batch_rows, len(sources))
            block = self._rows_for(sources[start:stop])[:, columns]
            select = (inverse >= start) & (inverse < stop)
            out[select] = block[inverse[select] - start]
        return out
```

`_entries` keeps its current budget semantics for `__getitem__`. Route its
Dijkstra calls through `_rows_for` so indexing also uses the cache. In
`Environment.get_distances_to_interior_bins`, when
`isinstance(self.distance_between_nodes_, LazyGraphDistances)`, return
`self.distance_between_nodes_.cross_distances(position_bin_inds,
interior_bin_indices)`. Keep the `np.ix_` path for dense arrays.

**Default cache size:** 64 MiB is about 500 rows at 16,930 bins. That is
enough to cover the distinct animal bins in one 256-row chunk with room left
to reuse them on replay. See [overview open question 3](overview.md#open-questions).

## D3. Result-store chunk lookup

Reader (`result_store.py:173-215`, `_ChunkedArray._getitem`). Published chunks
are already sorted by `IncrementalResultWriter.complete` (`result_store.py:116`)
and checked as contiguous when the store is opened. Cache their starts once in
`__init__` and find each row's chunk with `searchsorted`:

```python
# __init__
self._chunks = spec["chunks"]
self._starts = np.array([chunk["start"] for chunk in self._chunks], dtype=np.int64)

# _getitem, replacing the loop over every chunk
owner = np.searchsorted(self._starts, indices[0], side="right") - 1
order = np.argsort(owner, kind="stable")
touched, first = np.unique(owner[order], return_index=True)
for chunk_number, lo, hi in zip(touched, first, [*first[1:], len(order)]):
    selected = order[lo:hi]
    chunk = self._chunks[chunk_number]
    ...  # existing mmap/shape check/copy, using `selected`
```

Chunk coverage is complete and contiguous, which `_validate_variable_chunks`
(`result_store.py:227-250`) checks when the store is opened, so every `owner`
is valid. The cost becomes O(n_rows log n_chunks + touched chunks).

Writer (`IncrementalResultWriter.write`, `result_store.py:85-109`). Keep a
per-variable sorted list of `(start, stop)` next to `spec["chunks"]` and check
overlap only against neighbours:

```python
import bisect

intervals = self._intervals[name]          # sorted list of (start, stop)
position = bisect.bisect_left(intervals, (start, stop))
before = intervals[position - 1] if position else None
after = intervals[position] if position < len(intervals) else None
if (before and before[1] > start) or (after and after[0] < stop):
    raise ValueError("Output chunks overlap")
bisect.insort(intervals, (start, stop))
```

## D4. Accuracy-relative sorted-emission test

Replace the comparison at
`test_phase7c_sorted_accumulation.py:358-374` (optimized vs per-neuron
float32 reference at `rtol=atol=1e-6`) with a comparison of both paths against
a float64 oracle emission. The oracle sums the same float32 inputs in float64
and rounds once to float32, through the same model.

```python
def float64_oracle(counts, rates, durations, summed_rates):
    c, r = np.asarray(counts, np.float64), np.asarray(rates, np.float64)
    d, s = np.asarray(durations, np.float64), np.asarray(summed_rates, np.float64)
    ll = sum(xlogy(c[:, n][:, None], r[n][None, :] * d[:, None]) for n in range(c.shape[1]))
    return jnp.asarray((ll - d[:, None] * s).astype(np.float32))
```

Assertions (the proposed tolerances need user approval, see
[phase 1a](phase-1a-sorted-matrix-emission.md#gate)):

1. Emission accuracy: for every element,
   `|ll_optimized - ll_f64| <= 16 * np.spacing(np.abs(ll_f64).astype(np.float32))`.
   Measured maximum: 6.3 ulp ([appendix E1](appendix.md#e1-sorted-emission-order-bought-agreement-not-accuracy)).
2. Posterior no worse than the reference: for each output variable,
   `max|post_opt - post_oracle| <= 2 * max|post_ref - post_oracle| + 1e-6`,
   where `post_oracle` is the predict run with the oracle emission
   monkeypatched in, the same way `per_neuron_jax_reference` is today.
