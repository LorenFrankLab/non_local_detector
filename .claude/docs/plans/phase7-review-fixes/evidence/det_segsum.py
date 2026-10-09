"""Prototype check for the deterministic segmented-scan row sum proposed in designs.md."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import non_local_detector.likelihoods.common as common


def _segmented_add(left, right):
    left_start, left_value = left
    right_start, right_value = right
    keep = right_start.reshape(right_start.shape + (1,) * (right_value.ndim - right_start.ndim))
    return left_start | right_start, jnp.where(keep, right_value, left_value + right_value)


@partial(jax.jit, static_argnames=("num_segments", "indices_are_sorted"))
def deterministic_segment_sum(values, segment_ids, num_segments, indices_are_sorted=False):
    n = values.shape[0]
    shape = (num_segments, *values.shape[1:])
    if n == 0 or num_segments == 0:
        return jnp.zeros(shape, values.dtype)
    ids = jnp.clip(segment_ids, -1, num_segments)
    if not indices_are_sorted:
        order = jnp.argsort(ids, stable=True)
        ids, values = ids[order], values[order]
    starts = jnp.concatenate([jnp.ones(1, bool), ids[1:] != ids[:-1]])
    _, prefix = jax.lax.associative_scan(_segmented_add, (starts, values))
    segments = jnp.arange(num_segments, dtype=ids.dtype)
    last = jnp.clip(jnp.searchsorted(ids, segments, side="right") - 1, 0, n - 1)
    present = ids[last] == segments
    mask = present.reshape((num_segments,) + (1,) * (values.ndim - 1))
    return jnp.where(mask, prefix[last], jnp.zeros((), values.dtype))


rng = np.random.default_rng(0)
# Correctness vs float64, sorted and unsorted, invalid ids, tails.
for sorted_ids in (True, False):
    for tail in ((), (5,), (2, 3)):
        n, k = 300, 17
        ids = rng.integers(-2, k + 2, n)
        if sorted_ids:
            ids = np.sort(ids)
        vals = rng.normal(size=(n, *tail)).astype(np.float32)
        expect = np.zeros((k, *tail))
        for i, v in zip(ids, vals.astype(np.float64)):
            if 0 <= i < k:
                expect[i] += v
        got = np.asarray(deterministic_segment_sum(jnp.asarray(vals), jnp.asarray(ids, jnp.int32), k, sorted_ids))
        seq = np.asarray(jax.ops.segment_sum(jnp.asarray(vals), jnp.asarray(ids), k))
        print(f"sorted={sorted_ids!s:5} tail={tail!s:7} max|scan-f64|={np.abs(got - expect).max():.2e} max|segsum-f64|={np.abs(seq - expect).max():.2e}")

# Empty and single.
print("empty", deterministic_segment_sum(jnp.zeros((0, 3)), jnp.zeros(0, jnp.int32), 4).shape)
print("all invalid", np.asarray(deterministic_segment_sum(jnp.ones((3, 2)), jnp.array([-1, 9, 9]), 4)).sum())
# NaN stays in its own row.
v = jnp.array([[1.0], [jnp.nan], [2.0], [3.0]])
print("nan rows", np.asarray(deterministic_segment_sum(v, jnp.array([0, 1, 1, 2]), 3, True)).ravel())

# Gradient wrt values equals gather of cotangent at owned rows (0 for invalid).
ids = jnp.array([2, -1, 2, 0, 9, 1, 2])
g = jax.grad(lambda x: (deterministic_segment_sum(x, ids, 4) * jnp.arange(4.0)[:, None]).sum())(jnp.ones((7, 1)))
print("grad", np.asarray(g).ravel())

# Run-to-run bitwise identity.
vals = jnp.asarray(rng.normal(-13, 7, (5000, 254)).astype(np.float32))
ids = jnp.asarray(np.sort(rng.integers(0, 256, 5000)).astype(np.int32))
ref = np.asarray(deterministic_segment_sum(vals, ids, 256, True))
print("bitwise 20 repeats", all(np.array_equal(ref.view(np.uint32), np.asarray(deterministic_segment_sum(vals, ids, 256, True)).view(np.uint32)) for _ in range(20)))

# CPU timing vs ordered loop and segment_sum.
def bench(fn, *a, r=20):
    fn(*a).block_until_ready(); fn(*a).block_until_ready()
    t = []
    for _ in range(r):
        s = time.perf_counter(); fn(*a).block_until_ready(); t.append(time.perf_counter() - s)
    return np.median(t) * 1e3

for n, cols, rows, srt in ((50, 16930, 256, True), (20000, 500, 500, False), (200000, 64, 256, True)):
    vals = jnp.asarray(rng.normal(size=(n, cols)).astype(np.float32))
    ids_np = rng.integers(0, rows, n)
    if srt:
        ids_np = np.sort(ids_np)
    ids = jnp.asarray(ids_np.astype(np.int32))
    scan = jax.jit(lambda v, i: deterministic_segment_sum(v, i, rows, srt))
    loop = jax.jit(lambda v, i: common._ordered_spike_row_sum(v, i, rows))
    seg = jax.jit(lambda v, i: jax.ops.segment_sum(v, i, num_segments=rows, indices_are_sorted=srt))
    print(f"n={n:6d} cols={cols:5d} rows={rows} sorted={srt}: scan {bench(scan, vals, ids):8.2f} ms | ordered loop {bench(loop, vals, ids):8.2f} ms | segment_sum {bench(seg, vals, ids):8.2f} ms")
