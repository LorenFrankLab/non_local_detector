# Clusterless log-KDE reduction

`estimate_log_joint_mark_intensity` computes, for each decoding spike `q` and
position bin `p`,

```text
log_joint[q, p] = max(log(mean_rate * sum_e w_e K_mark(e, q) K_pos(e, p) / occupancy[p]), LOG_EPS)
```

over encoding samples `e` with normalized weights `w_e`. This note records how
the GEMM path evaluates that sum, what it guarantees, and what it costs.

## Contract

- Both paths (`use_gemm=True` and the `use_gemm=False` reference) return values
  floored at `LOG_EPS`. Every caller already floored them; the GEMM path now
  floors before returning, so `block_estimate_log_joint_mark_intensity` and the
  default predict path no longer clamp again.
- Above the floor, the result matches a float64 brute-force logsumexp to float32
  roundoff, whatever the other decoding spikes in the call, the number of mark
  features, or the encoding and position tile sizes.
- Below the floor the reduction is not evaluated exactly; such entries become
  `LOG_EPS`.
- `NaN` inputs propagate to the affected decoding spikes and position bins.

## Why the previous reduction was batch dependent

The earlier fast path stabilized each encoding sample by its largest mark kernel
over all decoding spikes in the block, then shared one global offset. A decoding
spike far in mark space from the rest of its block therefore had its kernels
flushed to zero relative to the block's best match. With eight mark features,
bandwidth 0.001, and occupancy 1e-14, a spike scored -9.8533 alone but `LOG_EPS`
(-34.5388) beside a matching spike. Its memory also grew with
`n_encoding x block_size` (3,100 MiB of temporaries for 40,000 encoding
samples and the default 10,000-spike block).

Mark distances also expanded `||x - y||^2` as `||x||^2 + ||y||^2 - 2 x.y`,
which cancels for features with large offsets: translating an encoding mark of 0
and a decoding mark of 1 by 10,000 moved their log kernel from -1.41894 to
-0.91894. Distances now subtract each feature pair before scaling.

## Algorithm

Encoding samples are processed in tiles of at most 2,048 rows. Each tile is one
matrix product over the whole decoding block and each position tile (all
positions unless `pos_tile_size` is given), accumulated across tiles. Temporaries
are `O(tile x (n_decoding + n_position))` plus the output, independent of the
number of encoding samples.

**Window.** Scale every mark factor by a peak `A` and every position factor by a
peak `B[p]`, then shift both up by `s = -log(2 n tiny) / 2` for `n` encoding rows,
where `tiny` is the smallest normal number. Each factor is at most `exp(s)`, so
the `n`-term sum stays below `1 / (2 tiny)`: it cannot overflow, and its
reciprocal (the derivative of its logarithm) stays normal. A term is lost only if
a factor or the product flushes below `tiny`, and each such loss is below
`tiny * exp(s)` in scaled units. Entries within

```text
W = log(4 * eps) - log(n * tiny) + s
```

nats of `A + B[p]` therefore lose less than four float epsilons; `W` is about
105 nats for 2,048 float32 rows and 100 nats for 40,000.

**Gate.** Before reducing, compute kernel-free peaks: the mark kernel's value at
zero distance (its Gaussian normalizer) and, per position bin, the largest
weighted position kernel. If `A + B[p] - W` is below the floor for every bin, no
entry at or above the floor can lose mass. For microvolt-scale features with
the default bandwidths the peaks exceed the floor by at most 12 nats on the
fixtures below, against a window of about 100. One scalar condition per call
then selects:

- **Common path.** Every tile uses the fixed peaks as scales, so tile products
  add linearly and a single logarithm is taken at the end. The scales depend on
  no decoding spike, which makes each spike's result independent of its batch.
- **Extreme normalizations** (for example 16 features with bandwidth 0.001 and
  occupancy 1e-14). Each tile scales every decoding spike and position bin by its
  own maximum over the tile, accumulates in log space, and flags entries whose
  possible loss could exceed four epsilons of both their estimate and the floor
  divided by the number of tiles, so the total loss stays within eight epsilons.
  Flagged entries receive an exact logsumexp, one decoding spike at a time to
  bound memory. This path is correct but slow; it did not run for any fixture
  below. Under `jax.vmap` both paths run, so callers map over blocks with
  `jax.lax.map`.

`LOG_EPS` is a flat floor on a density whose units depend on the mark units and
feature count, which is why rescaling marks (for example microvolts to volts
with eight features) can push the margin past the window. If the flat floor is
replaced by a unit-consistent background density, the margin no longer depends
on mark units; revisit whether the exact path is still needed then.

The gate uses the float32 window, the narrowest dtype the products can use, so it
stays conservative when x64 is enabled. Loop carries are explicitly typed, so
float32 inputs and float64 inputs both work with x64 enabled.

## Performance against the previous reduction

Warm medians of the raw kernel on place-field fixtures (12 units, 4 microvolt
features, `waveform_std=24`, 6 cm position bandwidth), compared with the
reduction this change replaces. Q is decoding spikes per call (the default
predict block is 10,000), E encoding samples, P position bins. GPU timings are
from an A100 and CPU timings from 16 pinned x86 cores, both with JAX 0.9.0.
"Memory" is XLA's compiled temporary allocation on CPU.

| Workload | A100 before -> after | CPU before -> after | Memory before -> after |
| --- | --- | --- | --- |
| 1D, Q=1,000, E=2,000, P=300 | 0.7 -> 0.8 ms | 20.4 -> 7.5 ms | 15.3 -> 19.9 MiB |
| 1D, Q=10,000, E=8,000, P=300 | 4.6 -> 5.0 ms | 362.6 -> 198.0 ms | 610.5 -> 172.5 MiB |
| 2D, Q=10,000, E=8,000, P=1,600 | 16.4 -> 17.3 ms | 374.9 -> 229.7 ms | 610.5 -> 290.9 MiB |
| 1D, Q=10,000, E=40,000, P=300 | 19.4 -> 21.3 ms | 1,300.2 -> 767.1 ms | 3,099.9 -> 172.6 MiB |
| Streaming, Q=10,000, E=8,000, tile 512 | 6.4 -> 6.3 ms | 579.0 -> 204.8 ms | |
| Streaming, Q=10,000, E=8,000, tile 2,048 | 5.4 -> 5.6 ms | 437.6 -> 196.2 ms | |

The remaining GPU overhead (up to 11% at the default block size, 0.1 ms for the
smallest workload) is the sequential encoding tiles that bound memory. After the
floor, outputs differ from the previous reduction by at most 8.8e-5 nats on these
fixtures, and from a float64 brute-force reduction by at most 1.4e-5 nats.
