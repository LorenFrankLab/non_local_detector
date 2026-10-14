# Batch-independent clusterless log-KDE

The default log-KDE reduction uses independently scaled matrix multiplication
with logaddexp accumulation across encoding tiles. Maxima belong to each query
and position, so adding a decoding query cannot flush an existing query's
kernel to zero. Defaults cap encoding tiles at 2,048 rows, position tiles at
256 columns, and decoding query tiles at 128 rows. Requested smaller tiles are
honored. The final tile overlaps earlier samples and masks that prefix, avoiding
a padded copy of a large precomputed encoding-by-position matrix. Padded public
decode events still drop out through the existing row reduction.

The historical private compensated-linear entry points retain their signatures
but also use the guarded matrix product. Waveform feature count no longer
selects a numerical approximation: low-dimensional queries can still have
arbitrarily large dynamic range. The explicit `use_gemm=False` legacy
probability-space reference remains available.

Mark distances subtract each feature pair before scaling and squaring. This
incorporates the distance repair already proposed in PR #70, including integer
marks and guarded subtraction overflow. The two PRs overlap on that function;
this branch includes it so the complete numerical fix can be reviewed and
merged independently. PR #70 is left untouched.

## Before and after

With eight mark dimensions, bandwidth 0.001, a zero encoding mark and occupancy
1e-14, the remote query `[sqrt(180)*0.001, 0, ...]` should score approximately
−9.8533. Previously it scored −34.5388 when batched with a matching query,
despite scoring correctly by itself. Both arrangements now return the
closed-form Gaussian log ratio. The regression covers encoding tiles as well.

Adding 10,000 to an encoding mark 0 and decoding mark 1 previously changed
their log Gaussian kernel from −1.41894 to −0.91894. Pairwise subtraction
preserves the original kernel.

The fast path bounds omitted scaled mass by three times the encoding tile size
times the dtype's smallest normal value, accounting for flushed factors and
products. A standard positive dot-product roundoff bound adjusts the comparison
with four dtype epsilons of the computed mass. Suspect positive and zero outputs
receive a direct logsumexp fallback. Scalar conditionals inside a sequential
query loop bound its temporary workspace; only suspect entries are replaced,
so another query's fallback cannot change a regular output. Existing
local/nonlocal, weighted, padded, streaming, float32/float64 and determinism
controls keep their tolerances; no stored numerical reference data is changed.

On CUDA/ROCm, small streamed tiles use a direct parallel logsumexp when their
dtype-aware encoding-by-query-by-position workspace is at most 16 MiB. Tiny
GEMMs otherwise spend substantial time launching kernels and checking fallback
predicates. This fixed limit keeps the reduction independent of recording,
decoder block, and grid sizes. JAX resolves the platform branch during lowering,
so CPU and larger tiles retain guarded GEMM. The direct branch uses the same
safe empty-support reduction and does not relax tail accuracy.

Independent review added analytical derivative regressions: all-zero encoding
weights have zero weight derivatives, and a completely masked prefix tile
cannot poison the gradients of later supported samples. Empty logsumexp and
logaddexp operands now use safe inactive branches. A length-one standalone
bandwidth normalizes every observed mark dimension when broadcast.

On a CPU workload with 1,000 encoding marks, 100 queries, 200 spatial bins and
four waveform dimensions, seven warm calls gave median 0.451 ms before and
6.826 ms after the initial logsumexp repair. These runs shared a host with
other validation work; they quantify a material throughput cost, not a
performance guarantee. Both outputs were finite with shape (100, 200).

The follow-up performance repair measured the existing seeded raw-kernel
workloads on the local CPU with device inputs, three warmups and seven
synchronized repetitions. Median time fell from 11.59 to 0.99 ms for the
1,000-encoding/100-query/512-bin case, from 11.74 to 1.04 ms with 16 mark
dimensions, and from 496.17 to 27.55 ms with 16,000 encodings and 256 queries.
Compiled temporary workspace fell from 15.77 to 2.94 MiB for the default case
and from 66.28 to 7.82 MiB for the large case. Compared with the prior direct
logsumexp outputs, maximum absolute float32 differences were 3.82e-6, 1.53e-5,
and 1.34e-5 respectively, from reordering positive reductions; all outputs
remained finite. These local measurements supplement the controlled CPU/GPU
performance review and do not predict other hardware's speed.

The repository-wide run passed 3,675 cases and skipped seven. Two existing
NaN-propagation tests exposed that the all-zero-weight shortcut hid invalid
encoding marks. The shortcut now bypasses reductions only when kernel inputs
contain no NaNs; malformed inputs retain their NaN evidence. The failed cases,
empty-tile value/gradient tests, new regressions and padding checks were rerun
after this correction. Stored golden comparisons passed unchanged.
