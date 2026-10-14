# Batch-independent clusterless log-KDE

The default log-KDE reduction now uses encoding and position tiles with online
logsumexp accumulation. The maxima used by a reduction belong to each output,
so adding a different decoding query cannot flush an existing query's kernel
to zero. Defaults cap encoding tiles at 256 rows and position tiles at 128
columns; decoding work is bounded by the predictor's existing block size.
Requested smaller tiles are honored. Padding retains zero-weight encoding rows
and drops padded decode events through the existing row reduction.

The historical private compensated-linear entry points retain their signatures
but also use logsumexp. Waveform feature count no longer selects a numerical
approximation: low-dimensional queries can still have arbitrarily large
dynamic range. The explicit `use_gemm=False` legacy probability-space reference
remains available; it is not the underflow-safe default.

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

This intentionally trades some compensated-GEMM throughput for reliable tail
likelihoods. A future fast path must demonstrate an error bound and fall back
for suspect positive outputs as well as zero outputs. Existing local/nonlocal,
weighted, padded, streaming, float32/float64 and determinism controls are kept
at their existing tolerances; no stored numerical reference data is changed.

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

The repository-wide run passed 3,675 cases and skipped seven. Two existing
NaN-propagation tests exposed that the all-zero-weight shortcut hid invalid
encoding marks. The shortcut now bypasses reductions only when kernel inputs
contain no NaNs; malformed inputs retain their NaN evidence. The failed cases,
empty-tile value/gradient tests, new regressions and padding checks were rerun
after this correction. Stored golden comparisons passed unchanged.
