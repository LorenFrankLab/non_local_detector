> **Snippets unverified.** The code below has not been executed. Per the process
> rule in [PLAN.md](PLAN.md), treat it as intent to be re-derived by prototyping,
> not as code to paste. The problem statements and measurements are verified.

# Phase 7 — Measured performance work

Last, because correctness comes first and because 6a/6b change the shapes these
touch. Write against post-6b code.

## Measurement protocol (applies to every task)

An earlier draft of this phase asserted a speedup from a single unsynchronized
timing loop. That is not evidence. Every task must produce:

1. **Output parity first.** Record baseline outputs to a pickle in the scratchpad
   *before* editing; after, assert parity at the stated tolerance. A speedup
   without a paired output check does not count.
2. **Synchronized timings.** `block_until_ready()` on every result; JAX is async
   and unsynchronized timings measure dispatch, not compute.
3. **Compile time excluded.** One warm-up call outside the timed region, and
   report compile time separately — a 100× "speedup" that moves work into
   compilation is not a win for a single-shot fit.
4. **Repeated trials.** ≥5 runs; report median and spread, not a single number.
5. **Peak memory.** `memory_analysis().temp_size_in_bytes` for jitted regions, or
   `jax.live_arrays()` sampling. A faster path that raises peak memory may be a
   regression on GPU.
6. **dtype parity.** Confirm the rewrite did not silently promote or demote.

Report all six per task in the PR description. If a task fails (1) or regresses
(5) beyond a stated budget, drop it rather than shipping it.

---

## Task 1 — Matmul instead of the per-neuron loop

`sorted_spikes_kde.py:399-405` and `sorted_spikes_glm.py:492-495` accumulate a
full `(n_time, n_interior_bins)` array once per neuron.
`sorted_spikes_diffusion.py:682-689` already does the right thing.

Measured (`n_time=20000, n_neurons=120, n_bins=900`, unsynchronized — **re-measure
under the protocol above**):

```
   per-neuron loop:  248.6 ms
     single matmul:    2.3 ms       ~108x
max abs diff 1.9e-06, max rel diff 2.1e-07
```

**Precondition the rewrite depends on:** interior fields are EPS-floored
(`sorted_spikes_kde.py:225-234`, `sorted_spikes_glm.py:351-354`), so `log(field)`
is finite and `xlogy(k, field) ≡ k·log(field)`. This fails if a field can be
exactly 0 — `xlogy(0,0) = 0` but `0·log(0) = NaN`. Assert the floor at fit time
rather than assuming it.

Move `_spike_counts_matrix` (`sorted_spikes_diffusion.py:548-567`) into
`common.py` rather than importing across siblings or copying; update the
diffusion module to import the shared version. The rewrite also removes the
inlined `np.bincount(np.digitize(...))` at `sorted_spikes_glm.py:488-491`.

Carry the `dt` terms from phase 6b — the matmul form there is the target shape.

---

## Task 2 — Bound `kde` JIT recompilation

`common.kde` (`common.py:332-338`) is jitted with `n_samples` in its shape
signature, and every unit has a different spike count. Verified: cache grows one
entry per distinct count.

Bucket sample counts and pad, carrying zero weights for padded rows so they drop
out of numerator and denominator. `log_kde` (`common.py:422-455`) already maps
zero weights to `-inf` and drops them from both sums; confirm linear `kde`
(`common.py:364-369`) does the same via its `jnp.sum(weights)` denominator.

**Bucket policy is not delegated — measure it.** Power-of-two buckets can nearly
double memory for a unit just above a boundary (a 513-spike unit padding to 1024).
Compare at least:

- powers of two,
- multiples of 256 above a floor of 256,
- powers of `sqrt(2)` rounded up.

on a realistic spike-count distribution (take one from a golden fixture rather
than inventing it). Choose by measured compile count **and** peak memory together,
and record the comparison in the PR. If no policy improves total wall-clock for a
realistic fit, drop this task — the compilation cost may be amortized already.

---

## Task 3 — Remove the per-block full-array copy in `clusterless_kde`

`block_estimate_log_joint_mark_intensity` (`clusterless_kde.py:137-158`) writes
each block with `dynamic_update_slice` into a full-size array, copying the whole
`(n_decoding_spikes, n_position_bins)` array per block. `common.block_kde` was
already migrated away from this, with the rationale at `common.py:411-414`.

```python
    blocks = [
        estimate_log_joint_mark_intensity(
            decoding_spike_waveform_features[start : start + block_size],
            encoding_spike_waveform_features,
            waveform_stds,
            occupancy,
            mean_rate,
            position_distance,
            encoding_weights,
        )
        for start in range(0, n_decoding_spikes, block_size)
    ]
    if not blocks:
        return jnp.zeros((0, n_position_bins))
    return jnp.clip(jnp.concatenate(blocks, axis=0), min=LOG_EPS, max=None)
```

**Caveat to measure, not assume:** the list holds every block alive until
`concatenate`, so peak is blocks + output ≈ 2× output, whereas
`dynamic_update_slice` peaks at output + one block. Whether this is a win depends
on whether XLA elides the copy. Measure peak memory both ways at
`n_decoding_spikes=5000, n_position_bins=900`; if the list form regresses peak,
keep `dynamic_update_slice` and instead donate the buffer, or drop the task.

The single final clip is existing behaviour (one floor, not per block) — preserve
it.

---

## Validation

| Test | Asserts | File |
|---|---|---|
| `test_matmul_matches_per_neuron_loop` | New non-local likelihood equals a reference per-neuron `xlogy` loop written in the test, `rtol=1e-5`. | `test_sorted_spikes_kde.py` |
| `test_interior_fields_are_floored` | Every interior field ≥ EPS at fit time, KDE and GLM — the matmul's precondition. | same + `test_sorted_spikes_glm.py` |
| `test_kde_bucketing_parity` | Padded and unpadded `kde`/`log_kde` agree, `rtol=1e-6`, for counts spanning two buckets. | `test_kde_common.py` |
| `test_kde_compile_count_bounded` | 32 units with 32 distinct counts produce ≤ 6 distinct `kde._cache_size()` entries. | same |
| `test_kde_padding_weights_are_zero` | Padded rows carry zero weight and do not shift the density. | same |
| `test_block_joint_intensity_parity` | Blocked equals unblocked exactly; a ragged final block is exercised. | `test_clusterless_kde.py` |

```bash
uv run pytest src/non_local_detector/tests/likelihoods -q
uv run pytest src/non_local_detector/tests/test_golden_regression.py -q
```

Goldens must be **unchanged** — every task is parity-preserving. A golden diff
means the rewrite is wrong, not that the baseline needs updating. Do not request
a snapshot-update approval for this phase.

## Review

Dispatch `code-reviewer` with the six measurement artefacts per task. Ask whether
task 2's padding can leak into any consumer reading `samples_`/`weights_` off a
`KDEModel` (`common.py:500-505`), and whether task 1 left the diffusion module
importing the shared `_spike_counts_matrix` rather than keeping a copy.

## Deliberately not in this phase

- Streaming the linear `clusterless_kde` position kernel — see
  [overview.md](overview.md#deferred-with-triggers).
- `GaussianMixture.score_samples` discarding responsibilities (`gmm.py:1068`).
