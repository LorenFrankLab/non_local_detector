> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 6b — Rates in Hz, intensities scaled by bin duration

Depends on phase 6a (needs `bin_durations` and `sample_cell_durations`).

## Contracts referenced

- [C3 — Time vocabulary](shared-contracts.md#c3--time-vocabulary)

## Problem

`weighted_mean_rate` (`common.py:185-202`) returns spikes per weighted *position
sample*. `sampling_frequency` is accepted and ignored by every fit;
`clusterless_kde_log.py:1389-1400` is the only place the assumption is written
down. A decode grid whose spacing differs from the position sampling interval is
mis-calibrated:

```
position fs= 30 Hz  mean_rate=0.166667/sample -> 5.000 Hz
position fs=500 Hz  mean_rate=0.010000/sample -> 5.000 Hz
encode at 30 Hz, decode at 500 Hz:
  expected spikes per decode bin (truth): 0.01
  model's summed intensity per bin      : 0.1735   (17.4x)
```

## Falsification

**Updating `weighted_mean_rate` callers is not sufficient.** Several backends
store rates that never pass through it:

- GLM `coefficients` encode expected counts per encoding sample
  (`sorted_spikes_glm.py:351` builds `place_fields` as `exp(X @ coef)`).
- MRF occupancy offsets are weighted sample counts (`sorted_spikes_mrf.py:823-827`).
- Diffusion exposure fields come from `pixellate_interior_fields`, also in sample
  counts.

Multiplying those by decode `dt` converts nothing — it applies an *extra* time
factor to a quantity that is already per-sample. Before implementing, write
`test_rate_invariant_to_position_sampling_rate` for **every** backend and confirm
each fails on `main` for the right reason. A backend that passes before the fix
is not yet converted.

## Tasks

### 1. Build the backend matrix first

Write this table into the PR description and fill it before editing anything.
It is the phase's real deliverable; the code follows from it.

| Backend | Fit exposure units | Stored rate/field units | Spike term `log(dt)` | Expected-count `dt` | Local formula | Non-local formula |
|---|---|---|---|---|---|---|
| `sorted_spikes_kde` | | | | | | |
| `sorted_spikes_glm` | | | | | | |
| `sorted_spikes_diffusion` | | | | | | |
| `sorted_spikes_mrf` | | | | | | |
| `clusterless_kde` | | | | | | |
| `clusterless_kde_log` | | | | | | |
| `clusterless_gmm` | | | | | | |
| `clusterless_diffusion` | | | | | | |
| `no_spike` | | | | | | |

Target for every row: exposure in **seconds**, stored rate in **Hz**, spike term
carries `+ k·log(dt)`, expected-count term carries `× dt`.

### 2. Exposure in seconds

`weighted_mean_rate` (`common.py:185`) takes a required `exposure_duration` and
returns Hz. Every fit computes it as

```python
    exposure_duration = float(np.sum(weights * sample_cell_durations(position_time)))
```

**not** `weight_sum * median(np.diff(position_time))`, which is wrong for
jittered or gapped timestamps. Store it in the encoding dict as
`encoding_exposure_duration`.

Grep for `weighted_mean_rate` to get the call sites rather than trusting a list.

### 3. Convert the backends that bypass `weighted_mean_rate`

- **GLM**: the design matrix rows are position samples of width
  `sample_cell_durations(position_time)`. Add `log(sample_dt)` as an offset in
  `fit_poisson_regression` so the fitted intercept is a log-Hz rate:
  `eta = X @ coef + log(sample_dt)`. Then `place_fields = exp(X_pred @ coef)` is
  already Hz and needs no post-hoc scaling.
- **MRF**: `occupancy_field` becomes `sum(weights * sample_dt)` per bin rather
  than a weighted count, so the Poisson offset is an exposure in seconds.
- **Diffusion**: same change inside `pixellate_interior_fields`.
- **`no_spike`**: `no_spike_rate` is already documented as Hz
  (`no_spike.py:41-43`); replace the median-interval multiply at `:79` with
  per-bin `dt`. This backend is the one already using the target convention —
  after this phase it stops being the odd one out.

### 4. Scale at predict

Each predictor computes `dt = bin_durations(time_edges)` once. Sorted non-local:

```python
    dt = jnp.asarray(bin_durations(time_edges))[:, None]      # (n_bins, 1)
    log_likelihood = (
        spike_counts @ log_interior_fields
        + spike_counts.sum(axis=1, keepdims=True) * jnp.log(dt)
        - dt * no_spike_part_log_likelihood[is_track_interior]
    )
```

The `Σ_n k_n · log dt` term is the part of `xlogy(k, λ·dt)` the matmul does not
carry. Derive the equivalent per backend from the matrix in task 1 rather than
copying this shape; the clusterless paths scale
`summed_ground_process_intensity` and each spike's joint intensity instead.

### 5. Remove the dead `sampling_frequency` parameters

Seven fits accept and ignore it. Either use it or delete it — leaving it is what
made this defect invisible. With exposure now derived from `position_time`, delete.

### 6. CHANGELOG

`Changed` (breaking): encoding rates are Hz, and Poisson intensities are scaled by
each decoding bin's duration. Previously rates were per position sample and
results were only calibrated when the decode bin width equalled the position
sampling interval — a 30 Hz position grid decoded at 500 Hz overstated intensity
by 17×. `sampling_frequency` is removed from the encoding-fit signatures.

## Validation

| Test | Asserts |
|---|---|
| `test_rate_invariant_to_position_sampling_rate[backend]` | Same spikes fit against 30 Hz and 500 Hz position grids give the same Hz `mean_rates` ± 1e-6. **Parametrized over every backend** — this is the falsification check. |
| `test_intensity_scales_with_bin_width[backend]` | Doubling decode bin width doubles the expected-count term. |
| `test_decode_dt_differs_from_position_dt` (slow) | End-to-end: 30 Hz position, 500 Hz decode; the summed intensity per bin matches the true expected count ± 5%. This is the 17.4× reproduction inverted. |
| `test_exposure_handles_gapped_timestamps` | With a gap in `position_time`, exposure equals the sum of sample-cell widths, not `n_samples × median_dt`. |
| `test_glm_offset_gives_hz_place_fields` | GLM `place_fields` for a known-rate simulated neuron recover the true Hz rate ± 5%. |

```bash
uv run pytest src/non_local_detector/tests -q
uv run pytest -m snapshot -q
```

**Approval gate.** All goldens move. In the four-part analysis, show that for a
fixture where decode `dt` equals `position_dt`, the change is a pure rescale by a
known constant — compute it and check it matches `dt` exactly. If the ratio is not
constant across bins for a uniform grid, a `dt` factor has been applied twice
somewhere.

## Review

Dispatch `code-reviewer` with the completed backend matrix. Ask them to verify
each row independently against the code rather than accepting the table.
