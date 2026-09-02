> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 2 — One GMM log-intensity policy

## Problem

The clusterless GMM log intensity is `log(rate) + log p(pos, mark) − log p(pos)`.
All three sites clamp the **numerator** to `LOG_EPS` before subtracting, which
destroys the cancellation that makes a tail ratio finite:

- `clusterless_gmm.py:722-724` — non-local, consumed by the combiner at `:99`
- `clusterless_gmm.py:754-759` — bin-tiled non-local
- `clusterless_gmm.py:876-878` — local, subtraction inline at `:883-885`

`LOG_EPS = log(1e-15) ≈ −34.54`. A joint density below that is routine for a
joint over position plus 4–6 mark dimensions at an atypical mark.

Reproduced:

```
realistic 6-D tail: log_joint=-60, log_occupancy=-8
  correct     : -52.00
  implemented : -26.54     -> 25.5 log units, ~1e11 likelihood ratio
```

The ground-process term has the same problem in probability space: it
exponentiates two separately-small densities and divides, non-local at
`clusterless_gmm.py:512-516` and local at `clusterless_gmm.py:898-900`.

**Note for the executor:** an earlier draft proposed
`maximum(log_rate + log_joint − log_occ, LOG_EPS)`. That is *also* wrong — for
the case above it returns `−34.54`, not `−52`. Flooring the ratio loses the
information just as flooring the numerator does. [C1](shared-contracts.md#c1--degeneracy-policy-for-log-intensities)
settles this: floor only `-inf`.

## Contracts referenced

- [C1 — Degeneracy policy](shared-contracts.md#c1--degeneracy-policy-for-log-intensities)

## Falsification

Before implementing, run this against your intended combiner:

```python
assert combiner(log_rate=0.0, log_joint=-60.0, log_occ=-8.0) == pytest.approx(-52.0)
assert jnp.isneginf(combiner(log_rate=0.0, log_joint=-jnp.inf, log_occ=-8.0)) is False
```

If the first fails, the policy is still a floor on a finite value. If the second
returns `-inf` rather than `LOG_EPS`, the degeneracy branch is missing.

## Tasks

### 1. Adopt the reference combiner

`clusterless_kde_log._log_joint_from_log_marginal`
(`clusterless_kde_log.py:41-99`) already implements [C1](shared-contracts.md#c1--degeneracy-policy-for-log-intensities)
correctly: it floors zero-occupancy bins and `-inf` marginals to `LOG_EPS`, and
deliberately excludes `NaN` so a broken computation reaches `core.py`'s
diagnostics. Read its docstring before writing anything.

Give `clusterless_gmm` the same policy in a shared helper, used by local and
non-local alike:

```python
def _log_intensity(
    log_rate: jnp.ndarray, log_joint: jnp.ndarray, log_occupancy: jnp.ndarray
) -> jnp.ndarray:
    """log(rate * p(pos, mark) / p(pos)), flooring only true zero mass.

    A finite value is preserved however small: flooring the finished ratio at
    LOG_EPS destroys a tail exactly as flooring log_joint did (log_joint=-60,
    log_occupancy=-8 gives -52, which max(-52, LOG_EPS) turns into -34.54).
    NaN is left untouched so it reaches core.py's diagnostics.
    """
    log_intensity = log_rate + log_joint - log_occupancy
    degenerate = (jnp.isneginf(log_intensity) | jnp.isneginf(log_occupancy)) & ~jnp.isnan(
        log_intensity
    )
    return jnp.where(degenerate, LOG_EPS, log_intensity)
```

Rewrite `_accumulate_log_likelihood_block` (`clusterless_gmm.py:89-102`) to call
it, and delete the three numerator clamps, keeping their reshapes:

- `:722-724` → `joint_logp_block = joint_logp_flat.reshape(block_size, n_bins)`
- `:754-759` → drop the `jnp.clip`, keep `.reshape(block_size, n_tile)`
- `:876-878` → `joint_logp = _gmm_logp(joint_gmm, eval_points)`, and at `:883-885`
  replace the inline expression with `_log_intensity(...)`

### 2. Ground-process term in log space, floored once

Both sites exponentiate small densities and divide. Replace with a log-space form
and — critically — **do not floor per electrode before summing**. Accumulate each
electrode's log intensity, combine across electrodes with `logsumexp`, then apply
the policy once.

Non-local, `clusterless_gmm.py:511-517`:

```python
        # Expected-counts term at bins, formed in log space so a bin where both
        # densities underflow keeps its finite ratio. Collected per electrode and
        # combined once below rather than floored here: a per-electrode floor
        # would add n_electrodes * EPS to an empty bin.
        log_gpi = _gmm_logp(gpi_gmm, interior_place_bin_centers)
        per_electrode_log_intensity.append(
            _log_intensity(safe_log(mean_rate, eps=EPS), log_gpi, log_occupancy)
        )
```

then after the electrode loop:

```python
    summed_ground_process_intensity = jnp.exp(
        jax.nn.logsumexp(jnp.stack(per_electrode_log_intensity, axis=0), axis=0)
    ) if per_electrode_log_intensity else jnp.zeros_like(log_occupancy)
```

Apply the analogous change to the local site at `clusterless_gmm.py:898-900`,
where `interp_pos` replaces `interior_place_bin_centers`.

### 3. CHANGELOG

`Changed` (numerical bound): clusterless GMM log intensities are no longer
floored at `LOG_EPS`; only true zero mass is floored. `Fixed`: the joint mark
density was clamped before dividing by occupancy, inflating the log intensity in
low-density bins by tens of log units. GMM decoding results change, most visibly
where the observed mark is atypical.

## Validation

New `src/non_local_detector/tests/likelihoods/test_clusterless_gmm_log_intensity.py`:

| Test | Asserts |
|---|---|
| `test_tail_ratio_preserved` | `_log_intensity(0, −60, −8)` is `−52` ± 1e-4. |
| `test_zero_mass_floors` | `log_joint = −inf` → `LOG_EPS`. |
| `test_zero_occupancy_floors` | `log_occupancy = −inf` → `LOG_EPS`. |
| `test_nan_propagates` | A NaN input yields NaN, not `LOG_EPS`, preserving the diagnostics contract. |
| `test_local_and_nonlocal_agree` (slow) | For a fixture with a deliberately atypical decode mark, local and non-local give the same per-spike log intensity at the animal's bin, ± 1e-4. |
| `test_ground_process_not_floored_per_electrode` | With N electrodes each contributing a degenerate bin, the summed intensity at that bin equals a single floor, not `N × EPS`. |

```bash
uv run pytest src/non_local_detector/tests/likelihoods -q
uv run pytest src/non_local_detector/tests/test_clusterless_likelihood_agreement.py -q
uv run pytest src/non_local_detector/tests/test_golden_regression.py -q
```

**Approval gate.** GMM goldens and snapshots move. Produce the four-part analysis
required by `.claude/skills/numerical-validation/SKILL.md`. The explanation must
show the diff is confined to bins where the pre-fix code clamped
(`log_joint < LOG_EPS`); if well-supported bins move, the change is wrong.

## Review

Dispatch `code-reviewer`. Ask whether any other module floors before forming a
ratio, and whether `_log_intensity` and
`clusterless_kde_log._log_joint_from_log_marginal` should now be one shared
function in `common.py` rather than two implementations of one policy.
