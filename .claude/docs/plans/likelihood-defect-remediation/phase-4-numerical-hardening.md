> **SUPERSEDED — DO NOT EXECUTE.**
> The implementation snippets below were written without being run and are known
> to be defective; see the readiness table in [PLAN.md](PLAN.md). The *problem
> statements and reproductions* in this file remain valid and are the reason the
> phase exists. Everything under "Tasks" must be re-derived by prototyping
> against the real code before this phase can ship.

# Phase 4 — Numerical hardening

Three defect clusters. GLM zero-exposure moved to phase 1 (it is reachable as
soon as the mask fix lands).

---

## Defect 1 — Compensated log-KDE NaN when the first tile is fully de-weighted

`_compensated_linear_marginal_chunked` seeds its online maximum with `-inf`
(`clusterless_kde_log.py:869-871`). If the first encoding tile has only zero
weights, `log_w_chunk` is all `-inf`, `chunk_total` is all `-inf`
(`clusterless_kde_log.py:846`), and `chunk_max` is `-inf`.

Reproduced (tile size 4, first 4 of 8 weights zero):

```
tiled   has NaN: True        untiled has NaN: False
control (zeros in the *second* tile) has NaN: False
```

The `all_zero_weight` guard at `clusterless_kde_log.py:893` only covers the case
where *every* weight is zero, so the partial case escapes. Reachable during EM
whenever a state de-weights the earliest encoding spikes.

### Falsification

An earlier draft guarded only the `running_sum` rescale at
`clusterless_kde_log.py:851-852`. **That does not fix the bug.** Verified:

```
running_sum rescale guarded OK : 1.0
log_scale = chunk_total - new_max -> [nan nan]      <- line 860, unguarded
sqrt_scale                        -> [nan nan]
```

`new_max` stays `-inf`, so `log_scale = chunk_total − new_max` is `-inf − (-inf)`
and `W`/`P` are still NaN. Before implementing, run the `test_zero_weight_first_tile`
test against your patch; if it passes only because of the `all_zero_weight` guard,
you have not fixed it.

### Fix

Guard **both** the running rescale and the per-chunk scale construction. A chunk
with no finite rows must contribute nothing and must not move the running max:

```python
        # A chunk whose rows are all -inf (fully de-weighted tile, or all
        # padding) contributes nothing. Both the running rescale and this
        # chunk's own scale must avoid -inf - (-inf): guarding only the former
        # still yields NaN in log_scale below.
        chunk_is_empty = jnp.isneginf(chunk_max)
        new_max = jnp.where(
            chunk_is_empty, running_max, jnp.maximum(running_max, chunk_max)
        )
        rescale = jnp.where(
            jnp.isneginf(running_max) | (new_max == running_max),
            1.0,
            jnp.exp(running_max - new_max),
        )
        running_sum = running_sum * rescale

        ...

        # Rows with no mass get log_scale = -inf (sqrt_scale = 0), never NaN.
        log_scale = jnp.where(
            jnp.isneginf(chunk_total) | chunk_is_empty, -jnp.inf, chunk_total - new_max
        )
        sqrt_scale = jnp.exp(0.5 * log_scale)
```

Keep the `all_zero_weight` guard; with this fix it is redundant for correctness
but still short-circuits a fully empty electrode. Say so in a comment.

---

## Defect 2 — diag/spherical Mahalanobis catastrophically cancels in float32

`_estimate_log_gaussian_prob` uses the expanded `Σμ²p − 2xᵀ(μp) + x²ᵀp` for
`diag` (`gmm.py:408-414`) and `spherical` (`gmm.py:415-421`). JAX is float32 here
(x64 is not enabled anywhere in the package).

Reproduced — unit variance, true squared distance exactly 1.0:

```
|mu|=1e2   maha =     1.0000     |mu|=1e4   maha =     0.0000
|mu|=1e3   maha =     1.0000     |mu|=1e5   maha = -2048.0000
```

A negative squared distance is impossible and inflates the log density without
bound. Raw waveform amplitudes in µV reach 1e4 routinely — the default
`waveform_std=24.0` implies that scale.

### Fix

Centred differences, using `lax.map` to match the deliberate choice in the `full`
branch (`gmm.py:386-398`, whose comment records that `vmap` materializes
`(K, N, D)` and OOMs on GPU — do not "optimize" it back):

```python
    elif covariance_type == "diag":
        # Centred difference rather than x^2 - 2*x*mu + mu^2: in float32 the
        # expanded form catastrophically cancels away from the origin (a true
        # squared distance of 1 comes out as -2048 at |mu|=1e5), producing
        # impossible negative Mahalanobis distances.
        def comp_diag(args: tuple[Array, Array]) -> Array:
            mu, p = args                      # (D,), (D,)
            Y = (X - mu) * p
            return jnp.sum(Y * Y, axis=1)

        maha = jax.lax.map(comp_diag, (means, precisions_chol)).T
    else:  # spherical: precisions_chol is (K,), so p is scalar per component
        def comp_spherical(args: tuple[Array, Array]) -> Array:
            mu, p = args                      # (D,), ()
            Y = (X - mu) * p
            return jnp.sum(Y * Y, axis=1)

        maha = jax.lax.map(comp_spherical, (means, precisions_chol)).T
```

---

## Defect 3 — the weighted GMM is not scale-invariant

`nk = resp.sum(axis=0) + eps` with `eps = 10 * finfo(X.dtype).eps`
(`gmm.py:343-344`) is a **fixed absolute** floor. Under small sample weights it
dominates `nk`, and `nk` divides the means (`gmm.py:345`), so the fitted density
depends on the *scale* of the weights — which is meaningless.

Reproduced (two clusters at 5 and 25):

```
weight scale 1        means [ 5.008 24.956]   sum(w)=1.000
weight scale 1e-06    means [ 4.978 24.808]   sum(w)=1.006
weight scale 1e-09    means [ 0.000  3.764]   sum(w)=6.960
```

An earlier draft proposed only `weights = nk / jnp.sum(nk)`. That restores a
unit-sum mixture while leaving the means destroyed — it fixes the symptom that is
easy to assert, not the defect.

The same bad denominator appears in initialization (`gmm.py:1218-1221`), and
KMeans initialization ignores `sample_weight` entirely (`gmm.py:1155-1158`).

### Falsification

Assert **scale invariance of the whole estimator**, not `sum(weights_)`:
fit at weight scales `1`, `1e-6`, `1e-9` and require `means_`, `covariances_`,
`weights_`, and `score_samples(X)` to agree within `rtol=1e-4`. A patch that only
normalizes `weights_` fails this immediately.

### Fix

Normalize sample weights to mean 1 before EM, so the epsilon floor is always
relative to O(1) responsibilities:

```python
    # Normalize to mean 1 so the fixed epsilon floor in _estimate_gaussian_parameters
    # stays negligible regardless of the caller's weight scale. A GMM density is
    # invariant to a global rescale of sample weights; without this, weights of
    # 1e-9 drive the means toward zero.
    if sample_weight is not None:
        sw_mean = jnp.mean(sample_weight)
        sample_weight = jnp.where(sw_mean > 0, sample_weight / sw_mean, sample_weight)
```

applied once in `fit` (`gmm.py:938-940`) so every downstream consumer —
`_initialize_parameters`, `_fit_single`, `_em_fit_while_loop` — sees the
normalized array. Then `weights = nk / jnp.sum(nk)` in `_m_step_func`
(`gmm.py:546-551`) for exact unit sum, and the same in `_initialize_parameters`
(`gmm.py:1218-1221`). Delete the now-unused `total_weight` lines.

Pass weights to KMeans (`gmm.py:1155-1158`): `sklearn.cluster.KMeans.fit` accepts
`sample_weight`. Confirm the installed sklearn's signature before relying on it.

### Also in scope — weighted EM objective

`lb = jnp.mean(log_prob_norm)` (`gmm.py:608`) is unweighted, but the M-step is
weighted. It drives the convergence test, `n_init` restart selection, and the
reported `lower_bound_`. Replace with the weighted objective:

```python
        lb = (
            jnp.mean(log_prob_norm)
            if sample_weight is None
            else jnp.sum(sample_weight * log_prob_norm) / jnp.sum(sample_weight)
        )
```

The pre-M-step evaluation is sklearn's own convention and is **not** changed.

---

## Validation

| Test | Asserts | File |
|---|---|---|
| `test_zero_weight_first_tile_is_finite` | Tiled equals untiled and is finite when tile 1 is fully de-weighted. | `test_clusterless_kde_log_optimization.py` |
| `test_all_zero_weight_electrode_floors` | Fully de-weighted electrode still yields `LOG_EPS`, not NaN. | same |
| `test_maha_nonnegative_at_large_coordinates` | `diag` and `spherical`: recovered Mahalanobis ≥ 0 and within `rtol=1e-4` of 1.0 at \|μ\| = 1e2…1e5. | `test_gmm.py` |
| `test_diag_matches_full_covariance` | A `diag` fit equals a `full` fit with diagonal covariance on `score_samples`, ± 1e-4. | same |
| `test_gmm_scale_invariance` | `means_`, `covariances_`, `weights_`, `score_samples` agree across weight scales 1 / 1e-6 / 1e-9, `rtol=1e-4`. | same |
| `test_mixture_weights_sum_to_one` | `sum(weights_)` is 1.0 ± 1e-6 at every scale. | same |
| `test_kmeans_init_uses_weights` | Zero-weight samples do not influence initial centers. | same |
| `test_weighted_lower_bound_matches_objective` | `lower_bound_` equals `sum(w·logp)/sum(w)` for the returned parameters' predecessor, not the unweighted mean. | same |

Baseline-vs-after for defect 2: before editing, record `score_samples` on a
well-conditioned `diag` fixture (coordinates ~O(1)); after, assert `rtol=1e-5`.
The fix must move ill-conditioned results only.

```bash
uv run pytest src/non_local_detector/tests/likelihoods -q
uv run pytest src/non_local_detector/tests/test_clusterless_gmm_optimization.py -q
uv run pytest src/non_local_detector/tests/test_golden_regression.py -q
```

**Approval gate.** GMM goldens may move where coordinates are large or weights
small. Produce the four-part numerical-validation analysis.

## Review

Dispatch `code-reviewer`. Ask whether `lax.map` preserved the memory
characteristics the `full` branch documents, and whether weight normalization
reaches *every* consumer of `sample_weight` — a path that sees raw weights while
another sees normalized ones would be worse than the original defect.
