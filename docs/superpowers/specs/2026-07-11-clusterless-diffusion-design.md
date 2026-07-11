# clusterless_diffusion — design

**Date:** 2026-07-11
**Status:** Approved design (pre-implementation)
**Scope:** A new opt-in clusterless likelihood, `clusterless_diffusion`. A sibling `clusterless_mrf` (penalized-Poisson GAM on the same spectral engine) is a planned follow-up with its own spec.

## Summary

Add a clusterless (marked-point-process) likelihood that estimates the same
mark intensity as `clusterless_kde` but replaces the **spatial** Gaussian kernel
with the environment's **graph heat kernel** `exp(-tL)` (`t = position_std**2 / 2`).
The mark kernel stays a Gaussian over waveform features. This is the clusterless
analog of `sorted_spikes_diffusion` (which is the diffusion analog of
`sorted_spikes_kde`).

The single substitution buys two things from one mechanism:

- **(A, primary) Geometry-respecting spatial smoothing.** On environments with
  non-trivial 2D geometry (barriers, holes, multi-arm mazes) `clusterless_kde`
  smooths marks in Euclidean space and leaks density across walls. Diffusion
  follows the manifold graph, so a spike's spatial evidence respects track
  topology.
- **(B, very important) Performance.** The heat kernel is a cached low-rank
  operator `Q (exp(-tΛ) ⊙ (Qᵀ ·))`. Predict becomes one rank-`m` matmul on the
  mark-weighted bin histograms instead of `clusterless_kde`'s pairwise
  `O(n_bins · n_enc · n_decode)` position kernel.

(C, principled occupancy / less tuning) is a nice-to-have, not a goal.

## Non-goals / scope decisions

- **Prob-space first.** Ship the linear-space version now. A log-space variant
  (`clusterless_diffusion_log`, parallel to `clusterless_kde_log`) is a likely
  follow-up but is **not** in this spec. Diffusion is inherently a linear-space
  matmul, so a log-space path would exp/log around each diffusion — extra cost
  that fights goal B; defer it.
- **`clusterless_rf`** is a separate modeling paradigm (non-parametric, tree-based,
  CPU/sklearn rather than JAX) and gets its own spec later.
- **Default unchanged.** `clusterless_kde` remains the default clusterless
  algorithm; this is purely additive and opt-in via
  `clusterless_algorithm="clusterless_diffusion"`.
- **New module** `likelihoods/clusterless_diffusion.py` with its own registry
  entry (Approach 1) — mirrors the one-module-per-algorithm convention and the
  `sorted_spikes_diffusion` ↔ `sorted_spikes_kde` relationship. Shared bits are
  factored into `likelihoods/common` where clean.

## Background

- **Marked point process (shared with `clusterless_kde`).** For electrode `e`,
  over the decode window:

  ```
  log L_e = Σ_{j: decode spikes on e} log λ_e(x, m_j)  −  ∫ λ̄_e(x) dt
  λ_e(x, m) = μ_e · p_e(x, m) / π(x)      (joint mark intensity)
  λ̄_e(x)    = μ_e · p_e^gpi(x) / π(x)      (ground-process / mark-marginalized)
  ```

  where `x` ranges over interior bins, `m` is the waveform mark, `π(x)` is
  occupancy, `μ_e` the mean rate.

- **Diffusion engine (`likelihoods/diffusion.py`).** Builds a finite-difference
  symmetric graph Laplacian `L` on interior bins and applies the heat kernel in
  the eigenbasis: `diffuse(eigvals, eigvecs, sigma, fields) = Q (exp(-tΛ) ⊙ (Qᵀ F))`
  for `fields` of shape `(n_bins, n_fields)`. The eigenbasis is component-aware,
  bandwidth-aware auto-truncated, and **cached on the `Environment`** (built once,
  reused across electrodes and EM refits). It is a coordinate-independent Gaussian
  of std `sigma` that respects graph geometry.

## Design

### 1. Core math — how diffusion enters the intensity

Only the **spatial** estimator changes. Replace the Gaussian position kernel
`N(x; x_i, σ_pos²)` with the graph heat kernel `H_t = exp(-tL)`,
`t = position_std²/2`. The mark kernel `K_mark` stays Gaussian (`waveform_std`).

For a decode spike `j` with mark `m_j`, using the per-encoding-spike EM weights
`w_i` (interpolated at encoding-spike times, exactly as `clusterless_kde` does):

```text
D_e[:, j]   = Σ_i w_i · K_mark(m_j, m_i) · onehot(bin(x_i))          # (n_bins × n_decode)
p_e(x, m_j) = [H_t · D_e][x, j] / ( (Σ_i w_i) · ΔV(x) )
```

- `D_e` is the **weight-and-mark-weighted** encoding-spike histogram over interior
  bins. Weights are load-bearing: partial EM weights must scale each encoding
  spike's contribution and divide by `Σ_i w_i` (matching `clusterless_kde`'s
  weighted marginal), or `p_e` is wrong.
- `K_mark` reuses `kde_distance` — **its Gaussian normalizer is already included**,
  so do not re-apply a mark normalizer.
- `H_t` is applied with the engine's mass-conserving, component-aware clip+rescale
  (§4), so `Σ_x (H_t D_e)[x,j] = Σ_i w_i K_mark(m_j, m_i)`.
- `ΔV(x)` is the per-bin measure (uniform grid → constant; linearized track →
  per-bin width, from the environment). Dividing by `(Σ_i w_i)·ΔV(x)` — **not** a
  normalize-to-unit-integral — makes `p_e` a proper *joint* density whose spatial
  integral recovers the weighted mark marginal:
  `∫ p_e(x, m_j) dx = Σ_i w_i K_mark(m_j, m_i) / Σ_i w_i = p_e(m_j)`.
  (Per-column normalization to `∫=1` would erase that marginal and is incorrect.)

Decode-independent pieces — **occupancy** `π(x)` and the **ground-process** field
`p_e^gpi(x)` (mark-marginalized) — are weighted-histogrammed and diffused **once at
fit**, like `sorted_spikes_diffusion`. The only per-decode-spike work is building
`D_e` (weighted mark kernel + scatter, `O(n_enc · n_decode)`) and one low-rank
diffusion `H_t D_e`.

**Correctness anchor:** on a 1D / open-field grid where the heat kernel ≈ a
Euclidean Gaussian, this reduces to `clusterless_kde` (agreement, §Testing); on a
barrier track it does not leak across walls.

### 2. Fit (encoding model)

Mirrors `clusterless_kde`'s encoding dict; diffusion parts replace the position
machinery.

**Inputs:** `position_time, position, spike_times, spike_waveform_features,
environment, sampling_frequency, position_std, waveform_std, weights` (EM),
`heat_kernel_rank` (truncation, default auto), `block_size` (requested cap) and
`memory_budget` (per-block byte budget, **default `536_870_912` = 512 MiB**; drives
the effective block size, §3), `disable_progress_bar`.

**Computed once at fit:**

- **Eigenbasis** — built once per `Environment`, reused across electrodes and EM
  refits (not re-decomposed per fit); cached on `environment`. Both cache functions
  return `(eigvals, eigvecs)` as **NumPy** (not the rank). To avoid a mode-dependent
  retrieval contract, **fit resolves the basis once and records the rank from it**:
  - Auto (default, `heat_kernel_rank is None`): `eigvals, eigvecs = cached_heat_kernel_eigenbasis(environment, position_std)`
    (2nd arg is **σ**, bandwidth-aware).
  - Explicit rank: `eigvals, eigvecs = cached_eigenbasis(environment, heat_kernel_rank)`.
  Then `resolved_rank = eigvecs.shape[1]` (stored). Predict does **not** call
  `cached_eigenbasis` directly — it goes through the single device-basis helper below.
  - **Device residency (goal B) — one retrieval helper.** The host basis is NumPy but
    predict does a JAX matmul, and converting `Q (n_bins × rank)` host→device *per
    fit* would repeat the transfer on every EM refit and duplicate `Q` across encoding
    groups sharing one environment. Define **one helper**, `get_device_basis(environment,
    resolved_rank)`:
    1. Look up the `Environment`'s transient device cache (`_diffusion_device_basis_`,
       a dict **keyed by `(resolved_rank, device, dtype)`**).
    2. **On hit:** return the device `(Λ, Q, component_labels)`.
    3. **On miss only** (first predict, post-load, post-refit-refit, or new device):
       retrieve the host basis via `cached_eigenbasis(environment, resolved_rank)`,
       convert to `jnp` on the current device, store under the key, return.

    So predict *always* calls `get_device_basis`; the host retrieval + conversion
    happens **only** on a device-cache miss. Keying by `device`/`dtype` returns arrays
    on the *right* device under a different JAX context (no silent cross-device
    transfer). Safe to key by `Device` because the cache is transient (below) — the
    unpickleable `Device` never reaches `pickle`. Shared across encoding groups and EM
    refits (parallel to `_diffusion_eigenbasis_`); the encoding model holds only
    `resolved_rank`, not its own `Q`. Two lifecycle hooks (both tested, §Testing):
    - **Invalidation:** `Environment.fit_place_grid()` already `del`s the diffusion
      graph/Laplacian/eigenbasis/rank caches when the grid is rebuilt; the device
      cache **must be `del`'d there too**. Note this only protects a *new* fit on the
      rebuilt grid — an existing encoding model is itself grid-bound and stale after a
      refit (its occupancy, `node_order`, bin indices, and ground-process field all
      describe the old grid), so it must be **re-fit**, not merely re-predicted (§3,
      §Testing). Optionally, predict records the environment's grid "generation" and
      raises a clear error on mismatch rather than silently using stale arrays.
    - **Serialization:** `Environment` is pickled directly (`save_model` pickles the
      whole detector + its environments), and JAX arrays / the device cache must
      **not** be pickled. Add `Environment.__getstate__` to drop the transient device
      cache; it is lazily rebuilt from the host basis on the next predict after load.
- **Diffused occupancy** `π(x) = H·O(x) / (Σ w_pos · ΔV(x))` `(n_interior,)`, where
  `O` is the weighted occupancy bin-histogram and `Σ w_pos` its total weight —
  the **same** density convention as `p_e` (§1), applied via `heat_kernel_apply`
  (clip + per-component mass-rescale). **Not** `to_density` (which forces `∫=1`, a
  different convention that would break parity with `p_e`, esp. on non-uniform
  linearized bins). **Safe branch:** all-zero EM weights make `Σ w_pos == 0` (a
  supported degeneracy); use a safe denominator (`where(Σ w_pos > 0, …, 1)`) so the
  fit stays finite (a degenerate model whose electrodes are all zero-rate, below),
  and emit a **diffusion-specific `UserWarning`** — note `clusterless_kde` accepts
  all-zero occupancy weights *silently*, so this is a new, intentionally-surfaced
  warning (asserted in §Testing), not "matching KDE".
- **`summed_ground_process_intensity`** `(n_interior,)` —
  `Σ_e μ_e · p_gpi_e(x) / π(x)` with `p_gpi_e(x) = H·S_e(x) / (Σ w_e · ΔV(x))`
  (`S_e` = weighted, mark-marginalized encoding-spike histogram), floored at `EPS`.
  **Safe branch:** an electrode with no effective spikes has `Σ w_e == 0`, so
  `p_gpi_e` is `0/0`; that electrode is marked **zero-rate at fit** (`weight_total_e = 0`,
  `μ_e = 0`), contributes **0** to the sum (skip its division), and **warns** — the
  fit-side of the §3 predict zero-rate guard, matching `clusterless_gmm`'s
  per-electrode zero-rate handling. Decode-independent (same key/role as in
  `clusterless_kde`). **`ΔV(x)` cancels** in every `p/π` ratio (`p_e/π`,
  `p_gpi_e/π`), so the *likelihood* is independent of the (possibly non-uniform)
  bin measure — `ΔV` matters only for the density's marginal property (§Testing),
  not the decode. All three densities (`π`, `p_gpi`, `p_e`) share one convention.
- **Per electrode:** `encoding_marks` (features, `n_enc_e × n_mark`),
  `encoding_bin_indices` (interior-bin index of each encoding spike, precomputed
  so predict skips bin lookup), **`encoding_weights`** (per-spike EM weights `w_i`,
  interpolated at encoding-spike times) and their **`weight_total`** `Σ_i w_i`
  (or store weights and let predict sum), and weighted `mean_rate` `μ_e`.

**Weights (EM):** occupancy, per-electrode histograms, mean rates, **and the
per-spike joint estimator `D_e`** are all weighted by `weights` (as in
`sorted_spikes_diffusion` / `clusterless_kde`), so EM refits produce the correct
`p_e(x,m)` and a fully de-weighted electrode is detectable (§4). A binary-weight
fit must equal a hard-subset fit (test in §Testing).

**Stored dict:** keeps `clusterless_kde`'s shared keys (`occupancy`,
`mean_rates`, `summed_ground_process_intensity`, bandwidths, `environment`);
swaps `encoding_positions` → `encoding_bin_indices`; adds per-electrode
`encoding_weights` / `weight_total`, `resolved_rank` + `node_order`
(interior ↔ full-grid mapping), and the memory knobs **`block_size`** (requested
cap) and **`memory_budget`**. Predict receives settings *only* via `**encoding_model`
(base prediction does not pass fit params separately), so any user-set
`memory_budget` / `block_size` **must** be stored here — otherwise a predict-time
override is silently lost and reverts to the default (P1). The eigenbasis is **not**
in the dict: the host NumPy basis is cached on `environment` (`_diffusion_eigenbasis_`);
the transient device basis is cached on `environment` keyed by `(resolved_rank,
device, dtype)`, retrieved by predict *only* through `get_device_basis` (§2, Device
residency — invalidated on grid refit, excluded from pickling).

### 3. Predict

Both paths mirror `clusterless_kde`'s predict, prob-space, with `safe_log`
flooring to `LOG_EPS`.

**Non-local** → `(n_time, n_bins)`:

```text
ll(x, t) = −summed_ground_process_intensity(x)          # precomputed, broadcast over time
for each electrode e:
    if weight_total_e == 0 (zero-rate: μ_e == 0):        # zero weights OR zero encoding spikes
        ll += segment_sum(LOG_EPS per decode spike → time bins)   # floor, do NOT divide (§4 contract)
        continue
    K    = K_mark(m_j, m_i)                              # (n_decode_e × n_enc_e), via kde_distance
    D_e  = scatter (w_i · K) onto bins via encoding_bin_indices   # (n_bins × n_decode_e), EM-weighted
    P_e  = heat_kernel_apply(Q, Λ, σ, D_e)              # low-rank matmul + component-aware clip & mass-rescale
    p_e  = P_e / ( weight_total_e · ΔV(x) )             # weight_total_e > 0 here; joint density
    lc   = safe_log(μ_e · p_e / π(x))                    # (n_bins × n_decode_e), floored to LOG_EPS
    ll  += segment_sum(lc, over spikes → time bins)
```

**Zero-rate guard (mirrors `clusterless_gmm` / `clusterless_kde`):** a zero-weight or
zero-encoding-spike electrode has `weight_total_e == 0`, so `D_e == 0` and
`P_e / weight_total_e` is `0/0 → NaN` — which `safe_log` does **not** repair, and
`μ_e == 0` does not cancel. So that electrode is floored to `LOG_EPS` per observed
decode spike **before** any division (exactly the merged `clusterless_gmm` zero-rate
handling); it is not skipped. For the general (`weight_total_e > 0`) path, still use
a double-`where` safe denominator as belt-and-suspenders.

`heat_kernel_apply` is a **JAX port of `diffusion.diffuse`** and must replicate its
semantics exactly: `Q (exp(−tΛ) ⊙ (Qᵀ D_e))`, then clip to ≥ 0, then rescale each
column back to its input mass **per graph component** (`component_labels`).
Clipping alone is not enough — without the mass-rescale the likelihood *magnitude*
depends on the truncation rank (see §4). The matmul is the entire spatial cost
(rank-`m`, GPU-friendly).

**Local** → `(n_time, 1)`: for each decode spike, evaluate the joint at the
animal's position at that spike's time. Because the clip+mass-rescale is a
per-column (over-all-bins) nonlinearity, the local value comes from the **same**
diffused column as non-local — local builds `D_e`, applies the identical
`heat_kernel_apply`, and **indexes the animal's nearest interior bin**
`bin(x_a(t_j))` for spike `j`; then `segment_sum` over spikes → time bins.

The exact identity is **per spike**: spike `j`'s local contribution equals the
non-local value at `(t_j-bin, bin(x_a(t_j)))`. It is **not** a per-time-bin
identity — within one decode time bin the animal can move across bins, so different
spikes index different columns and no single non-local column equals the local
time-bin sum. The §Testing equality is therefore asserted **per spike** (or, as a
coarser check, on single-spike / stationary time bins). Interpolation is
**nearest-bin** for the first cut; barrier-safe *linear* interpolation (as
`sorted_spikes_diffusion` supports) is a documented follow-up. The
`h_j·(Gᵀ(w∘K_j))` single-bin shortcut with `G_i = exp(−tΛ)⊙Q[bin_i]` (derived at
predict from the cached eigenbasis + `encoding_bin_indices`, **not stored**) is a
valid optimization *only* when clipping is inactive (high/full rank, no negative
lobes); it is not the primary path. The ground-process term is the fit-time
diffused field looked up at `x_a(t)`, matching `clusterless_kde`'s local
`summed_expected_counts`.

**Memory / block size:** block over decode spikes. The **inherited clusterless
default block (`10_000`) is unsafe here** — large grids are the primary use case,
and the per-block live set is dominated by several `block`-scaled arrays, not just
one. Count them all with the **dtype item size** (not a hardcoded 4): per block,
`K` (`n_enc × block`), `D_e`/`P_e` (`n_bins × block`), the spectral projection
`Qᵀ D_e` (`rank × block`), the log-intensity `lc` (`n_bins × block`), and the
clip/rescale temporaries (~`n_bins × block`). So

```text
bytes_per_col ≈ itemsize · (n_enc + rank + c · n_bins)      # c = 4 (D, P, lc, temporaries)
effective_block = clip( floor(memory_budget / (bytes_per_col · safety)), 1, requested_block )
```

**Concrete policy:** `c = 4`; `safety = 2` (headroom for XLA workspace);
`itemsize` from the working dtype (float32 → 4). `memory_budget` is a new **fit /
predict parameter**, default **exactly `536_870_912` bytes (512 MiB)**, a positive
`int` users can raise/lower (validated, §4); `effective_block` is `floor`-rounded and
`≥ 1`, and logged. Never silently use the inherited `10_000` on a large grid —
resolve it through this policy.

### 4. Bandwidths, normalization, degeneracy policy

**Bandwidths:**
- `position_std` → heat-kernel σ (`t = σ²/2`): physical, grid-independent, follows
  geometry.
- `waveform_std` → mark Gaussian — unchanged from `clusterless_kde`.
- `heat_kernel_rank` → eigenbasis truncation. Default is the diffusion engine's
  **tolerance-based** auto-selection (retain modes whose heat-kernel weight
  `exp(-tλ)` exceeds a tolerance; fall back to the dense basis past a fraction) —
  it does **not** cap-and-warn (unlike `sorted_spikes_mrf`'s rank cap; the engine
  has no such warning). No new rank warning is introduced. (If a hard cap + warning
  is ever wanted, it must be defined explicitly; out of first-cut scope.)

**Normalization (spelled out in §1):** `p_e(x, m_j) = (H_t D_e)[x,j] / ((Σ_i w_i)·ΔV(x))`.
Three rules make this a correct *joint* density that matches KDE on simple geometry:
1. The **mark** normalizer lives in `kde_distance` (`K_mark`) — do **not** add another.
2. Normalize by `Σ_i w_i` (the weighted encoding count), **not** `n_enc`, and by the
   per-bin measure `ΔV(x)` (uniform → constant; linearized track → per-bin width).
3. **Do not** normalize each diffused column to `∫=1`. Because `H_t` conserves mass,
   this convention gives `∫ p_e(x,m_j) dx = p_e(m_j)` (the weighted mark marginal) —
   forcing unit integral would erase that marginal and inflate low-mass marks.

**Negative lobes / mass-rescale:** a truncated heat kernel can produce non-tiny
negative lobes; clipping to ≥ 0 alone inflates the total, so the diffused column
must be **rescaled back to its input mass, per graph component** — exactly what
`diffusion.diffuse` does. The predict path's `heat_kernel_apply` (a JAX port) must
do the same; otherwise the `Σ_i w_i` mass identity is violated. Note the *density*
and the pointwise likelihood **do** legitimately change with rank (lower rank =
coarser smoothing); what must stay rank-invariant is the **integrated mass**
`Σ_x (H D_e)[:,j] = Σ_i w_i K`. Guarded by the mass-invariance test in §Testing.

**Degeneracy policy — identical to the merged #35 contract:**
- *Tier 1 (raise):* non-finite in-window features / positions (`validate_finite`),
  non-positive `position_std` / `waveform_std`, invalid `weights`
  (`validate_weights`), and **`heat_kernel_rank`** — must be `None` or a positive
  `int` (reject float/bool/zero/negative before it reaches SciPy/cache slicing), and
  surface the engine's component-count constraint (`rank ≥ n_components`, from
  `_require_rank_covers_components`) as a `ValidationError` rather than an incidental
  failure. **`memory_budget`** must be a finite positive integer number of bytes
  (reject `0` / negative / non-finite / non-integer); **`block_size`** a positive int.
- *Tier 2 (warn):* none specific to the rank here — the engine's auto-truncation is
  tolerance-based with a dense fallback and does not warn (see Bandwidths). (Warns
  from the shared occupancy/EM machinery still apply.)
- *Zero-rate electrode* (zero weights **or** zero encoding spikes → `μ_e = 0`):
  each observed decode spike floors to `LOG_EPS` — **not skipped** — matching the
  marked-point-process contract shared by `clusterless_kde` / `clusterless_gmm`;
  its ground-process term is 0.
- *Occupancy / empty:* zero-occupancy bins and empty mark-weighted histograms
  floor to `EPS` / `LOG_EPS`.

## Testing & benchmark plan

Follows `scientific-tdd` (tests first) + `numerical-validation` for invariants.
Default stays `clusterless_kde`, so existing golden / snapshot are untouched.

- **Density correctness (guards the normalization, §1/§4):** these catch
  spatially-constant errors that rank/argmax posterior tests *cannot*.
  - *Mark-marginal recovery:* `Σ_x p_e(x, m_j)·ΔV(x)` equals the weighted mark-KDE
    marginal `Σ_i w_i K_mark(m_j, m_i) / Σ_i w_i` (to tolerance) — verifies the
    `Σ_i w_i` / `ΔV` normalization and the mass-conserving diffuse.
  - *Absolute log-likelihood:* on a small hand-checkable fixture, assert the actual
    `log λ_e` value (not just posterior shape), so a wrong constant `Z` is caught.
  - *Mass-invariance across ranks:* the integrated mass `Σ_x (H D_e)[:,j]` (equiv.
    `Σ_x p_e·ΔV`) is **rank-invariant** — fails if `heat_kernel_apply` clips without
    the per-component mass-rescale. Do **not** assert pointwise log-likelihood
    invariance across ranks (the density legitimately changes with rank); instead
    check pointwise *convergence* only among high ranks above a stated tolerance.
- **Weighted-EM correctness:** a **binary-weight** fit equals a
  **hard-subset** fit (drop the zero-weight encoding spikes), mirroring
  `test_clusterless_weights`'s binary-weights-match-subset test; and a graded EM
  weight scales `D_e` as specified. **Zero-rate at fit and predict** (`Σ w_pos == 0`
  occupancy; `Σ w_e == 0` electrode): the fit produces finite outputs (no `0/0`
  NaN), the zero-rate electrode adds 0 ground-process intensity, and its decode
  spikes contribute exactly `LOG_EPS` each — guards the fit and predict zero-rate
  branches. Assert the **`UserWarning`s** fire (`Σ w_pos == 0` occupancy warning;
  per-electrode `Σ w_e == 0` warning), since both are intentionally surfaced (unlike
  `clusterless_kde`, which is silent).
- **Input validation (Tier-1):** `heat_kernel_rank` of a float / bool / `0` /
  negative / below `n_components` raises `ValidationError` (not an incidental
  SciPy/cache error).
- **Local ≡ non-local (per spike, internal):** compare the **internal per-spike
  `lc`** (the `log λ_e` term *before* `segment_sum` and *before* the
  `−summed_ground_process_intensity`) — spike `j`'s `lc` at `bin(x_a(t_j))` equals
  the local per-spike value. Do **not** compare a final non-local output cell (it
  aggregates every spike in the time bin plus the ground-process term). The
  coarser public-output variant is valid only on **one-spike stationary** time bins
  (subtract the ground-process term).
- **Device-cache lifecycle:** (1) **refit invalidation** — build the device cache
  (fit + predict once), then `environment.fit_place_grid(new grid)`; assert the
  device cache is dropped, **re-fit the encoding model** on the new grid (the old
  encoding model is grid-bound and stale), and assert predict then works against the
  rebuilt basis (no stale `Q` / shape error). If the optional grid-generation guard
  is implemented, also assert predicting with the *stale* encoding model raises a
  clear error rather than using stale arrays. (2) **save/load parity** — `save_model`
  → `load_model` succeeds (device cache excluded from the pickle, so no JAX-`Device`
  pickling error), and predict on the loaded detector matches pre-save within
  tolerance (cache lazily rebuilt). (3) **block-size (deterministic)** — assert
  `effective_block` from the formula is `≥ 1`, `≤ requested`, and *shrinks as
  `n_bins` grows*; and that decoding at the reduced block gives the **same
  likelihood** as a single large block (block-parity). (4) **override survives
  handoff** — a non-default `memory_budget` passed at fit is stored in the encoding
  dict and drives predict's `effective_block` (assert the resolved block differs from
  the default's), so a user override is not silently lost; likewise `block_size`.
  Measured peak memory is a **non-gating benchmark only** — JAX
  allocator/preallocation makes peak an unstable unit-test assertion.
- **Agreement with `clusterless_kde` (anchor for A):** on a 1D track + a 2D open
  field (no barriers), assert the decoded posterior agrees with `clusterless_kde`
  by rank-correlation / argmax on simulated data — the style of the existing
  `test_clusterless_likelihood_agreement` (kde-vs-gmm). Not bit-identical (different
  smoothers), but tracks closely. (Complements, not replaces, the absolute tests
  above.)
- **Geometry win (the point of A):** a 2D barrier / multi-arm scenario where a
  near-wall spike's mark evidence must not cross the wall; quantify leaked mass on
  the wrong side (`clusterless_kde` smears, `clusterless_diffusion` respects the
  barrier). Plus a grid-independence check (posterior stable across bin sizes —
  `position_std` is physical), mirroring `sorted_spikes_diffusion`.
- **Performance (B):** benchmark predict (and fit) vs `clusterless_kde` on a large
  grid + many spikes; record baseline → compare on realistic data (optimization
  workflow). Documented result, not a flaky pass/fail.
- **Invariants (property tests):** finite log-likelihoods, non-negative densities,
  and the §4 degeneracy cases (zero-rate electrode → `LOG_EPS`/spike;
  zero-occupancy → floor).
- **Integration:** registry entry `"clusterless_diffusion"`; end-to-end
  `ClusterlessDecoder` / `NonLocalClusterlessDetector` fit + predict smoke.

## Sequencing

Because B is a priority and the perf win is a hypothesis (cached low-rank matmul
beats pairwise KDE), sequence an early **correctness + perf spike** — the core
non-local predict on simulated data, measured against `clusterless_kde` for both
posterior agreement and predict time — to de-risk before the full production
build (validators, local path, blocking, tests, docs). The writing-plans step
orders this.

## Risks / open items

- **Normalization fidelity.** The normalization is now specified exactly (§1/§4:
  weight by `w_i`, divide by `Σ_i w_i · ΔV(x)`, mark normalizer only in
  `kde_distance`, no per-column unit-integral). The remaining risk is
  *implementation* faithfulness — guarded by the mark-marginal-recovery and
  absolute-log-likelihood tests, which the spike runs first.
- **JAX `heat_kernel_apply` must match `diffusion.diffuse` exactly** — clip **and**
  per-component rescale to input mass. A clip-only port breaks mass conservation
  (the density's integral drifts with rank). Guarded by the mass-invariance test;
  prefer reusing/porting `diffuse`'s exact logic over reimplementing.
- **EM weights are load-bearing** in `D_e` (`Σ_i w_i` normalization), not only in
  occupancy/mean-rate. Guarded by the binary-weights-match-subset test.
- **Local path.** First cut is nearest-bin, correct-by-construction equal to the
  non-local value at that bin (asserted). Barrier-safe *linear* interpolation (à la
  `sorted_spikes_diffusion`) and the clip-free single-bin shortcut are documented
  follow-ups, not first-cut requirements.
- **Device-basis cache lifecycle (the newest / most fragile area).** The perf-driven
  device cache on `Environment` adds two lifecycle obligations: it must be
  invalidated in `fit_place_grid` (alongside the existing diffusion caches) and
  excluded from pickling via `Environment.__getstate__` (lazily rebuilt after load).
  Both are easy to forget and silent when wrong (stale `Q` after refit; unpickleable
  detector). Guarded by the refit-invalidation and save/load-parity tests. If these
  prove fiddly, a fallback is to drop the cache and convert per predict (simpler,
  slower) — a measured B-vs-simplicity call for the spike.
