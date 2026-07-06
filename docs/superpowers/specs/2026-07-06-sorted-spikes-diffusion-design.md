# Sorted-spikes diffusion / MRF likelihood — design

**Date:** 2026-07-06
**Branch:** `sorted-spikes-diffusion`
**Status:** design, pending implementation plan

## Summary

Add opt-in sorted-spikes place-field likelihoods built on a **shared spectral engine**:
the eigendecomposition of a geometry-respecting graph Laplacian `L` on the environment's
interior bins. The same `eig(L)` powers two estimators:

- **Now — diffusion smoother** (`sorted_spikes_diffusion`): a linear heat-kernel smoother
  `exp(-tL)`, `t = position_std²/2`, giving manifold-aware place fields that don't smear
  across walls/holes, in 1D and N-D.
- **Fast-follow — MRF-GAM** (`sorted_spikes_mrf`): a penalized-Poisson GAM whose
  reduced-rank basis is the smoothest eigenmodes of `L` and whose penalty is the same `L`.
  More principled (occupancy as a log-offset, REML smoothness, Bayesian SE), fit for the
  whole population at once (NeMoS-style).

The two are the *same operator* viewed two ways
(`exp(-tL) = Σⱼ exp(-tλⱼ)vⱼvⱼᵀ`; the MRF basis is `[v₁…v_rank]` with penalty `diag(λⱼ)`),
so `eig(L)` is computed once per environment and reused by both, across all neurons and
all EM iterations.

### Decisions (settled)

- Estimator = graph heat-kernel diffusion, not Gaussian KDE (fixes cross-wall smearing).
- Operator `L` = **finite-difference graph Laplacian**, edge weight ∝ `1/d²` (verified to
  give bandwidth = `position_std`, grid-independent — neurospatial's `exp(-d²/2σ²)`
  weighting instead yields ≈ σ·bin_size, a latent grid-dependence worth reporting upstream).
- Apply via a **cached eigendecomposition** `eig(L)`, not per-call `expm_multiply`
  (approved: build eig in from the start; it is also the MRF-GAM basis).
- Substrate = the environment's manifold graph — `track_graphDD` (N-D + 1D grids),
  `track_graph_with_bin_centers_edges_` (linearized `track_graph`). Adapter consumes them.
- Sourcing = port a small numerical unit into nld; no runtime dependency on neurospatial.
- MRF-GAM is a designed-for **fast-follow phase**, fit with the **NeMoS population trick**
  (shared design matrix `B`, response `(n_bins, n_neurons)`, coefficient matrix
  `(rank, n_neurons)`), mirroring `sorted_spikes_glm`.

## Motivation

Gaussian KDE smooths in the coordinate metric and smears across walls/holes in 2D (the
package sidesteps this by linearizing to 1D). Graph diffusion fixes this directly, in any
dimension: reflecting (Neumann) boundaries are intrinsic to the graph (no edges to absent
bins), the operator conserves mass (`1ᵀL = 0` ⇒ `exp(-tL)` column-stochastic), and
smoothing follows the diffusion distance on the domain. The MRF-GAM adds the correct
statistical treatment of a Poisson rate: occupancy as exposure (offset), automatic
smoothness (REML), and uncertainty (SE).

## The shared spectral engine

### Finite-difference Laplacian `L`

Given the interior-bin graph (below) with Euclidean edge `distance` `d`, on a
**face-adjacent** graph: edge weight `w = 1/d²`; `L = D - W` (sparse, **symmetric**,
σ-independent). On a regular grid `L ≈ -∂²`, so `exp(-tL)` with `t = σ²/2` is a Gaussian of
std σ **independent of bin size** (verified: std/σ = 1.000 at bin sizes 0.5/1/2/4 in 1D and
on a 2D face-adjacent grid). The N-D adapter **drops Moore/diagonal edges**: with `1/d²`
weighting the diagonals inflate the small-wavenumber diffusion coefficient, oversmoothing by
≈√2 in 2D (verified: 8-connected std/σ = 1.413 vs face-only 1.000).

### Eigendecomposition (built once, cached)

`L = QΛQᵀ` via dense `scipy.linalg.eigh` (default), or truncated `scipy.sparse.linalg.eigsh(...,
sigma=-1e-8, which="LM")` for the `rank` smallest eigenpairs when `n_bins` is large — a
**negative** shift, because `sigma=0` factorizes the singular Laplacian and is unreliable
(`which="SM"` is the no-shift-invert fallback). Truncation keeps **all** zero modes (one per
connected component, `rank ≥ n_components`), preserving component-wise mass. A cached wrapper
`cached_eigenbasis(environment, rank)` owns the cache: a dict on the `Environment` **keyed by
`rank`** (`_diffusion_eigenbasis_[rank]`), invalidated in `fit_place_grid` like
`_bin_distance_matrix_`. Reused across all neurons and across EM refits
(`fit_encoding_model` is called every EM M-step, `base.py:1991`, with the graph and σ
constant, so the eig is invariant).

### Diffusion application (all neurons at once)

`exp(-tL) F = Q · (exp(-tΛ)[:,None] ⊙ (Qᵀ F))`, `t = σ²/2`, with `F` the
`(n_bins, n_fields)` matrix of occupancy + all neuron count-fields — a single batched
matmul over neurons (the population "fit many at once" is inherent for the linear smoother).
Truncated modes give a principled low-pass approximation for large `n_bins`; validated
against the exact `expm(-tL)` on a moderate grid.

## Estimator 1 — diffusion smoother (ship first)

### Densities and `bin_sizes`

`exp(-tL)` conserves field sums. Place fields are densities, so a smoothed count field is
normalized to ∫=1 via `density_i = smoothed_i / Σⱼ(smoothed_j · bin_sizes_j)` using per-bin
volumes from the adapter. On a uniform grid the volume factor cancels in the
`marginal/occupancy` ratio; on uneven bins it is required. No separate `transition`/`density`
mode and no `M⁻¹L` — the volume correction lives entirely in this normalization.

### Place field and contract

Mirrors `sorted_spikes_kde.py` exactly (so units match; pinned by the KDE drop-in test):

- Smooth an **occupancy** count field weighted by `weights` (KDE convention — default ones)
  and, per neuron, a **spike** count field weighted by `weights_at_spike_times` — the
  posterior `weights` interpolated to each spike time (`sorted_spikes_kde.py:178`), **not**
  unweighted spike counts. `mean_rate_k` = `weights_at_spike_times.sum() / weight_sum`. This
  matters: EM refits pass posterior `weights` (`base.py:1991`), so the spike fields must be
  weighted (parity test with non-uniform weights).
- Normalize each smoothed field to an ∫=1 density; `place_field_k =
  mean_rate_k · marginal_k / occupancy` with KDE's zero-occupancy guard
  (`where(occ>0, marginal/occ, EPS)`) and `EPS` floor.
- **Scatter the interior-bin rate into a FULL-GRID array** (`jnp.zeros((n_total_bins,)).at[
  is_track_interior].set(...)`), so `place_fields` is `(n_neurons, n_total_bins)` and
  `no_spike_part_log_likelihood` is `(n_total_bins,)` — exactly like KDE
  (`sorted_spikes_kde.py:204-219`). Interior-only storage silently breaks gap/barrier envs
  because `get_bin_ind` (local predict) returns full-grid indices.
- Returns the same dict keys as `sorted_spikes_kde` **plus** the cached spectral engine
  handle, `node_order`, `bin_sizes` (the base class splats the whole dict into predict —
  every key must be a predict param).

`position_std` is the bandwidth in coordinate units (verified grid-independent). σ-guard
warns when `position_std` < ~1 bin width.

## Estimator 2 — MRF-GAM (fast-follow phase, same engine)

Penalized-Poisson GAM: `n_ik ~ Poisson(o_i · exp(η_ik))`, `η_k = B γ_k`, with `B` the
`rank` smoothest eigenmodes (from the shared engine) and penalty `λ·γ_kᵀ diag(λ_eig) γ_k`
— **exactly a generalized-Ridge penalty** in the eigenbasis. **Occupancy `o` is a shared
log-offset (exposure), never a denominator.** `λ` chosen by REML (Wood 2011); Bayesian SE
and edf available. Emits `η = log-rate` directly, which is what the HMM Poisson emission
`Σ[count·log λ − λ]` consumes.

**NeMoS population trick:** `B` and the offset `log(o)` are shared across neurons; only the
response `n_k` and coefficients `γ_k` differ. So the whole population fits jointly — design
`B (n_bins, rank)`, response `(n_bins, n_neurons)`, coefficient matrix `Γ (rank, n_neurons)`,
generalized-Ridge weights `λ_eig` — via batched/`vmap`'d penalized IRLS in JAX, no
per-neuron loop. (Whether to depend on NeMoS or implement the batched fit directly is a
decision for that phase; the pattern is the same either way.) Mirrors `sorted_spikes_glm`.

This phase is **not built in the first PRs**; the engine, adapter, and `bin_sizes` are
designed so it drops on without rework.

## Performance

- **eig cached on the `Environment`** (dict keyed by `rank`, via `cached_eigenbasis`),
  reused across neurons and EM refits — the largest win, since EM refits encoding each
  M-step with `L` unchanged.
- **All neurons in one batched matmul** (diffusion) / one population fit (MRF-GAM).
- **Truncated eigsh** (`sigma=-1e-8`) for large `n_bins` (low-pass approximation; keep all
  per-component null modes).
- **Skip zero-spike neurons** (empty field → `EPS` floor).
- **Optional `float32`** for large problems (adequate for smoothed densities).
- One-time `eig` is `O(n³)` dense (or truncated) with `O(n²)` `Q`; amortized across neurons
  and EM iterations. A baseline-measurement task records fit / per-EM-iteration cost on a
  representative environment before any further tuning; size-adaptive dense-vs-truncated is
  chosen from that measurement.

## Goals / non-goals

### Goals

- `sorted_spikes_diffusion` registered and selectable; manifold-aware place fields for N-D
  grids, 1D grids (both via `track_graphDD`), and linearized `track_graph`.
- Shared spectral engine + adapter + `bin_sizes`, cached on the `Environment`.
- Same encoding dict contract and Poisson log-likelihood as `sorted_spikes_kde`.
- Numerical validation: bandwidth invariance, analytic-Gaussian, mass conservation,
  mode-reconstruction, no cross-wall/arm leakage, KDE drop-in equivalence.

### Non-goals / deferred

- **MRF-GAM** (`sorted_spikes_mrf`) — designed-for fast-follow, not in the first PRs.
- Clusterless (mark-space) diffusion; LOO/REML bandwidth for the diffusion smoother;
  adaptive per-bin bandwidth; Richardson; sub-bin local decoding.

## Architecture

### Module layout

```text
src/non_local_detector/likelihoods/
  diffusion.py                  # finite-diff L, eig(L) engine, diffuse-via-modes, adapter
  sorted_spikes_diffusion.py    # fit_/predict_ mirroring sorted_spikes_kde.py
  # sorted_spikes_mrf.py        # fast-follow: MRF-GAM population Poisson fit (later phase)
  __init__.py                   # register in _SORTED_SPIKES_ALGORITHMS

src/non_local_detector/tests/likelihoods/
  test_sorted_spikes_diffusion.py
  diffusion_oracle.py           # OPTIONAL dense finite-diff / spatstat oracle (tests only)
```

### `diffusion.py`

- `build_laplacian(graph) -> sparse L` — finite-difference (`w=1/d²`) symmetric Laplacian
  from edge `distance` attributes; σ-independent. Caller passes a **face-adjacent** graph.
- `diffusion_eigenbasis(L, rank=None) -> (eigvals, eigvecs)` — dense `eigh` (full) or
  truncated `eigsh(sigma=-1e-8)` (`rank` smallest, all per-component null modes).
- `cached_eigenbasis(environment, rank=None) -> (eigvals, eigvecs)` — the cache-owning
  wrapper: builds the graph + `L` + eigenbasis on a miss and stores it in
  `environment._diffusion_eigenbasis_[rank]`; returns the cached basis on a hit (or slices a
  cached full-rank entry). This is what `diffusion_eigenbasis(L, rank)` alone cannot do —
  it takes `L`, not the environment. Returns `(eigvals, eigvecs)` **only**; callers get
  `node_order`/`bin_sizes` from `environment_graph` (also cached, so the graph is built once).
- `diffuse(eigvals, eigvecs, sigma, fields) -> (n_bins, n_fields)` — `Q(exp(-tΛ)⊙(QᵀF))`.
- `to_density(smoothed, bin_sizes)` — normalize a smoothed field to ∫=1.
- `environment_graph(environment) -> (graph, node_order, bin_sizes)` — the adapter (N-D
  branch drops diagonal edges → face-adjacent).

### Environment → interior-bin graph adapter

Builds `L` on `place_bin_centers_[is_track_interior]` (interior bins, flat order),
consuming the environment-built graph. **Two branches:**

- **N-D grid (incl. 1D grid), `track_graph is None`:** subgraph of `track_graphDD` induced
  by interior nodes, relabeled `0..n_interior-1` in `np.where(is_track_interior_.ravel())[0]`
  order (track_graphDD node ids = flat bin index over all bins,
  `make_nD_track_graph_from_environment`, `environment.py:1599`; non-interior nodes are
  isolated). Edges carry `distance` (`environment.py:1659`).
- **Linearized `track_graph`:** consume `track_graph_with_bin_centers_edges_` +
  `place_bin_centers_nodes_df_`, reduced to interior bin-center adjacency (bin centers are
  `is_bin_edge=False`; `node_id == -1` marks gap bins). Junctions connect arms through
  shared original nodes; gap bins (`is_track_interior=False`, `edge_spacing>0`) stay
  disconnected so arms don't leak. Edges carry `distance`. This is the trickiest piece and
  carries the most implementation risk.

`bin_sizes`: N-D = product of per-dim `np.diff(edges_[d])` widths, meshed `'ij'`,
interior-masked (exclude padding bins, `environment.py:1068`); 1D linearized =
`np.diff(place_bin_edges_)` interior-masked (exclude wide gap bins, `environment.py:1481`).

Round-trip tested for **both** branches (unit field at interior bin `k` → bump centered on
`k`); junction/gap tested for the linearized branch.

### `sorted_spikes_diffusion.py`

- `fit_sorted_spikes_diffusion_encoding_model(position_time, position, spike_times,
  environment, weights=None, sampling_frequency=500, position_std=sqrt(12.5), rank=None,
  block_size=100, disable_progress_bar=False) -> dict` — `rank` must be an explicit
  parameter or `sorted_spikes_algorithm_params={"rank": …}` is dropped by the signature
  filter (`base.py:3886`). Call `environment_graph(environment)` for
  `(graph, node_order, bin_sizes)` and `cached_eigenbasis(environment, rank)` for the basis
  (`cached_eigenbasis` returns the eigenbasis only); pixellate weighted occupancy + weighted
  neuron spike fields; batch-`diffuse`; `to_density`; `mean_rate_k · marginal_k / occupancy`
  (guard + EPS); scatter to full-grid; return KDE dict keys + engine handle / `node_order` /
  `bin_sizes`.
- `predict_sorted_spikes_diffusion_log_likelihood(...)` — **dedicated** function (the base
  class splats the encoding dict as kwargs, so it can't call the KDE predict whose
  signature requires `marginal_models`/`occupancy_model`). Non-local branch copies KDE's
  non-local body (uses only `place_fields`/`no_spike_part_log_likelihood`/`is_track_interior`);
  local branch indexes place fields by the animal's interpolated bin (`get_bin_ind`).

### Data flow

```text
Environment (N-D grid, 1D grid, or linearized track_graph)
  └─ environment_graph → (interior-bin graph, node_order, bin_sizes)
       └─ build_laplacian → eig(L)   [cached on Environment; reused across EM & neurons]
occupancy + neuron count-fields (n_bins, 1+n_neurons)
  └─ diffuse (Q exp(-tΛ) Qᵀ, batched) → to_density
       └─ place_field_k = mean_rate_k · marginal_k / occupancy   [guard, EPS]
            └─ encoding dict (KDE keys + engine/node_order/bin_sizes)
                 └─ HMM predict: Σ_k [count_k·log λ_k − λ_k]   (JAX, unchanged)
```

## Error handling

Reuse `_validation` / `ValidationError`: require a fitted `Environment` with
`place_bin_centers_`, `is_track_interior_`, and the relevant graph; reject non-positive
`position_std`; **σ-guard** warns when `position_std` < ~1 bin width; degenerate cases —
empty spike train → zero field; zero-occupancy bin → `where(occ>0,…,EPS)` guard;
disconnected/single-node interior → `exp(-tL)=I` there (no smoothing, no error); floor at
`EPS`; clip the summed no-spike term once, matching `sorted_spikes_kde`.

## Testing strategy (TDD order — each RED first)

1. **Adapter round-trip** — unit field at interior bin `k` → bump on `k`; interior relabel
   matches `place_bin_centers_[is_track_interior]`. Both adapter branches.
2. **Mode reconstruction** — `Σⱼ exp(-tλⱼ)vⱼvⱼᵀ` (full modes) equals `expm(-tL)` on a
   moderate grid (from `mgcv_mrf`'s `test_basis_are_diffusion_modes`); truncated-rank
   approximation within tolerance.
3. **Bandwidth invariance** — recovered std = `position_std` across ≥3 bin sizes (B1 guard).
4. **Analytic-Gaussian equivalence** — single point ≈ exact Gaussian of std `position_std`,
   `<2%`, 1D and N-D.
5. **Mass conservation** — `Σ smoothed = Σ field` to ~1e-10; density normalization ∫=1.
6. **Non-uniform-grid correctness** — uneven 1D bins: `to_density` + ratio vs dense oracle.
7. **No cross-wall / arm leakage** — 2D barrier and linearized W-track: no leak across
   gap/between arms; Euclidean-KDE-smeared layout stays separated.
8. **σ-guard** — warns when `position_std` < ~1 bin width.
9. **Drop-in KDE equivalence** — wall-less open field: diffusion `place_fields` ≈ KDE
   `place_fields` at interior bins away from the boundary, ~5% (boundary bins differ; pins
   the density normalization / units).
10. **eig cache reuse** — refitting with new `weights` (EM) reuses the cached eig; no rebuild.
11. **Invariants** (`property`) — place fields ≥ 0, finite; posteriors sum to 1; no NaN/Inf.
12. **End-to-end** — sorted-spikes decoder with `sorted_spikes_diffusion` on simulated
    replay; recover trajectory; new syrupy snapshot; smallest-real-slice smoke test.

Invariant tests (3/4/5/7) adapt spatstat's `test_density_heat.py`. Validation per
`CLAUDE.md`: `pytest -m property`, golden regression, the new snapshot. `jax` and
`numerical-validation` skills apply during implementation.

## Tradeoffs

**Wins:** manifold-aware, mass-conserving place fields in 1D and N-D with verified
grid-independent physical bandwidth; a shared spectral engine reused across neurons and EM
(and by the future MRF-GAM); all neurons fit at once; drop-in at the model level; no new
dependency.

**Costs:** a small ported/derived numerical unit maintained in nld; one-time `eig` is
`O(n³)`/`O(n²)` (amortized, cached; truncated for scale); deliberate deviation from
neurospatial's edge weighting (report upstream); the linearized `track_graph` adapter is
the riskiest piece; 2D `1/d²` isotropy is approximate (validated, not assumed); the MRF-GAM
fast-follow adds an iterative population Poisson fit (heavier, JAX/`vmap`).

## Out of scope

Clusterless; adaptive/per-bin bandwidth; Richardson; sub-bin local decoding. The MRF-GAM is
deferred to a fast-follow phase (engine designed to support it).
