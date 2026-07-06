# Sorted-spikes diffusion likelihood — design

**Date:** 2026-07-06
**Branch:** `sorted-spikes-diffusion`
**Status:** design, pending implementation plan

## Summary

Add a new, opt-in sorted-spikes likelihood algorithm, `sorted_spikes_diffusion`,
that estimates place fields with a **graph heat-kernel diffusion** smoother instead of
the current Gaussian kernel density estimate (KDE). The diffusion runs on the
`Environment`'s existing manifold graph, so it respects track geometry (walls, holes,
junctions) and works uniformly across 1D linearized tracks and N-D open fields.

The estimator is **ported from the author's `neurospatial` package** (MIT,
`neurospatial.ops.smoothing`), not built from scratch and not taken as a runtime
dependency. Specifically we port the graph-Laplacian construction and **apply it with
batched `scipy.sparse.linalg.expm_multiply`** rather than materializing a dense kernel —
a scalability strategy taken from the spatstat diffusion port and explicitly flagged as
an unbuilt "deferred stretch goal" in neurospatial's own docstring.

### How we got here (decisions)

- Estimator = heat-kernel diffusion, not Gaussian KDE (fixes cross-wall smearing).
- Substrate = the `Environment`'s manifold graph (`track_graphDD` for N-D, the
  linearized graph for 1D), not a separate pixel grid — reflecting boundaries and mass
  conservation are intrinsic to the graph; works in 1D and N-D; no transpose/ravel adapter.
- Sourcing = **port** neurospatial's kernel construction into `non_local_detector`
  (no new dependency, no `Environment`-bridging), applied via `expm_multiply`.
- The spatstat NumPy port (`density_heat.py`) is at most an **optional test oracle**,
  now largely redundant given the analytic-Gaussian equivalence test.

## Motivation

Ordinary Gaussian KDE (`common.KDEModel`, used by `sorted_spikes_kde`) smooths in the
coordinate metric. In 2D it smears probability across walls and holes; the package
currently sidesteps this by linearizing to 1D via `track_graph`. Graph diffusion fixes
cross-wall smearing directly, in any dimension, and is a principled intensity estimator:

1. **Reflecting (Neumann) boundaries** — no probability flux across a wall (the animal
   cannot cross it, so neither should its estimated density).
2. **Mass conservation** — the operator is column-stochastic, so total intensity is
   preserved; downstream Poisson rates are correctly normalized.
3. **Intrinsic (diffusion) distance** — smoothing follows the diffusion distance on the
   domain, correct around junctions and holes where Euclidean is wrong.

## Estimator: what we port and how we apply it

### Ported construction (from `neurospatial.ops.smoothing.compute_diffusion_kernels`)

Given a graph whose edges carry a Euclidean `distance` attribute:

1. **Gaussian edge weights** `w_uv = exp(-d_uv² / (2σ²))` (Belkin–Niyogi heat-kernel
   weighting).
2. **Graph Laplacian** `L = D - W` (sparse).
3. **Volume correction (density mode only):** `L ← M⁻¹ L`, `M = diag(bin_sizes)`, so
   bins of unequal size integrate correctly.

The heat kernel is `K = exp(-t L)`, `t = σ² / 2`.

### Applied via batched `expm_multiply` (the spatstat lesson)

We never materialize the dense `K`. The place-field pipeline only needs the operator
applied to a handful of fields (occupancy + one count field per neuron), so we compute

```text
smoothed = expm_multiply(-t · L, fields)      # fields shape (n_bins, n_fields)
```

stacking occupancy and all neuron count-fields as columns of one `(n_bins, n_neurons+1)`
matrix and diffusing them in a single call. This is `O(nnz)` memory instead of `O(n²)`,
exact (adaptive Krylov; no `Nstep`/`pmax`/stability tuning), and removes neurospatial's
dense >3000-bin blow-up — the one axis on which spatstat scaled better.

**Mass conservation is automatic.** For a symmetric graph Laplacian `1ᵀL = 0`, so
`exp(-tL)` is exactly column-stochastic; `expm_multiply` conserves mass in `transition`
mode with no explicit renormalization (a clip of tiny negative Krylov noise is retained).

### Modes

- **`transition` (default):** mass-conserving; correct for count fields (occupancy,
  spike counts) on a uniform grid, where all `bin_sizes` are equal.
- **`density`:** volume-corrected; used only when bin areas vary (e.g. linearized tracks
  with uneven bins). Requires `bin_sizes`; area-weighted normalization.

### Where this runs (NumPy at fit time; JAX inherits the result)

The diffusion is a **host-side fit-time computation** in NumPy/SciPy (`expm_multiply`),
exactly like the existing KDE fit uses `scipy.interpolate`. Only the resulting
`place_fields` arrays enter JAX for the HMM likelihood. No JAX diffusion kernel is
needed, and the JAX code path is unaffected.

## Goals / non-goals

### Goals

- New registered `sorted_spikes_diffusion` algorithm selectable via
  `sorted_spikes_algorithm="sorted_spikes_diffusion"`.
- Manifold-aware place fields in 1D (linearized `track_graph`) and N-D (`track_graphDD`).
- Same encoding-model dict contract and Poisson log-likelihood as `sorted_spikes_kde`
  (drop-in at the model level).
- Rigorous numerical validation (analytic Gaussian, mass conservation, no cross-wall
  leakage) plus a scaling guard.

### Non-goals (YAGNI, filed for later)

- Clusterless (mark-space) diffusion — marks are not spatial; out of scope.
- Leave-one-out bandwidth selection (spatstat's per-point kernels enable choosing σ by
  LOO likelihood instead of a fixed `position_std`) — deferred.
- Adaptive / per-bin bandwidth (spatstat's per-pixel σ) — breaks the single-`expm`
  formulation on a graph; noted as a real capability gap, deferred.
- Richardson extrapolation, explicit `Nstep`/`pmax` CFL machinery, connect 4/8 toggling —
  not ported (moot under `expm_multiply`; connectivity is fixed by the environment graph).

## Architecture

### Module layout

```text
src/non_local_detector/likelihoods/
  diffusion.py                  # graph→L build + batched expm_multiply application
  sorted_spikes_diffusion.py    # fit_/predict_ mirroring sorted_spikes_kde.py
  __init__.py                   # register in _SORTED_SPIKES_ALGORITHMS

src/non_local_detector/tests/likelihoods/
  test_sorted_spikes_diffusion.py
  density_heat_oracle.py        # OPTIONAL vendored spatstat NumPy port (test oracle only)
```

### `diffusion.py` (the well-bounded numerical unit)

- `build_diffusion_operator(graph, sigma, *, bin_sizes=None, mode="transition") -> sparse L`
  — port of neurospatial's Gaussian-weighted Laplacian + volume correction. Keeps `L`
  sparse (does **not** exponentiate).
- `diffuse(L, sigma, fields, *, mode, bin_sizes=None) -> ndarray (n_bins, n_fields)`
  — `expm_multiply(-t·L, fields)`, `t=σ²/2`; clip tiny negatives; apply density-mode
  area normalization when requested.
- `interior_subgraph(environment) -> (graph, node_order)` — the adapter (below).

### Environment → interior-bin graph adapter

The likelihoods operate on `place_bin_centers_[is_track_interior]` (interior bins, flat
order). We build `L` on exactly those bins in that order:

- **N-D:** take the subgraph of `track_graphDD` induced by interior nodes and relabel
  `0..n_interior-1` in the order `np.where(is_track_interior_.ravel())[0]`. Edges already
  carry `distance` (`make_nD_track_graph_from_environment`).
- **1D:** use the linearized graph / `distance_between_nodes_`, restricted to interior
  bins in the same order.

No pixel grid, no transpose — just a node relabeling. Correctness is pinned by a
round-trip test (a unit field at interior bin `k` diffuses to a bump centered on `k`).

### `sorted_spikes_diffusion.py`

Mirrors `sorted_spikes_kde.py`:
- `fit_sorted_spikes_diffusion_encoding_model(position_time, position, spike_times,
  environment, weights=None, sampling_frequency=500, position_std=sqrt(12.5),
  mode="transition", block_size=100, disable_progress_bar=False) -> dict`
  - build & cache the diffusion operator on the interior-bin graph;
  - pixellate occupancy (weighted by dt) and each neuron's spike positions to count
    fields on interior bins;
  - **batch** occupancy + all neuron fields through one `diffuse` call;
  - `place_field_k(x) = mean_rate_k · marginal_k(x) / occupancy(x)`, floored at `EPS`
    (identical formula and clipping convention to `sorted_spikes_kde`);
  - return the same dict keys as `sorted_spikes_kde` (`environment`, `occupancy`,
    `mean_rates`, `place_fields`, `no_spike_part_log_likelihood`, `is_track_interior`,
    `disable_progress_bar`), plus the cached operator / ordering.
- `predict_sorted_spikes_diffusion_log_likelihood(...)`
  - **non-local:** reuse `sorted_spikes_kde`'s non-local path verbatim (it uses only
    `place_fields`, `no_spike_part_log_likelihood`, `is_track_interior`);
  - **local:** index place fields by the animal's interpolated bin (`get_bin_ind`)
    rather than re-running a smoother (bin-resolution; consistent discretization).

`position_std` keeps its meaning (equivalent-Gaussian σ in coordinate units), mapped to
diffusion time `t=σ²/2`.

### Data flow

```text
Environment (fitted, 1D or N-D)
  └─ interior-bin graph + edge distances
       └─ build & cache sparse L(σ, mode)                         [once, at fit]
position samples ─ pixellate(weights=dt) → occupancy field  ┐
neuron k spikes  ─ pixellate            → marginal_k field   ┤ stack columns
                                                             ├→ expm_multiply(-tL, U)  [batched, NumPy]
                                                             ┘
  occupancy, marginals on interior bins
    └─ place_field_k(x) = mean_rate_k · marginal_k(x) / occupancy(x)   [floored EPS]
         └─ encoding-model dict (same keys as sorted_spikes_kde)
              └─ HMM predict:  Σ_k [ count_k · log λ_k − λ_k ]   (JAX, unchanged)
```

## Error handling

Reuse `_validation` / `ValidationError` conventions:
- require a fitted `Environment` with `place_bin_centers_`, `is_track_interior_`, and the
  manifold graph (`track_graphDD` for N-D, linearized graph for 1D); clear error if missing;
- reject non-positive `position_std`; reject invalid `mode`;
- **σ ≥ bin-width guard:** warn when `position_std` is below ~1 bin width, since the
  Gaussian edge weights collapse and `K ≈ I` (silent near-no-op) — a guard implied by
  spatstat's `Nstep` calibration;
- floor place fields at `EPS`, and clip the summed no-spike term once (not per neuron),
  matching `sorted_spikes_kde`.

## Testing strategy (TDD order — each RED first)

Invariant tests are adapted from spatstat's `test_density_heat.py`, which is a stronger
suite than we would write from scratch.

1. **Adapter round-trip** — a unit field at interior bin `k` diffuses to a bump centered
   on `k`; interior-node relabeling matches `place_bin_centers_[is_track_interior]` order.
2. **Analytic-Gaussian equivalence** — single binned point far from boundaries ≈ exact
   Gaussian with bandwidth σ (`<2%`), in 1D and N-D (spatstat test 2).
3. **Mass conservation** — `Σ (smoothed) = Σ (field)` in `transition` mode to ~1e-10;
   `Σ intensity·bin_size = N` for the intensity form (spatstat test 1).
4. **No cross-wall leakage** — barrier / two-arm environment: zero mass crosses the gap;
   a layout Euclidean KDE smears across stays separated under diffusion.
5. **1D correctness** — linear track ≈ 1D analytic Gaussian; W-track respects arm gaps.
6. **σ ≥ bin-width guard** — warns (does not smooth) when `position_std` < ~1 bin width.
7. **Drop-in equivalence anchor** — wall-less open field, well-sampled trajectory:
   diffusion `place_fields` ≈ KDE `place_fields` within a few percent (also pins the
   `mean_rate · marginal / occupancy` normalization so units match).
8. **Invariants** (`property`) — place fields ≥ 0 and finite; posteriors sum to 1;
   no NaN/Inf.
9. **Scaling guard** — the `expm_multiply` path stays `O(nnz)` memory (no dense
   `n_bins × n_bins` materialization); a large-bin environment that would warn under a
   dense kernel runs without it.
10. **End-to-end** — fit a sorted-spikes decoder with `sorted_spikes_diffusion` on
    simulated replay; recover the trajectory; capture a **new** syrupy snapshot; then a
    smallest-real-slice smoke test.

Numerical validation per `CLAUDE.md`: `pytest -m property`, the golden-regression suite,
and the new snapshot. The `jax` and `numerical-validation` skills apply during
implementation. Optionally cross-check `diffuse` against the vendored spatstat oracle on
a 2D open-field grid.

## Tradeoffs

**Wins:** manifold-aware, mass-conserving place fields in 1D and N-D; exact heat kernel
(no time-stepping error) applied at `O(nnz)` memory via `expm_multiply`; a single
geometry representation (the graph); drop-in at the model level; no new dependency; kernel
math reused from the author's own MIT package rather than reinvented.

**Costs:** a small ported numerical unit duplicated across two MIT repos (divergence
risk); `expm_multiply` cost grows with σ (more Krylov steps) and with `nnz`; bin-resolution
local decoding; the Gaussian graph-Laplacian↔continuous-Laplacian correspondence is clean
only for near-regular bin lattices (true here; validated in tests 2/5) and under-smooths
when σ < bin width (guarded).

## Out of scope

Clusterless; LOO bandwidth selection; adaptive/per-bin bandwidth; Richardson; sub-bin
local decoding. Both 1D and N-D grid environments are supported.
