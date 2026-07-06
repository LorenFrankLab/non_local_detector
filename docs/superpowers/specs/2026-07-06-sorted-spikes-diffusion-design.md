# Sorted-spikes diffusion likelihood — design

**Date:** 2026-07-06
**Branch:** `sorted-spikes-diffusion`
**Status:** design, pending implementation plan

## Summary

Add an opt-in sorted-spikes likelihood algorithm, `sorted_spikes_diffusion`, that
estimates place fields with a **graph heat-kernel diffusion** smoother instead of the
current Gaussian KDE. The diffusion runs on the `Environment`'s existing manifold graph,
so it respects track geometry (walls, holes, junctions) and works uniformly across 1D
linearized tracks and N-D open fields.

From `neurospatial` (MIT, the author's other package) we take the **application
strategy** — apply the heat operator with batched `scipy.sparse.linalg.expm_multiply`
instead of materializing a dense kernel — and the **graph substrate** idea. We do **not**
port neurospatial's Gaussian edge weighting: it makes the effective bandwidth
grid-dependent (≈ σ·bin_size). Instead we build a **finite-difference Laplacian**
(edge weight ∝ 1/d²), which gives a grid-independent physical bandwidth equal to
`position_std` — the same calibration spatstat's diffusion estimator uses.

### Design decisions (settled)

- Estimator = heat-kernel diffusion, not Gaussian KDE (fixes cross-wall smearing).
- Operator = **finite-difference graph Laplacian** (weight ∝ 1/d²), applied via batched
  `expm_multiply`. Verified: effective bandwidth = `position_std`, independent of bin
  size (neurospatial's `exp(-d²/2σ²)` weighting instead yields ≈ σ·bin_size — a latent
  grid-dependence in neurospatial's own `diffusion_kde`, worth reporting upstream).
- Substrate = the `Environment`'s manifold graph — `track_graphDD` for N-D grids
  (incl. 1D grids), `track_graph_with_bin_centers_edges_` for linearized `track_graph`
  environments. Both are built by the environment; the adapter consumes them.
- Sourcing = **port** a small numerical unit into `non_local_detector` (no runtime
  dependency on neurospatial, no cross-`Environment` bridging).
- The spatstat NumPy port is at most an optional test oracle; a dense finite-difference
  kernel is a cheaper oracle for the non-uniform-grid case.

## Motivation

Gaussian KDE smooths in the coordinate metric; in 2D it smears probability across walls
and holes, which the package currently sidesteps by linearizing to 1D. Graph diffusion
fixes cross-wall smearing directly, in any dimension, and is a principled estimator:

1. **Reflecting (Neumann) boundaries** — no probability flux across a wall; intrinsic to
   the graph (no edges to absent bins ⇒ boundary nodes keep more mass at home).
2. **Mass conservation** — a symmetric graph Laplacian has `1ᵀL = 0`, so `exp(-tL)` is
   exactly column-stochastic; `expm_multiply` conserves the field sum with no explicit
   renormalization (a clip of tiny negative Krylov noise is retained).
3. **Intrinsic (diffusion) distance** — smoothing follows the diffusion distance on the
   domain, correct around junctions and holes where Euclidean is wrong.

## Estimator

### Finite-difference graph Laplacian (the B1 fix)

Given a graph whose edges carry a Euclidean `distance` attribute `d`:

- **Edge weight** `w = 1/d²` (finite-difference, not `exp(-d²/2σ²)`). Diagonal edges
  (longer `d`) are down-weighted by `1/d²`, which mitigates connect-8 anisotropy.
- **Laplacian** `L = D - W` (sparse, **symmetric**). `L` depends only on the graph
  geometry, **not** on σ — so it is built once and reused for any bandwidth.

The heat operator is `exp(-t L)`, `t = σ²/2`. On a regular grid `L ≈ -∂²`, so
`exp(-tL)` is a Gaussian of std σ **independent of bin size** (verified: effective
std/σ = 1.000 at bin sizes 0.5/1/2/4, versus 0.499/0.990/1.922/3.409 for the Gaussian
edge weighting).

### Applied via batched `expm_multiply`

We never materialize the dense operator. The place-field pipeline applies the operator
to a handful of fields (occupancy + one count field per neuron), so:

```text
smoothed = expm_multiply(-t · L, fields)      # fields shape (n_bins, n_fields)
```

stacking occupancy and all neuron count-fields as columns of one `(n_bins, n_neurons+1)`
matrix and diffusing them in a single call — `O(nnz)` memory instead of `O(n²)`, and
`expm_multiply` is tolerance-bounded adaptive Krylov (no `Nstep`/stability tuning).

### Densities and `bin_sizes`

The operator is mass-conserving on **field sums** (`Σ smoothed = Σ field`). Place fields
are densities, so we convert with per-bin volumes `bin_sizes` (from the adapter): a
smoothed count field is normalized to an ∫=1 density via
`density_i = smoothed_i / Σ_j (smoothed_j · bin_sizes_j)`. On a uniform grid the volume
factor cancels in the `marginal/occupancy` ratio; on non-uniform grids (uneven 1D bins)
it is required. There is **no** separate `transition`/`density` mode and no `M⁻¹L`
operator — the volume correction lives entirely in this field normalization.

### Bandwidth and where it runs

`position_std` is the bandwidth in coordinate units (verified grid-independent), mapped
to `t = position_std²/2`. A **σ-guard** warns when `position_std` is below ~1 bin width
(the operator then barely smooths). The diffusion is a **host-side fit-time computation**
in NumPy/SciPy, like the existing KDE fit's `scipy.interpolate`; only the resulting
`place_fields` enter JAX for the HMM likelihood, so the JAX path is unaffected.

## Goals / non-goals

### Goals

- Registered `sorted_spikes_diffusion` selectable via `sorted_spikes_algorithm="…"`.
- Manifold-aware place fields for N-D grids, 1D grids (both via `track_graphDD`), and
  linearized `track_graph` environments (via `track_graph_with_bin_centers_edges_`).
- Same encoding-model dict contract and Poisson log-likelihood as `sorted_spikes_kde`
  (drop-in at the model level).
- Numerical validation: bandwidth invariance across bin sizes, analytic-Gaussian
  equivalence, mass conservation, no cross-wall/arm leakage, KDE drop-in equivalence.

### Non-goals (YAGNI, filed for later)

- Clusterless (mark-space) diffusion — marks are not spatial.
- Leave-one-out bandwidth selection (a spatstat capability) — deferred.
- Adaptive / per-bin bandwidth — breaks the single-`exp` formulation on a graph; deferred.
- Richardson extrapolation, `Nstep`/`pmax` machinery, connect 4/8 toggling — not ported.

## Architecture

### Module layout

```text
src/non_local_detector/likelihoods/
  diffusion.py                  # finite-diff L build + batched expm_multiply + adapter
  sorted_spikes_diffusion.py    # fit_/predict_ mirroring sorted_spikes_kde.py
  __init__.py                   # register in _SORTED_SPIKES_ALGORITHMS

src/non_local_detector/tests/likelihoods/
  test_sorted_spikes_diffusion.py
  diffusion_oracle.py           # OPTIONAL dense finite-diff / spatstat oracle (tests only)
```

### `diffusion.py`

- `build_diffusion_operator(graph) -> sparse L` — finite-difference (`w=1/d²`) symmetric
  Laplacian from edge `distance` attributes. σ-independent; not exponentiated.
- `diffuse(L, sigma, fields) -> ndarray (n_bins, n_fields)` — `expm_multiply(-t·L, fields)`,
  `t=σ²/2`; clip tiny negatives.
- `to_density(smoothed, bin_sizes) -> ndarray` — normalize a smoothed field to ∫=1.
- `environment_graph(environment) -> (graph, node_order, bin_sizes)` — the adapter.

### Environment → interior-bin graph adapter

The likelihoods operate on `place_bin_centers_[is_track_interior]` (interior bins, flat
order). The adapter builds `L` on exactly those bins in that order, consuming the graph
the environment already builds. **Two branches:**

- **N-D grid (incl. 1D grid), `track_graph is None`:** take the subgraph of
  `track_graphDD` induced by interior nodes and relabel `0..n_interior-1` in the order
  `np.where(is_track_interior_.ravel())[0]`. Node ids of `track_graphDD` equal the flat
  bin index over **all** bins (`make_nD_track_graph_from_environment`, `environment.py:1599`),
  and non-interior nodes are isolated; the relabeling must be in exactly that order and
  contiguous so `L` aligns with `place_bin_centers_[is_track_interior]` and the field
  columns. Edges already carry `distance` (`environment.py:1659`).
- **Linearized `track_graph`:** consume `track_graph_with_bin_centers_edges_` and
  `place_bin_centers_nodes_df_` (`environment.py` `get_track_grid`). Reduce to interior
  bin-center adjacency (bin-center nodes are `is_bin_edge=False`; `place_bin_centers_nodes_df_.node_id`
  maps bin → node, `-1` for gap bins). Junctions connect arms through shared original
  nodes; gap bins (`is_track_interior=False`, `edge_spacing>0`) stay disconnected so
  arms don't leak. Edges carry `distance`.

`bin_sizes` (per interior bin volume) come from the environment edges: N-D = product of
per-dim `np.diff(edges_[d])` widths, meshed in `'ij'` order, then interior-masked
(exclude padding bins, `environment.py:1068`); 1D linearized = `np.diff(place_bin_edges_)`
interior-masked (exclude the wide gap "bins", `environment.py:1481`).

Correctness pinned by round-trip tests (a unit field at interior bin `k` diffuses to a
bump centered on `k`) for **both** branches, plus a junction/gap test for the linearized
branch.

### `sorted_spikes_diffusion.py`

Mirrors `sorted_spikes_kde.py` structure and its `mean_rate · marginal/occupancy` form:

- `fit_sorted_spikes_diffusion_encoding_model(position_time, position, spike_times,
  environment, weights=None, sampling_frequency=500, position_std=sqrt(12.5),
  block_size=100, disable_progress_bar=False) -> dict`
  - build & cache the operator `L`, `node_order`, `bin_sizes` from the adapter;
  - pixellate occupancy (weighted by `weights`, **mirroring KDE's convention — default
    ones, not dt**) and each neuron's spike positions to count fields on interior bins;
  - **batch** occupancy + all neuron fields through one `diffuse` call;
  - **normalize each smoothed field to an ∫=1 density** (`to_density`), so `marginal` and
    `occupancy` match KDE's normalized densities;
  - `mean_rate_k` computed as in KDE (weighted spike count / weight sum);
  - `place_field_k = mean_rate_k · marginal_k / occupancy`, with KDE's zero-occupancy
    guard (`jnp.where(occupancy>0, marginal/occupancy, EPS)`) and `EPS` floor;
  - return the same dict keys as `sorted_spikes_kde` **plus** `diffusion_operator`,
    `node_order`, `bin_sizes` (the base class splats the whole dict into predict as
    kwargs — every key must be a predict parameter).
- `predict_sorted_spikes_diffusion_log_likelihood(...)` — a **dedicated** function
  (not the KDE predict: the base class splats the encoding dict as kwargs, so signature
  must match the diffusion dict keys exactly). Non-local branch copies KDE's non-local
  body (uses only `place_fields`/`no_spike_part_log_likelihood`/`is_track_interior`).
  Local branch indexes place fields by the animal's interpolated bin (`get_bin_ind`).

`position_std` keeps its meaning (bandwidth in coordinate units → `t=σ²/2`).

### Data flow

```text
Environment (fitted; N-D grid, 1D grid, or linearized track_graph)
  └─ environment_graph → (interior-bin graph + distances, node_order, bin_sizes)
       └─ build & cache finite-difference L (σ-independent)        [once, at fit]
position samples ─ pixellate(weights) → occupancy count field  ┐
neuron k spikes  ─ pixellate           → marginal_k count field ┤ stack columns
                                                                ├→ expm_multiply(-tL, U)  [NumPy]
                                                                ┘
  smoothed → to_density (÷ Σ·bin_sizes) → marginal_k, occupancy densities
    └─ place_field_k = mean_rate_k · marginal_k / occupancy   [zero-occ guard, EPS floor]
         └─ encoding dict (KDE keys + operator/node_order/bin_sizes)
              └─ HMM predict: Σ_k [ count_k·log λ_k − λ_k ]   (JAX, unchanged)
```

## Error handling

Reuse `_validation` / `ValidationError`:
- require a fitted `Environment` with `place_bin_centers_`, `is_track_interior_`, and the
  relevant graph (`track_graphDD` or `track_graph_with_bin_centers_edges_`); clear error
  if missing;
- reject non-positive `position_std`;
- **σ-guard:** warn when `position_std` < ~1 bin width (operator barely smooths);
- **degenerate cases:** empty spike train → zero field (well-defined); zero-occupancy
  interior bin → KDE's `where(occ>0, …, EPS)` guard; disconnected component / single-node
  interior → `exp(-tL)=I` on that node (no smoothing, no error);
- floor place fields at `EPS`; clip the summed no-spike term once, matching `sorted_spikes_kde`.

## Testing strategy (TDD order — each RED first)

1. **Adapter round-trip** — unit field at interior bin `k` → bump centered on `k`;
   interior relabeling matches `place_bin_centers_[is_track_interior]` order. Both the
   `track_graphDD` branch and the linearized `track_graph` branch.
2. **Bandwidth invariance** — single binned point far from boundaries: recovered std =
   `position_std` within tolerance across ≥3 bin sizes (the B1 regression guard).
3. **Analytic-Gaussian equivalence** — that point ≈ exact Gaussian of std `position_std`,
   `<2%`, in 1D and N-D.
4. **Mass conservation** — one self-consistent invariant: `Σ smoothed = Σ field` (field
   sum) to ~1e-10; and the ∫=1 density normalization integrates to 1.
5. **Non-uniform-grid correctness** — uneven 1D bins: `to_density` + ratio matches a dense
   finite-difference oracle kernel; guards against the volume-normalization bug.
6. **No cross-wall / arm leakage** — 2D barrier and a linearized W-track: zero mass
   crosses the gap / between arms; a layout Euclidean KDE smears across stays separated.
7. **σ-guard** — warns when `position_std` < ~1 bin width.
8. **Drop-in KDE equivalence** — wall-less open field, well-sampled trajectory: diffusion
   `place_fields` ≈ KDE `place_fields` at **interior bins away from the boundary**, within
   a stated tolerance (~5%); boundary bins legitimately differ (different kernels) and are
   excluded — this also pins the density normalization so units match.
9. **Invariants** (`property`) — place fields ≥ 0 and finite; posteriors sum to 1; no NaN/Inf.
10. **Scaling** — the `expm_multiply` path stays `O(nnz)` (no dense `n_bins×n_bins`
    materialization) on a large-bin environment.
11. **End-to-end** — fit a sorted-spikes decoder with `sorted_spikes_diffusion` on
    simulated replay; recover the trajectory; capture a **new** syrupy snapshot; then a
    smallest-real-slice smoke test.

Invariant tests (2/3/4/6) are adapted from spatstat's `test_density_heat.py`. Numerical
validation per `CLAUDE.md`: `pytest -m property`, golden regression, the new snapshot.
The `jax` and `numerical-validation` skills apply during implementation.

## Tradeoffs

**Wins:** manifold-aware, mass-conserving place fields in 1D and N-D with a **verified
grid-independent physical bandwidth**; exact-in-time operator applied at `O(nnz)` memory
via `expm_multiply`; a single geometry representation (the environment's graph); drop-in
at the model level; no new dependency.

**Costs:** a small ported/derived numerical unit maintained in nld (finite-difference
Laplacian + `expm_multiply`); we deliberately deviate from neurospatial's edge weighting
(and neurospatial's `diffusion_kde` carries the same grid-dependence — file upstream);
`expm_multiply` cost grows with σ and `nnz`; bin-resolution local decoding; the linearized
`track_graph` adapter (bin-center reduction with junction/gap handling) is the trickiest
piece and carries the most implementation risk; 2D `1/d²` 8-connected isotropy is
approximate (validated by the analytic-Gaussian test, not assumed).

## Out of scope

Clusterless; LOO bandwidth selection; adaptive/per-bin bandwidth; Richardson; sub-bin
local decoding.
