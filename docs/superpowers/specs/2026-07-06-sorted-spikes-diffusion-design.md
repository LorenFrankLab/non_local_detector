# Sorted-spikes diffusion likelihood — design

**Date:** 2026-07-06
**Branch:** `sorted-spikes-diffusion`
**Status:** design, pending implementation plan

## Summary

Add a new, opt-in sorted-spikes likelihood algorithm, `sorted_spikes_diffusion`,
that estimates place fields with a **heat-kernel diffusion** intensity estimator
(the estimator behind `spatstat.explore::densityHeat.ppp`) instead of the current
Gaussian kernel density estimate (KDE). The diffusion is run as a random walk on
the `Environment`'s existing manifold graph, so it respects track geometry
(walls, holes, junctions) and generalizes across 1D linearized tracks and N-D
open fields without a separate pixel-grid representation.

The user brought four Downloads files as the starting point
(`density_heat.py`, `density_heat_jax.py`, `place_field.py`, `test_density_heat.py`).
The decision (see "Substrate", below) is to **adapt the spatstat algorithm onto our
own graph machinery** rather than vendor its pixel-grid implementation into the
production path. The spatstat NumPy code is vendored into the test suite only, as a
reference oracle.

## Motivation

Ordinary Gaussian KDE (`common.KDEModel`, used by `sorted_spikes_kde` and
`clusterless_kde`) smooths in the coordinate metric. In 2D it smears probability
across walls and holes (two points close in Euclidean space but far along the real
track get spurious shared density). The package currently sidesteps this by
linearizing to 1D via `track_graph`, which requires a hand-built graph and collapses
to 1D.

The diffusion estimator fixes cross-wall smearing directly, in whatever dimension,
and is a statistically principled intensity estimator:

1. **Reflecting (Neumann) boundaries** — no probability flux across a wall; the
   physically correct condition for an arena the animal cannot cross.
2. **Mass conservation** — the evolution operator is (sub)stochastic, so total
   intensity stays exactly `N`; downstream Poisson rates are correctly normalized
   with no geometry-dependent bias.
3. **Intrinsic (diffusion) distance** — smoothing follows the diffusion distance on
   the domain, correct around junctions and holes where Euclidean is wrong.

## Statistical foundation

`densityHeat` treats the binned point pattern as the initial condition of a diffusion
equation and evolves it for "time" `t = σ²`. The evolution operator is the heat
kernel `exp(-tL)`, where `L` is the graph Laplacian of the adjacency graph. On flat
unbounded space this is identical to Gaussian KDE with bandwidth `σ`. The discrete
update `u ← u A` is the explicit-Euler discretization of `∂u/∂t = -L u`. spatstat
runs this on a pixel-grid graph; we run the same diffusion on the `Environment`'s
manifold graph.

**Why graph diffusion is the principled choice (vs the two alternatives):**

- **Geodesic-distance KDE** (Gaussian of shortest-path distance, the mechanism
  `base.py` already uses for the non-local penalty and local-position kernel) is
  *not* a heat kernel: it is not mass-conserving, it double-counts mass at junctions,
  and it is only the small-`σ`, flat-space asymptotic limit of the heat kernel
  (Varadhan). It is acceptable as a soft *prior*, but biased as a *density estimator*
  feeding a Poisson likelihood — the bias becomes decoded-rate error. **Rejected.**
- **Pixel-grid diffusion** (vendor spatstat as-is) is equally principled as an
  estimator but cannot represent 1D linearized tracks (the dominant use case) and
  introduces a second discretization of the geometry (pixel grid + mask) that must be
  reconciled with the `Environment` grid via an error-prone transpose. **Rejected for
  production; retained as a test oracle.**
- **Graph diffusion** *is* spatstat's estimator, run on the manifold graph we already
  build. Reflecting boundaries and mass conservation are automatic from the graph's
  connectivity (no edges to non-interior bins ⇒ boundary nodes keep more mass at
  home). One source of truth for the geometry; general in 1D and N-D. **Chosen.**

## Goals / non-goals

**Goals**
- New registered `sorted_spikes_diffusion` algorithm selectable via
  `sorted_spikes_algorithm="sorted_spikes_diffusion"` in the decoder/classifier models.
- Manifold-aware place fields in 1D (linearized `track_graph`) and N-D (`track_graphDD`).
- Same encoding-model dict contract and Poisson log-likelihood as `sorted_spikes_kde`,
  so it is a drop-in at the model level.
- Rigorous numerical validation against the spatstat oracle and analytic Gaussians.

**Non-goals (YAGNI)**
- Clusterless (mark-space) diffusion — marks are not spatial; out of scope.
- Standard errors / leave-one-out / lagged arrivals / Richardson extrapolation —
  vendored in the oracle, not wired into the production estimator.
- Sub-bin-resolution local decoding (local decoding indexes place fields by the
  animal's bin; same discretization tradeoff as the grid).

## Substrate decision (resolved)

The manifold-aware estimator smooths on the **`Environment` graph**:
- **N-D:** `track_graphDD` — nodes are interior bin centers, edges are Moore-neighborhood
  adjacency between interior bins (`make_nD_track_graph_from_environment`), edge weight
  `distance` = Euclidean distance between adjacent centers.
- **1D:** the linearized `track_graph_with_bin_centers_edges_` / `distance_between_nodes_`.

Because edges only connect interior bins, the reflecting boundary is intrinsic to the
graph — no separate mask, no transpose/ravel adapter.

## Architecture

### Module layout

```
src/non_local_detector/likelihoods/
  diffusion_model.py            # DiffusionModel (KDEModel-compatible) + graph→operator build
  sorted_spikes_diffusion.py    # fit_/predict_ mirroring sorted_spikes_kde.py
  __init__.py                   # register in _SORTED_SPIKES_ALGORITHMS

src/non_local_detector/tests/likelihoods/
  density_heat_oracle.py        # vendored spatstat NumPy port (test oracle ONLY)
  test_sorted_spikes_diffusion.py
```

### `DiffusionModel` (the well-bounded unit)

Mirrors `common.KDEModel`'s interface so `sorted_spikes_diffusion.py` is
`sorted_spikes_kde.py` with the estimator swapped:

- `DiffusionModel(std, environment, connect=8, symmetric=False, block_size=None)`
- `.fit(samples, weights=None) -> self` — pixellate `samples` onto graph nodes
  (weighted counts per interior bin), store as the diffusion initial condition.
- `.predict(eval_points) -> density at eval_points` — diffuse the stored initial
  condition through the cached operator `A`, then read the density at `eval_points`
  by bin lookup (`environment.get_bin_ind`). Returns values on interior bins when
  `eval_points` are the interior bin centers.

Because diffusion is linear, multiple initial conditions (occupancy + every neuron's
marginal) are stacked as columns and evolved together in one `jax.lax.scan`.

### Transition operator `A` (built once, cached)

`A` depends only on `(graph, σ, connect, symmetric)` — not the samples — so it is
built once at `fit` time and cached on the encoding-model dict, then reused for
occupancy, every neuron's marginal, and predict.

Construction on the graph (nodes = interior bins, edge lengths `d_ij`):
- off-diagonal `A[i, j]` = jump probability `i → j`, a function of `d_ij`, `σ`, `Nstep`;
- diagonal `A[i, i] = 1 − Σ_j A[i, j]` (mass that does not jump stays);
- boundary/low-degree nodes keep more mass at home ⇒ automatic reflecting BC;
- **calibration:** choose per-step jump probability `p` and step count `Nstep` so that
  `Nstep × (per-step mean-squared displacement) = σ²` (matching Gaussian variance);
- **stability guards** ported from spatstat: enforce `Σ_j A[i,j] ≤ pmax < 1` and
  `Nstep = max(16, ⌈σ² / (2 · pmax · minstep²)⌉)`; raise on violation rather than
  silently producing negative/unstable probabilities.

`A` is stored as a JAX-friendly sparse structure (neighbor index arrays + value
arrays), so the walk is a batched segment-sum / sparse mat-vec inside the scan.

### Data flow

```
Environment (fitted, 1D or N-D)
  └─ manifold graph (track_graphDD or linearized graph) + edge lengths
       └─ build & cache A(σ, connect)                     [once, at fit]
position samples ─ pixellate → occupancy initial condition ┐
neuron k spikes  ─ pixellate → marginal_k initial condition ┤ stack columns
                                                            ├─→ one lax.scan: U ← Aᵀ U (Nstep)
                                                            ┘
  occupancy, marginals on interior bins
    └─ place_field_k(x) = mean_rate_k · marginal_k(x) / occupancy(x)   [floored at EPS]
         └─ encoding-model dict (same keys as sorted_spikes_kde)
              └─ HMM predict:  Σ_k [ count_k · log λ_k − λ_k ]   (unchanged)
```

### Interface contract (unchanged at the model level)

`fit_sorted_spikes_diffusion_encoding_model(position_time, position, spike_times,
environment, weights=None, sampling_frequency=500, position_std=sqrt(12.5),
connect=8, symmetric=False, block_size=100, disable_progress_bar=False) -> dict`

Returns the same keys `sorted_spikes_kde` returns (`environment`, `occupancy`,
`mean_rates`, `place_fields`, `no_spike_part_log_likelihood`, `is_track_interior`,
`disable_progress_bar`, plus the cached diffusion operator and models), so
`predict_sorted_spikes_diffusion_log_likelihood(...)` can share the non-local path of
`predict_sorted_spikes_kde_log_likelihood` verbatim (it uses only `place_fields`,
`no_spike_part_log_likelihood`, `is_track_interior`). The local path indexes place
fields by the animal's interpolated bin (`get_bin_ind`) instead of re-running a
smoother.

`position_std` keeps its current meaning (the equivalent-Gaussian standard deviation,
in coordinate units) — the calibration maps it to diffusion time `σ²`, so users do not
learn a new bandwidth concept.

## Error handling

Reuse `_validation` / `ValidationError` conventions:
- require a fitted `Environment` with `place_bin_centers_` and `is_track_interior_`;
- require the manifold graph to be present (`track_graphDD` for N-D, linearized graph
  for 1D); raise a clear error if missing;
- reject non-positive `position_std`;
- reject `connect` outside `{4, 8}`; raise on the spatstat stability-guard violation;
- floor place fields at `EPS`, matching `sorted_spikes_kde` (`min=EPS`), and clip the
  summed no-spike term once (not per neuron), matching the existing convention.

## Testing strategy (TDD order — each RED first)

The spatstat NumPy port is vendored into `tests/likelihoods/density_heat_oracle.py`
as the reference oracle (never imported by production code).

1. **Oracle port** — port `test_density_heat.py` to pytest (seeded, `property` /
   `integration` markers): mass conservation, single-point ≈ analytic Gaussian
   (<2%), connect 4/8 agreement, mask, weights, symmetric, varying-σ.
2. **Operator calibration** — flat/unbounded region: graph diffusion of a single
   binned point ≈ analytic Gaussian with bandwidth `σ` (far from boundaries), and
   ≈ the spatstat oracle on a 2D open-field grid.
3. **Mass conservation** — Σ (intensity × bin area) = `N` to ~1e-6, in 1D and N-D.
4. **No cross-wall leakage** — barrier environment: zero mass crosses the gap; a two-arm
   layout that Euclidean KDE smears across must stay separated under diffusion.
5. **1D correctness** — linear track ≈ 1D analytic Gaussian; W-track respects arm gaps.
6. **Graph-Laplacian consistency** — on the regular interior-bin lattice, the simple
   edge-length weighting is a consistent discretization of the continuous Laplacian
   (bandwidth recovered within tolerance); documents the assumption explicitly.
7. **Drop-in equivalence anchor** — wall-less open field, well-sampled trajectory:
   diffusion `place_fields` ≈ KDE `place_fields` within a few percent (also pins the
   `mean_rate · marginal / occupancy` normalization so units match exactly).
8. **Invariants** (`property`) — place fields ≥ 0 and finite; posteriors sum to 1;
   no NaN/Inf; JAX: no unexpected recompilation, shapes as expected.
9. **End-to-end** — fit a sorted-spikes decoder with `sorted_spikes_diffusion` on
   simulated replay; recover the trajectory; capture a **new** syrupy snapshot (not a
   diff of an existing one); then a smallest-real-slice smoke test.

Numerical validation per `CLAUDE.md`: run `pytest -m property`, the golden-regression
suite, and the new snapshot. The `jax` and `numerical-validation` skills apply during
implementation.

## Tradeoffs

**Wins:** manifold-aware, mass-conserving, statistically principled place fields in 1D
and N-D; single geometry representation (the graph); reuses the model-level contract so
it is a drop-in; no new hard dependencies (numpy/scipy/jax already present).

**Costs:** new numerical code (graph-Laplacian diffusion + σ calibration + stability
guards) rather than a straight vendor, so the validation burden is the crux — mitigated
by the spatstat oracle cross-check. Bin-resolution local decoding. `Nstep` grows with
`(σ / min-edge-length)²`, so fine bins or large σ cost more diffusion steps. New
parameters (`connect`, `symmetric`) to document. Graph-Laplacian↔continuous-Laplacian
correspondence is clean only for near-regular bin lattices (true here; validated in
test 6, not assumed).

## Out of scope

1D-vs-N-D is both supported; clusterless, SE/LOO/lagged/Richardson, and sub-bin local
decoding are out of scope for this change.
