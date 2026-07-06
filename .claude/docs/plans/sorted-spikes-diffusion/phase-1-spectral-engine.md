# Phase 1 — Spectral engine + environment-graph adapter

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Builds the geometry-only numerical core (`likelihoods/diffusion.py`) and its tests. No
likelihood, no model wiring — this phase is independently reviewable and ships green on its
own.

**Inputs to read first:**

- [designs.md](designs.md) — all six component algorithms (this phase implements
  [Laplacian](designs.md#laplacian), [eig+diffuse](designs.md#eig),
  [density](designs.md#density), [adapter-nd](designs.md#adapter-nd),
  [adapter-1d](designs.md#adapter-1d)).
- [shared-contracts.md](shared-contracts.md#engine-api) — engine signatures & invariants;
  [eig cache](shared-contracts.md#eig-cache).
- [src/non_local_detector/environment.py:1574-1664](../../../../src/non_local_detector/environment.py#L1574-L1664)
  — `make_nD_track_graph_from_environment` (node ids = flat bin index; edges carry
  `distance`).
- [src/non_local_detector/environment.py:1406-1539](../../../../src/non_local_detector/environment.py#L1406-L1539)
  — `get_track_grid` (produces `track_graph_with_bin_centers_edges_`,
  `place_bin_centers_nodes_df_`, gap bins).
- [src/non_local_detector/environment.py:478-481](../../../../src/non_local_detector/environment.py#L478-L481)
  — `_bin_distance_matrix_` cache invalidation pattern to mirror for the eig cache.
- [appendix.md](appendix.md) — B1/B2 verification numbers the tests reproduce.

**Contracts referenced:** [Engine API](shared-contracts.md#engine-api),
[Environment eig cache](shared-contracts.md#eig-cache) — do not weaken the ordering /
mass-conservation invariants.

## Tasks

- Create `src/non_local_detector/likelihoods/diffusion.py` with `build_laplacian`,
  `diffusion_eigenbasis`, `cached_eigenbasis` (the Environment-cache-owning wrapper —
  `diffusion_eigenbasis` takes `L` and cannot touch the cache), `diffuse`, `to_density`,
  `environment_graph` per
  [shared-contracts.md](shared-contracts.md#engine-api). Implement the math from
  [designs.md](designs.md) verbatim. Load-bearing details the review caught:
  - `build_laplacian` uses **finite-difference `1/d²`** on a **face-adjacent** graph.
  - truncated `diffusion_eigenbasis` uses `eigsh(..., sigma=-1e-8, which="LM")` (**not
    `sigma=0`** — singular/unreliable) and keeps **all** zero modes (one per connected
    component, `rank ≥ n_components`); dense `eigh` is the default.
  - `diffuse` via modes; `to_density` per [designs.md](designs.md#density). NumPy/SciPy only.
- Implement `environment_graph`'s **two branches** ([adapter-nd](designs.md#adapter-nd),
  [adapter-1d](designs.md#adapter-1d)) plus `bin_sizes` derivation for each. The N-D branch
  **must drop Moore/diagonal edges, keeping only face-adjacent pairs** (diagonals with
  `1/d²` oversmooth ≈√2 in 2D — verified). The linearized branch (chain+junction, already
  face-adjacent) is the highest-risk item — implement the primary contraction construction;
  if it proves brittle, switch to the documented fallback and note which was used. Both must
  pass the round-trip + junction + gap tests below.
- Add the eig cache on `Environment`: a lazily-populated `_diffusion_eigenbasis_` attribute
  (populated by the engine, not the dataclass), and add its invalidation next to the
  existing `_bin_distance_matrix_` reset in `fit_place_grid`
  ([environment.py:480-481](../../../../src/non_local_detector/environment.py#L480-L481)). This
  is the only edit to `environment.py`.
- Guard rails in the engine: raise `ValidationError` for a missing graph / non-positive
  inputs; a `UserWarning` when the caller's `sigma` < ~1 bin width (σ-guard) — put the
  σ-guard in a small helper the Phase-2 fit calls, since the engine itself is
  σ-parameterized per `diffuse` call.
- Docstrings: NumPy-style with array shapes; no references to this plan or phase numbers.

## Deliberately not in this phase

- `sorted_spikes_diffusion.py`, the registry entry, and any `base.py` model wiring → Phase 2.
- The place-field formula / `mean_rate·marginal/occupancy` / Poisson likelihood → Phase 2
  (the engine returns smoothed densities; it does not know about spikes or rates).
- MRF-GAM / eigenmode basis consumption → Phase 3.
- CHANGELOG entry → Phase 2 (the engine is internal; nothing user-facing ships yet).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_laplacian_symmetric_zero_rowsum` | `L == L.T`; `L @ ones ≈ 0`; PSD (min eigenvalue > −1e-9). |
| `test_adapter_roundtrip_nd` | unit field at interior bin `k` (2D grid env) diffuses to a bump whose argmax is `k`; `node_order == np.where(is_track_interior.ravel())[0]`. |
| `test_adapter_roundtrip_linearized` | same round-trip on a `track_graph` env (built as in `tests/environment/test_multi_edge_track_graph.py`). |
| `test_mode_reconstruction` | full-rank `(eigvecs*exp(-t·eigvals)) @ eigvecs.T` vs `scipy.linalg.expm(-t·L)`: max abs diff < 1e-8; truncated rank within stated tol. |
| `test_truncated_eigsh_robust` | truncated `diffusion_eigenbasis` succeeds (no singular-factor error) and returns the correct smallest eigenpairs incl. the zero mode; equals a dense-`eigh` slice. |
| `test_bandwidth_invariance` | recovered smoothing std = `sigma` within 5% across bin sizes {0.5,1,2,4}, **in 2D** (face-adjacency); a Moore/8-connected graph would fail at ≈1.41× (guards [appendix.md](appendix.md) #3). |
| `test_analytic_gaussian` | single interior point ≈ exact Gaussian std `sigma`, max rel err < 2% away from boundary (1D and 2D). |
| `test_disconnected_components_mass` | on a two-arm (disconnected interior) env, each component conserves its own mass; both null modes present in a truncated basis. |
| `test_mass_conservation` | `Σ diffuse(...) == Σ field` to 1e-10; `to_density` output integrates (`bin_sizes @ ·`) to 1. |
| `test_nonuniform_bins` | uneven 1D bins: `to_density`+ratio matches a dense finite-difference oracle kernel. |
| `test_no_leak_barrier` (`integration`) | 2D barrier + linearized W-track: intensity across gap / into far arm < 2% of peak. |
| `test_eig_cache_reuse` | second `environment_graph`/eig call returns the cached basis (identity check); `fit_place_grid` invalidates it. |

## Fixtures

Reuse `tests/conftest.py`: `simple_2d_environment` ([conftest.py:122](../../../../src/non_local_detector/tests/conftest.py#L122)),
`simple_1d_environment`, `simple_100_environment`. Build a `track_graph` env inline from the
pattern in `tests/environment/test_multi_edge_track_graph.py`. Analytic-Gaussian / barrier
fixtures synthesized in the test module (seeded `np.random.default_rng`). No real data.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented as specified; engine is NumPy/SciPy only; `environment.py` edit is
  limited to the eig-cache attribute + its invalidation.
- "Deliberately not in this phase" honored — no likelihood/registry/model code.
- Validation slice passes; `integration` test marked.
- Tests exercise real behavior (bandwidth numbers, mass, ordering), not tautologies; shared
  setup in fixtures.
- Docstrings/test/module names don't reference this plan or phase numbers.
- Engine invariants (ordering = interior flat-bin order; no dense kernel; `1ᵀL=0`) hold.
