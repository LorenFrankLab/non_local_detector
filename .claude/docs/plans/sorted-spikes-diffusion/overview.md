# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Source of truth for the design: `docs/superpowers/specs/2026-07-06-sorted-spikes-diffusion-design.md`.

## Current codebase integration points

- `src/non_local_detector/likelihoods/__init__.py:29-38` — `_SORTED_SPIKES_ALGORITHMS` (39-52 is `_CLUSTERLESS_ALGORITHMS`)
  registry. **Add** a `"sorted_spikes_diffusion"` entry (Phase 2) and later
  `"sorted_spikes_mrf"` (Phase 3), each `(fit_fn, predict_fn)`. Existing entries untouched.
- `src/non_local_detector/models/base.py:3861` — `encoding_algorithm, _ =
  _SORTED_SPIKES_ALGORITHMS[self.sorted_spikes_algorithm]`; `:3886` —
  `sig = inspect.signature(encoding_algorithm)` filters `self._encoding_model_data` to the
  fit function's parameters; `:3890` calls it. **Implication:** the fit function's
  parameter names must match the encoding-model-data keys it wants
  (`position_time`, `position`, `spike_times`, `environment`, `weights`,
  `sampling_frequency`, …). No base.py change needed.
- `src/non_local_detector/models/base.py:4043-4074` — `_, likelihood_func =
  _SORTED_SPIKES_ALGORITHMS[...]`; `likelihood_func(time, position_time, position,
  spike_times, **self.encoding_model_[likelihood_name], is_local=…)`. **The entire encoding
  dict is splatted as kwargs** → every dict key must be a predict parameter (see
  [shared-contracts.md](shared-contracts.md#encoding-dict)).
- `src/non_local_detector/models/base.py:1991` — EM M-step calls
  `self.fit_encoding_model(**self._encoding_model_data, weights=local_state_weights)`. The
  encoding model (and thus the diffusion) is **refit every EM iteration** with the graph
  and `position_std` constant → the eig cache must live on the `Environment`, not be
  rebuilt per fit.
- `src/non_local_detector/environment.py:480-481` — `fit_place_grid` invalidates the lazy
  `_bin_distance_matrix_` cache. The new eig cache follows the same pattern: a lazily
  populated attribute invalidated here.
- `src/non_local_detector/likelihoods/sorted_spikes_kde.py` — the module Phase 2 mirrors
  (fit at `:65`, predict at `:237`, place-field formula `:204-219`, zero-occupancy guard
  `:203-217`).
- `src/non_local_detector/likelihoods/sorted_spikes_glm.py:223,359` — fit/predict the
  Phase 3 MRF-GAM mirrors structurally (Poisson GLM family).
- `CHANGELOG.md` — `## [Unreleased] / ### Added`: Phase 2 and Phase 3 each add an entry.

## Scope and dependency policy

### Goals

- A registered, opt-in `sorted_spikes_diffusion` sorted-spikes likelihood: manifold-aware
  place fields for N-D grids, 1D grids (both via `track_graphDD`), and linearized
  `track_graph` environments (via `track_graph_with_bin_centers_edges_`).
- A shared spectral engine (finite-difference `L`, `eig(L)`, diffuse-via-modes) cached on
  the `Environment`, reused across neurons and EM refits.
- Model-level drop-in: same encoding dict contract + Poisson log-likelihood as
  `sorted_spikes_kde`.
- Fast-follow `sorted_spikes_mrf` (penalized-Poisson MRF-GAM) on the same engine.

### Non-Goals

- Clusterless (mark-space) diffusion.
- Adaptive / per-bin bandwidth; LOO/REML bandwidth for the *diffusion* smoother (REML is
  the MRF-GAM's job).
- Richardson extrapolation; sub-bin local decoding.
- Modifying or replacing `sorted_spikes_kde` / any existing algorithm — this is additive.

### Dependency policy

No new hard dependency. `scipy` (`scipy.linalg.eigh`, `scipy.sparse.linalg.eigsh`,
`scipy.sparse`) and `networkx` are already dependencies. Phase 3 fits the population GLM in
JAX (already the backend); depending on NeMoS vs implementing the batched fit directly is a
Phase-3 decision recorded there — default is to implement directly (no new dependency).

## Metrics

- **Bandwidth invariance:** recovered smoothing std = `position_std` within 5% across bin
  sizes {0.5, 1, 2, 4}× **in 2D** (face-adjacency; the B1 + Moore-oversmoothing guard; see
  [appendix.md](appendix.md)).
- **Analytic-Gaussian:** single interior point ≈ exact Gaussian, max rel. error < 2% away
  from boundaries.
- **Mass conservation:** `Σ smoothed = Σ field` to ≤ 1e-10; density normalization ∫=1.
- **Mode reconstruction:** `Σⱼ exp(-tλⱼ)vⱼvⱼᵀ` vs `expm(-tL)` max abs diff < 1e-8 (full rank).
- **KDE drop-in:** diffusion vs KDE `place_fields` agree ≤ 5% at interior bins away from the
  boundary on a wall-less open field.
- **No leakage:** intensity across a wall / into a disconnected arm < 2% of peak.
- **Invariants:** place fields ≥ 0 & finite; posteriors sum to 1; no NaN/Inf.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Grid-dependent bandwidth (the B1 blocker) | Finite-difference `1/d²` weighting, verified grid-independent; bandwidth-invariance test across bin sizes is a hard gate. |
| Density normalization wrong on non-uniform bins (B2) | Explicit `to_density` via `bin_sizes`; non-uniform-grid oracle test. |
| Interior-bin ordering / relabel mismatch | Round-trip test (unit field at bin k → bump on k) for both adapter branches; relabel in `np.where(is_track_interior.ravel())[0]` order. |
| Linearized `track_graph` adapter (junctions/gaps) — highest risk | Isolated in its own Phase-1 task with junction + gap tests; a fallback construction is documented in [designs.md](designs.md#adapter-1d). |
| Predict signature mismatch (base.py splats the dict) | Dedicated predict whose params == encoding-dict keys; enforced by an end-to-end model test. |
| Truncated-eig approximation error at scale | Default dense `eigh`; truncation only above a bin-count threshold, validated by the mode-reconstruction test at moderate size. |
| Interior-only `place_fields` breaks gap/barrier envs (`get_bin_ind` is full-grid) | Store FULL-GRID `place_fields`/`no_spike_part_log_likelihood` like KDE (scatter into zeros); test on a gap/barrier env. |
| 2D Moore-diagonal edges oversmooth ≈√2 with `1/d²` | N-D adapter keeps only face-adjacent edges; bandwidth-invariance test runs in 2D. |
| `eigsh(sigma=0)` singular/unreliable | Truncated solver uses `sigma=-1e-8` (or `which="SM"`); robustness test. |
| eig cache under-keyed by `rank` (first-caller-wins) | Cache is a dict keyed by `rank`; full-rank entry slices for smaller ranks. |
| Disconnected interior loses a component's null mode | Truncation requires `rank ≥ n_components` and keeps all zero modes; per-component mass test. |

## Rollout Strategy

Purely additive and opt-in. Users select `sorted_spikes_algorithm="sorted_spikes_diffusion"`
(or later `"sorted_spikes_mrf"`); the default remains `"sorted_spikes_kde"`. No existing
code path, output, or snapshot changes. New snapshots are added, not diffed. No
backwards-compatibility or deprecation concerns.

## Open Questions

1. **Truncated-eig bin-count threshold and default `rank`.** Deferred — the baseline-
   measurement task in Phase 2 records fit / per-EM-iteration cost; the threshold is chosen
   from that. Default before measurement: dense `eigh`.
2. **Phase 3: depend on NeMoS vs implement the batched population fit directly.** Deferred to
   Phase 3; default is to implement directly (no new dependency).
3. **Phase 3: shared vs per-neuron REML `λ`.** Deferred to Phase 3; default is a single
   shared `λ` (cheaper, standard for population fits), revisited if fields over/under-smooth.

## Estimated Effort

- Phase 1: ~350–500 LOC (`diffusion.py`) + ~300 LOC tests.
- Phase 2: ~200–300 LOC (`sorted_spikes_diffusion.py`) + registry + ~250 LOC tests + docs.
- Phase 3: ~300–400 LOC (`sorted_spikes_mrf.py`) + registry + ~250 LOC tests + docs.
