# Phase 3 (fast-follow) — `sorted_spikes_mrf` population Poisson GAM

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#mrf)

Adds a second, more principled estimator on the **same spectral engine**: a penalized-
Poisson MRF-GAM with occupancy as a log-offset and REML smoothness, fit for the whole
population at once. Ships after Phase 2; the engine, adapter, and `bin_sizes` are already in
place.

**Inputs to read first:**

- [designs.md](designs.md#mrf) — the population-fit algorithm.
- `/Users/edeno/Downloads/mgcv_mrf.py` — reference implementation (penalized IRLS
  `_fit_dense`, REML `_reml_dense`, `mrf_basis`, occupancy offset). See
  [appendix.md](appendix.md).
- [src/non_local_detector/likelihoods/sorted_spikes_glm.py:223,359](../../../../src/non_local_detector/likelihoods/sorted_spikes_glm.py#L223)
  — the Poisson-GLM fit/predict family this mirrors structurally.
- [shared-contracts.md](shared-contracts.md#engine-api) (eigenmodes as the shared basis),
  [encoding-dict contract](shared-contracts.md#encoding-dict).
- [NeMoS](https://github.com/flatironinstitute/nemos) `PopulationGLM` (appendix) — the
  shared-design-matrix population pattern.

**Contracts referenced:** [engine API](shared-contracts.md#engine-api) (reuse
`diffusion_eigenbasis` output as the basis `B`, eigenvalues as penalty weights),
[encoding-dict + predict](shared-contracts.md#encoding-dict).

## Tasks

- Create `src/non_local_detector/likelihoods/sorted_spikes_mrf.py`:
  - `fit_sorted_spikes_mrf_encoding_model(...)` — obtain `B` = `rank` smoothest eigenmodes
    and penalty weights `d` from the cached engine; pixellate spike counts
    `N (n_bins, n_neurons)` and occupancy `o`; fit the population penalized-Poisson GAM with
    occupancy as a shared log-offset and generalized-ridge penalty `λ·diag(d)`, vectorized
    over neurons in JAX (`vmap`) — the NeMoS trick ([designs.md](designs.md#mrf)); select `λ`
    by REML (default a single shared `λ`, Open Question 3). Return `place_fields = exp(η)`
    and the same [encoding-dict keys](shared-contracts.md#encoding-dict) as Phase 2 (+ any
    MRF-specific handles), so predict/HMM are unchanged.
  - `predict_sorted_spikes_mrf_log_likelihood(...)` — reuse the Phase-2 predict body
    (place-field-based non-local; bin-indexed local); factor the shared body into a small
    helper if it avoids duplication.
- Resolve Open Question 2 (depend on NeMoS vs implement the batched fit directly) at the top
  of this phase; default is implement directly (no new dependency). If depending on NeMoS,
  add it under an optional extra and record it in the dependency policy.
- Register `"sorted_spikes_mrf"` in
  [likelihoods/__init__.py:39-52](../../../../src/non_local_detector/likelihoods/__init__.py#L39-L52).
- Docs: CHANGELOG `### Added` entry (new option + when to prefer it: correct low-occupancy
  handling, automatic smoothness, SE); module docstring.

## Deliberately not in this phase

- Engine / adapter changes → Phase 1 (reuse as-is; eigenmodes are already exposed).
- The linear diffusion smoother → Phase 2 (shipped).
- Bayesian SE / edf surfaced through the *model* API — compute internally if needed for
  fitting, but exposing per-bin uncertainty to users is a separate follow-up.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_penalty_is_eigenvalue_ridge` | penalty weights returned equal the engine eigenvalues; basis columns are the smoothest modes (ascending eigenvalue). |
| `test_fullrank_matches_glm` | full-rank MRF at fixed `λ` matches an independent penalized-Poisson reference to < 1e-6 (mirrors `test_mgcv_mrf.py:test_fullrank_equals_glm`). |
| `test_occupancy_offset_no_denominator` | zero/low-occupancy bins produce finite rates (no division blow-up), unlike the ratio estimator. |
| `test_population_fit_matches_loop` | vectorized population fit == a reference per-neuron loop to tolerance (the NeMoS trick preserves results). |
| `test_reml_recovers_field` (`integration`) | REML selects a sensible `λ`; recovered field peak/centre/correlation match a simulated place cell (mirrors `test_mgcv_mrf.py:test_reml_recovers_field`). |
| `test_no_leak_barrier` (`integration`) | penalty does not smooth across a wall (far arm ≈ 0), since `L` is built on the masked graph. |
| `test_end_to_end_decoder` (`integration`) | `SortedSpikesDetector(sorted_spikes_algorithm="sorted_spikes_mrf")` decodes simulated replay; new snapshot. |

## Fixtures

Reuse Phase-1/2 fixtures plus a simulated place-cell trajectory + spikes (seeded), paralleling
`test_mgcv_mrf.py`'s simulation. No real data beyond the Phase-2 smoke slice.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Reuses the Phase-1 engine (no re-implementation of `L`/eig); population fit is vectorized
  (no per-neuron Python loop); occupancy enters as an offset, never a denominator.
- Registry entry added; Phase-2 code and all existing algorithms untouched.
- REML/`λ` selection is deterministic (seeded) and documented.
- New snapshot approved via the CLAUDE.md snapshot process; `integration` tests marked.
- CHANGELOG + docstring updated; no plan/phase references in code.
- Dependency decision (NeMoS vs direct) recorded; if added, it's an optional extra.
