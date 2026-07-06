# Phase 2 — `sorted_spikes_diffusion` likelihood + registration

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#density)

Wires the Phase-1 engine into a registered, opt-in sorted-spikes likelihood that is a
model-level drop-in for `sorted_spikes_kde`.

**Inputs to read first:**

- [src/non_local_detector/likelihoods/sorted_spikes_kde.py](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py)
  — the module to mirror: fit ([:65](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L65)),
  predict ([:237](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L237)),
  place-field formula + guard ([:203-219](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L203-L219)),
  non-local predict body ([:339-366](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L339-L366)).
- [src/non_local_detector/likelihoods/__init__.py:39-52](../../../../src/non_local_detector/likelihoods/__init__.py#L39-L52)
  — `_SORTED_SPIKES_ALGORITHMS`.
- [shared-contracts.md](shared-contracts.md#encoding-dict) — the splat contract (every dict
  key is a predict param) and the exact key list.
- [designs.md](designs.md#density) — density normalization + place-field formula.
- [src/non_local_detector/models/base.py:3861-3890](../../../../src/non_local_detector/models/base.py#L3861-L3890),
  [:4043-4074](../../../../src/non_local_detector/models/base.py#L4043-L4074) — dispatch + splat.

**Contracts referenced:** [encoding-dict + predict](shared-contracts.md#encoding-dict),
[engine API](shared-contracts.md#engine-api), [eig cache](shared-contracts.md#eig-cache).

## Tasks

- Create `src/non_local_detector/likelihoods/sorted_spikes_diffusion.py`:
  - `fit_sorted_spikes_diffusion_encoding_model(position_time, position, spike_times,
    environment, weights=None, sampling_frequency=500, position_std=sqrt(12.5),
    block_size=100, disable_progress_bar=False)` — parameter names must match
    `_encoding_model_data` keys (filtered by `inspect.signature`, base.py:3886). Call the
    engine (`environment_graph` → cached `diffusion_eigenbasis`); apply the σ-guard; pixellate
    occupancy (weights = KDE convention, default ones) + per-neuron spike-count fields onto
    interior bins in `node_order`; batch-`diffuse`; `to_density`; compute
    `mean_rate·marginal/occupancy` with the KDE guard + `EPS` floor ([designs.md](designs.md#density)).
    Return the [encoding-dict keys](shared-contracts.md#encoding-dict) + `node_order`,
    `bin_sizes`, engine handle.
  - `predict_sorted_spikes_diffusion_log_likelihood(...)` — **dedicated** function; params ==
    encoding-dict keys (do not call the KDE predict). Non-local branch = copy of KDE's
    non-local body ([sorted_spikes_kde.py:339-366](../../../../src/non_local_detector/likelihoods/sorted_spikes_kde.py#L339-L366));
    local branch indexes `place_fields` by the animal's interpolated bin
    (`environment.get_bin_ind`), returning `(n_time, 1)`.
- Register in [likelihoods/__init__.py:39-52](../../../../src/non_local_detector/likelihoods/__init__.py#L39-L52):
  import the two functions and add
  `"sorted_spikes_diffusion": (fit_…, predict_…)` to `_SORTED_SPIKES_ALGORITHMS`. Existing
  entries untouched.
- Baseline-measurement task (per CLAUDE.md optimization workflow): a script/notebook (not a
  test) recording eig-build time, per-fit `diffuse` time, and per-EM-iteration cost on a
  representative env (`n_bins`, `n_neurons`, `position_std`); note the dense-vs-truncated
  crossover. Record numbers in the PR description; resolves Open Question 1.
- Docs (ship with the change): CHANGELOG `## [Unreleased] / ### Added` entry describing the
  new `sorted_spikes_algorithm="sorted_spikes_diffusion"` option and when to prefer it
  (geometry-respecting place fields); a NumPy-style module docstring paralleling
  `sorted_spikes_kde.py`'s.

## Deliberately not in this phase

- Engine internals → Phase 1 (consume `diffusion.py` as a black box).
- MRF-GAM / `sorted_spikes_mrf` → Phase 3.
- Any change to `sorted_spikes_kde` or other algorithms — additive only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_fit_encoding_dict_keys_and_shapes` | dict has all [contract keys](shared-contracts.md#encoding-dict); `place_fields` shape `(n_neurons, n_interior)`, all > 0, finite. |
| `test_kde_dropin_equivalence` | wall-less `simple_2d_environment`, well-sampled trajectory: diffusion vs KDE `place_fields` agree ≤ 5% at interior bins away from the boundary (pins units). |
| `test_predict_shapes_local_nonlocal` | non-local `(n_time, n_interior)`, local `(n_time, 1)`; finite; mirrors `test_sorted_spikes_kde.py`. |
| `test_predict_signature_matches_dict` | `set(inspect.signature(predict).parameters) ⊇ set(encoding_dict) ∪ {time,is_local}` (guards the splat contract). |
| `test_invariants` (`property`) | place fields ≥ 0 & finite; per-time posteriors from the LL sum to 1; no NaN/Inf. |
| `test_end_to_end_decoder` (`integration`) | fit a `SortedSpikesDetector` with `sorted_spikes_algorithm="sorted_spikes_diffusion"` on simulated replay; recovers the trajectory; **new** syrupy snapshot (not a diff of an existing one). |
| `test_real_slice_smoke` (`integration`, `slow`) | smallest real-data slice fits + decodes without NaN/Inf. |

## Fixtures

`simple_2d_environment`, `simple_1d_environment`, `synthetic_spike_data`
([conftest.py:232](../../../../src/non_local_detector/tests/conftest.py#L232)); snapshot via
`syrupy` following [test_regression_snapshots.py](../../../../src/non_local_detector/tests/likelihoods/test_regression_snapshots.py).
Real slice: smallest available fixture referenced by an existing `integration` test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Fit/predict mirror `sorted_spikes_kde` semantics; predict is dedicated (not the KDE
  predict); every encoding-dict key is a predict parameter.
- Registry entry added; no existing algorithm touched.
- Snapshot is newly added and reviewed as an approved baseline (follow the CLAUDE.md
  snapshot-analysis process — do not `--snapshot-update` an existing one).
- KDE drop-in tolerance met; invariants hold; `integration`/`slow` marked.
- CHANGELOG + module docstring updated in this PR (not deferred).
- No plan/phase references in code, tests, or docstrings.
