# Fitted state-space object

Written 2026-11-05 against main c80eaebf. Execute after the core-internals branch
(EM transition-statistics deletion, time_edges callbacks) and the detector-contracts
branch have merged; re-verify every line number below before editing.

## Goal

Assemble the fitted state-bin layout and continuous transition once per fit into
one frozen `StateSpace` (`detector.state_space_`). Every reader uses its fit-time
facts, never the current constructor params, so `set_params` after fit cannot reach
decoding.

## Done when

- `fit` → `set_params(local_position_std=0.0 | state_names=… | observation_models=…)`
  → `predict` gives exactly the pre-`set_params` result (today: broadcast error).
- Golden regression and all suites pass unchanged.
- A before/after harness shows `np.array_equal` posteriors, `Dataset.identical`
  results, and byte-identical `.nld` members (ignoring `package_version`).
- The `main-*` compat pickles decode as before; released pickles still load and
  inspect (released 0.5/0.6 pickles are already refused at decode on main —
  `test_stored_artifact_compatibility.py:547` — keep that behaviour).
- base.py loses about 250 lines of assembly code.

## Where current params leak in after fit (main c80eaebf)

- base.py:2273, 2358 (`local_position_std` in likelihood rows); base.py:1018
  (`_decode_reads_position`), 1863, 2126.
- _input_validation.py:1159, 1475 (`observation_models` → encoding keys).
- _results.py:65 (`has_place_bins`), 246–247, 259 (current `state_names` /
  `observation_models`).
- _checkpointed.py:419–431.
- model_format/_detector.py:435 (`_data_blocks` reads current
  `continuous_transition_types`).
- `_fit` overwrites a constructor param at base.py:1516, which `_rebuild_continuous`
  undoes at _detector.py:848.

Reference counts in src (incl. docstrings): `state_ind_` 28 refs / 6 files,
`discrete_state_transitions_` 30 / 5, `continuous_state_transitions_` 23 / 5,
`is_track_interior_state_bins_` 18 / 5, `n_discrete_states_` 14 / 3.

## Fixed constraints (and why)

1. **EM-updated values stay plain detector attributes**: `discrete_state_transitions_`,
   `discrete_transition_coefficients_` / `_smoothness_` / `_coefficient_covariance_`,
   `initial_conditions_`, `continuous_initial_conditions_`,
   `discrete_initial_conditions_`, `transition_time_bin_width_`. Spyglass reads
   `vars(classifier).get("discrete_transition_coefficients_")` (a property would
   silently return None); EM edits them in place (_em.py:1474, 1497);
   `_restore_state_on_error` (base.py:2878) snapshots `__dict__` shallowly; the
   `.nld` `_STORED` reader uses `setattr`.
2. **The object never enters `jit`.** Plain frozen dataclass, not a pytree; no `jnp`
   conversion; no stored interior products (a second n² copy repeats the past +17%
   GPU peak regression). Core keeps receiving the same NumPy arrays from the same
   indexing (base.py:2801–2836).
3. **`state_space_` is not a constructor param** (trailing `_`), so
   clone/get_params/set_params are unaffected.
4. **`.nld` members stay the same**: the object is rebuilt on load, not stored;
   state.json key names unchanged.
5. Behaviour-preserving except the drift fix in C3: any other output change is a bug.

## Design

New module `src/non_local_detector/models/_state_space.py` (its path becomes a
pickle contract — keep it stable).

```python
@dataclass(frozen=True, eq=False)
class StateSpace:
    """Fit-time layout of the state bins and the continuous transition."""
    state_names: tuple[str, ...]
    observation_models: tuple[ObservationModel, ...]
    continuous_transition_types: tuple[tuple[Any, ...], ...]  # as configured
    local_position_std: float | None
    state_ind: np.ndarray          # (n_state_bins,) int
    bin_sizes: np.ndarray          # (n_states,) int
    is_track_interior: np.ndarray  # (n_state_bins,) bool
    continuous_transitions: np.ndarray | LazyDenseTransition
    transition_operator: Any | None
    transition_representation: TransitionRepresentation
    max_dense_transition_bytes: int

    n_states / n_state_bins / interior_state_ind  (properties)
    has_place_bins(state) -> bool        # replaces _results.has_place_bins
    likelihood_is_local(state) -> bool   # obs.is_local and std is None
    densified() -> StateSpace            # dataclasses.replace; EM "auto" and load
```

- Free functions moved out of base.py: `state_layout(...)` (base.py:1382–1426),
  `continuous_state_transitions(...)` (1471–1593), initial conditions (1428–1469)
  taking the layout. `_fit` keeps its order and builds `StateSpace` at the end.
- Read-only detector properties (released in 0.6.x, used in tutorials 02/04):
  `state_ind_`, `n_state_bins_`, `n_discrete_states_`, `bin_sizes_`,
  `is_track_interior_state_bins_`, `continuous_state_transitions_`. Unfitted →
  `NotFittedError` (an `AttributeError`, so `hasattr` behaves as today).
- Dropped (unreleased): `_continuous_transition_operator_`,
  `_transition_representation_`, `_max_dense_transition_bytes_`.
- Stays on the detector: `environments_`, `encoding_model_`, `mark_reference_`,
  `time_contract_`, `log_likelihood_`, covariate design structures.
- Intentionally decode-time settings: `sampling_frequency`, `no_spike_rate`,
  `non_local_position_penalty` / `_std`, algorithm execution settings.

### Persistence

- `.nld`: `_REBUILT` gains `state_space_`, loses the moved names. `_STORED`
  `transition_representation` / `max_dense_transition_bytes` keep file names but
  feed the builder. `_build_detector` / `_rebuild_continuous` (_detector.py:816–854)
  call the builder; the param-restore hack goes away. `save_model` refuses when
  `get_params()` differs from the object's recorded params (user decision pending:
  refuse vs store fit-time params — must be settled before the 0.7.0 freeze).
- Pickles: `_upgrade_state_space` in `_upgrade_saved_state` (_persistence.py:82),
  after the `environments_` backfill (:151): pop legacy keys, build `StateSpace`
  from the exact stored arrays; defaults representation `"dense"`, max bytes
  `DEFAULT_MAX_DENSE_BYTES`, operator `None`; check `bin_sizes.sum() ==
  len(state_ind)` and per-state counts, else raise asking for a refit.
- New test `test_unpickled_detector_holds_its_fitted_state_space` over all `MODELS`.

## Commit sequence (each keeps full suite + golden green)

- C0 (uncommitted): reference harness at the base commit, saved to scratchpad —
  dense/structured/auto; `local_position_std` ∈ {None, 0, 2}; MultiEnvironment,
  NoSpikeContFrag, ContFrag; covariate `DiscreteNonStationaryDiagonal` with
  dense and checkpointed EM; checkpointed predict; `most_likely_sequence`; pickle
  and `.nld` round trips with bytes of every member.
- C1: move-only extraction of the three builders; base methods become thin wrappers.
- C2: add `StateSpace`, `state_space_`, properties, writers (`_choose_em_representation`
  _em.py:755–760 uses `densified()`), `.nld` load path, pickle backfill; delete
  `_initialize_state_index` / `_initialize_initial_conditions` /
  `_initialize_continuous_state_transition`; migrate tests (test_base_initialization
  ~16 sites, test_local_position_std ~36, transitions/test_transition_operators 1,
  four private-name test files). Readers untouched → identical outputs.
- C3: switch every leak site to the object (the bug fix); `_compute_local_position_kernel`
  takes sigma; `has_place_bins` callers use the method (incl.
  test_result_schema.py:25); save-on-drift refusal; drift regression tests;
  CHANGELOG "Fixed".
- C4: package code reads `self.state_space_.*` directly; properties remain for users.
- C5: docs (result_layout.md, `_DetectorBase` docstring, CHANGELOG "Changed").

Validation per commit: compare against C0; `tests/models`, `tests/transitions`,
`tests/model_format`, `tests/models/test_stored_artifact_compatibility.py`, golden;
`jax.log_compiles` on two predicts (no new compiles); transition dtype/shape passed
to core unchanged.

## Module splits

- `core.py`: do not split. checkpointed_inference.py imports 12 names (10 private);
  26 test files import from core; test_row_slice_callback.py:281–284 monkeypatches
  `core._filter_internal` etc. — a split would make those patches silently stop
  intercepting.
- `discrete_state_transitions.py`: after the EM-statistics deletion, move the
  estimation block (CovariateTransitionFit … `_discrete_transition_from_statistics`)
  to `_transition_estimation.py`, move-only; update patch targets in
  test_discrete_transitions_estimation.py. Keep the patsy spline transforms where
  they are (`dmatrix` resolves `quantile_bs` / `cyclic_bs` from that module's
  namespace; they are in its `__all__`).

## Decisions (user, 2026-11-05)

1. Save after param drift: **refuse** with a clear error (set the params back or
   refit). No new stored fields, no schema change.
2. Document `state_space_`; **do not export** `StateSpace` from the package root.
3. Covariate design structures **stay on the detector**, outside the object;
   revisit after the EM-statistics deletion.
4. `no_spike_rate` **stays a decode-time setting** for now. The user thinks it
   should probably be fixed at fit eventually — leave it out of this refactor; do
   not design around preventing that later.
5. Do not add tests or rules that force edits on routine future changes (adding a
   param, field or state type) — see the user's no-defensive-machinery preference.

Known remaining gap: `initialize_environments` on a fitted detector can still
desync `environments_` from the layout (as today).
