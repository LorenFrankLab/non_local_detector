# Phase 4: declare when each constructor parameter is read

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Line numbers are for `0b88f984` and drift; find the code by symbol. This phase
does not depend on phases 1 to 3.

## Goal

Make the lifetime of every constructor parameter explicit, and treat a
fit-time parameter changed after a fit the same way whichever one it is.

A detector's constructor parameters are read at three different times:

- **at fit**: everything that shapes the fitted model (states, environments,
  transitions, the likelihood algorithm and its parameters);
- **at decode**: `sampling_frequency`, `no_spike_rate`, the two non-local
  penalty parameters, and each algorithm's execution settings (block sizes,
  progress bars);
- **by EM only**: the transition prior (`discrete_transition_concentration`,
  `_stickiness`, `_regularization`, `_prior_weight`), read only in
  `models/_em.py` (1197, 1216-1219, 1667-1669). `estimate_parameters` refits
  first, so these never act on a model from a plain `fit`.

`frozen_discrete_transition_rows` looks EM-only but is fit-time: `fit` reads it
to snapshot the frozen rows (`_initialize_discrete_state_transition`,
`models/base.py:1344-1350`), and `save_model` stores that snapshot
(`model_format/_detector.py:130-133`).

Nothing in the code says which is which. A fit records seven fit-time
parameters in `state_space_` (`layout_parameters`,
`models/_state_space.py:204`); decoding warns when one of those changed and
`save_model` refuses. The other fit-time parameters are not covered. Reproduced
on a simulated sorted-spikes detector, and re-run at `0b88f984` with the same
behaviour (the size of the change in the third row depends on the data):

| After `fit` | Result today |
| --- | --- |
| `set_params(sorted_spikes_algorithm_params={"position_std": 50.0})`, then `predict`, then `save_model` | Identical decode, no warning; the saved file's parameters say 50 for a model fitted at the default |
| `set_params(infer_track_interior=False)`, then `predict`, `save_model` | No warning; saved |
| `set_params(no_spike_rate=5.0, non_local_position_penalty=10.0)`, then `predict` | State probabilities change by up to 0.998 (intended: these are decode-time); the result records neither value |
| `fit_encoding_model` on an unfitted detector | `AttributeError: ... no attribute 'environments_'` |
| `set_params(sorted_spikes_algorithm_params={"position_std": 30.0})`, then `fit_encoding_model`, `predict`, `save_model` | The refit uses the new value (place fields move by up to 15), no warning, saved. This is correct and must stay so |

## Done when

- One place in the code names the decode-time and the EM-time constructor
  parameters. Any parameter not named there is fit-time, so a parameter added
  later needs no entry unless it is an exception.
- After a fit, changing any fit-time parameter and decoding warns once
  (`ParameterWarning`) naming every changed parameter, and `save_model`
  refuses, as both do for the seven today.
- Changing a decode-time or EM-time parameter, or an algorithm's execution
  setting, does neither.
- After `fit_encoding_model` refits with changed algorithm parameters, decoding
  does not warn about them and `save_model` accepts them.
- Results carry the decode-time parameter values that produced them.
- `fit_encoding_model` on a detector without fitted environments raises
  `NotFittedError` with a hint.
- Decoding results are unchanged, and every stored model, pickle and result
  written before this phase loads as before.

## Fixed decisions and constraints

- **No new stored field, no format-version bump, no migration** (user,
  2026-11-05). The record of fit-time values is kept in memory. A model loaded
  from a saved file rebuilds it from the file's parameters, which are the
  fit-time ones because `save_model` refuses a detector whose fit-time
  parameters changed.
- **The default is fit-time.** Reason: the user rejects rules that force an
  edit on routine changes. Defaulting to fit-time is also the safe direction: a
  new parameter left unclassified produces a warning, not a silent mismatch.
- **`no_spike_rate` stays decode-time** (user, 2026-11-05). Do not design
  against moving it to fit-time later.
- **The seven recorded parameters and what decoding reads from `state_space_`
  do not change.** This phase adds the remaining fit-time parameters to the
  comparison; it does not alter `StateSpace`'s existing fields or their use.
- **A refit of the encoding model updates the record of the algorithm
  parameters.** `fit_encoding_model` reads `*_algorithm_params` again, so after
  `set_params` and that refit the model really was fitted with the new values,
  and its saved file is right. Other fit-time parameters are read only by `fit`,
  so such a refit leaves them flagged.
- **The algorithm name keeps its stricter check.** Decoding with
  `sorted_spikes_algorithm` or `clusterless_algorithm` different from the one
  that fitted the encoding model raises (`models/base.py:1543-1565`). Leave
  that as it is.
- **Compare what the fit used, not how it was written.** For
  `*_algorithm_params`, compare the effective fit parameters: the fit
  function's own defaults, overlaid with the registry's `default_params` when
  the detector's value is None or with the user's dict otherwise (a user dict
  replaces the registry defaults, `likelihoods/__init__.py:427`), with the
  algorithm's `execution_settings` removed. Reason: `None` and a dict that
  spells out the same values must not warn, and execution settings are
  decode-time by design (`_execution_settings`, `likelihoods/__init__.py:468`).
- **Comparison is by saved form**, as for the seven
  (`_stored_form`, `model_format/_detector.py:648`): two values are the same if
  a saved model would store them the same. A value with no saved form cannot be
  compared and does not warn.
- **The EM-time set is the four prior parameters.**
  `frozen_discrete_transition_rows` is not in it (see "Goal"): it takes the
  fit-time default, so a file never holds rows that disagree with its stored
  snapshot.
- **EM-time parameters are only kept out of the fit-time warning.** Whether a
  saved file should record the prior that EM used is overview open question 5
  and is not built here.
- **Result attributes are added, none renamed.** The new ones are named after
  the parameters (`sampling_frequency`, `no_spike_rate`,
  `non_local_position_penalty`, `non_local_penalty_std`) and written only for
  parameters the detector has. Planner's call; overview open question 4.
- **Fitted attributes stay plain attributes** with their names; see
  [Stored artifacts](shared-contracts.md#stored-artifacts-that-must-not-change).

## Context to read

- `models/_state_space.py`: `StateSpace` (50; its docstring at 58-61 already
  says three of its fields are "recorded only so a later change ... can be
  detected"), `layout_parameters` (204), `recorded_layout_parameters` (236).
- `model_format/_detector.py`: `changed_layout_parameters` (668),
  `_stored_form` (648), the save refusal (741-750), `_fitted_attributes` (606)
  and `_REBUILT` (151), which decide which detector attributes `save_model`
  accepts, `_rebuild_state_space` (936), `_build_detector` (1273).
- `models/base.py`: `_fit` (1420-1524; the `StateSpace` is built at
  1514-1523), `_warn_if_layout_parameters_changed` (1633), called once from
  `_require_decodable` (`models/_input_validation.py:2206`), the algorithm-name
  check (1543-1565), `_fit_encoding_groups` (2625) and the family
  `_fit_encoding_model` bodies (3171, 4253), `_get_environment_by_name`
  (2843-2864), `set_params` (607).
- `models/_persistence.py`: `_upgrade_saved_state` (146) and
  `_upgrade_state_space` (253), which rebuild a `StateSpace` for older pickles.
- `likelihoods/__init__.py`: `LikelihoodAlgorithm.execution_settings` (194-200,
  240), `_resolve_fit_params` (383-465), `_execution_settings` (468).
- `models/_em.py`: `_em_results` (1843), which builds dense EM results through
  `convert_results_to_xarray`.
- `models/_results.py`: `convert_results_to_xarray` (186; attributes at
  298-305). `models/_checkpointed.py`: the checkpointed attributes (418-426).
- `docs/result_layout.md` section "Attributes" (line 89) and
  `tests/models/test_result_schema.py`, which checks that table against real
  results and round-trips attributes through `save_results`.
- `docs/warnings.md:12`, `docs/saving_models.md`, and the `set_params`
  docstring (`models/base.py:608-632`), which describe today's behaviour.

## Suggested approach

1. **Capture the reference first**: results of `predict` (in-memory and
   checkpointed) and `estimate_parameters`, a saved model's bytes, and a pickle,
   for one sorted and one clusterless model, at the base commit.
2. **Declare the lifetimes** as two small sets of parameter names next to
   `layout_parameters`, with a sentence each on why the parameter is read when
   it is.
3. **Record the remaining fit-time values at fit.** Extending `StateSpace` with
   one field that has a default (so older pickled `StateSpace` objects still
   work) keeps all change detection in one object; a private detector attribute
   listed in `_REBUILT` also works. Either way it must be rebuilt, not stored,
   on load from a saved file, and tolerated as absent on an older pickle.
   Record the algorithm parameters where the encoding model is fitted, after
   that fit succeeds: `fit` reaches it through `fit_encoding_model`, so one place
   covers `fit`, `fit_encoding_model` and EM's refits. `StateSpace` is built only
   by `_fit`, so the private attribute may suit these better.
4. **Add an "effective fit parameters" helper** beside `_resolve_fit_params`
   and use it for the algorithm parameters.
5. **Extend the comparison** so that the warning and the save refusal cover the
   seven and the rest in one list, in a stable order.
6. **Write the decode-time values** in both places results get their
   attributes, and add the rows to `docs/result_layout.md`.
7. **Raise `NotFittedError` from `fit_encoding_model`** when `environments_` is
   missing, with a hint to call `fit` (or `initialize_environments`) first.
   `NotFittedError` is also an `AttributeError`, so existing `hasattr` checks
   behave the same.
8. Documentation in the same pull request: `docs/warnings.md`,
   `docs/saving_models.md`, `docs/result_layout.md`, the `set_params` and
   `_DetectorBase` docstrings (which list the seven), and a `CHANGELOG.md`
   entry under `[Unreleased]`.

Use the `scientific-tdd` skill: each row of the table in "Goal" becomes a
failing test first.

## Validation

| Check | Asserts | Threshold |
| --- | --- | --- |
| Reference harness | Posteriors and state probabilities equal with `np.array_equal(a, b, equal_nan=True)`; result datasets equal once the new attributes are dropped; saved-model members byte-identical apart from the package version | hard |
| `tests/test_golden_regression.py` | Passes with the golden data untouched | hard |
| `tests/models/test_stored_artifact_compatibility.py`, `tests/model_format` (including the frozen files in `tests/compat_data/`) | Pass without regenerating anything; a detector loaded from an older pickle or file decodes without a change warning | hard |
| `tests/models/test_result_schema.py` | Passes; the documented attribute table matches, and the new attributes survive `save_results` and `load_results` with their types | hard |
| Existing change-warning tests for the seven | Pass unedited | hard |
| New tests (below) | As listed | hard |

New tests, in `tests/models/`, for one sorted and one clusterless model where
it matters:

| Test | Asserts |
| --- | --- |
| fit-time parameter changed | After `fit`, `set_params` of `infer_track_interior`, of `continuous_initial_conditions_types`, of `frozen_discrete_transition_rows`, or of a non-execution key of `*_algorithm_params` makes `predict` warn `ParameterWarning` naming it, leaves the decode identical, and makes `save_model` raise |
| same values written differently | `*_algorithm_params` changed from None to a dict of the effective values (and back) does not warn and saves |
| execution setting changed | Changing a key in the algorithm's `execution_settings` does not warn and saves |
| decode-time and EM-time parameters | Changing each named decode-time or EM-time parameter does not warn and saves |
| unclassified parameter is fit-time | A detector subclass with one extra constructor parameter, changed after `fit`, warns, without any edit to the two sets |
| results record decode-time values | `predict` results carry the attributes with the detector's current values, in-memory and checkpointed, and `estimate_parameters` results too |
| encoding refit with new algorithm parameters | After `fit`, `set_params` of a non-execution key of `*_algorithm_params`, then `fit_encoding_model`, `predict` does not warn about it and `save_model` saves; another fit-time parameter changed first is still flagged |
| encoding fit before fit | `fit_encoding_model` on a new detector raises `NotFittedError` |
| loaded model | A model saved, loaded and decoded does not warn; changing a fit-time parameter on the loaded model then warns |

The "unclassified parameter" test is what keeps the default honest; it does not
list parameters, so adding one later needs no edit.

## Fixtures

Existing simulated-data fixtures, `tmp_path` for saved models and results, and
the older-version artifacts already in `tests/compat_data/`.

## Deliberately not in this phase

- Recording in a saved file the prior that EM used (open question 5).
- Moving `no_spike_rate` to fit-time.
- Grouping the fitted attributes into objects (overview, "Deliberately not in
  this plan").
- A warning when the detector's `infer_track_interior` overrides an
  environment's own setting (overview, same section).
- Anything in phases 1 to 3.

## Review

Before opening the pull request, have an independent reviewer read the diff
and confirm:

- No stored name was added to the saved-model format, and no format version or
  migration was introduced.
- The comparison uses effective values, and the "same values written
  differently" test would fail without that.
- An encoding refit updates the record of the algorithm parameters.
- The default for an unclassified parameter is fit-time, and no test or table
  has to be edited when a parameter is added.
- Older pickles and files load and decode without the new warning, with the
  compatibility tests' output attached.
- Documentation that listed the seven parameters now describes the rule.
- No name in code, tests or docstrings refers to this plan.
