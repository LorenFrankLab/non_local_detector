# Phase 2: one data argument, data methods written once

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md) · [appendix](appendix.md)

Line numbers are for `0b88f984` and drift; phase 1 will also have moved code.
Find the code by symbol.

## Goal

Give the detectors' six data methods a single data argument and a single
implementation. Today `SortedSpikesDetector` and `ClusterlessDetector` each
define `fit`, `fit_encoding_model`, `compute_log_likelihood`, `predict`,
`estimate_parameters` and `most_likely_sequence`, with the same body and
almost the same docstring; the clusterless ones take one more positional array.

This is the one change that cannot be added after a release: two positional
spike arrays cannot grow to cover another kind of data, and one data argument
can later accept several sources. It is why this phase is scheduled before
0.7.0.

Users call these methods in every script, so the change needs a migration note,
and Spyglass PR #1618 needs its six call sites updated in step.

## Done when

- Each of the six methods is defined once, on `_DetectorBase`, with one
  docstring that describes the data argument for both kinds of data.
- `SortedSpikes` and `ClusterlessSpikes` are exported from the package root and
  documented.
- The API reference still shows the six methods on `SortedSpikesDetector` and
  `ClusterlessDetector`.
- The calls below work, and the old clusterless form (two spike arrays) and the
  old keyword `spike_times=` raise a `TypeError` from Python, not a silent
  misreading.
- Every call site in this repository is updated: tests, tutorials, getting
  started, README, notebooks, benchmarks, scripts.
- Spyglass #1618 has a commit that updates its six detector calls and passes
  its decoding tests against this branch.
- Outputs are unchanged.

*Interface sketch* (execution options are untouched in this phase and elided):

```python
from non_local_detector import ClusterlessSpikes, SortedSpikes

# sorted spikes: a plain sequence of spike-time arrays is accepted
detector.fit(position_time, position, spike_times, is_training=is_training)
results = detector.predict(spike_times, time_edges=edges, position=position,
                           position_time=position_time)

# clusterless
spikes = ClusterlessSpikes(spike_times, spike_waveform_features)
detector.fit(position_time, position, spikes)
results = detector.predict(spikes, time_edges=edges, position=position,
                           position_time=position_time)

# signatures
fit(position_time, position, neural_data, *, is_training=None, ...)
fit_encoding_model(position_time, position, neural_data, *, is_training=None, ...)
compute_log_likelihood(position_time, position, neural_data, *,
                       time_edges, is_missing=None, row_slice=None,
                       valid_position_intervals=None)
predict(neural_data, *, time_edges, position=None, position_time=None, ...)
estimate_parameters(position_time, position, neural_data, *, time_edges, ...)
most_likely_sequence(neural_data, *, time_edges, position=None, ...)
```

## Fixed decisions and constraints

- **The data argument is `neural_data`** (planner's call; overview open
  question 1). If the user chooses to keep the name `spike_times`, only the
  name in this file changes: the sorted keyword call sites then stay as they
  are and the clusterless ones still change.
- **A sorted-spikes detector accepts a plain sequence of spike-time arrays** as
  well as a `SortedSpikes` (planner's call). Reason: it is the most common call
  and the meaning is unambiguous. A clusterless detector requires a
  `ClusterlessSpikes`; given anything else it raises `ValidationError` with an
  example of the right call. A detector given the other family's data class
  raises `ValidationError` naming both.
- **One call form.** The old forms are removed in this phase, with no alias and
  no deprecation period. Reason: 0.7.0 is unreleased and already changes every
  `predict` call; the user's rule is not to keep parallel old and new paths.
- **The family classes stay.** `SortedSpikesDetector` and `ClusterlessDetector`
  keep their names, module, constructors and constructor parameters, and name
  their family. Reason: stored pickles, saved models and Spyglass rows name
  them; see
  [Stored artifacts](shared-contracts.md#stored-artifacts-that-must-not-change).
- **`is_missing` and `row_slice` become keyword-only in
  `compute_log_likelihood`** (planner's call). Reason: they are positional
  today, so after the merge an old clusterless call
  `compute_log_likelihood(position_time, position, spike_times, features,
  time_edges=...)` would bind the features to `is_missing` instead of failing.
  The HMM drivers already pass both by keyword
  (`_call_log_likelihood_chunk`, `core.py:2371-2396`). The other five methods
  already take everything after the data argument by keyword.
- **A replaced `compute_log_likelihood` now receives
  `(position_time, position, neural_data, ...)`**, the public signature, with
  `neural_data` always the family's data object, whichever form the caller
  passed (planner's call). Reason: a replacement then sees one type, and a
  sorted caller's choice of the plain list cannot change what it receives. This
  is a visible change for anyone who replaced it; it goes in the changelog and
  the migration note.
- **A `_DetectorBase` subclass without a family keeps working.** The abstract
  placeholders go, so the base's concrete methods are what such a subclass
  overrides. `tests/test_discrete_transitions_estimation.py:64` is the example
  (see phase 1). The base's own methods may raise a clear error when called
  without a family; a subclass that overrides `compute_log_likelihood`,
  `fit_encoding_model`, `_fit_encoding_model` and `_validate_time_contract`
  must not reach that error. Edit that test file only where its overrides have
  to follow the new signatures.
- **No output changes.**
- **Execution options are not touched** (that is phase 3). Leave `n_chunks`,
  `inference_mode` and the rest exactly as they are, so the two public changes
  can be reviewed separately.

## Context to read

- `models/base.py`, the per-family public methods to merge:
  - clusterless: `fit_encoding_model` 3081, `fit` 3248,
    `compute_log_likelihood` 3412, `predict` 3506, `estimate_parameters` 3732,
    `most_likely_sequence` 3962.
  - sorted: `fit_encoding_model` 4176, `fit` 4299, `compute_log_likelihood`
    4424, `predict` 4514, `estimate_parameters` 4649, `most_likely_sequence`
    4877.
  - The abstract placeholders on the base (1526-1529, 2690-2717) and
    `_PACKAGE_LOG_LIKELIHOODS` (4949), which lists the two family functions.
  - The decorators `requires_time_edges` (`time_edges.py:50`) and
    `row_slice_aware` (`core.py:1448`), which the merged methods keep.
- `src/non_local_detector/__init__.py`: exports and `__all__`.
  `tests/test_public_api.py` checks them (`test_spyglass_imports_are_listed`
  at 121, `test_detector_constructor_types_are_top_level` at 127).
- `docs/internal_modules.txt`: a new public module must define `__all__`.
- `mkdocs.yml` (`inherited_members: false`, filter `!^_`) and
  `docs/gen_ref_pages.py`. `models.base.__all__` names only the two family
  classes, so with these settings a method moved to `_DetectorBase` leaves the
  API reference, and the docs build still passes.
- The ten model classes' docstring examples (`models/decoder.py`,
  `cont_frag_model.py`, `nospike_cont_frag_model.py`,
  `multienvironment_model.py`, `non_local_model.py`).
- `tests/models/_helpers.py` `_decoder` (62) and the fixtures in
  `tests/conftest.py`: many tests fit through these.
- `docs/time_grid_migration.md`: the existing 0.7.0 migration note, as a model
  for the new section. `CHANGELOG.md` `[Unreleased]`.
- `tests/test_executable_docs.py`: runs the fenced code in the docs.
- [appendix.md](appendix.md): call-site counts here, and the Spyglass call
  sites and worktree.

## Suggested approach

1. Move the six methods to `_DetectorBase`, each taking `neural_data` and
   converting it with the family object (which accepts the sorted shorthand).
   Write one numpydoc docstring per method. Delete the per-family copies, the
   abstract placeholders and anything they alone used.
2. Export the data classes from a public module with `__all__` and from the
   package root.
3. Update call sites with a script rather than by hand: parse with `ast`,
   rewrite clusterless calls to wrap the two arrays in `ClusterlessSpikes`, and
   rename the `spike_times=` keyword in detector calls. Leave calls to the
   likelihood functions alone: their `spike_times` and
   `spike_waveform_features` arguments do not change. Dict literals that are
   later passed as `**kwargs` to a detector need the same edit; the ones that
   feed likelihood functions do not. Review the script's diff by directory.
4. Notebooks under `notebooks/` are edited through their paired scripts (see
   the `jupyter-notebook-editor` skill). Three notebooks under
   `notebooks/02_likelihood_models/` were already broken by an earlier import
   move; do not fix that here, and say so in the pull request.
5. Documentation in the same pull request. For the API reference, show the six
   inherited methods on the `models.base` page, for example by naming them in
   `inherited_members` for that page in `docs/gen_ref_pages.py`. Also README,
   `docs/getting-started/`, `docs/tutorials/*.py`, `docs/data_contract.md`,
   `docs/likelihood_mark_calibration.md`, a "Passing neural data" section in
   `docs/time_grid_migration.md` (or a sibling page linked from it) showing the
   before and after for both kinds of data and for a replaced
   `compute_log_likelihood`, and a `CHANGELOG.md` entry under `[Unreleased]`
   marked as a breaking change.
6. Spyglass: in the worktree named in the appendix, update the six calls. The
   clusterless `predict` call wraps the two lists that `spikes_for_sequence`
   returns. Run Spyglass's `tests/decoding` against this branch. Committing
   there is a local commit on that pull request's branch; pushing needs the
   user's approval.

## Validation

| Check | Asserts | Threshold |
| --- | --- | --- |
| Phase 1's reference harness, with its calls updated to the new form | Every array equal with `np.array_equal(a, b, equal_nan=True)`; saved-model members byte-identical apart from the package version | hard |
| `tests/test_golden_regression.py` | Passes with the golden data untouched | hard |
| New tests of the data argument (see below) | The accepted and rejected forms behave as "Fixed decisions" says, for one model of each family | hard |
| `tests/models`, `tests/model_format`, `tests/test_public_api.py`, `tests/test_executable_docs.py`, `tests/models/test_stored_artifact_compatibility.py` | Pass | hard |
| `tests/test_discrete_transitions_estimation.py` | Passes: a `_DetectorBase` subclass with its own likelihood and no family still fits and runs EM | hard |
| `sklearn.base.clone(detector)` and `type(d)(**d.get_params(deep=False))` for the ten classes (existing tests cover this) | Unchanged | hard |
| Search for leftover call forms: detector calls with `spike_waveform_features=` or `spike_times=`, or with two positional spike arrays | None in `src`, `tests`, `docs`, `README.md`, `benchmarks`, `scripts`, `notebooks`. Historical records are out of scope: `docs/spyglass_migration/`, `docs/plans/`, `docs/tasks/`, `docs/performance_artifacts/`, `docs/time_grid_snapshot_reference.patch`, `CHANGELOG.md` and the "before" examples of migration notes | hard |
| Docs build (`mkdocs build`, which runs the tutorials) | Succeeds | hard |
| Built API page of `models.base` | The six methods appear under both family classes | hard |
| Spyglass `tests/decoding` in the #1618 worktree against this branch | Pass | hard |
| Lines removed from `models/base.py` | About 700; report the number | target |

New tests, in `tests/models/`, parametrized over one sorted and one clusterless
model:

| Test | Asserts |
| --- | --- |
| data object accepted by every data method | `fit`, `fit_encoding_model`, `compute_log_likelihood`, `predict`, `estimate_parameters` and `most_likely_sequence` run with the family's data class and return what the pre-change form returned on the same data |
| sorted shorthand | A plain list of spike-time arrays and `SortedSpikes(list)` give equal results |
| wrong data for the family | A clusterless detector given a plain list, and each detector given the other family's class, raise `ValidationError` whose message names the expected class |
| replaced likelihood | A replaced `compute_log_likelihood` receives a `SortedSpikes` when the caller passed a plain list |
| old call forms | `predict(spike_times=...)`, `fit(position_time, position, spike_times, features)` and `compute_log_likelihood(position_time, position, spike_times, features, time_edges=...)` raise `TypeError` |
| data object is not modified | The arrays passed in are the same objects, with the same contents and flags, after `fit` and `predict` |

Keep these few. They should not enumerate algorithms or parameters, so that
adding one later needs no edit here.

## Fixtures

Existing simulated-data fixtures. The new tests reuse
`tests/models/_helpers.py`.

## Deliberately not in this phase

- The execution options (`inference`): phase 3.
- Family-neutral preset classes that take the likelihood as a parameter, and
  any new kind of data: later work
  ([Rules for later sources](shared-contracts.md#rules-for-later-sources)).
- Renaming `spike_times` or `spike_waveform_features` inside the likelihood
  functions or the encoding models.
- The three already-broken notebooks under `notebooks/02_likelihood_models/`.

## Review

Before opening the pull request, have an independent reviewer read the diff
and confirm:

- The six methods exist once; no per-family copy or alias remains.
- The harness comparison is exact, with its output attached.
- The migration note and changelog entry describe the change for both kinds of
  data and for a replaced `compute_log_likelihood`.
- The call-site script changed detector calls only, not likelihood-function
  calls.
- Constructors, class names and stored names are untouched.
- The new tests exercise behaviour, and none of them has to be edited when an
  algorithm or parameter is added.
- No name in code, tests or docstrings refers to this plan.
