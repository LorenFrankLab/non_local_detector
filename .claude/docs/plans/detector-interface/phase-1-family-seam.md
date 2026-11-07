# Phase 1: route every family-specific step through one family object

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

Line numbers are for `0b88f984` and drift; find the code by symbol.

## Goal

Make the detector base class independent of the kind of neural data. Today
`_DetectorBase` checks whether it is a `ClusterlessDetector`, its input step
works out the family from the length of an untyped spike tuple, and it relies
on methods that the two family classes override. After this phase it holds one
data object and asks one family object for everything that differs.

This phase changes no public signature and no output. It is the groundwork for
phase 2, which then only has to change how the public methods receive their
data. It also makes a later kind of data a matter of adding a data class and a
family object.

## Done when

- `_DetectorBase` and its mixins contain no `isinstance(self,
  ClusterlessDetector)` (or `SortedSpikesDetector`), no family worked out from
  the length of the spike data, no `spike_data[0]`, and no `clusterless=` flag
  passed from a detector method.
- `_ValidatedInputs` and `_DecodeContext` hold the data object in place of their
  `spike_data` tuples, and `_validated_inputs` takes it.
- The two family classes override none of the base's internals. Today they
  override `_likelihood_predict_function`, `_likelihood_predict_kwargs`,
  `_likelihood_execution_settings`, `_no_spike_mark_log_likelihood`,
  `_validate_time_contract`, `_refit_encoding_model` and `_fit_encoding_model`,
  and set `_PREDICT_EXAMPLE`. After this phase each
  holds its constructor, its algorithm-parameter helpers, the thin public
  wrappers, and the name of its family.
- The internal fit of `estimate_parameters` passes the data object down. Today
  it calls the public family `fit` with `*spike_data`
  (`models/base.py:2301-2314`).
- The two data classes and two family objects of
  [shared-contracts.md](shared-contracts.md) exist, private for now.
- A direct subclass of `_DetectorBase` that supplies its own likelihood and has
  no family still works (see the constraint below).
- Every public method of both families has the signature and docstring it has
  today, and returns exactly what it returns today.
- A user who replaced `compute_log_likelihood` (on an instance, a subclass or a
  family class) still has it called with the arguments it receives today.

## Fixed decisions and constraints

- **No output changes.** Likelihoods, posteriors, marginal log likelihoods,
  Viterbi paths and saved-model bytes are identical. Reason: this is a
  behaviour-preserving refactor; any difference is a bug to find, not to
  explain.
- **No public change.** Signatures, docstrings, warnings and error messages of
  public methods stay. Reason: phase 2 makes the public change in one reviewed
  step.
- **Input checks stay per call**, and `SpikeTimeOrder` stays scoped to one
  decoding call. See
  [Input checks stay per call](shared-contracts.md#input-checks-stay-per-call).
- **The data classes hold references only.** No copying, sorting or freezing of
  the caller's arrays. Numeric conversion of object-dtype spike times stays
  where it is today: the stages of `_validated_inputs` that keep the checked
  times get them from `_check_spike_data`
  (`models/_input_validation.py:2208`). A stage that converts holds a new data
  object with the converted times; the caller's object is not changed.
- **Likelihood algorithms are untouched.** Same functions, same registries,
  same keyword arguments, same call order per state.
- **Stored names are untouched.** See
  [Stored artifacts](shared-contracts.md#stored-artifacts-that-must-not-change).
  In particular `mark_reference_` and `encoding_model_` stay plain detector
  attributes, and EM keeps the mark reference unchanged across encoding refits.
- **The EM encoding update keeps today's weights** (the summed posterior of the
  Local states). Handing it the per-state posterior is a later need and is not
  built here.
- **The engines stay usable without a family.**
  `tests/test_discrete_transitions_estimation.py:64` defines
  `_FixedLikelihoodDetector(_DetectorBase)`, which overrides
  `compute_log_likelihood`, `fit_encoding_model`, `_fit_encoding_model` and
  `_validate_time_contract` and calls `_run_em` directly; it is the only direct
  subclass outside `models/base.py`. `_predict`, `_run_em` and `_run_viterbi`
  must not ask for a family when the likelihood is replaced, and EM's encoding
  update must keep calling the overridable `_fit_encoding_model`. That test
  file passes unedited. Reason: the transition-estimation tests rely on a
  detector whose likelihood is given, and so may users.
- **Messages stay per family.** The two encoding fits log different lines
  (`"Fitting clusterless spikes..."` at `models/base.py:3179`,
  `"Fitting place fields..."` at 4255) and show different example calls in
  validation errors (`_PREDICT_EXAMPLE`, 3360 and 4407, used at 2122). The
  family supplies both.

## Context to read

- [shared-contracts.md](shared-contracts.md): the data classes and the table of
  what the family provides, with where each piece lives today.
- `models/base.py`:
  - `_DecodeContext` (206-242), whose `spike_data` tuple becomes the data
    object, and `_decode_context` (1691-1736).
  - `_validate_decode_population` (1531-1606): the remaining subclass check.
  - The hooks (1611-1631) and their overrides (3368-3408, 4414-4420).
  - `_log_likelihood_rows` (1787-1969): the No-Spike branch (1875-1899) and the
    likelihood call (1902-1916).
  - `_compute_spike_log_likelihood` (1653), `_predict_spikes` (2091),
    `_estimate_spike_parameters` (2207; `encoding_fit_arguments` at 2342-2351),
    `_most_likely_spike_sequence` (2354).
  - `_fit_encoding_groups` (2625-2688), and both `_fit_encoding_model` bodies
    (3171-3246 clusterless, 4253-4297 sorted): they differ only in the feature
    check, the feature masking, and the mark reference. Each family's `fit`
    calls `self.fit_encoding_model` (3348, 4395), which a subclass may override.
  - `_replaced_log_likelihood` and `_decode_likelihood_callback` (1738-1785),
    and `_PACKAGE_LOG_LIKELIHOODS` (4949).
- `models/_input_validation.py`, the input step, now the main thing this phase
  changes:
  - `_InputStage` (680), `_ValidatedInputs` (684-826), whose `spike_data` tuple
    becomes the data object, and `_validated_inputs` (1918-2142).
  - `_check_spike_data` (2208-2251), which works out the family at 2245.
  - `_validate_spike_structure` (350), `_numeric_spike_times_list` (543),
    `_validate_spike_data` (563).
- `models/_em.py`: `_refit_encoding_model` (714), `_em_update_encoding_model`
  (1378-1454).
- `likelihoods/__init__.py`: the protocols (79-162), `_PREDICT_KEYWORDS` (287),
  `_resolve_fit_params` (383), `_execution_settings` (468).
- `likelihoods/no_spike.py` `predict_no_spike_log_likelihood`;
  `likelihoods/mark_reference.py` `fit_mark_references`,
  `mark_reference_log_likelihood`.
- Tests that name private members this phase may move:
  `_refit_encoding_model` (5), `_resolve_clusterless_algorithm_params` (3),
  `_resolve_sorted_spikes_algorithm_params` (2), `_validate_spike_data` (1).
  Find them with `grep -rlE "\b<name>\b" tests`.

## Suggested approach

1. **Capture the reference first.** Before editing, run a harness at the base
   commit and save its outputs outside the repository (the scratchpad). Cover
   both families with their default algorithm, and at least one other
   algorithm each; `predict` dense and checkpointed (compact and spatial);
   `estimate_parameters` with and without `estimate_encoding_model`;
   `most_likely_sequence`; `compute_log_likelihood` whole and with a
   `row_slice`; a No-Spike model, a `local_position_std` model, a
   multi-environment model and a 2-D environment with bins outside the track;
   `save_model` bytes of every member. The
   `fitted-state-space` plan's harness (same directory tree) covered a similar
   list and is a reasonable starting point if its script is still in a
   scratchpad; otherwise write it fresh.
2. **Add the data classes and the family objects** in a private module under
   `models/`. Start each family method as a move of the existing code, keeping
   the statements and their order.
3. **Switch the base class over** one responsibility at a time (the input step
   and its checks, the decode context, the likelihood call, No-Spike, the encoding fit, the EM
   update), comparing against the reference after each. The public wrappers in
   the two family classes build the data object from their array arguments and
   pass it down.
4. **Merge the two `_fit_encoding_model` bodies** into one on the base, with the
   family supplying the group selection and the extra fitted pieces.
5. **Keep `compute_log_likelihood` replacements working**: when a replacement
   is found, core is still handed `(position_time, position, *arrays)` as
   today, built from the data object.
6. Either keep the private names the tests use or update those tests; say which
   in the pull request.

Use the `safe-refactoring` skill for the moves and the `jax` skill if any array
reaching a jitted function changes type or identity.

## Validation

| Check | Asserts | Threshold |
| --- | --- | --- |
| Reference harness, before and after | Every saved array equal with `np.array_equal(a, b, equal_nan=True)`; result datasets `identical`; saved-model members byte-identical apart from the package version | hard |
| `tests/test_golden_regression.py` | Passes with the golden data untouched | hard |
| `tests/models`, `tests/likelihoods/test_algorithm_contract.py`, `tests/model_format`, `tests/models/test_stored_artifact_compatibility.py` | Pass; the only test edits are for moved private names | hard |
| `tests/models/test_uniform_grid_contract.py`, `tests/models/test_chunk_likelihood_preparation.py` | Pass unedited: the engines still validate direct calls and prepare each chunk as before | hard |
| `tests/test_discrete_transitions_estimation.py` | Passes unedited: a `_DetectorBase` subclass with its own likelihood and no family still fits and runs EM | hard |
| Replaced likelihood (existing tests that set or override `compute_log_likelihood`; 29 test files mention it) | The replacement receives the same arguments as before | hard |
| Search of `models/base.py` and `models/_input_validation.py` | None of the patterns in "Done when" remain in `_DetectorBase` or its mixins, and the family classes define none of the overrides listed there | hard |
| `jax.log_compiles` over two successive `predict` calls, both families | No compilation that the base commit does not also show | target |
| `uv run --no-sync ruff check src/`, `ruff format --check src/`, mypy as CI runs it | Clean | hard |

## Fixtures

No new fixtures. The harness uses
`non_local_detector.simulate.sorted_spikes_simulation.make_simulated_data` and
the clusterless simulation helpers the tests already use
(`tests/models/_helpers.py`, `tests/conftest.py`).

## Deliberately not in this phase

- Exporting the data classes or accepting them in public methods: phase 2.
- Deleting the per-family public methods: phase 2. Here they become thin
  wrappers.
- Splitting the spike-only keys out of the shared encoding-model type, the
  per-state posterior for EM, several sources: later work, described in
  [Rules for later sources](shared-contracts.md#rules-for-later-sources).
- Family-specific code outside the detector (`model_checking/`,
  `visualization/static.py:124`, the saved-model format's `mark_reference`
  entry). They are consumers of one family's fitted model and are correct as
  they are.

## Review

Before opening the pull request, have an independent reviewer (the
`code-reviewer` agent or a person) read the diff and confirm:

- The harness comparison was run and is exact, with its output attached.
- Nothing listed under "Deliberately not in this phase" was done.
- No public signature, docstring, warning or error message changed.
- The family objects hold no state and the data classes copy nothing.
- No name in code, tests or docstrings refers to this plan.
