# Overview: scope, integration, risks

[← back to PLAN.md](PLAN.md)

Line numbers are for `0b88f984` and drift; find the code by symbol.

## Why

A design review at `2f297194` found the numerical core and the documented
contracts in good shape and the remaining design debt in the detector layer:

- Sorted and clusterless data are modelled by subclassing. The base class
  checks which subclass it is, passes spike arrays as an untyped tuple, and each
  family repeats six public methods with their docstrings (about 1,400 lines in
  two copies). A third kind of data would add a third copy.
- Ten of `predict`'s 18 parameters choose how inference runs, and they
  constrain each other: the checkpointed driver forbids two of them and the
  in-memory driver forbids six (`_validate_inference_arguments`,
  `models/_input_validation.py:110`). Chunks are counted two ways (`n_chunks`,
  `chunk_size`). `estimate_parameters` has 28 parameters.
- Constructor parameters are read at three different times (fit, decode, EM),
  and nothing declares which. A fit records seven of them; changing any other
  fit-time parameter afterwards changes nothing, warns about nothing, and is
  saved as if the model had been fitted with it.

The user wants more kinds of data later (calcium traces, LFP as its own
likelihood, several sources in one detector). Two public signatures cannot be
widened after a release without a deprecation cycle, so they are settled first.

## Current codebase integration points

| Area | Where | What changes | What stays |
| --- | --- | --- | --- |
| Detector base | `models/base.py:342` `_DetectorBase` | Loses its checks of which subclass it is (phase 1); gains the six public data methods (phase 2) | Mixin structure; `__init__`; `set_params` |
| Family classes | `models/base.py:2867` `ClusterlessDetector`, `:4036` `SortedSpikesDetector` | Lose their method copies (phase 2) | Class names, module path, constructors |
| Ten model classes | `models/decoder.py`, `cont_frag_model.py`, `nospike_cont_frag_model.py`, `multienvironment_model.py`, `non_local_model.py` | Docstring examples only | Constructors, names, defaults |
| Decode path | `models/base.py:1787` `_log_likelihood_rows`, `:206` `_DecodeContext` | Reads data through the family object (phase 1) | Output values, dtype, order of operations |
| Input checks | `models/_input_validation.py:350-677`; the input step `_validated_inputs` (1918) and `_ValidatedInputs` (684) | The input step holds the data object and reaches the checks through the family object (phase 1) | The checks themselves, their stages, and that they run per call |
| Execution options | `models/base.py:2091` `_predict_spikes`, `:2207` `_estimate_spike_parameters`; `models/_em.py:139-354`; `models/_input_validation.py:110` | Nine of `predict`'s ten execution flags and five of `estimate_parameters`' eight become one `inference` argument (phase 3) | The two drivers and their numerics; `cache_likelihood` |
| Change detection | `model_format/_detector.py:668` `changed_layout_parameters`; `models/base.py:1633` | Covers every fit-time parameter (phase 4) | `StateSpace` and what decoding reads from it |
| Results | `models/_results.py:298`, `models/_checkpointed.py:418` | New attributes for decode-time parameter values (phase 4) | Dimensions, coordinates, existing attributes |
| Likelihood algorithms | `likelihoods/` | Nothing | Fit and predict functions, registries, encoding-model dicts |
| HMM core | `core.py`, `checkpointed_inference.py`, `transition_operators.py` | Nothing | Everything |
| Saved-model format | `model_format/` | One rebuilt (not stored) attribute name (phase 4) | Every stored name |
| API reference | `mkdocs.yml`, `docs/gen_ref_pages.py` | Shows the inherited data methods on the two family classes (phase 2) | Everything else |

## Scope

### Goals

- A detector's public methods take one data argument, and each method is
  written once for every kind of data.
- The base class reaches data only through one family object, so a new kind of
  data needs a data class, a family object and its likelihood algorithms, and no
  edit to the detector classes.
- `predict` and `estimate_parameters` take one argument that says how inference
  runs. Apart from `cache_likelihood`, an option exists only on the strategy it
  applies to.
- Changing any fit-time parameter after a fit is detected the way the seven
  recorded ones are today.
- Outputs are unchanged throughout. Stored models, results and Spyglass rows
  keep loading.
- A Spyglass parameter row can choose either inference strategy, as it can
  today.

### Non-goals

- Calcium traces, LFP, or several sources in one detector. The
  [rules for later sources](shared-contracts.md#rules-for-later-sources) record
  what this plan leaves possible.
- Changing the likelihood algorithms, the encoding model's dict form or the
  algorithm registries.
- Changing constructor parameters, class names or stored names.

### Deliberately not in this plan

- **Removing the engines' repeated input checks.** The design discussion
  proposed validating inputs once at the public method. On reading the code,
  the repetition is deliberate and tested (see
  [Input checks stay per call](shared-contracts.md#input-checks-stay-per-call)),
  and the per-chunk cost is already avoided by the prepared time grid
  (`_DecodeTimeGrid`). Nothing is left worth a phase.
- **One transition input for every HMM driver.** The scan bodies in `core.py`
  already take a transition operator (`_filter_scan` and `_smoother_scan`,
  `core.py:958-1096`). What remains is about 80 lines of branching in the model
  layer (`models/base.py:2506-2542`, `2775-2800`;
  `models/_checkpointed.py:169-208`). Unifying the driver
  signatures would touch functions that 27 test files import, and an earlier
  change to scan inputs raised GPU peak memory by 17% and was reverted.
  Revisit if a new transition form or a second source would add a fourth
  branch.
- **Grouping the EM-updated fitted values into one object.** Spyglass reads
  them from `vars(detector)` and EM edits some in place.
- **`infer_track_interior` on both the detector and `Environment`.** The
  detector's value silently overrides the environment's (reproduced:
  `Environment(infer_track_interior=False)` with a default detector inferred 17
  of 144 bins, with no warning). A one-line warning would fix it; it is
  unrelated to these phases and can be its own small change.

## Metrics

- **Outputs (hard):** for every phase, posteriors, state probabilities,
  marginal log likelihoods, Viterbi paths and saved-model bytes are identical
  before and after on the phase's reference cases. The golden regression test
  passes untouched. Arrays are compared with
  `np.array_equal(a, b, equal_nan=True)`: bins outside the track are NaN in the
  posteriors and log likelihoods, so plain `np.array_equal` is False even for
  an array and its copy.
- **Stored artifacts (hard):** `tests/compat_data/` and the stored-artifact
  compatibility tests pass without regenerating anything.
- **Compiles (target):** `jax.log_compiles` shows no additional compilation on
  two successive `predict` calls after a phase compared with before it.
- **Size (target):** phase 2 removes roughly 700 lines from `models/base.py`
  (one of the two copies of six methods). Report the actual number.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Phase 1 changes numbers while moving code (order of operations, dtype, which arrays reach a jitted function) | Reference harness captured before the phase, compared with NaN-aware `np.array_equal`; no test file needs editing in phase 1 beyond the few that name moved private members |
| The call-form change is large (77 files here, up to 115 counting dict literals) and a missed call site lingers | A renamed keyword raises `TypeError` at the call, so missed sites fail loudly; the phase ends with a search that finds none |
| Spyglass rows that name `n_chunks` stop having an effect, or raise, after phase 3; or a translation changes which EM path a row takes | Phase 3 includes the Spyglass translation in the same update of #1618 and spells out the mapping that keeps each row's path; `cache_likelihood`, the other released option, stays a keyword |
| Spyglass can no longer choose checkpointed inference: DataJoint cannot pack `InMemory` or `Checkpointed`, and flat keys such as `inference_mode` are dropped with only a logged warning | Rows hold a plain `inference` dict that the adapter turns into the object, and the adapter refuses the flat keys (phase 3) |
| A stored model or pickle stops loading | No stored name changes; the compatibility tests are part of every phase's validation |
| Phase 4 warns about a parameter that did not really change (the same value written another way) | Compare resolved values: algorithm parameters with the registry's and the fit function's defaults filled in |
| Phase 4 flags a model after a legitimate `fit_encoding_model` refit with new algorithm parameters | The record of the algorithm parameters follows the encoding fit, not `fit` alone (phase 4) |
| Moving the methods onto the private base class removes them from the API reference, and the docs build still passes | Phase 2 shows the inherited data methods on the family classes and checks the built page |
| A user replaced `compute_log_likelihood` and relies on its argument list | Phase 1 keeps the argument list; phase 2 changes it once, in the changelog and the migration note |

## Rollout

- 0.7.0 is unreleased, and it already changes every `predict` call (`time`
  became `time_edges`). Phases 2 and 3 land before the release, with no
  deprecation period and no parallel old call form.
- Phase 1 has no public effect. Phase 4 adds warnings and result attributes.
- Spyglass PR #1618 is updated alongside phases 2 and 3. It already pins
  `>=0.7.0,<0.8`.
- Order: 1, then 2, then 3. Phase 4 can run at any point; if it runs in
  parallel with 2 or 3, expect conflicts in the decode methods of
  `models/base.py` and rebase it last.

## Open questions

Each has a current best answer, which the phase files assume.

1. **Name of the data argument.** Best answer: `neural_data`. Renaming breaks
   sorted calls that name `spike_times=` (90 calls in 43 files here, plus
   dict literals, plus three in Spyglass; see the appendix). The cheaper
   alternative keeps the name `spike_times` for the data argument, so only
   clusterless calls change; the cost is a wrong name once a non-spike kind of
   data exists, and a deprecation cycle to fix it after release.
2. **`inference` versus keeping the released options flat.** Best answer: one
   `inference` argument taking `InMemory(n_chunks)` or `Checkpointed(...)`,
   with `cache_likelihood` left as a keyword because `estimate_parameters`
   needs it settable while EM still chooses its path. That leaves one rule in
   `predict` (`cache_likelihood=True` with `Checkpointed`). The cheaper
   alternative also keeps `n_chunks` as a plain keyword and groups only the
   seven unreleased checkpoint options into one argument. That needs no
   Spyglass translation and fewer call-site edits, but `estimate_parameters`
   would still need `transition_representation` to force in-memory EM, with its
   rules. Either way the checkpoint options are an object that a Spyglass row
   cannot store, so rows use a plain dict (phase 3).
3. **Strict structured EM.** With `inference`, `estimate_parameters` loses
   `transition_representation`: `None` chooses as `"auto"` does today,
   `InMemory` runs in-memory EM, `Checkpointed` runs structured EM.
   `"structured"`, which also refuses dense fallback blocks in the internal
   fit, is no longer selectable for EM (it stays on `fit`). Eight test files
   pass `transition_representation` to `estimate_parameters`, several with all
   three values (listed in phase 3). Best answer: accept the loss and assert
   the refusal on `fit` where a test needs it.
4. **Decode-time values in results.** Best answer: write `sampling_frequency`,
   `no_spike_rate` and, where the model has them, the two non-local penalty
   parameters into the result attributes. Changing `no_spike_rate` and the
   penalty after a fit moved state probabilities by up to 0.998 on a simulated
   recording, and today nothing in the result says which values were used.
5. **EM-time parameters after EM.** The prior parameters are read only by
   `estimate_parameters`, which refits first. If they are changed after EM and
   the model is saved, the file states a prior the transitions were not
   estimated under. Best answer: leave this alone; phase 4 only keeps these
   parameters from triggering the fit-time warning.

## Estimated effort

Rough diff sizes, for expectations only:

- Phase 1: about 400 lines added (data classes, family object), 300 moved or
  removed in `models/base.py`; a handful of test edits.
- Phase 2: about 700 lines removed from `models/base.py`; mechanical edits in
  77 to 115 files here, six call sites and one test in Spyglass.
- Phase 3: about 150 lines added, 250 removed; call-site edits in about 21
  detector-level test files (nine more under `tests/core/` name an execution
  option in driver calls that do not change) and four documentation pages;
  about 30 lines in Spyglass.
- Phase 4: about 200 lines added; three documentation pages.
