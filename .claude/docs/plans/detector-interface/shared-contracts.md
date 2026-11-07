# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Data classes](#data-classes)
- [Family contract](#family-contract)
- [Input checks stay per call](#input-checks-stay-per-call)
- [Rules for later sources](#rules-for-later-sources)
- [Stored artifacts that must not change](#stored-artifacts-that-must-not-change)

Line numbers are for `0b88f984` and drift; find the code by symbol.

## Data classes

A data class holds one kind of neural data as the caller gave it. Phase 1
introduces the two below for internal use; phase 2 exports them and makes them
the argument of the public methods.

*Interface sketch:*

```python
@dataclass(frozen=True, eq=False, repr=False)
class SortedSpikes:
    spike_times: Sequence[ArrayLike]              # one 1-D array of times (s) per neuron

@dataclass(frozen=True, eq=False, repr=False)
class ClusterlessSpikes:
    spike_times: Sequence[ArrayLike]              # one 1-D array of times (s) per electrode
    spike_waveform_features: Sequence[ArrayLike]  # (n_spikes, n_features) per electrode
```

Fixed:

- They hold references to the caller's arrays. They do not copy, sort, convert
  or freeze them. Waveform features can be gigabytes, and setting an array
  read-only would change the caller's own object.
- Field names follow the package's existing argument names
  (`spike_times`, `spike_waveform_features`), which the likelihood functions and
  Spyglass already use.
- They set `eq=False` and `repr=False`. A generated `==` over arrays raises, and
  a generated `repr` would print the spikes into error messages.
- They carry no fitted state and no cache (see
  [Input checks stay per call](#input-checks-stay-per-call)).
- A later kind of data (calcium traces, LFP) is another class of this sort with
  its own fields, for example sample times and a `(n_time, n_channels)` array.
  Nothing in the detector base class may assume a data class has `spike_times`.

## Family contract

A *family* is one kind of neural data together with everything the detector
does differently for it. Today the detector base class asks which subclass it
is; after phase 1 it asks its family object instead. There is one stateless
family object per data class.

Everything in this table is family-specific today and moves behind the family
object. The right-hand column is where the behaviour lives now.

| The family provides | Today (`src/non_local_detector/models/base.py` unless noted) |
| --- | --- |
| Its data class, unit name ("neuron", "electrode") and algorithm registry | `isinstance(self, ClusterlessDetector)` at 1537; `clusterless=len(spike_data) > 1` in `_check_spike_data` (`models/_input_validation.py:2245`); `_SORTED_SPIKES_ALGORITHMS`, `_CLUSTERLESS_ALGORITHMS` (`likelihoods/__init__.py:301`, `:326`) |
| The name of the constructor parameter that selects the algorithm | `"clusterless_algorithm"` / `"sorted_spikes_algorithm"` at 1546-1548 |
| Checks of the data on entry: structure, numeric and finite times, range warnings | `_check_spike_data` (`models/_input_validation.py:2208`), called at each stage of `_validated_inputs` (1918), which calls `_validate_spike_data` (563) with `clusterless=` |
| The positional data arguments of the likelihood functions | `*spike_data` at 1905; the `spike_data` tuples of `_DecodeContext` (239) and `_ValidatedInputs` (`models/_input_validation.py:714`) |
| The data of one encoding group (events inside the group's support) | `fit_group` in each `_fit_encoding_model`: 3192-3217 (clusterless), 4268-4284 (sorted), using `_group_spike_mask` 245 |
| Fitted pieces beyond the encoding model | Clusterless only: `_fit_mark_reference` 3039-3079 and `mark_reference_`, kept unchanged by EM refits (`_refit_encoding_model` 2985-3003; `refit_mark_reference` in `_fit_encoding_model` 3171-3246) |
| Extra keywords for every likelihood evaluation | `_likelihood_predict_kwargs`: `{}` (1615) or `{"mark_reference": ...}` (3376) |
| The No-Spike log likelihood | Poisson term from `predict_no_spike_log_likelihood` called with `spike_data[0]` at 1875-1899, plus the clusterless mark term `_no_spike_mark_log_likelihood` 3381-3395 |
| The check that decode data match the fit (units, feature dimensions) | `_validate_decode_population` 1531-1606 (`place_fields` vs `mean_rates`, `mark_dimensions`) |
| What a fitted detector must hold before decoding | Clusterless: `mark_reference_` present, `_validate_time_contract` 3397 |
| The EM encoding update | `_refit_encoding_model` (`models/_em.py:714`, overridden at `base.py:2985`) |
| Its log line for an encoding fit and its example call for validation errors | `"Fitting clusterless spikes..."` 3179, `"Fitting place fields..."` 4255; `_PREDICT_EXAMPLE` 3360, 4407 |

Fixed:

- A detector may have no family. A direct subclass of `_DetectorBase` can
  supply its own likelihood and encoding fit
  (`tests/test_discrete_transitions_estimation.py:64` does), and the HMM
  engines (`_predict`, `_run_em`, `_run_viterbi`) work for it without one.

- The family object is stateless and private (the names are the executor's
  choice). The detector passes it what it needs; it does not hold a detector.
- The likelihood algorithms keep their fit and predict functions, their
  registries and their encoding-model dicts exactly as they are. The family
  object calls them; it does not replace them.
- `SortedSpikesDetector` and `ClusterlessDetector` stay as classes in
  `models/base.py` with their current constructors. They name their family.
  See [Stored artifacts](#stored-artifacts-that-must-not-change).

## Input checks stay per call

The checks of spike data run on every public call today, from the stages of
`_validated_inputs` (`models/_input_validation.py:1918`), and keep doing so. Do
not move them to the data class's constructor and do not cache their results on
the data object.

Why: the code relies on re-checking. `SpikeTimeOrder`
(`likelihoods/common/spike_rows.py:95`) is created per decoding call, keyed by
the identity of the input arrays, and its docstring says "a fresh instance on
the next call rechecks even arrays modified in place. Never store this object
on a detector". A data object that validated once and cached the result would
miss an array changed in place between calls.

The engines' repeated grid checks also stay. `_predict` and `_run_viterbi`
re-validate the time grid "as the engine's guard for direct calls"
(`base.py:2463-2470`, `2756-2762`), and two tests call them directly to assert
it (`tests/models/test_uniform_grid_contract.py:43`, `:114`).

## Rules for later sources

Nothing here is built by this plan. These rules are what phases 1 and 2 must
leave possible, and what a later plan for calcium traces, LFP or several
sources should start from.

1. **One source now, several later.** A *source* is one data object with the
   likelihood fitted to it. A detector has one source. Sources are independent
   given the state and position, so their log likelihoods add; a detector with
   several would sum them. Phases 1 and 2 keep this additive by making the base
   class reach data only through the family object and by giving the public
   methods a single data argument, which can later accept a mapping of named
   data objects in the same slot.
2. **Every source gives every state a normalized density over the same
   observation.** For continuous-valued data, zero is not a neutral log
   likelihood, so a source cannot contribute nothing for a state it does not
   model. The mark reference is the existing example: No-Spike scores marks
   with a reference density so that mark densities cancel between states
   (`base.py:3376-3379`, `fit_encoding_model` docstring 3146-3155).
3. **A bin where a source has no sample contributes zero to every state.**
4. **A likelihood may depend on the state and not on position.** A state with
   no place bins takes one column. A state with place bins must return one
   column per interior bin: the detector refuses any other shape rather than
   broadcast it (`_wrong_likelihood_shape`, `base.py:1934-1938`). A state-only
   likelihood repeats its column, or a later plan declares it to that check.
5. **EM will need to hand a source the per-state posterior.** Today the
   encoding update sums the Local states' posterior into one weight per position
   sample (`models/_em.py:1389-1403`). A likelihood with parameters per state
   and no position labels can only be fitted by EM from the full posterior.
6. **Some shared encoding-model keys are spike-only.** `rate_units` and
   `encoding_exposure_seconds` are required of every fit
   (`likelihoods/encoding_model.py:25`), and `_TIME_CONTRACT` records
   `rate_units` (`models/_input_validation.py:74`). A non-spike family needs
   them moved to a spike-level type first. The stored contents of existing
   models would not change.
7. **A third family should not add five more detector classes.** The ten
   existing classes are five presets times two families. Presets that take the
   likelihood as a parameter would serve any family.

## Stored artifacts that must not change

Spyglass stores fitted models, results and parameter rows and reloads them for
months. These names are read from stored artifacts or by Spyglass code, so the
phases leave them as they are:

- The ten model class names, their module paths, and every constructor
  parameter name, including `sorted_spikes_algorithm`,
  `sorted_spikes_algorithm_params`, `clusterless_algorithm` and
  `clusterless_algorithm_params`. Rows store `class_name` with
  `get_params(deep=False)`. scikit-learn reads parameters from the `__init__`
  signature, so each class keeps an explicit signature (a `**kwargs`
  constructor makes `get_params()` return `{}`; checked on scikit-learn 1.8).
- `models.base.ClusterlessDetector`, `models.base.SortedSpikesDetector` and
  `models._state_space.StateSpace` as import paths: pickles name them.
- Fitted attributes as plain instance attributes, with their names:
  `encoding_model_` (keyed by `(environment_name, encoding_group)`),
  `environments_`, `mark_reference_`, `initial_conditions_`,
  `discrete_state_transitions_`, `discrete_transition_coefficients_` and the
  other names in `_STORED` (`model_format/_detector.py:119`). Spyglass reads
  some through `vars(detector)`, where a property would not appear. `n_iter_`,
  `converged_`, `em_monotonicity_violations_` and `degenerate_timesteps_` are
  already read-only properties over `_em_diagnostics_`; their stored names stay.
- The saved-model file's member and field names, and the pickle path
  (`models/_persistence.py:146` `_upgrade_saved_state`).
- The result dataset's dimensions, coordinates and existing attribute names
  (`docs/result_layout.md`). New attributes may be added.
- `tests/compat_data/` and `tests/models/test_stored_artifact_compatibility.py`
  hold artifacts written by earlier versions. They must keep passing without
  being regenerated.
