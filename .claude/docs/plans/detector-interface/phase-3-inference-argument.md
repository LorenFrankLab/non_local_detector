# Phase 3: one `inference` argument

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md)

Line numbers are for `0b88f984` and drift; phases 1 and 2 will also have moved
code. Find the code by symbol.

## Goal

Replace the execution flags of `predict` and `estimate_parameters` with one
argument that names how inference runs and carries only the options that apply
to it.

Today `predict` has ten such flags: `cache_likelihood`, `n_chunks`,
`inference_mode`, `output_mode`, `chunk_size`, `result_path`,
`checkpoint_dir`, `selected_intervals`, `max_read_bytes`,
`replay_cache_bytes`. Which are legal depends on the others:
`_validate_inference_arguments` (`models/_input_validation.py:110-162`) rejects
two with the checkpointed driver and six with the in-memory one.
`estimate_parameters` has eight (`cache_likelihood`, `store_log_likelihood`,
`n_chunks`, `transition_representation`, `max_dense_transition_bytes`,
`chunk_size`, `result_path`, `replay_cache_bytes`) with their own rules
(`models/_em.py:156-354`). A user has to learn the rules from error messages.

Only `n_chunks`, `cache_likelihood` and `store_log_likelihood` existed in a
released version (appendix), so the rest can be reshaped freely before 0.7.0.

## Done when

- `predict` takes `inference` in place of nine flags (all but
  `cache_likelihood`), and `estimate_parameters` takes it in place of five
  (`n_chunks`, `transition_representation`, `chunk_size`, `result_path`,
  `replay_cache_bytes`).
- Every option that applies to one strategy is a field of that strategy. In
  `predict` three rules remain: `cache_likelihood=True` with `Checkpointed`, a
  spatial `output_mode` without a `result_path`, and a compact `output_mode`
  with `return_outputs` asking for spatial outputs
  (`models/_checkpointed.py:314-330`).
- `InMemory` and `Checkpointed` are exported from the package root and
  documented.
- Results, including the result attributes that record how inference ran, are
  unchanged for equivalent calls.
- Spyglass #1618 translates a stored `n_chunks` into `inference` on both of its
  paths, and a row can choose either strategy with a plain dict.

*Interface sketch:*

```python
@dataclass(frozen=True)
class InMemory:
    """Filter and smooth with the whole recording's arrays on the device."""
    n_chunks: int = 1

@dataclass(frozen=True, eq=False)  # selected_intervals may be an array
class Checkpointed:
    """Bounded device memory: checkpoint the forward pass and replay chunks."""
    chunk_size: int | None = None
    output_mode: Literal["compact", "spatial"] | None = None
    result_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    selected_intervals: ArrayLike | None = None
    replay_cache_bytes: int | None = None
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES

predict(neural_data, *, time_edges, position=None, position_time=None,
        is_missing=None, valid_position_intervals=None,
        discrete_transition_covariate_data=None, cache_likelihood=False,
        return_outputs=None, inference=None)

estimate_parameters(position_time, position, neural_data, *, time_edges, ...,
                    cache_likelihood=True, store_log_likelihood=False,
                    return_outputs=None, ...,
                    max_dense_transition_bytes=DEFAULT_MAX_DENSE_BYTES,
                    inference=None)
```

## Fixed decisions and constraints

All of these are the planner's call unless a reason cites a user decision; see
overview open questions 2 and 3.

- **`predict`: `inference=None` means `InMemory()`.** Reason: it is today's
  default (`inference_mode="dense"`), and a result that silently became lazy or
  compact would surprise callers.
- **`estimate_parameters`: `inference=None` chooses**, exactly as
  `transition_representation="auto"` with no other option does today
  (`_fit_em_model` and `_choose_em_representation`, `models/_em.py:733-830`).
  `InMemory(n_chunks=k)` runs in-memory EM for any `k`: today that is
  `"dense"`, or `"auto"` with `n_chunks != 1` (`_resolve_em_options`,
  `models/_em.py:331-345`). `Checkpointed` requires the structured path
  whether or not any of its fields is set. Today that requires naming a
  checkpoint option (`structured_required`, `models/base.py:2316-2318`), so
  decide it from the type of `inference`, not from which fields are non-None.
  `transition_representation` leaves `estimate_parameters`; it stays on `fit`.
- **`cache_likelihood` stays a keyword of both methods**, with today's
  defaults (False for `predict`, True for `estimate_parameters`). Reason: in
  `estimate_parameters` it has to be settable while EM still chooses its path,
  which today's rows can rely on: only `n_chunks != 1` forces in-memory EM,
  `cache_likelihood` never does. As today, it applies to in-memory inference:
  `predict` rejects `cache_likelihood=True` with `Checkpointed`
  (`models/_input_validation.py:139-146`), and structured EM ignores it (the
  `chunk_size` entry of the `estimate_parameters` docstring says so).
- **`Checkpointed.output_mode=None`** means state probabilities only
  (`"compact"`) when no `result_path` is given, and the spatial posteriors
  written lazily (`"spatial"`) when one is. `"spatial"` without a `result_path`
  still raises, as today (`models/_checkpointed.py:314-319`). Reason: today's
  default (`"spatial"`) makes a bare checkpointed call an error. `return_outputs`
  does not change this choice: `Checkpointed()` with spatial `return_outputs`
  raises, and its hint names `result_path`, since `output_mode="spatial"` alone
  would then fail for want of a path.
- **`estimate_parameters` with `Checkpointed`** uses `chunk_size`,
  `result_path` and `replay_cache_bytes`. Its final E-step always returns the
  same outputs as in-memory EM (`models/_em.py:1729-1760`), so it rejects
  `selected_intervals` and an explicit `output_mode="compact"` with a
  `ValidationError`. Pass `checkpoint_dir` and `max_read_bytes` through to the
  driver rather than rejecting them, unless that turns out to be more than a
  keyword each; say which was done.
- **`store_log_likelihood` stays a keyword of `estimate_parameters`**, and its
  one rule stays: it cannot be combined with a `result_path`
  (`models/_em.py:236-246`).
- **`most_likely_sequence` keeps `n_chunks`.** Reason: it has one execution
  option and no rule to remove; the checkpointed driver does not do Viterbi.
- **`max_dense_transition_bytes` stays** on `fit` and `estimate_parameters`.
- **The result attributes keep their names and values**: `inference_mode`,
  `output_mode`, `em_transition_representation`, `em_n_chunks`,
  `em_chunk_size`, `em_replay_cache_bytes` (`docs/result_layout.md`, section
  "Attributes", line 89). Reason: stored results carry them and a test checks
  the documented table against real results
  (`tests/models/test_result_schema.py`).
- **No output changes** for equivalent calls. The drivers and their numerics
  are not touched.
- **One form.** The removed flags have no alias. Reason: as in phase 2.
- **Stored Spyglass rows keep working.** A row's `decoding_kwargs` can name
  `n_chunks`. On Spyglass's fit and predict path an unknown key is dropped with
  a logged warning; on its `estimate_parameters` path it raises `TypeError`
  (appendix). The translation, in the same update of #1618 as this phase,
  keeps each row's behaviour:
  - predict path: `n_chunks=n` becomes `inference=InMemory(n_chunks=n)`;
  - `estimate_parameters` path: `n_chunks=n` with `n != 1` becomes
    `inference=InMemory(n_chunks=n)`; `n_chunks=1` is dropped, so EM still
    chooses its path as it does for that row today. Forcing in-memory EM there
    could run out of memory on a large 2-D environment where the automatic
    choice is structured.
  - `cache_likelihood` and `store_log_likelihood` pass through unchanged.
- **Spyglass rows choose a strategy with a plain dict.** `decoding_kwargs` is a
  LONGBLOB, and DataJoint refuses to pack a dataclass (checked on 0.14.9), so a
  row cannot hold `InMemory` or `Checkpointed`. A row holds
  `inference={"strategy": "in_memory", "n_chunks": 4}` or
  `inference={"strategy": "checkpointed", "chunk_size": 20_000}`, and the adapter
  builds the object before either path runs. The keys other than `strategy` are
  the class's fields, so a key that does not belong raises `TypeError`.
- **Spyglass refuses the flat keys it would otherwise drop.** A row can choose
  checkpointed `predict` today with `inference_mode`, `output_mode` and the
  rest, which `get_valid_kwargs` passes through. After this phase it would drop
  them with only a logged warning and run in memory, which can run out of memory
  or return different outputs. So the adapter raises `ValueError` naming
  `inference` when a row holds `inference_mode`, `output_mode`, `chunk_size`,
  `result_path`, `checkpoint_dir`, `selected_intervals`, `max_read_bytes` or
  `replay_cache_bytes`, or holds both `inference` and `n_chunks`. None of these
  was released (appendix), so no row written for 0.6.9 is affected.

## Context to read

- `models/base.py`: `_predict_spikes` (2091-2205), where the flags are
  validated and the driver is chosen; `_estimate_spike_parameters` (2207-2352);
  the public `predict` and `estimate_parameters` (after phase 2, one copy each
  on `_DetectorBase`) and their docstrings, which document each flag.
- `models/_input_validation.py`: `_validate_inference_arguments` (110-162),
  which this phase reduces to the `cache_likelihood` rule;
  `_validate_transition_arguments` (90), which stays.
- `models/_em.py`: `_structured_em_options` (139), `_validate_em_options`
  (156), `_resolve_em_options` (309), `_fit_em_model` (733),
  `_choose_em_representation` (777), `_em_settings` (1070; the result
  attributes at 1142-1151), `_em_final_e_step` (1713-1801).
- `models/_checkpointed.py`: `_predict_checkpointed` (275-454), including the
  `output_mode` and `result_path` checks (314-330).
- `checkpointed_inference.py`: `checkpointed_forward_backward` (399), for the
  meaning and validation of each option, and `default_chunk_size` (212),
  `default_replay_cache_bytes` (195). Its own arguments do not change.
- `types.py`: `InferenceMode` (112) and `OutputMode` (115).
- Documentation that describes the flags: `docs/performance_prediction.md`,
  `docs/hardware_settings.md`, `docs/result_layout.md`,
  `docs/tutorials/02_decoding_sorted_spikes.py`.
- Tests:
  - 21 detector-level test files pass a removed flag directly in a `predict`
    or `estimate_parameters` call, and others pass one through `**kwargs`. Nine
    files under `tests/core/` name the flags in calls to the core drivers, which
    keep their arguments.
  - Eight test files pass `transition_representation` to
    `estimate_parameters`, several through a parameter that takes all three
    values: `tests/models/test_checkpointed_em.py`,
    `test_checkpointed_diagnostics.py`, `test_em_transition_recovery.py`,
    `test_encoding_update_local_observations.py`,
    `test_frozen_transition_rows.py`, `test_position_input_contract.py`,
    `test_result_schema.py`, and `tests/test_em_monotonicity.py`. All need
    editing, including `test_result_schema.py`, which is also one of this
    phase's checks.
- [appendix.md](appendix.md): the Spyglass adapter (`clusterless.py:330`,
  `sorted_spikes.py:271`) and `get_valid_kwargs` (`utils.py:158`).

## Suggested approach

1. **Capture the reference first**: at the base commit, save the results of
   `predict` for in-memory (one chunk and several, cached and not) and
   checkpointed (compact; spatial with a `result_path`; with
   `selected_intervals`; replay cache zero and default), and of
   `estimate_parameters` on each of today's three paths, for one sorted and one
   clusterless model. Keep the datasets with their attributes.
2. Add the two classes in a public module with `__all__`; export them from the
   package root. Validate field values where they are validated today, so
   messages stay the same where possible.
3. Change the two methods. `_predict_spikes` picks the driver from the type of
   `inference`. The EM option resolution in `models/_em.py` reduces to the
   three cases above; delete what no longer has a caller.
4. For tests that ran `estimate_parameters` with each representation: `"auto"`
   becomes `None`, `"dense"` becomes `InMemory()`, and `"structured"` becomes
   `Checkpointed()`. `"structured"` also refused dense fallback blocks in the
   internal fit, which `Checkpointed()` does not. Where a test depends on that
   refusal, assert it on a `fit(transition_representation="structured")`
   instead. If a test cannot be expressed either way, stop and report
   (overview open question 3).
5. Update call sites, the four documentation pages, the migration note and the
   changelog (a breaking change) in the same pull request. Update the hints and
   warnings in `src` too: about 35 strings under `models/` name a removed flag,
   such as the dense-EM memory warning that says to pass
   `transition_representation="auto"` to `estimate_parameters`.
6. Spyglass, in the #1618 worktree: where `adapter_kwargs` is built, apply the
   translation and the refusal above for both tables and both paths, and add
   tests with a `decoding_kwargs` that names `n_chunks` (1 and more than 1) on
   each path, an `inference` dict for each strategy, and a flat key.
   Local commit only; pushing needs the user's approval.

## Validation

| Check | Asserts | Threshold |
| --- | --- | --- |
| Reference harness, calls rewritten to `inference` | Every array equal with `np.array_equal(a, b, equal_nan=True)`; every result attribute equal in name, type and value | hard |
| `tests/test_golden_regression.py` | Passes with the golden data untouched | hard |
| `tests/models/test_result_schema.py` | Passes after its `estimate_parameters` calls are rewritten; the documented attribute table still matches | hard |
| `tests/models`, `tests/core`, `tests/test_executable_docs.py`, `tests/test_public_api.py` | Pass | hard |
| New tests (below) | As listed | hard |
| Search for the removed keywords in detector calls across `src`, `tests`, `docs`, `README.md`, `benchmarks`, `scripts`, `notebooks`, and for them in message strings under `src` | None left, apart from the historical records phase 2 lists and `docs/performance_validation.md` | hard |
| Docs build | Succeeds | hard |
| Spyglass `tests/decoding` in the #1618 worktree, including the new translation tests | Pass | hard |

New tests, in `tests/models/`, for one sorted model unless noted:

| Test | Asserts |
| --- | --- |
| default is in-memory | `predict(...)` and `predict(..., inference=InMemory())` return identical datasets |
| checkpointed defaults | `Checkpointed()` returns state probabilities with no `state_bins` dimension; `Checkpointed(result_path=tmp)` returns lazily loaded spatial posteriors; `Checkpointed(output_mode="spatial")` without a path raises `ValidationError` |
| the rules that remain | `predict(..., cache_likelihood=True, inference=Checkpointed())` raises `ValidationError`; `predict(..., return_outputs="all", inference=Checkpointed())` raises `ValidationError` whose hint names `result_path` |
| EM path follows `inference` | `em_transition_representation` is `"dense"` for `InMemory()` and `"structured"` for `Checkpointed()` with no field set; `None` gives the same attribute as today's default on the same data |
| EM rejects what it cannot honour | `Checkpointed(selected_intervals=...)` and `Checkpointed(output_mode="compact")` raise `ValidationError` from `estimate_parameters` |
| removed keywords | `predict(n_chunks=2)` and `predict(inference_mode="checkpointed")` raise `TypeError` |
| wrong type | `inference="dense"` raises `ValidationError` naming the two classes |

## Fixtures

Existing simulated-data fixtures and `tmp_path` for result stores.

## Deliberately not in this phase

- Choosing the driver automatically in `predict` from the recording's size.
- Changing `fit`'s `transition_representation` default, or having `fit` pick
  the representation. One small case decoded identically after a structured
  and a dense fit, but fit time and memory on real 2-D environments have not
  been measured.
- The core drivers' own signatures (see overview, "Deliberately not in this
  plan").
- The data argument: phase 2.

## Review

Before opening the pull request, have an independent reviewer read the diff
and confirm:

- The harness comparison is exact, attributes included, with its output
  attached.
- No removed flag survives as an alias, and no validation that only policed
  flag combinations is left behind.
- `Checkpointed()` with no field set runs structured EM.
- The documentation pages, migration note and changelog are updated.
- The Spyglass translation covers both of its paths, keeps `n_chunks=1` rows on
  the automatic EM path, lets a row choose either strategy, refuses the flat
  keys, and has tests.
- The new tests do not need editing when an option is added to either class.
- No name in code, tests or docstrings refers to this plan.
