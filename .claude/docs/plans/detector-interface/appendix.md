# Appendix: Spyglass references, call-site counts, released options

[← back to PLAN.md](PLAN.md)

## Spyglass PR #1618

- Repository `LorenFrankLab/spyglass`, pull request #1618, branch
  `decoding-params-serialization-roundtrip`, head `f7b3d4386` (open, not a draft,
  2026-11-06). A local worktree of that branch is at
  `/Users/edeno/Documents/GitHub/spyglass-nld-contracts`.
- It already pins `non-local-detector>=0.7.0,<0.8` (`pyproject.toml:60` and the
  four `environments/*.yml` files).
- Pushing to that branch, or opening a pull request there, needs the user's
  approval each time.

### Detector calls (all by keyword)

| File (under `src/spyglass/decoding/v1/`) | Line | Call |
| --- | --- | --- |
| `clusterless.py` | 369 | `classifier.estimate_parameters(position_time=, position=, spike_times=, spike_waveform_features=, time_edges=, **estimate_kwargs)` |
| `clusterless.py` | 384 | `classifier.fit(position_time=, position=, spike_times=, spike_waveform_features=, **fit_kwargs)` |
| `clusterless.py` | 430 | `classifier.predict(position_time=, position=, spike_times=sequence_spikes, spike_waveform_features=sequence_marks, time_edges=, **interval_kwargs)` |
| `sorted_spikes.py` | 310 | `classifier.estimate_parameters(...)`, `spike_times=` by keyword |
| `sorted_spikes.py` | 324 | `classifier.fit(...)`, `spike_times=` by keyword |
| `sorted_spikes.py` | 368 | `classifier.predict(...)`, `spike_times=` by keyword |

- `utils.py:383` `spikes_for_sequence(spike_times, edges, *, shared_stop,
  shared_boundary, spike_waveform_features=None)` returns `(times, marks)` for
  one decode interval. The `predict` call sites pass its result.
- One test file under `tests/decoding/` builds a `dict` with a `spike_times`
  key.

### Stored keyword arguments

`DecodingParameters` rows hold a user dict, `decoding_kwargs`, that Spyglass
passes to the detector's methods:

- **Fit and predict path:** `utils.py:158` `get_valid_kwargs(classifier,
  decoding_kwargs, logger)` reads the signatures of `classifier.fit` and
  `classifier.predict`, keeps the keys each accepts, and logs a warning naming
  the keys neither accepts. It does not raise.
- **`estimate_parameters` path:** `clusterless.py:369` and
  `sorted_spikes.py:310` pass `**estimate_kwargs` (`decoding_kwargs` plus
  `is_missing` and the covariates) straight through, so a key the method does
  not accept raises `TypeError`.
- `clusterless.py:330` (and `sorted_spikes.py:271`) already pop adapter-only
  keys out of `decoding_kwargs` into `adapter_kwargs` before either path. A
  translation of renamed keys belongs there.
- Which keys real rows hold is not known. The two Spyglass notebooks
  (`notebooks/py_scripts/41_Decoding_Clusterless.py`,
  `42_Decoding_SortedSpikes.py`) use an empty dict.
- `decoding_kwargs` is a LONGBLOB (`core.py:35`), and
  `convert_classes_to_dict` applies only to `decoding_params`. DataJoint 0.14.9
  packs a flat dict of strings and numbers, and refuses a dataclass:
  `Packing object of type <class 'Checkpointed'> currently not supported!`

## What 0.6.9 released

Stored rows and user scripts can only name options that a released version
accepted. In `v0.6.9` (`git show v0.6.9:src/non_local_detector/models/base.py`):

| Option | In 0.6.9 |
| --- | --- |
| `n_chunks`, `cache_likelihood`, `store_log_likelihood` | yes |
| `inference_mode`, `output_mode`, `chunk_size`, `result_path`, `checkpoint_dir`, `selected_intervals`, `max_read_bytes`, `replay_cache_bytes` | no |
| `transition_representation`, `return_outputs`, `time_edges` | no |

0.6.9's `predict(spike_times, time, position=None, position_time=None,
is_missing=None, discrete_transition_covariate_data=None,
cache_likelihood=False, n_chunks=1)` took `time`, which 0.7.0 replaces with
`time_edges`. Every `predict` call written for 0.6.9 already has to change for
0.7.0.

## Call sites that change when the spike arguments become one data argument

Counted at `0b88f984` by parsing each file and classifying calls of `fit`,
`predict`, `estimate_parameters`, `most_likely_sequence`,
`compute_log_likelihood` and `fit_encoding_model` by how they pass spike data.
The positional classification is a heuristic, and the last column counts every
`dict(spike_times=...)` or `{"spike_times": ...}` literal, including ones that
feed likelihood functions rather than detectors, so it is an upper bound.

| Where | Sorted, `spike_times=` by keyword | Clusterless, features by keyword | Clusterless, positional | Dict with a `spike_times` key (upper bound) |
| --- | --- | --- | --- | --- |
| `tests/` | 62 calls, 29 files | 35 calls, 14 files | 73 calls, 24 files | 116 in 52 files |
| `docs/tutorials/` | 1 | 1 | 9 calls, 4 files | 0 |
| `docs/getting-started/` | 1 | 0 | 0 | 0 |
| `README.md` | 1 | 2 | 0 | 0 |
| `notebooks/` | 25 calls, 11 files | 3 calls, 2 files | 3 calls, 2 files | 0 |
| `benchmarks/` | 0 | 0 | 2 calls, 1 file | 6 in 5 files |
| `scripts/` | 0 | 0 | 0 | 6 in 3 files |
| Spyglass #1618 `src/` | 3 | 3 | 0 | 0 |

77 files in this repository hold a call that changes, or at most 115 with the
files that hold such a dict literal. With the argument renamed, a sorted call
that names `spike_times=` changes; a sorted call that passes the list by
position does not, so those are not counted. Two of the six methods are also
called from inside `models/base.py`: the internal `fit` and `fit_encoding_model`
calls that phase 1 replaces.

The script that produced the table is not kept in the repository. To recount,
parse each `.py` file, each notebook code cell and each fenced Python block with
`ast`, and classify `ast.Call` nodes whose `func.attr` is one of the six method
names by their keywords (`spike_times`, `spike_waveform_features`) and by the
number of positional arguments after the leading `position_time, position`
(two for `fit`, `estimate_parameters`, `compute_log_likelihood` and
`fit_encoding_model`; none for `predict` and `most_likely_sequence`).
