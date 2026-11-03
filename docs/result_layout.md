# Result layout

`predict` and `estimate_parameters` return an [xarray](https://docs.xarray.dev)
`Dataset`. This page lists its dimensions, coordinates, data variables and
attributes, for single- and multi-state models, for each `return_outputs`
option, and for dense and checkpointed inference
([Checkpointed prediction](performance_prediction.md)). Dense and checkpointed
results have the same layout apart from the differences noted here.

## Dimensions

| Dimension | Length | Absent when |
| --- | --- | --- |
| `time` | Returned decode bins: every bin of `time_edges`, or the bins `selected_intervals` selects in checkpointed mode | Never |
| `state_bins` | Every (state, position bin) pair of the model, including bins off the track interior | Checkpointed `output_mode="compact"` |
| `states` | Discrete states | The model has one state (decoders) |

A single-state model, such as `SortedSpikesDecoder` or `ClusterlessDecoder`,
has no `states` dimension: its state probabilities are `(time,)` and its
`states`, `environments` and `encoding_groups` coordinates are scalars. Use
`np.atleast_1d(results.coords["states"].values)` to get the state names as an
array for any model.

## Coordinates

| Coordinate | Dimension | Meaning |
| --- | --- | --- |
| `time` | `time` | Bin centers, in seconds |
| `time_bin_start`, `time_bin_end` | `time` | Bin edges, in seconds |
| `time_bin_width` | `time` | `time_bin_end - time_bin_start`, in seconds |
| `time_bin_end_inclusive` | `time` | True only for the final bin of the decode grid, which owns its right edge |
| `is_missing` | `time` | Bins decoded without a likelihood: the `is_missing` you passed, plus bins without tracked position for models that need position |
| `source_row` | `time` | Checkpointed only: each row's index in the full decode grid, so selected rows can be placed in the recording |
| `states` | `states` | State names |
| `environments` | `states` | Each state's environment name |
| `encoding_groups` | `states` | Each state's encoding group |
| `state_bins` | `state_bins` | A MultiIndex with levels `state` and the position dimensions: `position` in 1-D; `x_position`, `y_position` in 2-D; `z_position`, ... beyond |
| `state_ind` | `state_bins` | Each bin's index into `states` (the detector's `state_ind_`) |

States without a position (No-Spike, and Local unless `local_position_std` is
set) have one bin each, with NaN position. Unstacking `state_bins` puts that
bin at the last position; `non_local_detector.analysis.get_spatial_posterior`
leaves those states out.

Bins follow the `time_bin_convention` attribute: each bin is closed on the left
and open on the right, except the final bin of the decode grid, which is closed
on both sides. See [Time bins and rates](time_grid_migration.md).

## Data variables

Variables along `state_bins` are float32 and are NaN at bins off the track
interior. State probabilities are float32 with JAX's default precision.

| Variable | Dimensions | Present when | Meaning |
| --- | --- | --- | --- |
| `acausal_posterior` | `(time, state_bins)` | Always, except checkpointed compact | Probability of each state and position given all the data (smoother) |
| `acausal_state_probabilities` | `(time, states)` | Always | Probability of each state given all the data |
| `causal_posterior` | `(time, state_bins)` | `"filter"`, except checkpointed compact | Probability given the data up to each bin (filter) |
| `causal_state_probabilities` | `(time, states)` | `"filter"` | State probability given the data up to each bin |
| `predictive_state_probabilities` | `(time, states)` | `"predictive"` | State probability given the data before each bin (one-step-ahead prediction) |
| `predictive_posterior` | `(time, state_bins)` | `"predictive"` or `"predictive_posterior"` | One-step-ahead predicted probability of each state and position |
| `log_likelihood` | `(time, state_bins)` | `"log_likelihood"` | Log likelihood of each bin's spikes at each state bin |

`return_outputs="all"` adds every optional variable. A string `"predictive"`
adds both predictive variables; in a list, `"predictive"` adds only
`predictive_state_probabilities`. Checkpointed compact results accept only
`"filter"` and `"predictive"`, and add only their state probabilities.
For a single-state model, the `(time, states)` variables are `(time,)`.

Checkpointed spatial results read their `state_bins` variables lazily from the
store at `result_path`; one read larger than `max_read_bytes` raises an error.

## Attributes

| Attribute | Present in | Meaning |
| --- | --- | --- |
| `marginal_log_likelihoods` | All | Array of marginal log likelihoods: one for `predict`, one per E-step for `estimate_parameters` |
| `time_bin_convention` | All | `"left_closed_right_open_final_closed"` |
| `time_units` | All | `"seconds"` |
| `non_local_detector_version` | All | Version that made the result |
| `em_transition_representation` | `estimate_parameters` | `"dense"` or `"structured"` EM |
| `em_objectives`, `em_previous_objectives` | `estimate_parameters` | The objectives EM's convergence checks compare (see `estimate_parameters`) |
| `em_n_chunks` | Dense EM | Chunks of the dense E-step |
| `em_chunk_size`, `em_replay_cache_bytes` | Structured EM | Checkpoint chunk rows and replay cache of the E-step |
| `marginal_log_likelihood` | Checkpointed | Marginal log likelihood of the final decode |
| `inference_mode`, `output_mode` | Checkpointed | `"checkpointed"`; `"compact"` or `"spatial"` |
| `conditioning`, `whole_recording_conditioning` | Checkpointed | Selected rows are conditioned on the whole recording |
| `recording_rows`, `n_recording_bins` | Checkpointed | Bins in the full decode grid |
| `transition_representation` | Checkpointed | Transition representation the decode used |
| `dtype`, `state_probability_dtype`, `evidence_accumulation`, `evidence_dtype` | Checkpointed | Precision of the outputs and of the log-evidence sum |
| `n_degenerate`, `n_nan`, `degenerate_row_mask_hex`, `nan_row_mask_hex`, `diagnostic_row_mask_encoding`, `diagnostics_are_global` | Checkpointed | Rows whose filter was degenerate or NaN, as counts and bit masks over the full decode grid |
| `max_chunk_rows`, `likelihood_evaluations`, `checkpoint_bytes`, `checkpoint_cache_entries`, `replay_cache_bytes`, `global_gradients_supported` | Checkpointed | Run diagnostics |
| `_native_metadata` | Checkpointed | Internal labels for reopening the store; `save_results` drops it |

`estimate_parameters` with a `result_path` returns a checkpointed result, with
both the EM and the checkpointed attributes.

## Saving and loading

`save_results(results, "results.nc")` writes netCDF, which cannot hold a
MultiIndex: it stores the `state_bins` levels as plain coordinates, booleans
in attributes as integers, and drops `_native_metadata`.
`load_results("results.nc")` rebuilds the `state_bins` MultiIndex, keeping
`state_ind` as a coordinate along it. `load_results(result_path)` reopens a
checkpointed store with lazy arrays.

## Most likely sequence

`most_likely_sequence` returns a pandas `DataFrame` indexed by `time`, with
one row per bin and columns `state`, the position dimensions, `environment`
and `encoding_group_names` for the most likely state bin.
