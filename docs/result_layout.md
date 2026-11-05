# Result layout

`predict` and `estimate_parameters` return an [xarray](https://docs.xarray.dev)
`Dataset`. This page lists its dimensions, coordinates, data variables and
attributes, for single- and multi-state models, for each `return_outputs`
option, and for dense and checkpointed inference
([Checkpointed prediction](performance_prediction.md)). Dense and checkpointed
results have the same layout apart from the differences noted here. The
test suite checks the tables on this page against the results the package
builds.

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

| Coordinate | Dimension | Present in | Meaning |
| --- | --- | --- | --- |
| `time` | `time` | All | Bin centers, in seconds |
| `time_bin_start`, `time_bin_end` | `time` | All | Bin edges, in seconds |
| `time_bin_width` | `time` | All | `time_bin_end - time_bin_start`, in seconds |
| `time_bin_end_inclusive` | `time` | All | True only for the final bin of the decode grid, which owns its right edge |
| `is_missing` | `time` | All | Bins decoded without a likelihood: the `is_missing` you passed, plus bins without tracked position for models that need position |
| `source_row` | `time` | Checkpointed | Each row's index in the full decode grid, so selected rows can be placed in the recording |
| `states` | `states` | All | State names |
| `environments` | `states` | All | Each state's environment name |
| `encoding_groups` | `states` | All | Each state's encoding group |
| `state_bins` | `state_bins` | With `state_bins` | A MultiIndex with levels `state` and the position dimensions: `position` in 1-D; `x_position`, `y_position` in 2-D; `z_position`, ... beyond |
| `state_ind` | `state_bins` | With `state_bins` | Each bin's index into `states` (the detector's `state_ind_`) |

"With `state_bins`" means dense results and checkpointed spatial results.

States without a position (No-Spike, and Local unless `local_position_std` is
set) have one bin each, with NaN position. Unstacking `state_bins` puts that
bin at the last position; `non_local_detector.analysis.get_spatial_posterior`
leaves those states out.

`state_ind` is a plain (non-index) coordinate along `state_bins`, so
`results.acausal_posterior.unstack("state_bins")` keeps it as a float
coordinate over `(state, *position)`, NaN where a state has no bin. Drop it
first if you do not want it:
`results.acausal_posterior.drop_vars("state_ind").unstack("state_bins")`.
`get_spatial_posterior` drops it.

Bins follow the `time_bin_convention` attribute: each bin is closed on the left
and open on the right, except the final bin of the decode grid, which is closed
on both sides. See [Time bins and rates](time_grid_migration.md).

## Data variables

Variables along `state_bins` are float32 and are NaN at bins off the track
interior. State probabilities are float32 with JAX's default precision.

| Variable | Dimensions | Added by `return_outputs` | In compact results | Meaning |
| --- | --- | --- | --- | --- |
| `acausal_posterior` | `(time, state_bins)` | Always | No | Probability of each state and position given all the data (smoother) |
| `acausal_state_probabilities` | `(time, states)` | Always | Yes | Probability of each state given all the data |
| `causal_posterior` | `(time, state_bins)` | `"filter"` | No | Probability given the data up to each bin (filter) |
| `causal_state_probabilities` | `(time, states)` | `"filter"` | Yes | State probability given the data up to each bin |
| `predictive_state_probabilities` | `(time, states)` | `"predictive"` | Yes | State probability given the data before each bin (one-step-ahead prediction) |
| `predictive_posterior` | `(time, state_bins)` | `"predictive_posterior"` | No | One-step-ahead predicted probability of each state and position |
| `log_likelihood` | `(time, state_bins)` | `"log_likelihood"` | No | Log likelihood of each bin's spikes at each state bin |

The column "Added by `return_outputs`" lists the option that adds each
variable, alone or in any collection (list, tuple, set); `return_outputs="all"` adds every one.
Checkpointed compact results accept only `"filter"` and `"predictive"`, and
hold only the variables marked "Yes".
For a single-state model, the `(time, states)` variables are `(time,)`.

Checkpointed spatial results read their `state_bins` variables lazily from the
store at `result_path`; one read larger than `max_read_bytes` raises an error.

## Attributes

| Attribute | Type | Present in | Meaning |
| --- | --- | --- | --- |
| `marginal_log_likelihoods` | float64 array | All | Marginal log likelihoods, one-dimensional: one for `predict`, one per E-step for `estimate_parameters` (the last is the final decode's) |
| `time_bin_convention` | str | All | `"left_closed_right_open_final_closed"` |
| `time_units` | str | All | `"seconds"` |
| `non_local_detector_version` | str | All | Version that made the result |
| `em_transition_representation` | str | `estimate_parameters` | `"dense"` or `"structured"` EM |
| `em_objectives`, `em_previous_objectives` | float64 array | `estimate_parameters` | The objectives EM's convergence checks compare (see `estimate_parameters`) |
| `em_n_chunks` | int | Dense EM | Chunks of the dense E-step |
| `em_chunk_size`, `em_replay_cache_bytes` | int | Structured EM | Checkpoint chunk rows and replay cache of the E-step |
| `inference_mode`, `output_mode` | str | Checkpointed | `"checkpointed"`; `"compact"` or `"spatial"` |
| `conditioning` | str | Checkpointed | Selected rows are conditioned on the whole recording |
| `whole_recording_conditioning` | bool | Checkpointed | True: selected rows are conditioned on the whole recording |
| `recording_rows`, `n_recording_bins` | int | Checkpointed | Bins in the full decode grid |
| `transition_representation` | str | Checkpointed | Transition representation the decode used |
| `dtype`, `state_probability_dtype`, `evidence_accumulation`, `evidence_dtype` | str | Checkpointed | Precision of the outputs and of the log-evidence sum |
| `n_degenerate`, `n_nan` | int | Checkpointed | Rows whose filter was degenerate or NaN, over the full decode grid |
| `degenerate_row_mask_hex`, `nan_row_mask_hex`, `diagnostic_row_mask_encoding` | str | Checkpointed | Those rows as bit masks over the full decode grid |
| `diagnostics_are_global` | bool | Checkpointed | True: the diagnostics cover the full decode grid |
| `max_chunk_rows`, `likelihood_evaluations`, `checkpoint_bytes`, `checkpoint_cache_entries`, `replay_cache_bytes` | int | Checkpointed | Run diagnostics |
| `global_gradients_supported` | bool | Checkpointed | Run diagnostic |
| `_native_metadata` | dict | Checkpointed | Internal labels for reopening the store; `save_results` drops it |

Arrays are NumPy arrays and the other types are Python's.

`estimate_parameters` with a `result_path` returns a checkpointed result, with
both the EM and the checkpointed attributes.

## Saving and loading

`save_results(results, "results.nc")` writes netCDF, which cannot hold a
MultiIndex: it stores the `state_bins` levels as plain coordinates, booleans
in attributes as integers, and drops `_native_metadata`. It records the dtype
of every integer variable and coordinate in the file's
`non_local_detector_integer_dtypes` attribute, because netCDF3 (written when
neither netCDF4 nor h5netcdf is installed) stores int64 as int32, and the
type of every attribute that is not a string in its
`non_local_detector_attribute_types` attribute, because netCDF reads a
boolean back as an integer, an integer or float as a NumPy scalar and a
one-element array as a scalar.
`load_results("results.nc")` rebuilds the `state_bins` MultiIndex, keeping
`state_ind` as an integer coordinate along it, gives the integer variables and
coordinates (`state_ind`, `encoding_groups`, checkpointed `source_row`) their
recorded dtypes, unless a variable was changed to a non-integer dtype after
saving, and gives the attributes their recorded types (the types in the table
above). The MultiIndex's `state` level holds objects, as in the results
saved; other string coordinates come back as xarray reads them (fixed-width
strings from netCDF4). Files saved by earlier versions, which did not record
attribute types, keep the attributes netCDF reads, except that
`marginal_log_likelihoods` is made one-dimensional. The netCDF data are read lazily, and closing the returned dataset
(or leaving a `with` block) closes the file. `load_results(result_path)`
reopens a checkpointed store with lazy arrays.

## Most likely sequence

`most_likely_sequence` returns a pandas `DataFrame` indexed by `time`, with
one row per bin and columns `state`, the position dimensions, `environment`
and `encoding_group_names` for the most likely state bin.
