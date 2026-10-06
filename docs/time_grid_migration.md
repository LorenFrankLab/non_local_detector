# Migrating to explicit decode bins and rates in Hz

This release changes spike ownership and encoding units. Refit saved detectors
and recompute downstream results. Renaming `time` alone is insufficient.

## Choose decode boundaries

All prediction, estimation, Viterbi, and spike-counting calls require
`time_edges=` as a keyword. N + 1 boundaries produce N rows. Bins are closed on
the left and open on the right, except the final bin of each call also owns its
right boundary. Detector/HMM bins must be uniform; standalone likelihoods can
use variable durations.

`detector.compute_log_likelihood` follows the detector's uniform-grid and
transition-clock contract too. For variable-duration likelihoods without an
HMM, call the registered predictors in `non_local_detector.likelihoods`
directly with freshly fitted Hz encoding fields.

```python
from non_local_detector import calculate_time_edges, time_edges_from_centers

# Here detector is a fitted SortedSpikesDecoder. NonLocal detectors also
# require position_time and position when evaluating their local state.
# An explicit acquisition/decode interval; trim drops any partial final bin.
edges = calculate_time_edges([start, stop], sampling_frequency=500, trim=True)
results = detector.predict(spike_times, time_edges=edges)

# Uniform existing sample timestamps, retaining their nominal row coordinates.
edges = time_edges_from_centers(position_time)
results = detector.predict(spike_times, time_edges=edges,
                           position_time=position_time, position=position)
sequence = detector.most_likely_sequence(spike_times, time_edges=edges)
```

For a runnable example with camera jitter, tracking gaps, and independent
intervals, see [`scripts/time_grid_migration_example.py`](../scripts/time_grid_migration_example.py).

The centered conversion preserves the number of rows, but between-sample spikes
now belong to the nearest sample's bin. Old `time` bins assigned them to the
preceding sample. This can change likelihoods and posteriors. Floating-point
rounding can also change the reconstructed centers by a few ulps: align data to
`results.time`, rather than relying on exact timestamp equality.

The removed `get_spike_time_bin_ind` helper is replaced by explicit selection
and counting. Keep spike times and marks from the same unit aligned:

```python
from non_local_detector.likelihoods.common import (
    get_spikecount_per_time_bin, select_spikes_in_rows, select_spike_rows,
)

n_bins = len(edges) - 1
unit_spike_times = spike_times[unit]
selection = select_spikes_in_rows(unit_spike_times, 0, n_bins, time_edges=edges)
selected_marks = select_spike_rows(spike_waveform_features[unit], selection)
event_rows = selection.bin_ind
counts = get_spikecount_per_time_bin(unit_spike_times, time_edges=edges)
```

`selection.indexer` selects original events and `event_rows` indexes the
requested likelihood rows. The final row can contain spikes; retain it in
replay/shuffle code.

## Keep encoding support separate from the decode grid

Encoding uses original position timestamps and physical recording support.
Uniform tracking defaults to endpoint half-cells, giving N * dt seconds of
coverage. To clip recording support, pass `encoding_time_range=[start, stop]`
to `fit`, `fit_encoding_model`, or `estimate_parameters`. A singleton position
sample needs explicit recording bounds or a tracking interval.

For camera jitter, irregular sampling, or disconnected epochs, explicitly
identify continuous tracking intervals:

```python
detector.fit(
    position_time, position, spike_times,
    encoding_time_range=[acquisition_start, acquisition_stop],
    valid_position_intervals=[[epoch1_start, epoch1_stop],
                              [epoch2_start, epoch2_stop]],
)
```

Each interval needs at least one finite position sample. Positions are linearly
interpolated within an interval and held at its endpoint samples. NaN rows split
support before fitting; do not drop them and concatenate across tracking gaps.
Acquisition bounds alone do not assert tracking continuity. Duplicate timestamps
must be resolved upstream with an explicit policy.

Adjacent declared encoding intervals stay separate. If they share a boundary,
an event on that boundary belongs to the interval on its right; a segment's
closing boundary is included when no following segment starts there. These
rules count each supported encoding event once.

Exposure integrates the same interpolation basis used for event weights. Sample
occupancy is weighted in seconds; spikes carry their dimensionless interpolated
weight once. Fitted `mean_rates`, `place_fields`, and ground-process intensities
are now Hz, independently of tracking frequency. Predictions multiply those
rates by each decode-bin duration. The unused `sampling_frequency` argument has
been removed from encoding fit functions; the detector setting remains useful
for grid construction. The GLM default L2 penalty is now 0.5 per second,
equivalent to the previous 0.001 at the 500 Hz reference.

Stored `mean_rates`, `place_fields`, and `summed_ground_process_intensity` are
Hz; `interior_log_place_fields` is log-Hz. The historical
`no_spike_part_log_likelihood` key holds the **sum of Hz rates across neurons**,
not a log likelihood. Predictors multiply it by duration and subtract it.
Direct predictors default to `rate_units="Hz"`: callers supplying rate arrays
themselves are responsible for those units. Passing an old per-sample array
without its marker cannot be detected; prefer freshly fitted dictionaries or
the guarded detector API.

## Align masks, covariates, and interval labels

Decode `is_missing` has N entries; training masks and encoding/environment labels
have one entry per original position sample. Encoding support does not
automatically mark decode gaps missing. For a uniform grid, explicitly mark bins
outside declared valid observation intervals; keeping only bins whose entire
span is supported is a conservative policy.

Encoding's held endpoint positions do not change prediction interpolation.
Local decoding interpolates the supplied `position_time`/`position`; concatenated
finite segments can otherwise interpolate across a gap, including the held
endpoint portions of encoding support. Either prepare held endpoints separately
for each decode segment, or conservatively restrict observed bins to continuous
finite sample spans, as here:

```python
import numpy as np

# Two finite tracking spans. Encoding could additionally hold endpoints over
# [-0.5, 1.5] and [9.5, 11.5]; this decode policy uses only measured spans.
position_time = np.array([0., 1., 10., 11.])
position = np.array([[0.], [0.], [100.], [100.]])
observation_intervals = np.array([[0., 1.], [10., 11.]])
edges = calculate_time_edges([0., 11.], sampling_frequency=500)
supported = np.any(
    (edges[:-1, None] >= observation_intervals[:, 0])
    & (edges[1:, None] <= observation_intervals[:, 1]),
    axis=1,
)
results = detector.predict(
    spike_times, time_edges=edges, position_time=position_time,
    position=position, is_missing=~supported,
)
```

Derive these observation spans from the actual finite tracking segments,
splitting at NaNs and declared discontinuities. This one-grid policy propagates
the HMM through masked gaps. For separate calls, use each segment's own position
timeline and aligned mask.

Interpolate continuous covariates onto decode centers using their original
sample timeline. Categorical labels and Boolean masks need an explicit ownership
rule; interpolating them as numbers can change their meaning. Covariate-driven
transition data must have one aligned row per decode bin. Slice or rebuild
per-interval arguments for every prediction instead of reusing a full-recording
mask.

Viterbi uses the fitted covariate-transition tensor. Its rows must already align
with the requested bins; a different row count raises before likelihood
evaluation. Fit/estimate with aligned covariates before Viterbi. `predict` also
accepts new aligned `discrete_transition_covariate_data` for another grid.

Custom likelihoods used with chunking must accept global `row_slice` and carry
the `@row_slice_aware` decorator. Always forward the full edges and row range:

```python
from non_local_detector.core import row_slice_aware

original_likelihood = detector.compute_log_likelihood

@row_slice_aware
def custom_likelihood(*args, time_edges, row_slice=None, is_missing=None):
    return original_likelihood(
        *args, time_edges=time_edges, row_slice=row_slice, is_missing=is_missing,
    )
```

An unmarked callback still works on the full grid, but partial/chunked requests
raise. Adjusting a chunk's closing edge changes its duration, so an edges-only
callback cannot preserve both exposure and shared-boundary ownership.

Use `results.time` and the saved `time_bin_start`, `time_bin_end`,
`time_bin_end_inclusive`, and effective `is_missing` coordinates when plotting,
joining, or saving. Give independent calls a sequence/interval identifier before
concatenation; the inclusion flag retains each call's final closed boundary.
NaN tracking does not change the identity of the requested interval.

## Decide whether gaps continue or reset the HMM

Missing observations on one uniform grid let the HMM propagate through a gap.
Separate prediction calls restart the sequence from its initial distribution.
These are different statistical models. Do not silently concatenate disconnected
recordings for EM or infer a reset from `is_missing`.

When adjacent independent calls share a boundary, filter spike times so that an
event is passed to only one call. Apply that same event mask to waveform marks.
For example, use `[start, stop)` for each earlier sequence and include the final
stop only in the last sequence. The decoder's final-closed rule otherwise counts
the shared event in both calls.

## Saved models and downstream pipelines

Legacy pickles can be loaded for inspection. Prediction, likelihood evaluation,
and Viterbi reject models without known Hz units and transition provenance,
including when likelihoods are cached. Call `fit` or `estimate_parameters` with
the original recording to establish the new contract. EM-learned transitions are
bound to `transition_time_bin_width_`; prediction at another width requires
refitting. Configuration-only transitions remain defined per HMM step.

Before likelihood or HMM work, the detector checks every
`(environment, encoding_group)` required by a spike state. A missing entry or
missing/unsupported Hz marker requires fitting again, even if other entries
are current or a likelihood array is cached. Unused entries do not block
decoding; NoSpike-only states use the constructor's already-Hz rate rather than
an encoding entry. Known model and transition-clock metadata are still required.

Unit markers live in fitted attributes and encoding dictionaries. Constructor
parameters and `get_params()` remain configuration only, so reconstruct from
those parameters and fit again. `vars(fitted_detector)` contains learned state
and is not a constructor-argument dictionary.

`RandomWalk.movement_var` is the displacement variance for one HMM step;
`movement_mean` is the mean displacement for that step. `EmpiricalMovement`
estimates transitions between successive selected position samples and applies
`speedup` by taking a matrix power of that transition. Changing decode-bin width
does not automatically rescale either movement model. Choose their configuration
for the intended decode interval; uniform grids alone do not provide continuous
time calibration. Learned discrete-transition widths are checked separately.

The public functions in `non_local_detector.core` operate on observation rows
and do not validate decode edges. Detector prediction, estimation, and Viterbi
enforce this time contract, including internal cached prediction paths.

Seven backend families preserve model units and predictions through pickle
save/load. GLM model pickling remains unsupported because Patsy `DesignInfo`
cannot be serialized; this pre-existing defect is deferred. Refit GLM models
from their original recording and configuration in the process that decodes
them. Result datasets can still be saved independently.

Spyglass adapters must construct uniform decode grids, preserve original
tracking support, align masks and covariates, and label actual result rows.
Do not pass irregular camera timestamps as decode edges. Replay/shuffle adapters
must stop dropping a supposed unreachable final row, use keyword-only counting
and likelihood APIs, and convert every Hz ground-process field into expected
counts with the appropriate row duration. Recompute stored likelihoods,
posteriors, shuffles, and derived statistics after migration.
