# Data contract

What the detectors expect of the arrays you give them, and what they do with
data that break it. Every public method (`fit`, `fit_encoding_model`,
`estimate_parameters`, `predict`, `most_likely_sequence`,
`compute_log_likelihood`) reads its data through the same checks, so the same
input gets the same answer at each method that takes it. Arrays that cannot be
read raise. Data that can be read but look wrong give one
[`DataQualityWarning`](warnings.md) per call and are used as given.

## Units and clocks

| Quantity | Unit |
| --- | --- |
| `position_time`, `spike_times`, `time_edges`, `encoding_time_range`, `valid_position_intervals` | seconds, on one clock |
| `position` | your choice (usually cm), the same as `place_bin_size` and `position_range` |
| `sampling_frequency` | decode bins per second (Hz) |
| fitted `mean_rates`, `place_fields` | Hz |

Use float64 timestamps: float32 resolves times near 1.7e9 s (a Unix clock) only
to 128 s. Spike times whose dtype cannot resolve a decode bin
(`np.spacing(max(abs(t)))` above `1 / sampling_frequency`) warn; `time_edges`
that cannot resolve their own bins raise.

## Shapes

| Argument | Shape | Meaning |
| --- | --- | --- |
| `position_time` | `(n_position_samples,)` | times of the position samples, strictly increasing |
| `position` | `(n_position_samples,)` or `(n_position_samples, n_dims)` | the animal's position; with a track graph, raw `(x, y)` (the detector linearizes it, and 1-D position raises `ValidationError`) |
| `spike_times` | list of `(n_spikes_i,)` arrays | sorted data: one array per neuron. Clusterless data: one array per electrode |
| `spike_waveform_features` | list of `(n_spikes_i, n_features)` arrays | clusterless data only: the waveform features (marks) of each spike on each electrode |
| `time_edges` | `(n_bins + 1,)` | decode bin edges; results have one row per bin |
| `is_missing` | `(n_bins,)` | bins to decode without their observations |
| `is_training`, `encoding_group_labels`, `environment_labels`, `weights` | `(n_position_samples,)` | one entry per position sample |
| `valid_position_intervals` | `(n_intervals, 2)` | ordered, non-overlapping `[start, stop]` spans of continuous tracking |
| `encoding_time_range` | `(2,)` | `[start, stop]` of the encoding support |
| `discrete_transition_covariate_data` | one row per decode bin, at the bin centers (`fit` takes the training values on any timeline) | covariates of the transition formula |

## NaN and infinite values

- **`spike_times`**: a NaN or infinite time raises `DataError`. Empty arrays are
  valid.
- **`position_time`, `time_edges`**: must be finite and strictly increasing.
- **`position`**: a row with a NaN marks an untracked sample; an infinite value
  raises when fitting. Fitting splits its support at NaN rows. Decoding marks
  every bin that would read a NaN position as missing: a bin that owns any time
  between the samples either side of a NaN sample, so isolated dropouts at
  30 Hz mask about 67 ms each. No warning fires; the results' `is_missing` says
  which bins these are.
- **`is_missing`** only concerns decoding: a bin marked missing has a log
  likelihood of 0, so its posterior follows the transitions alone. It does not
  exclude encoding data, also in `estimate_parameters`, whose encoding refits
  read position samples: use `is_training` to leave samples out of the
  encoding model.

## Time grid

Decoding takes bin **edges**. Bins are closed on the left and open on the
right, except that the final bin of a call also owns its right boundary. Detector
bins must be uniform, because the HMM applies one transition per bin, and
`1 / sampling_frequency` wide, within the larger of 4 ulp of the largest edge
and 0.1% of the width; other grids raise. A model whose transitions were learned
by `estimate_parameters` decodes only at the width they were learned at.

Build edges with `detector.calculate_time_edges([start, stop])` (bins of
`1 / sampling_frequency` seconds) or `time_edges_from_centers(position_time)`
(one bin per position sample). A detector decoding one bin per position sample
is constructed with the position sampling rate. Spikes are assigned to bins by
these edges alone, so align anything else (masks, covariates) to the bin
centers, `(time_edges[:-1] + time_edges[1:]) / 2`. See
[Time bins and rates](time_grid_migration.md) for migrating code and for
independent decoding sequences.

## Tracking coverage

Position is read between samples by interpolation, so what the samples leave
out has to be declared.

**Uniform `position_time`** is one continuous recording: its steps match their
mean within the tolerance for bins above. NaN rows mark the gaps.

**Irregular `position_time`** (dropped rows, camera jitter, disconnected
epochs) needs `valid_position_intervals`, the spans in which tracking was
continuous. Do not drop NaN rows and concatenate across a gap without it:

- `fit`, `fit_encoding_model` and `estimate_parameters` raise `DataError`;
- `predict`, `most_likely_sequence` and `compute_log_likelihood` interpolate
  position across the gap, so the animal appears to glide from one side to the
  other. Spikes in the gap then disagree with that invented path, which reads
  as replay. They warn once if a bin not already missing reaches a gap of five
  or more position samples; a dropped frame or a jittery clock does not warn.

With `valid_position_intervals`, a decoding method marks as missing every bin
that owns a time outside the intervals. As for the tracked period as a whole,
a time within half a position sample of an interval counts as inside, so the
bins centered on an interval's first and last samples stay observed, and an
interval covering all of the tracking changes nothing. In `estimate_parameters`
these bins are missing in every E-step. A model that does not read position at
decode (no Local state and no non-local position penalty) decodes every bin and
ignores the intervals.

A decoding method warns once when a bin it reads falls in an interval with no
position sample, since position there comes from samples outside the interval,
and when position or the intervals leave every bin missing and the caller did
not ask for that, since the transitions' steady state looks like a decode. Pass
the tracking intervals of the decoded period, which need not be those of the
fit.

Decode bins that reach more than half a sample outside the tracked period are
always missing, with one warning: position there could only be extrapolated.

**Decoding without tracking**, such as sleep after fitting on a run, takes a
model that does not read position: `predict(spike_times, time_edges=...)` with
`position=None`. A Local state needs position, so a model with one raises
`ValidationError` without it.

**Encoding support** is where spikes and occupancy count. By default it is the
samples' span plus half a sample at each end, N * dt seconds in all; a single
position sample needs `encoding_time_range` or an interval. `encoding_time_range`
replaces those outer bounds, and does not assert that tracking was continuous: inside the samples it clips the support, and past them it
extends it, holding the position at the first or last sample, which warns if
it passes the samples by over one sample interval. With
`valid_position_intervals`, support is bounded by the intervals; each needs a
finite position sample, and positions are interpolated within it and held at
its endpoint samples. NaN rows inside an interval split it.

Spikes, position and time edges are checked against each other: spike times far
from the position span or the decode edges (other units or another clock; a
spike list that puts under 1% of its spikes in the position span and reaches
over ten spans past it also warns, so pass one epoch's spikes rather than a
day's), a
position span a hundred times longer than the spikes (position in ms, spikes in
s), no spike inside the position span, and positions more than one place bin
outside a fitted grid each warn. A track graph projects every position onto its
track, so nothing warns about positions off it. See [Warnings](warnings.md).
