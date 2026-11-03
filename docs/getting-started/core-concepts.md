# Core concepts

Every model in `non_local_detector` is a hidden Markov model whose hidden state
has two parts: a **discrete state** (for example Local or Non-Local Continuous)
and a **position** on a grid of spatial bins. A model is defined by five
choices, each covered below: the environment, the states, the likelihood model
of the spikes, the transitions, and the time bins.

## Inputs

| Argument | Shape | Meaning |
| --- | --- | --- |
| `position_time` | `(n_position_samples,)` | times of the position samples, in seconds |
| `position` | `(n_position_samples,)` or `(n_position_samples, n_dims)` | the animal's position, e.g. in cm |
| `spike_times` | list of `(n_spikes_i,)` arrays | sorted data: one array of spike times per neuron. Clusterless data: one array per electrode |
| `spike_waveform_features` | list of `(n_spikes_i, n_features)` arrays | clusterless data only: the waveform features (marks), such as peak amplitudes, of each spike on each electrode |

Spike times and position times are in seconds on the same clock.

## Environments

An `Environment` discretizes space into position bins.

- **Open field.** With `Environment(place_bin_size=2.0)`, `fit` builds a
  regular grid (1-D or 2-D, following the shape of `position`) over the
  positions it sees and, by default, marks the bins the animal visited as the
  track interior.
- **Track graph.** For mazes, describe the track as a
  [NetworkX](https://networkx.org) graph whose nodes have a `pos` attribute and
  whose edges have `distance` and `edge_id` attributes, and give the order in
  which the edges are laid out in 1-D (`edge_order`) and the gap between
  consecutive edges (`edge_spacing`). Position is then 2-D `(x, y)`; the model
  linearizes it with [track_linearization](https://github.com/LorenFrankLab/track_linearization)
  and moves only along the track, so a trajectory cannot jump between arms
  that happen to be adjacent in the 1-D layout.

Pass environments to a model with `environments=[Environment(...)]`. Models
with more than one environment, such as
`MultiEnvironmentSortedSpikesClassifier`, decode which environment the
activity represents.

## Models and their states

| Model | States |
| --- | --- |
| `SortedSpikesDecoder`, `ClusterlessDecoder` | Continuous |
| `ContFragSortedSpikesClassifier`, `ContFragClusterlessClassifier` | Continuous, Fragmented |
| `NoSpikeContFragSortedSpikesClassifier`, `NoSpikeContFragClusterlessClassifier` | No-Spike, Continuous, Fragmented |
| `NonLocalSortedSpikesDetector`, `NonLocalClusterlessDetector` | Local, No-Spike, Non-Local Continuous, Non-Local Fragmented |
| `MultiEnvironmentSortedSpikesClassifier`, `MultiEnvironmentClusterlessClassifier` | one state per environment |

- **Local**: the spikes represent the animal's current position.
- **Continuous** / **Non-Local Continuous**: the represented position follows
  a random walk from bin to bin.
- **Fragmented** / **Non-Local Fragmented**: the represented position can jump
  anywhere on the track from one bin to the next.
- **No-Spike**: the neurons are nearly silent; the firing model does not depend
  on position.

## Likelihood models

The likelihood model is the encoding model of how the spikes depend on
position. Choose it with `sorted_spikes_algorithm` or `clusterless_algorithm`,
and set its options with `sorted_spikes_algorithm_params` or
`clusterless_algorithm_params`.

| Data | Algorithm | Encoding model |
| --- | --- | --- |
| sorted | `sorted_spikes_kde` (default) | kernel density estimate of each neuron's place field |
| sorted | `sorted_spikes_glm` | Poisson GLM with a spline basis over position |
| sorted | `sorted_spikes_diffusion` | like `sorted_spikes_kde`, but smoothed with a heat kernel on the environment's graph, so smoothing respects walls, holes and junctions |
| sorted | `sorted_spikes_mrf` | penalized Poisson model on the smoothest eigenmodes of the environment graph's Laplacian |
| clusterless | `clusterless_kde` (default) | kernel density estimate over position and waveform features |
| clusterless | `clusterless_gmm` | Gaussian mixture over position and waveform features |
| clusterless | `clusterless_diffusion` | like `clusterless_kde`, with the graph heat kernel over position |
| clusterless | `clusterless_mark_glm` | mixture over waveform features with a Poisson GLM rate map per component |

Fitted firing rates are in Hz.

## Transitions

- **Continuous transitions** set how position moves within and between
  states: `RandomWalk` (Gaussian steps, `movement_var` per time bin),
  `Uniform` (jump anywhere), `Identity` (stay put), `Discrete` (the state has
  no position, as in Local and No-Spike) and `EmpiricalMovement` (learned
  from the animal's own movement).
- **Discrete transitions** set how the states follow one another:
  `DiscreteStationaryDiagonal` and `DiscreteStationaryCustom` for fixed
  probabilities, and `DiscreteNonStationaryDiagonal` and
  `DiscreteNonStationaryCustom` for probabilities that depend on covariates
  such as speed.

## Time bins

Decoding takes bin **edges**, `time_edges` with shape `(n_bins + 1,)`, and
returns one row per bin. Build them with `detector.calculate_time_edges([start, stop])`
(bins of `1 / sampling_frequency` seconds) or with
`time_edges_from_centers(position_time)` (one bin per position sample). Detector
bins must be uniform. See [Time bins and rates](../time_grid_migration.md) for
the details.

## Fit, estimate and predict

- `fit(position_time, position, spike_times, ...)` fits the encoding model;
  `is_training` selects the samples to fit it on, for example running periods
  only.
- `estimate_parameters(...)` fits the encoding model and then re-estimates the
  discrete transitions (and, optionally, the encoding model) by
  expectation-maximization, before decoding.
- `predict(spike_times, time_edges=..., ...)` decodes. It returns an
  [xarray](https://docs.xarray.dev) `Dataset` with
  `acausal_posterior` (`(n_time, n_state_bins)`: the probability of each state
  and position given all the spikes) and `acausal_state_probabilities`
  (`(n_time, n_states)`). Ask for more with `return_outputs`, e.g.
  `"filter"` for the causal (past-only) estimates or `"log_likelihood"`.
