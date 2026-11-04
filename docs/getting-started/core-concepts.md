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
  that happen to be adjacent in the 1-D layout. `edge_id` can be any unique
  integer label for each edge. If you also call
  `track_linearization.get_linearized_position` yourself, number `edge_id` in
  the order NetworkX lists the edges (`graph.edges`), as
  `track_linearization.make_track_graph` does: that function assumes it.

1-D position with no track graph is decoded as a 1-D open field: the model
moves between neighboring bins of that line and knows nothing of the track's
topology. For a linear track or a maze, pass the raw 2-D `(x, y)` position with
a `track_graph`, `edge_order` and `edge_spacing` instead of position you
linearized yourself; with a track graph, 1-D position raises `ValidationError`.

Pass environments to a model with `environments=[Environment(...)]`. Models
with more than one environment, such as
`MultiEnvironmentSortedSpikesClassifier`, decode which environment the
activity represents. A model fits its own copies of the environments it is
given, so after `fit` the fitted grids are in `model.environments_`, while
`model.environments` keeps the environments you passed, unchanged, so they can
be shared between models.
`Environment.get_params()` returns an environment's configuration, from which
`Environment(**params)` rebuilds it.

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
  from the animal's own movement). Each is a `ContinuousTransition`: its
  `make_state_transition(environments, context=None)` returns the block of
  probabilities from each source bin to each destination bin, and `context`
  (a `TransitionContext`) holds the fit's position and labels. A custom
  transition with this method and an `environment_name` attribute (the
  environment of its source bins, or `None` if it has no place bins) can be
  used in the same place. The block it returns must have one row per bin of
  the source state and one column per bin of the destination state, be
  non-negative, and have rows that sum to at most 1 (to 1, or 0 for bins
  nothing can leave); the detectors raise `ValidationError` otherwise. The
  environments a transition names must be its states': `environment_name`
  is the source state's environment and `environment2_name` (as on
  `Uniform`), when set, the destination's, otherwise `environment_name` is
  both. The method is not called for a block between a one-bin state
  (No-Spike, or Local without `local_position_std`) and a spatial state:
  from the one-bin state the block is uniform over the interior of the
  destination environment (so `Discrete`, which names none, cannot be used
  there), and into it every bin moves there with probability 1. A custom
  transition has no structured operator unless it implements
  `transition_operators.SupportsStructuredBlock` and defines
  `structured_block` in its own class (a subclass of a package transition
  that does not is treated as custom), so
  `transition_representation="structured"` rejects it. `"auto"` builds its
  dense block, except when an environment's graph distances are deferred,
  as for an open-field grid fit with a non-dense representation; use
  `"dense"` there.
- **Discrete transitions** set how the states follow one another:
  `DiscreteStationaryDiagonal` and `DiscreteStationaryCustom` for fixed
  probabilities, and `DiscreteNonStationaryDiagonal` and
  `DiscreteNonStationaryCustom` for probabilities that depend on covariates
  such as speed. Diagonal values must be in [0, 1] and custom matrices
  row-stochastic, checked when the transition is made. Each is a
  `DiscreteTransition` (`is_stationary`, `n_states`, `smoothness` and
  `make_state_transition(covariate_data)`), so a class with these members
  can be used instead.

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
  (`(n_time, n_states)`; `(n_time,)` for single-state models such as
  decoders). Ask for more with `return_outputs`, e.g. `"filter"` for the
  causal (past-only) estimates or `"log_likelihood"`. Every dimension,
  coordinate, variable and attribute of the result is listed in
  [Result layout](../result_layout.md).

States without a position (Local and No-Spike) have no position bins, so
unstacking `acausal_posterior` yourself misplaces their probability. For
models that have them, use
[`get_spatial_posterior`](../api/non_local_detector/analysis/posterior.md) to get
per-position probabilities.
