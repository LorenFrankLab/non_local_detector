# non_local_detector

`non_local_detector` decodes an animal's represented position from neural
spiking and detects when that representation departs from where the animal
is: hippocampal replay and other **non-local** activity. It works with sorted
spikes (one spike train per neuron) and with **clusterless** data (spike
waveform features per electrode, no spike sorting), and runs on CPU or GPU
through [JAX](https://docs.jax.dev).

The models are hidden Markov models (state-space models) over position and a
small set of discrete states. A detector, for example, distinguishes:

- **Local**: the spikes represent the animal's current position.
- **Non-Local Continuous**: the represented position follows a random walk
  away from the animal, as in a replayed trajectory.
- **Non-Local Fragmented**: the represented position jumps anywhere on the
  track from one time bin to the next.
- **No-Spike**: the neurons are nearly silent, so the spikes say nothing about
  position.

Fitting estimates an encoding model of how each neuron (or electrode) fires as
a function of position; prediction returns, for every time bin, the
probability of each state and the posterior over position.

## Where to start

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md): detect simulated replay events
  in about 20 lines.
- [Core concepts](getting-started/core-concepts.md): environments, states,
  likelihood models, transitions and time bins.
- [API reference](api/non_local_detector/index.md)

## Citing

If you use `non_local_detector`, please cite the software (see
[CITATION.cff](https://github.com/LorenFrankLab/non_local_detector/blob/main/CITATION.cff))
and the paper that describes the method:

> Denovellis, E. L., Gillespie, A. K., Coulter, M. E., Sosa, M., Chung, J. E.,
> Eden, U. T., & Frank, L. M. (2021). Hippocampal replay of experience at
> real-world speeds. *eLife*, 10, e64505.
> [doi:10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505)
