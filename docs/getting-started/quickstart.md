# Quickstart

This page fits a replay detector to simulated sorted spikes and finds the
simulated replay events. The code block is complete; it runs as written and
takes about 10 seconds on a laptop CPU.

## Detect simulated replay

`make_simulated_data` simulates a rat running back and forth on a 1-D track,
30 place cells, and five brief periods (`event_times`) in which the cells fire
in a fast sequence that sweeps across the track, the way replay does, while
the rat stays still.

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np

from non_local_detector import NonLocalSortedSpikesDetector, time_edges_from_centers
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

(
    speed,
    position,  # (n_time,) position in cm, sampled at 1,500 Hz
    spike_times,  # one array of spike times (s) per neuron
    time,  # (n_time,) sample times in s
    event_times,  # (n_events, 2) start and end of each simulated replay
    sampling_frequency,
    is_event,  # (n_time,) True inside a simulated replay
    place_fields,
) = make_simulated_data(n_neurons=30)

# Learn each neuron's place field from the run, outside the replay events.
detector = NonLocalSortedSpikesDetector().fit(
    time, position, spike_times, is_training=~is_event
)

# Decode in one time bin per position sample.
results = detector.predict(
    spike_times=spike_times,
    time_edges=time_edges_from_centers(time),
    position=position,
    position_time=time,
)

state_probability = results.acausal_state_probabilities  # (n_time, n_states)
non_local = state_probability.sel(
    states=["Non-Local Continuous", "Non-Local Fragmented"]
).sum("states")

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
axes[0].plot(time, position, color="black")
axes[0].set_ylabel("Position (cm)")
axes[1].plot(time, non_local, color="tab:red")
axes[1].set_ylabel("P(non-local)")
axes[1].set_xlabel("Time (s)")
for start, stop in event_times:
    for ax in axes:
        ax.axvspan(start, stop, color="gold", alpha=0.4)

for start, stop in event_times:
    in_event = (time >= start) & (time <= stop)
    print(
        f"{start:5.2f}-{stop:5.2f} s: P(non-local) = {non_local[in_event].mean():.2f}"
    )
```

The printout lists the mean non-local probability in each simulated event:

```text
 6.50- 6.77 s: P(non-local) = 0.96
19.50-19.58 s: P(non-local) = 0.87
32.50-32.77 s: P(non-local) = 0.96
45.50-46.00 s: P(non-local) = 0.15
58.50-58.50 s: P(non-local) = 0.06
```

The first three events contain spikes, and the detector assigns them to the
non-local states. The last two contain no spikes at all: the simulation pauses
the cells there, and the detector puts that time in the **No-Spike** state
instead, since silence says nothing about which position is represented.

The fifth event is zero-length (58.50-58.50 s): it falls within a single time
bin, so its mean is over one bin.

## What happened

1. **Environment.** The detector built a grid of 2 cm position bins over the
   positions seen during fitting. With 1-D position and no track graph, the
   track is treated as a 1-D open field (the detector warns about this); see
   [Core concepts](core-concepts.md#environments) for track graphs.
2. **Encoding.** `fit` estimated each neuron's firing rate as a function of
   position with a kernel density estimate (`sorted_spikes_kde`, the default),
   using only the samples where `is_training` is true.
3. **Decoding.** `predict` combined those rate maps with a hidden Markov model
   over four states (Local, No-Spike, Non-Local Continuous and Non-Local
   Fragmented) and position, and returned the smoothed (`acausal`) state
   probabilities and position posterior for every time bin.

## Next steps

- [Core concepts](core-concepts.md) explains each of these pieces and the
  options for each.
- [Time bins and rates](../time_grid_migration.md) explains `time_edges` and
  how spikes are assigned to bins.
