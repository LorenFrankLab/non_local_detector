# %% [markdown]
# # Classifying replay dynamics
#
# During rest and sharp-wave ripples, hippocampal cells fire in sequences that
# represent positions away from the animal: replay. The represented position
# does not always move the same way. It can sweep smoothly along the track
# (**continuous**), jump between distant places (**fragmented**), or stay in
# one place (**hover**), and a single event can mix these.
#
# A decoder with one movement model cannot tell these apart. A **classifier**
# has several discrete states, each with its own model of how the represented
# position moves from one time bin to the next, and the hidden Markov model
# infers at each time which state, and which position, best explains the
# spikes. The approach follows
# [Denovellis et al. (2021)](https://doi.org/10.7554/eLife.64505).
#
# In this tutorial we:
#
# 1. simulate a session of clusterless data: running on a track, then a rest
#    period containing six simulated replay events of different types,
# 2. fit a `NoSpikeContFragClusterlessClassifier` on the running periods,
# 3. look at its movement models (continuous and discrete transitions),
# 4. decode the **whole session**, and
# 5. look at how each replay event is classified.
#
# It builds on the [clusterless decoding tutorial](../03_decoding_clusterless/);
# see it for clusterless data and the encoding model.

# %%
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np

from non_local_detector import (
    Discrete,
    Environment,
    NoSpikeContFragClusterlessClassifier,
    RandomWalk,
    Uniform,
    time_edges_from_centers,
)
from non_local_detector.simulate.clusterless_simulation import (
    make_continuous_replay,
    make_fragmented_continuous_fragmented_replay,
    make_fragmented_hover_fragmented_replay,
    make_fragmented_replay,
    make_hover_continuous_hover_replay,
    make_hover_replay,
    make_simulated_run_data,
)

# On CPU, JAX cannot reuse the memory of arrays the decoder hands back to it
# ("donated buffers") and warns each time; the results are unaffected.
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

STATE_COLORS = {
    "No-Spike": "#7f7f7f",
    "Continuous": "#521b65",
    "Fragmented": "#ff6944",
}

# %% [markdown]
# ## Simulate a session
#
# ### Running
#
# As in the previous tutorial, `make_simulated_run_data` simulates about six
# minutes of a rat running back and forth on a 175 cm track, sampled at
# 1,000 Hz, with 20 place cells on five tetrodes (four marks per spike).

# %%
run = make_simulated_run_data(seed=0)

sampling_frequency = 1000  # Hz, the simulator's default
run_time = run.position_time  # (n_run_time,) s
run_position = run.position[:, 0]  # (n_run_time,) cm
print(f"Running period: {run_time[0]:.0f}-{run_time[-1]:.0f} s")

# %% [markdown]
# ### Replay events
#
# The replay simulators each return `(replay_time, multiunits)`, where
# `multiunits` has shape `(n_time, n_features, n_tetrodes)` at 1,000 Hz: a
# spike's marks where a tetrode spiked, NaN elsewhere. In the events, each
# cell fires at the mark values it has during running, so the encoding model
# can recognize it:
#
# - `make_continuous_replay`: each cell fires once, in the order of its place
#   field, as if the represented position swept from one end of the track to
#   the other.
# - `make_hover_replay`: one cell fires repeatedly, so the represented
#   position stays at that cell's place field.
# - `make_fragmented_replay`: five spikes from cells with place fields far
#   apart, in quick succession.
#
# The replay simulators default to the same cells and marks as
# `make_simulated_run_data`, and `make_hover_continuous_hover_replay`,
# `make_fragmented_hover_fragmented_replay` and
# `make_fragmented_continuous_fragmented_replay` build the three mixed events
# from these.

# %%
_, continuous = make_continuous_replay()
_, hover = make_hover_replay()  # the middle cell
_, fragmented = make_fragmented_replay()
_, hover_continuous_hover = make_hover_continuous_hover_replay()
_, fragmented_hover_fragmented = make_fragmented_hover_fragmented_replay()
_, fragmented_continuous_fragmented = make_fragmented_continuous_fragmented_replay()

replay_events = {
    "Continuous": continuous,
    "Hover": hover,
    "Fragmented": fragmented,
    "Hover-continuous-hover": hover_continuous_hover,
    "Fragmented-hover-fragmented": fragmented_hover_fragmented,
    "Fragmented-continuous-fragmented": fragmented_continuous_fragmented,
}
for name, multiunits in replay_events.items():
    n_spikes = np.any(~np.isnan(multiunits), axis=1).sum()
    print(f"{name:34s} {multiunits.shape[0]:4d} ms, {n_spikes:3d} spikes")

continuous_speed = 175 / (continuous.shape[0] / sampling_frequency)  # cm/s
print(f"The continuous replay crosses the track at ~{continuous_speed / 100:.0f} m/s")

# %% [markdown]
# ### Rest with replay
#
# After running, the animal rests for 8 s at the end of the track, and we place
# one replay event every second. To keep the events easy to see, the cells are
# silent during rest outside the events. (In real data, cells also fire for
# the animal's current position during rest; telling that local activity
# apart from replay is the job of the `NonLocalClusterlessDetector`.)

# %%
rest_duration = 8.0  # s
rest_time = run_time[-1] + np.arange(1, rest_duration * sampling_frequency + 1) / (
    sampling_frequency
)
time = np.concatenate([run_time, rest_time])  # (n_time,) s
position = np.concatenate([run_position, np.full(rest_time.size, run_position[-1])])

spike_times = [list(times) for times in run.spike_times]
spike_waveform_features = [list(marks) for marks in run.spike_waveform_features]
event_times = {}  # name -> (start, stop) in s
for event_ind, (name, multiunits) in enumerate(replay_events.items()):
    start = rest_time[(event_ind + 1) * sampling_frequency]
    event_times[name] = (start, start + multiunits.shape[0] / sampling_frequency)
    sample_ind, tetrode_ind = np.nonzero(np.any(~np.isnan(multiunits), axis=1))
    for sample, tetrode in zip(sample_ind, tetrode_ind, strict=True):
        spike_times[tetrode].append(start + sample / sampling_frequency)
        spike_waveform_features[tetrode].append(multiunits[sample, :, tetrode])

spike_times = [np.asarray(times) for times in spike_times]
spike_waveform_features = [np.asarray(marks) for marks in spike_waveform_features]
print(f"Session: {time[0]:.0f}-{time[-1]:.0f} s")
for name, (start, stop) in event_times.items():
    print(f"  {name:34s} {start:.3f}-{stop:.3f} s")

# %% [markdown]
# The whole session: position (top) and every spike on every tetrode, with
# the rest period shaded. The bottom panel zooms in on the rest period; each
# short burst of spikes is one replay event.

# %%
n_tetrodes = len(spike_times)
fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
axes[0].plot(time, position, color="black", linewidth=1)
axes[0].set_ylabel("Position (cm)")
axes[0].set_xlabel("Time (s)")
for tetrode, times in enumerate(spike_times):
    axes[1].scatter(
        times, np.full_like(times, tetrode), marker="|", s=20, color="black"
    )
    axes[2].scatter(
        times, np.full_like(times, tetrode), marker="|", s=200, color="black"
    )
axes[1].set_xlabel("Time (s)")
axes[2].set_xlim(rest_time[0], rest_time[-1])
axes[2].set_xlabel("Time (s)")
for ax in axes[:2]:
    ax.axvspan(rest_time[0], rest_time[-1], color="gold", alpha=0.4, label="rest")
    ax.set_xlim(time[0], time[-1])
axes[0].legend(loc="upper left")
for ax in axes[1:]:
    ax.set_ylabel("Tetrode")
    ax.set_yticks(range(n_tetrodes))
for name, (start, _) in event_times.items():
    axes[2].annotate(
        name, (start, n_tetrodes - 0.4), rotation=20, fontsize=8, ha="left"
    )
axes[2].set_ylim(-0.5, n_tetrodes + 0.8)
axes[2].set_title("Rest period")

# %% [markdown]
# ## The classifier
#
# `NoSpikeContFragClusterlessClassifier` has three discrete states:
#
# - **Continuous**: the represented position moves by a Gaussian random walk
#   (`RandomWalk`).
# - **Fragmented**: the represented position can jump anywhere on the track
#   from one bin to the next (`Uniform`).
# - **No-Spike**: the cells are nearly silent; the state has no position.
#
# We use it rather than `ContFragClusterlessClassifier` (Continuous and
# Fragmented only) because we decode the whole session, and most of the rest
# period is silence. Silence says nothing about how the represented position
# moves, so without a No-Spike state the model would split those stretches
# between Continuous and Fragmented according to its prior, with a flat
# posterior over position. With it, silence goes to No-Spike.
#
# There is no separate "stationary" state. A hovering position is a random
# walk that happens not to move, so hover events fall in the Continuous state
# (see the gallery below). To tell hovering apart from movement, add a state
# whose position transition is `Identity` with the `continuous_transition_types`
# argument.
#
# ### Movement models
#
# The model works on 2 ms time bins (its default `sampling_frequency` of
# 500 Hz), and transition probabilities are per bin. The default random walk
# has `movement_var=6.0` cm² per bin, a standard deviation of about 2.4 cm per
# 2 ms. The simulated continuous replay moves faster than that, so we widen
# the random walk to `movement_var=32.0` cm² per bin (about 5.7 cm per 2 ms).
# (With the default, much of the continuous event below is classified as
# Fragmented: the random walk cannot keep up.)
# The other transitions are the class defaults: `Discrete` to and from
# No-Spike (which has no position) and `Uniform` into Fragmented and between
# the two spatial states.

# %%
movement_var = 32.0  # cm² per 2 ms bin
continuous_transition_types = [
    # to:  No-Spike    Continuous                           Fragmented
    [Discrete(), Uniform(), Uniform()],  # from No-Spike
    [Discrete(), RandomWalk(movement_var=movement_var), Uniform()],  # from Continuous
    [Discrete(), Uniform(), Uniform()],  # from Fragmented
]

classifier = NoSpikeContFragClusterlessClassifier(
    continuous_transition_types=continuous_transition_types,
    environments=Environment(place_bin_size=2.0),
    clusterless_algorithm="clusterless_kde",
    clusterless_algorithm_params={
        "position_std": np.sqrt(12.5),  # cm
        "waveform_std": 1.0,  # mark units, see the previous tutorial
    },
)
print("States:", classifier.state_names)
print(f"Decoding bin width: {1000 / classifier.sampling_frequency:.0f} ms")

# %% [markdown]
# ## Fit on the running periods
#
# The encoding model is fit only where the animal runs faster than 4 cm/s, so
# the rest period, with its replay events, is not used for encoding. The
# decoder warns that 1-D position without a track graph is treated as a 1-D
# open field, which is fine for a straight track.

# %%
speed = np.abs(np.gradient(position, time))  # cm/s
is_running = speed > 4.0

classifier.fit(
    time,
    position[:, np.newaxis],
    spike_times,
    spike_waveform_features,
    is_training=is_running,
)

# %% [markdown]
# ### Continuous transitions
#
# After fitting, `continuous_state_transitions_` holds the probability of
# moving from each position bin (row) to each position bin (column), for each
# pair of states. Here are the four blocks between the two spatial states. A
# narrow diagonal band (Continuous to Continuous) means the position moves
# only a few bins per time bin; a uniform block means it can land anywhere.
# Each row sums to 1.

# %%
state_ind = classifier.state_ind_  # (n_state_bins,) state of each bin
transitions = classifier.continuous_state_transitions_
bin_edges = classifier.environments[0].place_bin_edges_[:, 0]  # cm
spatial_states = ["Continuous", "Fragmented"]

fig, axes = plt.subplots(
    2, 2, figsize=(7, 6.5), sharex=True, sharey=True, constrained_layout=True
)
for row, from_state in enumerate(spatial_states):
    for col, to_state in enumerate(spatial_states):
        from_bins = state_ind == classifier.state_names.index(from_state)
        to_bins = state_ind == classifier.state_names.index(to_state)
        block = transitions[np.ix_(from_bins, to_bins)]
        image = axes[row, col].pcolormesh(
            bin_edges, bin_edges, block, cmap="Blues", vmin=0.0, vmax=0.1
        )
        axes[row, col].set_title(f"{from_state} → {to_state}")
        axes[row, col].set_aspect("equal")
for ax in axes[-1]:
    ax.set_xlabel("Position at time t (cm)")
for ax in axes[:, 0]:
    ax.set_ylabel("Position at time t-1 (cm)")
fig.colorbar(image, ax=axes, label="Probability", extend="max")

# %% [markdown]
# The No-Spike state has a single bin and no position. Leaving it, the
# position starts uniformly over the track:

# %%
no_spike_bins = state_ind == classifier.state_names.index("No-Spike")
continuous_bins = state_ind == classifier.state_names.index("Continuous")
no_spike_to_continuous = transitions[np.ix_(no_spike_bins, continuous_bins)]
print(
    "No-Spike -> Continuous is uniform over the "
    f"{continuous_bins.sum()} position bins:",
    np.allclose(no_spike_to_continuous, 1 / continuous_bins.sum()),
)

# %% [markdown]
# ### Discrete transitions
#
# The discrete transition matrix sets how likely the model is to stay in a
# state from one bin to the next (the diagonal) or switch (off-diagonal).
# A self-transition probability `p` corresponds to an expected stay of
# `bin width / (1 - p)`. We keep the defaults.

# %%
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
classifier.plot_discrete_state_transition(ax=ax)

bin_width_ms = 1000 / classifier.sampling_frequency
stay_probability = np.diag(classifier.discrete_state_transitions_)
for name, p in zip(classifier.state_names, stay_probability, strict=True):
    print(f"Expected stay in {name}: {bin_width_ms / (1 - p):.0f} ms")

# %% [markdown]
# ## Decode the whole session
#
# We decode every 2 ms bin from the start to the end of the recording,
# running and rest alike. `calculate_time_edges` builds bins of
# `1 / sampling_frequency` over a time range; here the range spanned by the
# 1 ms position samples, with `trim=True` to drop a partial final bin. These
# classifiers do not use position at decoding time, so `predict` needs only
# the spikes and their marks. For long sessions, see
# [Checkpointed prediction](../../performance_prediction/) and
# [Settings for your hardware](../../hardware_settings/).

# %%
session_range = time_edges_from_centers(time)[[0, -1]]  # s
time_edges = classifier.calculate_time_edges(session_range, trim=True)
results = classifier.predict(
    spike_times, spike_waveform_features, time_edges=time_edges
)
print(results)

# %% [markdown]
# `acausal_state_probabilities` holds the probability of each state in each
# time bin. Over the whole session:

# %%
state_probabilities = results.acausal_state_probabilities
decoded_time = results.time.to_numpy()

is_event = np.zeros(decoded_time.shape, dtype=bool)
for start, stop in event_times.values():
    is_event |= (decoded_time >= start) & (decoded_time < stop)
is_rest = decoded_time > run_time[-1]
periods = {
    "Running period": ~is_rest,
    "Rest, outside events": is_rest & ~is_event,
    "Rest, inside events": is_event,
}
for period, in_period in periods.items():
    mean_probability = state_probabilities[in_period].mean("time")
    summary = ", ".join(
        f"{state} {mean_probability.sel(states=state).item():.2f}"
        for state in classifier.state_names
    )
    print(f"{period:22s} {summary}")

all_spike_times = np.sort(np.concatenate(spike_times))
windows = {
    "Whole session": (time[0], time[-1]),
    "Running, 100-103 s": (100.0, 103.0),
    "Rest period": (rest_time[0], rest_time[-1]),
}
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), constrained_layout=True)
for ax, (title, (start, stop)) in zip(axes, windows.items(), strict=True):
    in_window = (decoded_time >= start) & (decoded_time <= stop)
    ax.stackplot(
        decoded_time[in_window],
        state_probabilities[in_window].to_numpy().T,
        colors=[STATE_COLORS[state] for state in classifier.state_names],
        labels=classifier.state_names,
    )
    ax.set_xlim(start, stop)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
# Mark the spikes above the probabilities in the zoom on running.
in_zoom = (all_spike_times >= 100.0) & (all_spike_times <= 103.0)
axes[1].eventplot(
    all_spike_times[in_zoom], lineoffsets=1.09, linelengths=0.1, colors="black"
)
axes[1].set_ylim(0, 1.15)
axes[1].set_yticks([0, 0.5, 1])
axes[1].text(103.02, 1.09, "spikes", va="center")
axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

# %% [markdown]
# During rest, silence goes to No-Spike, and each replay event switches the
# model into Continuous or Fragmented. The model has no state for "local"
# activity: while the animal runs, Continuous tracks its current position.
#
# While running, the model is mostly in Continuous, but the zoom shows the
# Fragmented probability rising in the gaps between spikes. With no spikes,
# nothing constrains how the position moves, and the discrete transitions
# favor Fragmented: its expected stay, printed above, is much longer than
# Continuous's. Each spike that fits a smooth path pulls the model
# back. The Fragmented probability while running grows with the time to the
# nearest spike:

# %%
nearest_spike = np.searchsorted(all_spike_times, decoded_time)
time_to_spike = np.minimum(
    np.abs(decoded_time - all_spike_times[np.clip(nearest_spike - 1, 0, None)]),
    np.abs(
        all_spike_times[np.clip(nearest_spike, None, all_spike_times.size - 1)]
        - decoded_time
    ),
)  # s
fragmented_probability = state_probabilities.sel(states="Fragmented").to_numpy()
gap_edges_ms = [0, 5, 10, 20, 50]
for lower, upper in itertools.pairwise(gap_edges_ms):
    in_gap = ~is_rest & (time_to_spike >= lower / 1000) & (time_to_spike < upper / 1000)
    print(
        f"{lower:2d}-{upper:2d} ms from the nearest spike: "
        f"P(Fragmented) = {fragmented_probability[in_gap].mean():.2f} "
        f"({in_gap.sum() / (~is_rest).sum():.1%} of running-period bins)"
    )

# %% [markdown]
# This is the prior at work, not evidence for fragmented replay during
# running. If it matters for your analysis, set the discrete transitions with
# `discrete_transition_type`, or learn them from the data with
# `estimate_parameters`.
#
# ## Replay gallery
#
# For each event we plot, around its time:
#
# 1. the spikes on each tetrode, colored by the amplitude on channel 1, which
#    tells the cells of a tetrode apart,
# 2. the probability of each state, and
# 3. the posterior probability of the represented position, summed over the
#    Continuous and Fragmented states.
#
# `acausal_posterior` is indexed by `state_bins`, a (state, position) index.
# We drop the No-Spike bins, which have no position, before unstacking it
# into a `(time, state, position)` array.

# %%
spatial_posterior = (
    results.acausal_posterior.where(results.state != "No-Spike", drop=True)
    .unstack("state_bins")
    .sum("state")
)  # (n_time, n_position_bins)
place_bin_centers = spatial_posterior.position.to_numpy()
bin_size = np.diff(place_bin_centers).mean()


def plot_replay_event(name, padding=0.02):
    """Plot spikes, state probabilities and posterior around one replay event."""
    start, stop = event_times[name]
    window = slice(start - padding, stop + padding)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 6.5),
        sharex=True,
        constrained_layout=True,
        height_ratios=[1, 1, 1.5],
    )
    for tetrode, (times, marks) in enumerate(
        zip(spike_times, spike_waveform_features, strict=True)
    ):
        in_window = (times >= window.start) & (times <= window.stop)
        spikes = axes[0].scatter(
            1000 * (times[in_window] - start),
            np.full(in_window.sum(), tetrode),
            c=marks[in_window, 0],
            cmap="viridis",
            vmin=0,
            vmax=30,
            marker="|",
            s=150,
            linewidths=2,
        )
    axes[0].set_yticks(range(n_tetrodes))
    axes[0].set_ylim(-0.5, n_tetrodes - 0.5)
    axes[0].set_ylabel("Tetrode")
    fig.colorbar(spikes, ax=axes[0], label="Amplitude,\nchannel 1")

    probabilities = state_probabilities.sel(time=window)
    event_ms = 1000 * (probabilities.time.to_numpy() - start)
    for state in classifier.state_names:
        axes[1].plot(
            event_ms,
            probabilities.sel(states=state),
            color=STATE_COLORS[state],
            linewidth=2,
            label=state,
        )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    posterior = spatial_posterior.sel(time=window)
    image = axes[2].imshow(
        posterior.to_numpy().T,
        origin="lower",
        aspect="auto",
        cmap="bone_r",
        vmin=0.0,
        vmax=0.3,
        interpolation="none",
        extent=(
            event_ms[0] - bin_width_ms / 2,
            event_ms[-1] + bin_width_ms / 2,
            place_bin_centers[0] - bin_size / 2,
            place_bin_centers[-1] + bin_size / 2,
        ),
    )
    axes[2].set_ylabel("Position (cm)")
    axes[2].set_xlabel("Time from event start (ms)")
    fig.colorbar(image, ax=axes[2], label="Posterior\nprobability", extend="max")
    for ax in axes:
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(1000 * (stop - start), color="gray", linestyle=":", linewidth=1)
    fig.suptitle(name)

    in_event = (decoded_time >= start) & (decoded_time < stop)
    mean_probability = state_probabilities[in_event].mean("time")
    print(
        f"{name}, mean state probability during the event: "
        + ", ".join(
            f"{state} {mean_probability.sel(states=state).item():.2f}"
            for state in classifier.state_names
        )
    )


# %% [markdown]
# Dotted lines mark the start and end of each event; the 20 ms before and
# after are silent rest.
#
# ### Continuous
#
# The cells fire in the order of their place fields, so the represented
# position sweeps from one end of the track to the other. Expect the
# Continuous state, with a posterior that moves smoothly.

# %%
plot_replay_event("Continuous")

# %% [markdown]
# ### Hover
#
# A single cell fires every 2 ms, so the represented position stays at its
# place field, in the middle of the track. Staying put is a random walk with
# small steps, so the model calls this Continuous, not Fragmented: under the
# random walk the next position is very likely near the current one, while
# under the uniform transition each spike would have to land back on the same
# place by chance.

# %%
plot_replay_event("Hover")

# %% [markdown]
# ### Fragmented
#
# Five spikes from cells with fields far apart, 2 ms apart. The random walk
# cannot move that far that fast, so expect the Fragmented state, with a
# posterior that jumps.

# %%
plot_replay_event("Fragmented")

# %% [markdown]
# ### Hover, continuous, hover
#
# The position hovers at one end of the track, sweeps to the other end, and
# hovers there. All three parts are Continuous.

# %%
plot_replay_event("Hover-continuous-hover")

# %% [markdown]
# ### Fragmented, hover, fragmented
#
# Fragmented jumps, then a hover, then jumps again. Expect the state to switch
# from Fragmented to Continuous during the hover and back.

# %%
plot_replay_event("Fragmented-hover-fragmented")

# %% [markdown]
# ### Fragmented, continuous, fragmented
#
# Jumps, a smooth sweep, then jumps again.

# %%
plot_replay_event("Fragmented-continuous-fragmented")

# %% [markdown]
# ## Summary
#
# - A classifier has several discrete states with different movement models;
#   `continuous_transition_types` sets how position moves within and between
#   states and the discrete transition matrix sets how often the states switch.
#   Both are per time bin, so they depend on the bin width.
# - `NoSpikeContFragClusterlessClassifier` has No-Spike, Continuous and
#   Fragmented states. The No-Spike state absorbs silence, which matters when
#   decoding a whole session.
# - Continuous includes hovering: there is no separate stationary state unless
#   you add one with an `Identity` transition.
# - Fit on running, then decode the whole session and read off the state
#   probabilities in each event.
