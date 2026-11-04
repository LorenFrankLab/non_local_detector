# %% [markdown]
# # Detecting non-local events across a session
#
# The main use of `non_local_detector` is to find the moments in a recording
# when the hippocampus represents somewhere other than where the animal is,
# such as replay during rest. This tutorial runs that analysis on a whole
# simulated session:
#
# 1. simulate a session of running laps and rest, with replay events during
#    rest;
# 2. fit the detector's place fields on the running periods;
# 3. optionally refine the state transitions with expectation-maximization;
# 4. decode the **entire session**;
# 5. extract non-local events, compare them with the simulated ones, and
#    describe each event: its duration, the trajectory it represents and how
#    far that is from the animal.
#
# The analysis steps (2 to 5) do not depend on the simulation: with real data,
# replace the simulated arrays with your own position and spike times.

# %%
import tempfile
import time as timer
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from non_local_detector import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.analysis import maximum_a_posteriori_estimate

rng = np.random.default_rng(0)

# %% [markdown]
# ## Simulate a session
#
# The simulated rat runs 14 laps on a 180 cm linear track and rests for 8 to
# 16 s at each end. Forty place cells have Gaussian place fields (standard
# deviations 6 to 12 cm, peak rates 10 to 30 Hz) and fire at 0.1 Hz outside
# their fields.
#
# During each rest the cells also fire one to three **replay events**. In a
# replay event the cells fire at up to 80 Hz for a position that is not the
# rat's:
#
# - **continuous** events (75 %) sweep along the track at 4 to 8 m/s, from a
#   random start to a point at least 60 cm away;
# - **fragmented** events (25 %) last 100 to 200 ms and represent a new
#   random position in every 2 ms time step, with no spatial continuity.
#
# The simulation runs on a 2 ms grid. Within each 2 ms bin the spike counts
# are Poisson and the spike times are uniform, so spike times are not tied to
# the grid. Like a camera, the position is sampled at 30 Hz.

# %%
dt = 0.002  # simulation time step, s
track_length = 180.0  # cm
n_laps = 14


def run_between(start: float, stop: float, peak_speed: float = 50.0) -> np.ndarray:
    """Positions (cm) every `dt` of a run from `start` to `stop`.

    Speed ramps up and down over 0.4 s and is `peak_speed` (cm/s) in between.
    """
    ramp = 0.4  # s
    duration = abs(stop - start) / peak_speed + ramp
    t = np.arange(0.0, duration, dt)
    speed_profile = np.minimum(1.0, np.minimum(t, duration - t) / ramp)
    fraction = np.cumsum(speed_profile)
    fraction = (fraction - fraction[0]) / (fraction[-1] - fraction[0])
    return start + (stop - start) * fraction


# Alternate rests and laps: rest at 0 cm, run to 180 cm, rest, run back, ...
track_ends = [0.0, track_length]
rest_durations = rng.uniform(8.0, 16.0, size=n_laps + 1)
pieces, is_rest_pieces = [], []
for lap in range(n_laps + 1):
    here = track_ends[lap % 2]
    pieces.append(np.full(round(rest_durations[lap] / dt), here))
    is_rest_pieces.append(np.ones(pieces[-1].size, dtype=bool))
    if lap < n_laps:
        pieces.append(run_between(here, track_ends[(lap + 1) % 2]))
        is_rest_pieces.append(np.zeros(pieces[-1].size, dtype=bool))

# Keep a whole number of 0.1 s so that the 2 ms grid and the 30 Hz camera
# end at the same time.
n_bins = np.concatenate(pieces).size // 50 * 50
true_position = np.concatenate(pieces)[:n_bins]  # (n_bins,) cm, one per 2 ms
is_rest = np.concatenate(is_rest_pieces)[:n_bins]
bin_start = np.arange(n_bins) * dt  # s
session_duration = n_bins * dt

# %% [markdown]
# Next, the replay events. `represented_position` holds the position the
# spikes represent in each 2 ms bin: the rat's own position, except during
# replay.

# %%
# Pick each event's onset: one to three events per rest, in separate slots
# at least 1 s from the rest's edges. Then make three in four continuous.
onsets = []
rest_edges = np.flatnonzero(np.diff(np.concatenate(([0], is_rest, [0]))))
for rest_start, rest_stop in rest_edges.reshape(-1, 2):
    n_events = rng.integers(1, 4)
    slots = np.linspace(rest_start * dt + 1.0, rest_stop * dt - 1.0, n_events + 1)
    onsets.extend(rng.uniform(slots[:-1], slots[1:] - 0.5))
n_continuous = round(0.75 * len(onsets))
event_types = rng.permutation(
    ["continuous"] * n_continuous + ["fragmented"] * (len(onsets) - n_continuous)
)

represented_position = true_position.copy()
in_replay = np.zeros(n_bins, dtype=bool)
simulated_events = []
for onset, event_type in zip(onsets, event_types, strict=True):
    if event_type == "continuous":
        from_position = rng.uniform(0.0, track_length)
        to_position = rng.uniform(0.0, track_length)
        while abs(to_position - from_position) < 60.0:
            to_position = rng.uniform(0.0, track_length)
        replay_speed = rng.uniform(400.0, 800.0)  # cm/s
        n_event_bins = round(abs(to_position - from_position) / replay_speed / dt)
        trajectory = np.linspace(from_position, to_position, n_event_bins)
    else:
        n_event_bins = round(rng.uniform(0.1, 0.2) / dt)
        trajectory = rng.uniform(0.0, track_length, size=n_event_bins)
    first_bin = round(onset / dt)
    event_bins = slice(first_bin, first_bin + n_event_bins)
    represented_position[event_bins] = trajectory
    in_replay[event_bins] = True
    simulated_events.append(
        {
            "start_time": bin_start[first_bin],
            "end_time": bin_start[first_bin] + n_event_bins * dt,
            "type": event_type,
        }
    )

simulated_events = pd.DataFrame(simulated_events)
print(f"Session: {session_duration:.1f} s, {is_rest.mean():.0%} of it at rest")
print(simulated_events["type"].value_counts().to_string())

# %% [markdown]
# Finally, the spikes. Each cell's rate depends on the represented position;
# during replay the peak rate rises to 80 Hz. Each spike also gets a
# four-channel waveform amplitude for the clusterless example at the end: the
# cells are recorded in groups of four on ten tetrodes, and each cell has its
# own mean amplitude on each channel.

# %%
n_cells = 40
place_field_centers = np.sort(rng.uniform(0.0, track_length, n_cells))  # cm
place_field_widths = rng.uniform(6.0, 12.0, n_cells)  # Gaussian SD, cm
peak_rates = rng.uniform(10.0, 30.0, n_cells)  # Hz
out_of_field_rate = 0.1  # Hz


def place_field_rates(position: np.ndarray, peak_rate: np.ndarray) -> np.ndarray:
    """Firing rate (Hz) of every cell at each position, (n_positions, n_cells)."""
    distance = (position[:, np.newaxis] - place_field_centers) / place_field_widths
    return out_of_field_rate + peak_rate * np.exp(-0.5 * distance**2)


rates = place_field_rates(represented_position, peak_rates)
rates[in_replay] = place_field_rates(
    represented_position[in_replay], np.full(n_cells, 80.0)
)
spike_counts = rng.poisson(rates * dt)  # (n_bins, n_cells)

spike_times = []  # one array of spike times (s) per cell
for cell in range(n_cells):
    bins_with_spikes = np.repeat(np.arange(n_bins), spike_counts[:, cell])
    spike_times.append(
        np.sort(bin_start[bins_with_spikes] + rng.uniform(0, dt, bins_with_spikes.size))
    )

# Waveform features: 4 channels per tetrode, mean amplitude 40-300 uV per cell,
# with 10 uV of noise on each spike.
n_tetrodes = 10
cell_tetrode = np.arange(n_cells) // (n_cells // n_tetrodes)
mean_amplitudes = rng.uniform(40.0, 300.0, size=(n_cells, 4))
tetrode_spike_times, tetrode_features = [], []
for tetrode in range(n_tetrodes):
    cells = np.flatnonzero(cell_tetrode == tetrode)
    times = np.concatenate([spike_times[cell] for cell in cells])
    features = np.concatenate(
        [
            mean_amplitudes[cell] + rng.normal(0.0, 10.0, (spike_times[cell].size, 4))
            for cell in cells
        ]
    )
    order = np.argsort(times)
    tetrode_spike_times.append(times[order])
    tetrode_features.append(features[order])

print(f"{sum(s.size for s in spike_times):,} spikes from {n_cells} cells")

# %% [markdown]
# The camera samples the position at 30 Hz. From here on, the analysis uses
# only `position_time`, `position` and the spike times, as it would with real
# data.

# %%
position_sampling_frequency = 30.0  # Hz
position_time = np.arange(round(session_duration * position_sampling_frequency) + 1) / (
    position_sampling_frequency
)
position = np.interp(position_time, bin_start + dt / 2, true_position)  # cm

fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
ax.plot(position_time, position, color="black", linewidth=1)
for start, stop in simulated_events[["start_time", "end_time"]].to_numpy():
    ax.axvspan(start, stop, color="tab:red", alpha=0.6, linewidth=2)
ax.set(
    xlabel="Time (s)",
    ylabel="Position (cm)",
    title="Simulated session (red: replay events)",
    xlim=(0, session_duration),
)

# %% [markdown]
# A closer look at one rest shows what the detector has to find: the spikes
# of the cells sorted by place-field center. Outside the event, the cells
# whose fields cover the rat's position fire, plus a few scattered
# out-of-field spikes. During the event, a sequence sweeps across the track.

# %%
example = simulated_events.query("type == 'continuous'").iloc[1]
window = (example.start_time - 1.0, example.end_time + 1.0)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for cell, times in enumerate(spike_times):
    in_window = times[(times >= window[0]) & (times <= window[1])]
    ax.vlines(in_window, cell + 0.6, cell + 1.4, color="black", linewidth=1)
ax.axvspan(example.start_time, example.end_time, color="tab:red", alpha=0.3)
ax.set(
    xlabel="Time (s)",
    ylabel="Cell (sorted by place-field center)",
    xlim=window,
    title=f"Rat resting at {np.interp(example.start_time, position_time, position):.0f} cm",
)

# %% [markdown]
# ## Fit the detector on the running periods
#
# The place fields are estimated from the running periods only: when the rat
# runs, the cells represent where it is. During rest the represented position
# is what we want to find, so rest is left out. Running is speed above
# 4 cm/s.
#
# `NonLocalSortedSpikesDetector` has four states: **Local** (the spikes
# represent the rat's position), **No-Spike** (the cells are nearly silent),
# **Non-Local Continuous** and **Non-Local Fragmented** (see
# [Core concepts](../../getting-started/core-concepts/)). Its default
# transitions assume 2 ms time bins: the detector's `sampling_frequency` is
# 500 Hz.
#
# The position here is 1-D, so the detector warns that it treats the track
# as a 1-D open field. On a straight track with no junctions this is the
# same as the track graph; for a maze, pass 2-D position and a track graph
# instead.

# %%
speed = np.abs(np.gradient(position, position_time))  # cm/s
is_running = speed > 4.0
print(f"Running: {is_running.mean():.0%} of position samples")

detector = NonLocalSortedSpikesDetector()
detector.fit(position_time, position, spike_times, is_training=is_running)

# %% [markdown]
# The fitted place fields, in Hz at each position bin inside the track, are
# in `detector.encoding_model_`. They follow the simulated fields:

# %%
environment = detector.environments_[0]
bin_centers = environment.place_bin_centers_[environment.is_track_interior_.ravel(), 0]
fitted_place_fields = next(iter(detector.encoding_model_.values()))["place_fields"]

fig, axes = plt.subplots(
    2, 4, figsize=(10, 4), sharex=True, sharey=True, constrained_layout=True
)
for ax, cell in zip(axes.flat, np.linspace(0, n_cells - 1, 8).astype(int), strict=True):
    ax.plot(
        bin_centers,
        place_field_rates(bin_centers, peak_rates)[:, cell],
        color="lightgray",
        linewidth=4,
        label="simulated",
    )
    ax.plot(bin_centers, fitted_place_fields[cell], color="black", label="fitted")
    ax.set_title(f"Cell {cell}")
axes[0, 0].legend(frameon=False)
fig.supxlabel("Position (cm)")
fig.supylabel("Firing rate (Hz)")

# %% [markdown]
# ## Decode the whole session
#
# Decoding takes the edges of the time bins. To decode the whole recording,
# `calculate_time_edges` builds 2 ms bins (`1 / sampling_frequency`) from the
# first to the last position sample. `predict` interpolates the position to
# the bin centers for the Local state.

# %%
time_edges = detector.calculate_time_edges([position_time[0], position_time[-1]])
print(f"{time_edges.size - 1:,} time bins of {np.diff(time_edges).mean() * 1e3:.0f} ms")

start = timer.perf_counter()
results = detector.predict(
    spike_times=spike_times,
    time_edges=time_edges,
    position=position,
    position_time=position_time,
)
print(f"Decoded in {timer.perf_counter() - start:.1f} s")
print(results)

# %% [markdown]
# JAX may warn here that "Some donated buffers were not usable". That is a
# note about reusing memory inside the compiled filter; it does not change
# the results.

# %% [markdown]
# `results` holds, for each time bin, the probability of each state
# (`acausal_state_probabilities`) and of each state and position
# (`acausal_posterior`). Both are *acausal*: they use the spikes before and
# after each bin.
#
# Stacked over the whole session, the four state probabilities show Local
# for most of the time, and the non-local states taking over for brief
# periods during rest (red marks: simulated events).

# %%
state_colors = {
    "Local": "tab:blue",
    "No-Spike": "tab:gray",
    "Non-Local Continuous": "tab:red",
    "Non-Local Fragmented": "tab:orange",
}


def plot_session(results):
    """Position and stacked state probabilities over the whole session."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 4.5), sharex=True, constrained_layout=True
    )
    axes[0].plot(position_time, position, color="black", linewidth=1)
    axes[0].set_ylabel("Position (cm)")
    axes[1].stackplot(
        results.time,
        results.acausal_state_probabilities.to_numpy().T,
        colors=[state_colors[state] for state in results.states.to_numpy()],
        labels=results.states.to_numpy(),
    )
    axes[1].plot(
        simulated_events.start_time,
        np.full(len(simulated_events), 1.04),
        "v",
        color="tab:red",
        clip_on=False,
    )
    axes[1].set(xlabel="Time (s)", ylabel="Probability", ylim=(0, 1))
    axes[1].set_xlim(results.time[0], results.time[-1])
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    return fig, axes


plot_session(results)

# %% [markdown]
# Zooming in on one rest, with the spikes of each cell (sorted by the peak of
# its fitted place field) and the probability of each state:

# %%
cell_order = np.argsort(np.argmax(fitted_place_fields, axis=1))


def plot_window(results, time_window):
    """Spikes and state probabilities in a time window."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, constrained_layout=True
    )
    for row, cell in enumerate(cell_order):
        times = spike_times[cell]
        times = times[(times >= time_window[0]) & (times <= time_window[1])]
        axes[0].vlines(times, row + 0.6, row + 1.4, color="black", linewidth=1)
    axes[0].set_ylabel("Cell (by place field)")
    state_probabilities = results.acausal_state_probabilities
    for state, color in state_colors.items():
        axes[1].plot(
            results.time,
            state_probabilities.sel(states=state),
            color=color,
            label=state,
            linewidth=1,
        )
    for ax in axes:
        for start, stop in simulated_events[["start_time", "end_time"]].to_numpy():
            ax.axvspan(start, stop, color="tab:red", alpha=0.15)
    axes[1].set(
        xlabel="Time (s)", ylabel="Probability", ylim=(-0.02, 1.02), xlim=time_window
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    return fig, axes


plot_window(results, window)

# %% [markdown]
# During the simulated event (shaded), the non-local states take over.
# Outside it there are also brief non-local blips. Each starts at a single
# out-of-field spike: one spike from a cell whose field is far from the rat
# is better explained by a non-local position inside that field than by the
# rat's position, where the cell should be almost silent. This is the model
# working as designed, not a decoding error, and the event extraction below
# has to exclude these blips.

# %% [markdown]
# ## Refine the transitions with expectation-maximization (optional)
#
# The default transition probabilities are a prior guess: for example, a
# Non-Local state stays for 50 ms on average (a self-transition of 0.96 per
# 2 ms bin). `estimate_parameters` fits the place fields like `fit`, then
# re-estimates the **discrete transition matrix** (how often states start
# and how long they last) and the initial state probabilities by
# expectation-maximization (EM), and returns the decoding of the data it was
# given. With `estimate_encoding_model=True` it would also re-fit the place
# fields from the Local state's posterior; that is off by default.
#
# Two defaults limit what EM changes. Each row of the matrix has a
# Dirichlet prior that favours staying in the state
# (`discrete_transition_stickiness`), far stronger for Local and No-Spike
# than for the Non-Local states. And the No-Spike row is frozen
# (`frozen_discrete_transition_rows`). So EM mostly learns how often the
# Non-Local states start, how long they last and where they go next.
#
# Use it when the default dwell times may not suit your data, for example a
# different bin width, or events that are much longer or shorter than
# 50 ms. It decodes the data many times, so it takes several times as long
# as `predict`. The transitions are learned per bin, so decode with the same
# bin width afterwards.

# %%
em_detector = NonLocalSortedSpikesDetector()
start = timer.perf_counter()
em_results = em_detector.estimate_parameters(
    position_time,
    position,
    spike_times,
    time_edges=time_edges,
    is_training=is_running,
)
log_likelihoods = em_results.attrs["marginal_log_likelihoods"]
print(f"EM took {timer.perf_counter() - start:.0f} s")

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.plot(np.arange(len(log_likelihoods)), log_likelihoods, "o-", color="black")
ax.set(xlabel="E-step", ylabel="Marginal log-likelihood")

# %% [markdown]
# EM stops when an iteration improves the log-likelihood (plus the log prior
# of the transitions) by less than `tolerance` nats per time bin, or after
# `max_iter` iterations. The transition matrices before and after:

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
short_names = ["Local", "No-Spike", "NL Cont.", "NL Frag."]
for ax, transitions, title in zip(
    axes,
    [detector.discrete_state_transitions_, em_detector.discrete_state_transitions_],
    ["Default", "After EM"],
    strict=True,
):
    image = ax.imshow(np.log10(transitions), vmin=-4, vmax=0, cmap="viridis")
    for (row, column), probability in np.ndenumerate(transitions):
        ax.text(
            column,
            row,
            f"{probability:.3f}",
            ha="center",
            va="center",
            color="white" if probability < 0.1 else "black",
        )
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=short_names,
        yticklabels=short_names,
        xlabel="To state",
        ylabel="From state",
        title=title,
    )
fig.colorbar(image, ax=axes, label="log$_{10}$ probability per 2 ms bin")

# %% [markdown]
# The self-transitions set how long each state lasts: on average
# `bin width / (1 - self-transition)`.

# %%
bin_width = np.diff(time_edges).mean()
pd.DataFrame(
    {
        "default": bin_width / (1 - np.diag(detector.discrete_state_transitions_)),
        "after EM": bin_width / (1 - np.diag(em_detector.discrete_state_transitions_)),
    },
    index=detector.state_names,
).mul(1e3).round(0).rename_axis("Mean duration (ms)")

# %% [markdown]
# EM made Non-Local Continuous last longer, closer to the length of the
# simulated sweeps, and Non-Local Fragmented shorter, with most of its exits
# going back to Local: that is the signature of the single-spike blips.
# Local lasts longer too, which its strong stickiness prior favours, and
# the frozen No-Spike row is unchanged. In the same rest as before, the
# blips are now narrower:

# %%
plot_window(em_results, window)

# %% [markdown]
# The rest of this tutorial uses the EM decoding, `em_results`, and compares
# it with the default one, `results`, where it matters.

# %% [markdown]
# ## Extract non-local events
#
# A non-local event is a period in which the activity represents a position
# other than the rat's, so its probability is the sum of the two non-local
# states. Candidate events are the contiguous runs of time bins in which
# that probability exceeds a threshold, here 0.8.


# %%
def non_local_probability(results):
    """Probability of either non-local state in each time bin, (n_bins,)."""
    return (
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .to_numpy()
    )


p_non_local = non_local_probability(em_results)
results_p_non_local = non_local_probability(results)


def find_events(probability, time_edges, threshold, min_duration=0.0):
    """Contiguous runs of time bins with `probability` above `threshold`.

    Parameters
    ----------
    probability : np.ndarray, shape (n_bins,)
    time_edges : np.ndarray, shape (n_bins + 1,)
        Edges of the time bins, in s.
    threshold : float
    min_duration : float, optional
        Shortest event kept, in s.

    Returns
    -------
    events : pd.DataFrame
        One row per event: the first and one-past-last time bin, and the
        start, end and duration in s.
    """
    is_above = np.concatenate(([0], probability > threshold, [0]))
    first_bin, stop_bin = np.flatnonzero(np.diff(is_above)).reshape(-1, 2).T
    events = pd.DataFrame(
        {
            "first_bin": first_bin,
            "stop_bin": stop_bin,
            "start_time": time_edges[first_bin],
            "end_time": time_edges[stop_bin],
        }
    )
    events["duration"] = events.end_time - events.start_time
    return events[events.duration >= min_duration].reset_index(drop=True)


def overlaps(events, other_events):
    """(n_events, n_other_events) True where two events overlap in time."""
    return (
        events.start_time.to_numpy()[:, np.newaxis] < other_events.end_time.to_numpy()
    ) & (events.end_time.to_numpy()[:, np.newaxis] > other_events.start_time.to_numpy())


candidates = find_events(p_non_local, time_edges, threshold=0.8)
candidates["overlaps_simulated"] = overlaps(candidates, simulated_events).any(axis=1)
print(f"{len(candidates)} candidate periods above 0.8")

# %% [markdown]
# Most candidates are the single-spike blips seen above. They are short: a
# few bins around one spike, while a replay event lasts tens to hundreds of
# milliseconds. A histogram of the candidate durations shows the two groups
# (colored here by whether they overlap a simulated event, which you would
# not know with real data):

# %%
duration_bins = np.arange(0, 0.30, 0.004)
fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
ax.hist(
    candidates.duration[~candidates.overlaps_simulated] * 1e3,
    bins=duration_bins * 1e3,
    color="tab:gray",
    label="outside simulated events",
)
ax.hist(
    candidates.duration[candidates.overlaps_simulated] * 1e3,
    bins=duration_bins * 1e3,
    color="tab:red",
    label="overlaps a simulated event",
)
ax.axvline(50, color="black", linestyle="--", label="minimum duration")
ax.set(xlabel="Candidate duration (ms)", ylabel="Count", yscale="log")
ax.legend(frameon=False)

# %% [markdown]
# Keeping the candidates that last at least 50 ms removes the blips. How to
# choose the threshold and the minimum duration depends on the question and
# the data; with real recordings they are often combined with other
# criteria, such as sharp-wave ripples or a minimum number of active cells.

# %%
detected_events = find_events(p_non_local, time_edges, threshold=0.8, min_duration=0.05)
print(f"{len(detected_events)} non-local events detected")

# %% [markdown]
# ## Compare with the simulated events
#
# A simulated event that overlaps a detected one is a **hit**, and one with
# no detected event is a **miss**. A detected event that overlaps no
# simulated event is a **false alarm**. One simulated event can be **split**
# into more than one detected event.


# %%
def compare_events(detected_events, simulated_events):
    """Hits, misses, false alarms and splits of detected vs simulated events."""
    is_overlapping = overlaps(detected_events, simulated_events)
    n_hits = is_overlapping.any(axis=0).sum()
    return {
        "detected": len(detected_events),
        "hits": n_hits,
        "misses": len(simulated_events) - n_hits,
        "false alarms": (~is_overlapping.any(axis=1)).sum(),
        "split": (is_overlapping.sum(axis=0) > 1).sum(),
    }


print(f"{len(simulated_events)} simulated events")
pd.DataFrame(
    {
        "default transitions": compare_events(
            find_events(results_p_non_local, time_edges, 0.8, min_duration=0.05),
            simulated_events,
        ),
        "after EM": compare_events(detected_events, simulated_events),
    }
)

# %% [markdown]
# Compare the two columns: with the default transitions the blips last
# longer (Non-Local Fragmented lasts 50 ms on average, rather than the much
# shorter time EM learned), so more of them pass the 50 ms minimum as false
# alarms.

# %% [markdown]
# For each hit, how far are the detected start and end from the simulated
# ones? (For a split event, the start of its first and the end of its last
# detected piece.)

# %%
is_overlapping = overlaps(detected_events, simulated_events)
detected_events["simulated_event"] = np.where(
    is_overlapping.any(axis=1), is_overlapping.argmax(axis=1), -1
)
hit_ids = np.flatnonzero(is_overlapping.any(axis=0))
onset_error = (
    np.array([detected_events.start_time[is_overlapping[:, i]].min() for i in hit_ids])
    - simulated_events.start_time[hit_ids].to_numpy()
)
offset_error = (
    np.array([detected_events.end_time[is_overlapping[:, i]].max() for i in hit_ids])
    - simulated_events.end_time[hit_ids].to_numpy()
)

fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
error_bins = np.arange(-60, 62, 4)
ax.hist(onset_error * 1e3, bins=error_bins, alpha=0.6, label="start")
ax.hist(offset_error * 1e3, bins=error_bins, alpha=0.6, label="end")
ax.set(xlabel="Detected minus simulated time (ms)", ylabel="Number of events")
ax.legend(frameon=False)
print(
    f"Start error: median {np.median(onset_error) * 1e3:.0f} ms, "
    f"range {onset_error.min() * 1e3:.0f} to {onset_error.max() * 1e3:.0f} ms"
)
print(
    f"End error:   median {np.median(offset_error) * 1e3:.0f} ms, "
    f"range {offset_error.min() * 1e3:.0f} to {offset_error.max() * 1e3:.0f} ms"
)

# %% [markdown]
# ## The represented trajectory of each event
#
# `get_conditional_non_local_posterior` returns the posterior over position
# *given* that the activity is non-local: it sums the two non-local states
# and normalizes over position in each time bin. Each panel below shows it
# for one detected event (between the gray lines, with 20 ms on either
# side), with the rat's position (blue) and, from the simulation, the
# position the spikes represented (red dots). Outside the event the
# non-local states are improbable, so the conditional posterior there says
# little.

# %%
non_local_posterior = em_detector.get_conditional_non_local_posterior(em_results)
place_bin_edges = environment.place_bin_edges_.squeeze()
bin_centers_time = em_results.time.to_numpy()
rat_position = np.interp(bin_centers_time, position_time, position)  # cm
padding = 10  # bins (20 ms) shown before and after each event

n_columns = 6
n_rows = int(np.ceil(len(detected_events) / n_columns))
fig, axes = plt.subplots(
    n_rows,
    n_columns,
    figsize=(12, 2.2 * n_rows),
    sharey=True,
    constrained_layout=True,
)
for ax, event in zip(axes.flat, detected_events.itertuples(), strict=False):
    rows = slice(max(event.first_bin - padding, 0), event.stop_bin + padding)
    relative_edges = (time_edges[rows.start : rows.stop + 1] - event.start_time) * 1e3
    relative_time = (bin_centers_time[rows] - event.start_time) * 1e3
    mesh = ax.pcolormesh(
        relative_edges,
        place_bin_edges,
        non_local_posterior.isel(time=rows).to_numpy().T,
        cmap="bone_r",
        vmin=0,
        vmax=0.3,
    )
    ax.plot(relative_time, rat_position[rows], color="tab:blue", linewidth=2)
    ax.plot(
        relative_time,
        np.where(in_replay[rows], represented_position[rows], np.nan),
        ".",
        color="tab:red",
        markersize=2,
    )
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.axvline(event.duration * 1e3, color="gray", linewidth=0.5)
    ax.set_ylim(-8, track_length + 8)
    label = (
        "false alarm"
        if event.simulated_event < 0
        else simulated_events.type[event.simulated_event]
    )
    ax.set_title(f"Event {event.Index}: {label}", fontsize=9)
for ax in axes.flat[len(detected_events) :]:
    ax.axis("off")
fig.supxlabel("Time from event start (ms)")
fig.supylabel("Position (cm)")
fig.colorbar(mesh, ax=axes, shrink=0.3, label="P(position | non-local)")

# %% [markdown]
# ## Describe each event
#
# Per event, we can summarize:
#
# - its **duration**;
# - which non-local **dynamic** dominates: the mean probability of Non-Local
#   Continuous versus Non-Local Fragmented;
# - **how far** the represented position is from the rat. The most likely
#   position in each bin (the maximum a posteriori estimate,
#   `maximum_a_posteriori_estimate` from `non_local_detector.analysis`) is
#   compared with the rat's position. On a 1-D track the distance is the
#   absolute difference; for a maze with a track graph, use the distance
#   along the track instead, for example with
#   `non_local_detector.analysis.get_ahead_behind_distance`.

# %%
state_probabilities = em_results.acausal_state_probabilities

summaries = []
for event in detected_events.itertuples():
    rows = slice(event.first_bin, event.stop_bin)
    map_position = maximum_a_posteriori_estimate(non_local_posterior.isel(time=rows))[
        :, 0
    ]
    distance = np.abs(map_position - rat_position[rows])
    p_continuous = state_probabilities.sel(states="Non-Local Continuous")[rows].mean()
    p_fragmented = state_probabilities.sel(states="Non-Local Fragmented")[rows].mean()
    summaries.append(
        {
            "start_time": event.start_time,
            "duration_ms": event.duration * 1e3,
            "dynamic": "continuous" if p_continuous > p_fragmented else "fragmented",
            "rat_position": rat_position[rows].mean(),
            "mean_distance": distance.mean(),
            "max_distance": distance.max(),
            "simulated_type": (
                simulated_events.type[event.simulated_event]
                if event.simulated_event >= 0
                else "none"
            ),
        }
    )
event_summary = pd.DataFrame(summaries)
event_summary.round(1)

# %%
print(
    "Dominant dynamic matches the simulated type in "
    f"{(event_summary.dynamic == event_summary.simulated_type).sum()} "
    f"of {(event_summary.simulated_type != 'none').sum()} detected events"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
for dynamic, color in [("continuous", "tab:red"), ("fragmented", "tab:orange")]:
    subset = event_summary[event_summary.dynamic == dynamic]
    axes[0].scatter(
        subset.duration_ms, subset.mean_distance, color=color, label=dynamic
    )
axes[0].set(xlabel="Duration (ms)", ylabel="Mean distance from rat (cm)")
axes[0].legend(title="Dominant dynamic", frameon=False)
axes[1].hist(event_summary.max_distance, bins=np.arange(0, 190, 10), color="tab:gray")
axes[1].set(xlabel="Maximum distance from rat (cm)", ylabel="Number of events")

# %% [markdown]
# ## Save the results
#
# `save_results` writes the decoding to a netCDF file and `load_results`
# reads it back; `save_model` and `load_model` do the same for the fitted
# detector (as a pickle: load only files you trust). The event table is a
# plain DataFrame. A temporary directory stands in for your results folder
# here.

# %%
with tempfile.TemporaryDirectory() as results_folder:
    results_path = Path(results_folder) / "session_results.nc"
    NonLocalSortedSpikesDetector.save_results(em_results, results_path)
    print(f"{results_path.name}: {results_path.stat().st_size / 1e6:.0f} MB")

    loaded = NonLocalSortedSpikesDetector.load_results(results_path)
    print(
        "Reloaded state probabilities match:",
        np.array_equal(
            loaded.acausal_state_probabilities, em_results.acausal_state_probabilities
        ),
    )
    loaded.close()

    em_detector.save_model(Path(results_folder) / "detector.pkl")
    event_summary.to_csv(Path(results_folder) / "non_local_events.csv", index=False)

# %% [markdown]
# The full posterior is most of that size: one value per time bin and state
# bin. For long sessions, keep only what you need (for example
# `em_results[["acausal_state_probabilities"]]`), or use checkpointed
# prediction, which writes to disk as it decodes (below).

# %% [markdown]
# ## Clusterless data
#
# With unsorted spikes, use `NonLocalClusterlessDetector`, which models the
# spikes' waveform features (here the four tetrode amplitudes) instead of
# cell identities. Only the detector and the spike arguments change, and
# the event analysis above is the same. This decoding uses the default
# transitions, so compare it with the default sorted decoding;
# `estimate_parameters` takes the same arguments as `fit` and `predict`
# here.

# %%
clusterless_detector = NonLocalClusterlessDetector()
clusterless_detector.fit(
    position_time,
    position,
    tetrode_spike_times,
    tetrode_features,
    is_training=is_running,
)
clusterless_results = clusterless_detector.predict(
    spike_times=tetrode_spike_times,
    spike_waveform_features=tetrode_features,
    time_edges=time_edges,
    position=position,
    position_time=position_time,
)
clusterless_events = find_events(
    non_local_probability(clusterless_results), time_edges, 0.8, min_duration=0.05
)
pd.Series(compare_events(clusterless_events, simulated_events), name="clusterless")

# %% [markdown]
# ## Long recordings
#
# A real session can be hours of 2 ms bins, too much to hold the full
# posterior in memory. [Checkpointed prediction](../../performance_prediction/)
# decodes in chunks and writes the results to disk, and
# [Settings for your hardware](../../hardware_settings/) lists memory and
# speed settings for CPUs and GPUs.
