# %% [markdown]
# # Decoding a session with sorted spikes
#
# A decoder estimates, in each small time bin, which position the spikes of a
# population of place cells represent. `SortedSpikesDecoder` does this for
# sorted spikes (one spike train per neuron) with a hidden Markov model that
# has a single state, Continuous, in which the represented position follows a
# random walk.
#
# This tutorial goes through a complete analysis of a simulated recording
# session:
#
# 1. simulate a session on a linear track and look at the spikes and position;
# 2. fit the encoding model (each neuron's place field) on the periods when
#    the animal is running, and compare the fitted place fields with the true
#    ones;
# 3. look at the transition model;
# 4. decode the whole session, running and resting, and look at the results;
# 5. measure how far the decoded position is from the animal's position;
# 6. save the results and load them back.
#
# Then it repeats the analysis on a circular track described by a track graph
# and in a 2-D open field. [Data format and
# environments](../01_data_and_environments/) explains the inputs and
# environments used here.

# %%
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from track_linearization import get_linearized_position, make_circular_track

from non_local_detector import Environment, RandomWalk, SortedSpikesDecoder
from non_local_detector.analysis import maximum_a_posteriori_estimate
from non_local_detector.simulate import simulate_poisson_spikes

# JAX warns "Some donated buffers were not usable" when it cannot reuse the
# memory of an input array for an output. It affects memory reuse only, not
# the results, so we hide it to keep the output readable.
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

rng = np.random.default_rng(0)

# %% [markdown]
# ## Simulate a session
#
# We simulate four minutes on a 180 cm linear track. The rat runs from one end
# to the other in 3 to 5 s, rests at the end for 2 to 8 s, and runs back.
# The simulation runs at 1 kHz; the "camera" records the rat's position at
# 30 Hz, as a real tracking system would.

# %%
SIMULATION_RATE = 1000.0  # Hz
TRACKING_RATE = 30.0  # Hz
TRACK_LENGTH = 180.0  # cm
SESSION_DURATION = 240.0  # s

n_samples = int(SESSION_DURATION * SIMULATION_RATE)
true_time = np.arange(n_samples) / SIMULATION_RATE

# Alternate runs (a smooth start and stop between the two ends) and rests.
segments = [np.zeros(int(rng.uniform(2, 8) * SIMULATION_RATE))]
current_end = 0.0
while sum(len(segment) for segment in segments) < n_samples:
    other_end = TRACK_LENGTH - current_end
    n_run = int(rng.uniform(3, 5) * SIMULATION_RATE)
    fraction = (1 - np.cos(np.pi * np.arange(1, n_run + 1) / n_run)) / 2
    segments.append(current_end + (other_end - current_end) * fraction)
    segments.append(np.full(int(rng.uniform(2, 8) * SIMULATION_RATE), other_end))
    current_end = other_end
true_position = np.concatenate(segments)[:n_samples]  # (n_samples,), cm

# The camera's samples.
position_time = np.arange(0.0, SESSION_DURATION, 1 / TRACKING_RATE)
position = np.interp(position_time, true_time, true_position)
speed = np.abs(np.gradient(position, position_time))  # cm/s

print(f"position_time: {position_time.shape}, position: {position.shape}")

# %% [markdown]
# Next, 40 place cells. Each has a Gaussian place field with a random center,
# width (standard deviation 5 to 12 cm) and peak rate (5 to 25 Hz). Spikes are
# drawn from a Poisson process with each neuron's rate at the rat's position,
# one count per 1 ms simulation step; each spike gets a time spread uniformly
# over its 1 ms step, so spike times are not tied to the simulation grid.

# %%
n_neurons = 40
field_centers = np.sort(rng.uniform(0, TRACK_LENGTH, n_neurons))  # cm
field_widths = rng.uniform(5, 12, n_neurons)  # cm (standard deviation)
peak_rates = rng.uniform(5, 25, n_neurons)  # Hz


def place_field_rate(
    position: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    peak_rates: np.ndarray,
) -> np.ndarray:
    """Firing rates (Hz) of Gaussian place fields, shape (n_neurons, n_position)."""
    distance = position[np.newaxis, :] - centers[:, np.newaxis]
    return peak_rates[:, np.newaxis] * np.exp(
        -0.5 * (distance / widths[:, np.newaxis]) ** 2
    )


true_rates = place_field_rate(true_position, field_centers, field_widths, peak_rates)
spike_times = []
for rate in true_rates:
    counts = simulate_poisson_spikes(rate, SIMULATION_RATE, rng=rng)
    times = np.repeat(true_time, counts) + rng.uniform(
        -0.5 / SIMULATION_RATE, 0.5 / SIMULATION_RATE, counts.sum()
    )
    spike_times.append(np.sort(times))

print(
    f"spike_times: {len(spike_times)} neurons, "
    f"{sum(len(times) for times in spike_times)} spikes"
)

# %% [markdown]
# The rat is running when its speed is above 4 cm/s. Here are the position,
# the speed and the spikes, with neurons sorted by place-field center.

# %%
is_running = speed > 4.0  # (n_position_samples,)
print(f"running {is_running.mean():.0%} of the session")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 7),
    sharex=True,
    constrained_layout=True,
    height_ratios=[1, 0.6, 2],
)
axes[0].plot(position_time, position, color="black", linewidth=1)
axes[0].set_ylabel("Position (cm)")
axes[1].plot(position_time, speed, color="black", linewidth=1)
axes[1].axhline(4.0, color="tab:red", linestyle="--", linewidth=1)
axes[1].set_ylabel("Speed (cm/s)")
axes[2].eventplot(spike_times, colors="black", linelengths=0.8, linewidths=0.5)
axes[2].set_ylabel("Neuron")
axes[2].set_xlabel("Time (s)")
axes[2].set_xlim(0, SESSION_DURATION)
axes[0].set_title("Simulated session (red line: running threshold)")
plt.show()

# %% [markdown]
# ## Fit the encoding model on the running periods
#
# The encoding model describes how each neuron's firing rate depends on
# position: its place field. We fit it only on the running periods, as is
# usual: place fields are defined during movement, and in real recordings,
# rest contains replay, when cells fire for positions other than the rat's.
# Pass the training mask as `is_training`, one entry per position sample.
#
# `fit` also builds the environment's grid from the positions it is given (2
# cm bins here) and the transition model. The rat's position is 1-D and there
# is no track graph, so the decoder warns that it treats the track as a 1-D
# open field. For a linear track that is what we want.

# %%
decoder = SortedSpikesDecoder(environments=[Environment(place_bin_size=2.0)]).fit(
    position_time, position, spike_times, is_training=is_running
)

# %% [markdown]
# The fitted encoding model is in `decoder.encoding_model_`, a dictionary with
# one entry per environment and encoding group, here the single key `("", 0)`.
# The default algorithm, `sorted_spikes_kde`, estimates each place field with
# a kernel density estimate; `place_fields` holds the fitted rates in Hz,
# shape `(n_neurons, n_position_bins)`.

# %%
encoding_model = decoder.encoding_model_[("", 0)]
fitted_place_fields = np.asarray(encoding_model["place_fields"])  # Hz
environment = decoder.environments_[0]
bin_centers = environment.place_bin_centers_[:, 0]  # cm
true_place_fields = place_field_rate(
    bin_centers, field_centers, field_widths, peak_rates
)
print(f"place_fields: {fitted_place_fields.shape}")

# %% [markdown]
# The fitted place fields (red) follow the true ones (black) closely. The KDE
# smooths each field with a Gaussian kernel whose width is `position_std`
# (about 3.5 cm by default), set through `sorted_spikes_algorithm_params`.

# %%
example_neurons = np.linspace(0, n_neurons - 1, 8).astype(int)
fig, axes = plt.subplots(
    2, 4, figsize=(11, 4.5), sharex=True, sharey=True, constrained_layout=True
)
for ax, neuron in zip(axes.flat, example_neurons, strict=True):
    ax.plot(bin_centers, true_place_fields[neuron], color="black", label="true")
    ax.plot(bin_centers, fitted_place_fields[neuron], color="tab:red", label="fitted")
    ax.set_title(f"neuron {neuron}")
axes[0, 0].legend()
fig.supxlabel("Position (cm)")
fig.supylabel("Firing rate (Hz)")
plt.show()

# %%
correlation = np.array(
    [
        np.corrcoef(true, fitted)[0, 1]
        for true, fitted in zip(true_place_fields, fitted_place_fields, strict=True)
    ]
)
print(
    f"correlation between true and fitted place fields: "
    f"median {np.median(correlation):.3f}, min {correlation.min():.3f}"
)

# %% [markdown]
# ## The transition model
#
# The transition model says how the represented position can move from one
# time bin to the next. The decoder's default is a `RandomWalk`: the position
# takes a Gaussian step with variance `movement_var` (6 cm² by default) in each
# time bin. `plot_continuous_state_transition` shows the transition matrix:
# row `i` is the probability of moving from position bin `i` to each bin.

# %%
decoder.plot_continuous_state_transition()
plt.gcf().set_size_inches(5, 5)
ax = plt.gca()
ax.set_xlabel("Position bin at time t")
ax.set_ylabel("Position bin at time t - 1")
ax.set_title("Random walk transition matrix")
plt.show()

# %% [markdown]
# Each row is a narrow Gaussian around the diagonal. One row, in cm:

# %%
transition = decoder.continuous_state_transitions_  # (n_bins, n_bins)
from_bin = np.argmin(np.abs(bin_centers - 90.0))
fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
ax.plot(bin_centers, transition[from_bin], color="black", marker="o", markersize=3)
ax.set_xlim(70, 110)
ax.set_xlabel("Position at time t (cm)")
ax.set_ylabel("Probability")
ax.set_title(f"From the bin at {bin_centers[from_bin]:.0f} cm")
plt.show()

# %% [markdown]
# A step with a standard deviation of √6 ≈ 2.4 cm per 2 ms bin allows
# represented trajectories far faster than the rat runs, fast enough for
# replay. When the spikes are informative, as here, they pull the estimate
# to the right place. Choose `movement_var` with your bin width in mind: it
# is per bin, so halving the bin width calls for half the variance for the
# same speed (see `RandomWalk`).
#
# ## Decode the whole session
#
# A real analysis decodes the whole recording, not just the running periods.
# `calculate_time_edges([start, stop])` makes uniform bins of
# `1 / sampling_frequency` seconds (2 ms by default) over the session.

# %%
time_edges = decoder.calculate_time_edges([0.0, SESSION_DURATION])
print(
    f"{len(time_edges) - 1} bins of {decoder.sampling_frequency:.0f} Hz "
    f"({1000 / decoder.sampling_frequency:.0f} ms)"
)

# %% [markdown]
# Memory grows with the number of time bins times the number of position
# bins. `n_chunks` splits the session into that many pieces and computes the
# likelihood of the spikes one piece at a time, instead of holding it for the
# whole session at once; the results are the same. We also ask for the
# causal (past-only) posterior with `return_outputs="filter"`.

# %%
results = decoder.predict(
    spike_times,
    time_edges=time_edges,
    n_chunks=4,
    return_outputs="filter",
)
print(results)

# %% [markdown]
# The returned dataset has one row per time bin:
#
# - `acausal_posterior`, shape `(n_time, n_state_bins)`: the probability of
#   each position given all the spikes, before and after. This is usually
#   the one to use.
# - `causal_posterior`: the same given only the spikes up to that bin (asked
#   for with `return_outputs="filter"`).
# - `acausal_state_probabilities` and `causal_state_probabilities`: the
#   probability of each discrete state. The decoder has only one state, so
#   they are always 1.
# - Coordinates: `time` (bin centers), `time_bin_start` and `time_bin_end`
#   (bin edges), `is_missing`, and `state_bins`, which pairs each state with a
#   position bin. For this single-state decoder, unstack it to get a
#   `(time, state, position)` array, and sum over `state` for the posterior
#   over position. Models with Local or No-Spike states need
#   `non_local_detector.analysis.get_spatial_posterior` instead.
#
# Only interior position bins appear in `state_bins`. The full posterior is
# `n_time × n_position_bins` float32 numbers, and `n_chunks` does not change
# that:

# %%
posterior = results.acausal_posterior.unstack("state_bins").sum("state")
causal_posterior = results.causal_posterior.unstack("state_bins").sum("state")
print(f"posterior over position: {dict(posterior.sizes)}")
print(f"acausal_posterior: {results.acausal_posterior.nbytes / 1e6:.0f} MB")

# %%
single_chunk = decoder.predict(spike_times, time_edges=time_edges)
print(
    "n_chunks=4 matches n_chunks=1:",
    np.allclose(single_chunk.acausal_posterior, results.acausal_posterior),
)

# %% [markdown]
# For long sessions where even the posterior does not fit in memory,
# checkpointed prediction (`inference_mode="checkpointed"`) keeps memory
# independent of the session length and can write the posterior to disk as it
# goes. See [Checkpointed prediction](../../performance_prediction/) and
# [Settings for your hardware](../../hardware_settings/) for how to choose
# these options.
#
# ### Look at the posterior
#
# Over the whole session, the posterior (dark is high probability) follows the
# rat's position (dashed line), during runs and rests alike:

# %%
# The animal's position at the center of each decode bin.
position_at_bins = np.interp(results.time, position_time, position)

fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
posterior.plot(
    x="time",
    y="position",
    ax=ax,
    cmap="bone_r",
    vmin=0.0,
    vmax=0.1,
    add_colorbar=False,
    rasterized=True,
)
ax.plot(
    results.time,
    position_at_bins,
    color="magenta",
    linestyle="--",
    linewidth=0.8,
    label="animal's position",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("Acausal posterior, whole session")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# Zooming in on 20 s: the spikes, the causal posterior and the acausal
# posterior. The causal posterior uses only the spikes up to each bin, so it
# is more spread out; the acausal posterior also uses the later spikes.

# %%
zoom = slice(30.0, 50.0)
in_zoom = (results.time >= zoom.start) & (results.time <= zoom.stop)
neuron_order = np.argsort(field_centers)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 8),
    sharex=True,
    constrained_layout=True,
)
axes[0].eventplot(
    [
        spike_times[neuron][
            (spike_times[neuron] >= zoom.start) & (spike_times[neuron] <= zoom.stop)
        ]
        for neuron in neuron_order
    ],
    colors="black",
    linelengths=0.8,
    linewidths=0.8,
)
axes[0].set_ylabel("Neuron")
for ax, name, values in [
    (axes[1], "Causal posterior", causal_posterior),
    (axes[2], "Acausal posterior", posterior),
]:
    values.sel(time=zoom).plot(
        x="time",
        y="position",
        ax=ax,
        cmap="bone_r",
        vmin=0.0,
        vmax=0.1,
        add_colorbar=False,
        rasterized=True,
    )
    ax.plot(
        results.time[in_zoom],
        position_at_bins[in_zoom],
        color="magenta",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_title(name)
    ax.set_ylabel("Position (cm)")
    ax.set_xlabel("")
axes[2].set_xlabel("Time (s)")
plt.show()

# %%
print("mean of the largest posterior probability in each bin:")
print(f"  causal:  {causal_posterior.max('position').mean().item():.2f}")
print(f"  acausal: {posterior.max('position').mean().item():.2f}")

# %% [markdown]
# ## How accurate is the decoding?
#
# The most likely position in each bin, the maximum a posteriori (MAP)
# estimate, is a point estimate to compare with the rat's position.
# `maximum_a_posteriori_estimate` returns it as shape `(n_time, n_dims)`. We
# split the error by whether the rat was running at the center of each bin.

# %%
map_position = maximum_a_posteriori_estimate(posterior)[:, 0]
decoding_error = np.abs(map_position - position_at_bins)  # cm
running_at_bins = np.interp(results.time, position_time, speed) > 4.0

for name, rows in [
    ("whole session", np.ones_like(running_at_bins)),
    ("running", running_at_bins),
    ("resting", ~running_at_bins),
]:
    print(
        f"{name:>13}: median error {np.median(decoding_error[rows]):.1f} cm, "
        f"90th percentile {np.percentile(decoding_error[rows], 90):.1f} cm"
    )

# %%
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for name, rows, color in [
    ("running", running_at_bins, "tab:blue"),
    ("resting", ~running_at_bins, "tab:orange"),
]:
    sorted_error = np.sort(decoding_error[rows])
    ax.plot(sorted_error, np.linspace(0, 1, sorted_error.size), color=color, label=name)
ax.set_xlim(0, 20)
ax.set_xlabel("|MAP - position| (cm)")
ax.set_ylabel("Fraction of bins")
ax.set_title("Cumulative distribution of the decoding error")
ax.legend()
print(f"bins off by more than 10 cm: {np.mean(decoding_error > 10):.1%}")
plt.show()

# %% [markdown]
# Most bins are decoded within a few centimeters, a couple of position bins,
# and few are off by more than 10 cm. The resting errors come in steps of
# 2 cm because the rat sits still at the end of the track and the MAP
# estimate is a bin center. This simulation has no replay, so during rest the
# posterior stays at the rat's position. In real data, the posterior leaves
# the rat's position during some rest periods: that is replay.
#
# ## Save and load the results
#
# `save_results` writes the results to a netCDF file and `load_results` reads
# them back, restoring `state_bins`. (The fitted decoder itself is saved
# with `save_model` and `load_model`; see
# [Saving and loading models](../../saving_models/).) We use a temporary
# directory here; in your analysis, use a path of your own.

# %%
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "decoded_session.nc"
    decoder.save_results(results, path)
    print(f"saved {path.name}: {path.stat().st_size / 1e6:.0f} MB")

    loaded = SortedSpikesDecoder.load_results(path)
    print(
        "loaded results match:",
        np.array_equal(loaded.acausal_posterior, results.acausal_posterior),
    )
    loaded.close()

# %% [markdown]
# ## A circular track with a track graph
#
# On any track other than a straight line, describe the track with a track
# graph. Here we use a circular track of radius 30 cm, approximated by 24
# straight edges with `make_circular_track`. Laid out in 1-D, the edges go
# around the circle in order with no gaps, starting and ending at node 0.

# %%
radius = 30.0  # cm
n_segments = 24
circle_graph = make_circular_track(radius=radius, n_segments=n_segments)
circle_edge_order = [(node, (node + 1) % n_segments) for node in range(n_segments)]
circle_length = sum(circle_graph.edges[edge]["distance"] for edge in circle_edge_order)
print(f"track length: {circle_length:.1f} cm")

# %% [markdown]
# The rat runs counterclockwise around the circle for two minutes, in runs of
# 4 to 8 s at a peak speed of 25 to 45 cm/s with rests of 1 to 4 s between
# them. Tracking gives its 2-D `(x, y)` position at 30 Hz. Twenty-four place
# cells have 2-D Gaussian fields (standard deviation 6 cm) spaced around the
# circle.

# %%
circle_duration = 120.0  # s
n_circle_samples = int(circle_duration * SIMULATION_RATE)
circle_true_time = np.arange(n_circle_samples) / SIMULATION_RATE

speeds = []  # cm/s at each simulation step
while sum(len(segment) for segment in speeds) < n_circle_samples:
    n_run = int(rng.uniform(4, 8) * SIMULATION_RATE)
    peak_speed = rng.uniform(25, 45)
    speeds.append(peak_speed * np.sin(np.pi * np.arange(n_run) / n_run) ** 2)
    speeds.append(np.zeros(int(rng.uniform(1, 4) * SIMULATION_RATE)))
circle_speed = np.concatenate(speeds)[:n_circle_samples]
angle = np.pi + np.cumsum(circle_speed) / SIMULATION_RATE / radius  # radians
circle_true_xy = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)

circle_position_time = np.arange(0.0, circle_duration, 1 / TRACKING_RATE)
circle_position = np.stack(
    [
        np.interp(circle_position_time, circle_true_time, circle_true_xy[:, dim])
        for dim in range(2)
    ],
    axis=1,
)  # (n_position_samples, 2)
circle_is_running = (
    np.linalg.norm(np.gradient(circle_position, circle_position_time, axis=0), axis=1)
    > 4.0
)

field_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
circle_field_centers = radius * np.stack(
    [np.cos(field_angles), np.sin(field_angles)], axis=1
)
squared_distance = (
    (circle_true_xy[np.newaxis] - circle_field_centers[:, np.newaxis]) ** 2
).sum(axis=-1)  # (n_neurons, n_samples)
circle_rates = 15.0 * np.exp(-0.5 * squared_distance / 6.0**2)  # Hz
circle_spike_times = []
for rate in circle_rates:
    counts = simulate_poisson_spikes(rate, SIMULATION_RATE, rng=rng)
    times = np.repeat(circle_true_time, counts) + rng.uniform(
        -0.5 / SIMULATION_RATE, 0.5 / SIMULATION_RATE, counts.sum()
    )
    circle_spike_times.append(np.sort(times))

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(circle_position[:, 0], circle_position[:, 1], color="lightgray", zorder=-1)
for neuron in range(0, 24, 4):
    spike_xy = np.stack(
        [
            np.interp(
                circle_spike_times[neuron],
                circle_position_time,
                circle_position[:, dim],
            )
            for dim in range(2)
        ],
        axis=1,
    )
    ax.scatter(spike_xy[:, 0], spike_xy[:, 1], s=4, label=f"neuron {neuron}")
ax.set_aspect("equal")
ax.set_xlabel("x position (cm)")
ax.set_ylabel("y position (cm)")
ax.set_title("Where six of the neurons fire")
ax.legend(loc="center", fontsize="small")
plt.show()

# %% [markdown]
# To decode on the track graph, give the `Environment` the graph,
# `edge_order` and `edge_spacing`, and fit with the 2-D position: the decoder
# linearizes it with the same graph.

# %%
circle_environment = Environment(
    place_bin_size=2.0,
    track_graph=circle_graph,
    edge_order=circle_edge_order,
    edge_spacing=0.0,
)
circle_decoder = SortedSpikesDecoder(environments=[circle_environment]).fit(
    circle_position_time,
    circle_position,
    circle_spike_times,
    is_training=circle_is_running,
)

# %% [markdown]
# On the track graph, the random walk measures distance along the graph. The
# first and the last position bins in 1-D are both next to node 0, so they are
# neighbors on the track, and the transition matrix wraps around: the corners
# of the matrix have the same band as the diagonal.

# %%
circle_decoder.plot_continuous_state_transition()
plt.gcf().set_size_inches(5, 5)
ax = plt.gca()
ax.set_xlabel("Position bin at time t")
ax.set_ylabel("Position bin at time t - 1")
ax.set_title("Random walk on the circular track")

circle_transition = circle_decoder.continuous_state_transitions_
print(f"P(first bin -> second bin) = {circle_transition[0, 1]:.3f}")
print(f"P(first bin -> last bin)   = {circle_transition[0, -1]:.3f}")
plt.show()

# %% [markdown]
# Decode the whole two minutes, then compare with the rat's linear position.
# To get it, interpolate the 2-D tracking to the bin centers and linearize it
# with `get_linearized_position` and the same graph. Distances on a circle
# wrap around too, so the error is the shorter way around.

# %%
circle_time_edges = circle_decoder.calculate_time_edges([0.0, circle_duration])
circle_results = circle_decoder.predict(
    circle_spike_times, time_edges=circle_time_edges
)
circle_posterior = circle_results.acausal_posterior.unstack("state_bins").sum("state")

circle_xy_at_bins = np.stack(
    [
        np.interp(circle_results.time, circle_position_time, circle_position[:, dim])
        for dim in range(2)
    ],
    axis=1,
)
circle_linear_position = get_linearized_position(
    circle_xy_at_bins,
    circle_graph,
    edge_order=circle_edge_order,
    edge_spacing=0.0,
).linear_position.to_numpy()

circle_map = maximum_a_posteriori_estimate(circle_posterior)[:, 0]
circle_error = np.abs(circle_map - circle_linear_position)
circle_error = np.minimum(circle_error, circle_length - circle_error)
print(
    f"median error {np.median(circle_error):.1f} cm, "
    f"90th percentile {np.percentile(circle_error, 90):.1f} cm"
)

# %%
fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
circle_posterior.sel(time=slice(0, 60)).plot(
    x="time",
    y="position",
    ax=ax,
    cmap="bone_r",
    vmin=0.0,
    vmax=0.1,
    add_colorbar=False,
    rasterized=True,
)
in_window = circle_results.time <= 60
ax.scatter(
    circle_results.time[in_window],
    circle_linear_position[in_window],
    color="magenta",
    s=0.2,
    label="animal's linear position",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Linear position (cm)")
ax.set_title("Circular track: acausal posterior, first 60 s")
ax.legend(loc="upper right", markerscale=20)
plt.show()

# %% [markdown]
# Each lap, the linear position jumps from the end of the 1-D layout back to
# 0 as the rat passes node 0, and the posterior jumps with it: on the track,
# these are neighboring places.
#
# ## A 2-D open field
#
# Decoding in 2-D works the same way with 2-D position and no track graph.
# We simulate three minutes of a rat foraging in an 80 × 80 cm box: its
# velocity is a smoothly varying random process (an Ornstein-Uhlenbeck process
# with a 1 s time constant), and it bounces off the walls. Thirty-six place
# cells have 2-D Gaussian fields (standard deviation 8 cm) on a 6 × 6 grid.

# %%
box_size = 80.0  # cm
box_duration = 180.0  # s
n_box_samples = int(box_duration * SIMULATION_RATE)
box_true_time = np.arange(n_box_samples) / SIMULATION_RATE

time_constant = 1.0  # s
velocity_std = 20.0  # cm/s, per dimension
decay = np.exp(-1 / (time_constant * SIMULATION_RATE))
innovations = rng.normal(
    scale=velocity_std * np.sqrt(1 - decay**2), size=(n_box_samples, 2)
)
# velocity[t] = decay * velocity[t - 1] + innovations[t]
velocity = lfilter([1.0], [1.0, -decay], innovations, axis=0)  # cm/s
unfolded = box_size / 2 + np.cumsum(velocity, axis=0) / SIMULATION_RATE
# Reflect at the walls: fold the unbounded path back into [0, box_size].
box_true_xy = box_size - np.abs(box_size - np.mod(unfolded, 2 * box_size))

box_position_time = np.arange(0.0, box_duration, 1 / TRACKING_RATE)
box_position = np.stack(
    [
        np.interp(box_position_time, box_true_time, box_true_xy[:, dim])
        for dim in range(2)
    ],
    axis=1,
)
box_speed = np.linalg.norm(np.gradient(box_position, box_position_time, axis=0), axis=1)
box_is_running = box_speed > 4.0
print(
    f"median speed {np.median(box_speed):.0f} cm/s, "
    f"running {box_is_running.mean():.0%} of the session"
)

grid = np.linspace(box_size / 12, box_size * 11 / 12, 6)
box_field_centers = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
squared_distance = (
    (box_true_xy[np.newaxis] - box_field_centers[:, np.newaxis]) ** 2
).sum(axis=-1)
box_rates = 15.0 * np.exp(-0.5 * squared_distance / 8.0**2)  # Hz
box_spike_times = []
for rate in box_rates:
    counts = simulate_poisson_spikes(rate, SIMULATION_RATE, rng=rng)
    times = np.repeat(box_true_time, counts) + rng.uniform(
        -0.5 / SIMULATION_RATE, 0.5 / SIMULATION_RATE, counts.sum()
    )
    box_spike_times.append(np.sort(times))

# %% [markdown]
# We use 4 cm bins: a 2-D grid has many more bins than a 1-D track, and memory
# and time grow with the number of bins.

# %%
box_decoder = SortedSpikesDecoder(environments=[Environment(place_bin_size=4.0)]).fit(
    box_position_time, box_position, box_spike_times, is_training=box_is_running
)

box_time_edges = box_decoder.calculate_time_edges([0.0, box_duration])
box_results = box_decoder.predict(
    box_spike_times, time_edges=box_time_edges, n_chunks=4
)
box_posterior = box_results.acausal_posterior.unstack("state_bins").sum("state")
print(f"posterior over position: {dict(box_posterior.sizes)}")

# %% [markdown]
# Summing the posterior over time gives the total time the decoder places
# the represented position in each bin, to compare with the time the rat
# spent there.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
summed = box_posterior.sum("time") * (box_time_edges[1] - box_time_edges[0])
summed.plot(
    x="x_position",
    y="y_position",
    ax=axes[0],
    cmap="bone_r",
    cbar_kwargs={"label": "Summed posterior (s)"},
)
axes[0].set_title("Posterior summed over time")
x_edges, y_edges = box_decoder.environments_[0].edges_
occupancy, _, _ = np.histogram2d(
    box_position[:, 0], box_position[:, 1], bins=[x_edges, y_edges]
)
mesh = axes[1].pcolormesh(x_edges, y_edges, occupancy.T / TRACKING_RATE, cmap="bone_r")
fig.colorbar(mesh, ax=axes[1], label="Time in bin (s)")
axes[1].set_title("Rat's occupancy")
for ax in axes:
    ax.set_xlabel("x position (cm)")
    ax.set_ylabel("y position (cm)")
    ax.set_aspect("equal")
plt.show()

# %%
environment_2d = box_decoder.environments_[0]
bin_xy = environment_2d.place_bin_centers_.reshape(*environment_2d.centers_shape_, 2)
distance_to_wall = np.minimum(bin_xy, box_size - bin_xy).min(axis=-1)
near_wall = (distance_to_wall < 8.0) & environment_2d.is_track_interior_
print("fraction of the session within 8 cm of a wall:")
print(f"  rat's occupancy:   {occupancy[near_wall].sum() / occupancy.sum():.2f}")
print(f"  summed posterior:  {summed.values[near_wall].sum() / box_duration:.2f}")

# %% [markdown]
# Both cover the whole box, but the decoder puts somewhat more probability
# near the walls than the rat spent there. The summed posterior reflects the
# model as well as the data: the random walk is not how a rat moves, and
# changing `movement_var` changes this difference.
#
# The MAP estimate is now an `(x, y)` pair for each bin. Over 20 s, it tracks
# both coordinates of the rat's position:

# %%
box_map = maximum_a_posteriori_estimate(box_posterior)  # (n_time, 2)
box_xy_at_bins = np.stack(
    [
        np.interp(box_results.time, box_position_time, box_position[:, dim])
        for dim in range(2)
    ],
    axis=1,
)
box_error = np.linalg.norm(box_map - box_xy_at_bins, axis=1)
print(
    f"median error {np.median(box_error):.1f} cm, "
    f"90th percentile {np.percentile(box_error, 90):.1f} cm"
)

window = box_results.time <= 20
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
for dim, (ax, name) in enumerate(zip(axes, ["x", "y"], strict=True)):
    ax.plot(
        box_results.time[window],
        box_map[window, dim],
        color="black",
        linewidth=0.8,
        label="MAP estimate",
    )
    ax.plot(
        box_results.time[window],
        box_xy_at_bins[window, dim],
        color="magenta",
        linestyle="--",
        linewidth=1.5,
        label="animal's position",
    )
    ax.set_ylabel(f"{name} position (cm)")
axes[0].legend(loc="upper right")
axes[1].set_xlabel("Time (s)")
axes[0].set_title("2-D open field: MAP estimate and the rat's position")
plt.show()

# %% [markdown]
# The MAP estimate moves in 4 cm steps, the size of the bins, so it cannot be
# more precise than that.
#
# ### The transition model matters
#
# The default `movement_var` (6 cm² per 2 ms bin) allows replay-speed
# movement. The rat forages far more slowly, so a smaller `movement_var` fits
# its movement better. Set it with `continuous_transition_types`, one
# transition per pair of states (here a single `RandomWalk`), and decode
# again:

# %%
slow_decoder = SortedSpikesDecoder(
    environments=[Environment(place_bin_size=4.0)],
    continuous_transition_types=[[RandomWalk(movement_var=2.0)]],
).fit(box_position_time, box_position, box_spike_times, is_training=box_is_running)
slow_results = slow_decoder.predict(
    box_spike_times, time_edges=box_time_edges, n_chunks=4
)
slow_posterior = slow_results.acausal_posterior.unstack("state_bins").sum("state")
slow_error = np.linalg.norm(
    maximum_a_posteriori_estimate(slow_posterior) - box_xy_at_bins, axis=1
)
for name, error in [("movement_var=6", box_error), ("movement_var=2", slow_error)]:
    print(
        f"{name}: median error {np.median(error):.1f} cm, "
        f"90th percentile {np.percentile(error, 90):.1f} cm"
    )

# %% [markdown]
# Here the smaller value gives smaller errors. Do not take it too far: with a
# step much smaller than a position bin, the represented position can barely
# move from bin to bin and sticks in place. To detect replay, which moves much
# faster than the rat, keep `movement_var` large enough for replay speeds.
#
# ## Summary
#
# - Fit the encoding model with `fit(position_time, position, spike_times,
#   is_training=...)`, using the running periods.
# - Decode the whole session with `predict(spike_times, time_edges=...)`,
#   with `time_edges` from `calculate_time_edges([start, stop])`. Use
#   `n_chunks` to compute the likelihood in pieces, and checkpointed
#   prediction for sessions whose posterior does not fit in memory.
# - For a single-state decoder, unstack `state_bins` and sum over `state` for
#   the posterior over position (models with Local or No-Spike states: use
#   `get_spatial_posterior`); `maximum_a_posteriori_estimate` gives the most
#   likely position.
# - Save results with `save_results` and read them with `load_results`.
# - On mazes, use a track-graph `Environment`; the random walk then follows
#   the track, including around loops.
