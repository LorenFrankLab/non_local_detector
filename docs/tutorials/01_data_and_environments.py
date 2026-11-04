# %% [markdown]
# # Data format and environments
#
# Every model in `non_local_detector` needs the same few inputs:
#
# - **position**: the animal's position (`position`) and the times it was
#   sampled (`position_time`);
# - **spikes**: for sorted data, the spike times of each neuron; for
#   clusterless data, the spike times on each electrode and the waveform
#   features (marks) of each of those spikes;
# - **decode bins**: the edges of the time bins to decode (`time_edges`);
# - **an environment**: a description of the space the animal moves in, which
#   the model divides into position bins.
#
# This tutorial shows what each input looks like, with its shape and a plot,
# and then shows how to describe space: open fields in one and two dimensions,
# and mazes described by a track graph. All times are in seconds and all
# positions in centimeters.

# %%
import matplotlib.pyplot as plt
import numpy as np
from track_linearization import (
    get_linearized_position,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)

from non_local_detector import (
    Environment,
    calculate_time_edges,
    time_edges_from_centers,
)
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

# %% [markdown]
# ## Position and sorted spikes
#
# `make_simulated_data` simulates a rat running back and forth on a 170 cm
# linear track for 65 s, with 30 place cells. (It also simulates a few
# replay-like events, which the [Quickstart](../../getting-started/quickstart/)
# uses; here we only look at the format of the data.)

# %%
(
    speed,
    position,
    spike_times,
    position_time,
    event_times,
    sampling_frequency,
    is_event,
    true_firing_rate,
) = make_simulated_data(n_neurons=30)

# %% [markdown]
# ### Position
#
# `position_time` holds the time of each position sample and `position` the
# position at that time, one row per sample. Position can be 1-D, with shape
# `(n_position_samples,)` or `(n_position_samples, 1)`, or 2-D, with shape
# `(n_position_samples, 2)` for `(x, y)`. This simulation samples position at
# 1,500 Hz. Camera tracking is usually 30 to 60 Hz, which works too: position
# samples do not have to line up with spikes or with the decode bins, because
# the models interpolate position between samples.

# %%
print(f"position_time: shape {position_time.shape}")
print(f"position:      shape {position.shape}")
print(f"sampling rate: {1 / np.median(np.diff(position_time)):.0f} Hz")
print(f"duration:      {position_time[-1] - position_time[0]:.1f} s")

# %%
fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
ax.plot(position_time, position, color="black", linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("Animal's position")
plt.show()

# %% [markdown]
# ### Sorted spikes
#
# For sorted data, `spike_times` is a list with one array per neuron. Each
# array holds that neuron's spike times in seconds, on the same clock as
# `position_time`, in increasing order. The arrays have different lengths
# because neurons fire different numbers of spikes; a neuron with no spikes is
# an empty array. Spike times are not binned: binning happens when you decode.

# %%
print(f"spike_times: list of {len(spike_times)} arrays")
for neuron in range(3):
    print(
        f"  neuron {neuron}: shape {spike_times[neuron].shape}, "
        f"first spikes {np.round(spike_times[neuron][:4], 4)} s"
    )

n_spikes = np.array([len(times) for times in spike_times])
print(f"spikes per neuron: {n_spikes.min()} to {n_spikes.max()}")

# %% [markdown]
# A raster plot shows each spike as a tick, one row per neuron. Each neuron
# has a place field, a part of the track where it fires, so the raster forms
# diagonal bands that follow the animal back and forth. The rows are ordered
# by where each neuron's field is. The near-vertical streaks while the animal
# rests at the far end are the simulated replay events, in which the neurons
# fire in a fast sequence.

# %%
fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 6),
    sharex=True,
    constrained_layout=True,
    height_ratios=[1, 2],
)
axes[0].plot(position_time, position, color="black", linewidth=1)
axes[0].set_ylabel("Position (cm)")
axes[1].eventplot(spike_times, colors="black", linelengths=0.8, linewidths=0.8)
axes[1].set_ylabel("Neuron")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(-0.5, len(spike_times) - 0.5)
axes[0].set_title("Position and spike raster")
plt.show()

# %% [markdown]
# The simulation also returns each neuron's true firing rate at every position
# sample (`true_firing_rate`, shape `(n_position_samples, n_neurons)`).
# Plotting it against position shows the true place fields, which are what a
# model learns when it is fitted:

# %%
order = np.argsort(position)
fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
ax.plot(position[order], true_firing_rate[order], linewidth=1)
ax.set_xlabel("Position (cm)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("True place fields of the 30 neurons")
plt.show()

# %% [markdown]
# ## Clusterless spikes
#
# Clusterless models skip spike sorting. Instead of assigning each spike to a
# neuron, they use every spike detected on an electrode together with features
# of its waveform, such as its peak amplitude on each channel of a tetrode.
# Spikes from the same neuron have similar features, and the model learns how
# features and position go together.
#
# The data come as two lists with one entry per electrode:
#
# - `spike_times[i]`: the times of the spikes on electrode `i`, shape
#   `(n_spikes_i,)`;
# - `spike_waveform_features[i]`: their features, shape
#   `(n_spikes_i, n_features)`. Row `j` describes spike `j` of
#   `spike_times[i]`, so the two must stay aligned.
#
# `make_simulated_run_data` simulates five tetrodes with four neurons each on
# a 175 cm track, with four waveform features per spike.

# %%
clusterless = make_simulated_run_data(n_runs=4)
clusterless_spike_times = clusterless.spike_times
spike_waveform_features = clusterless.spike_waveform_features

print(f"position_time: shape {clusterless.position_time.shape}")
print(f"position:      shape {clusterless.position.shape}")
print(f"spike_times:             list of {len(clusterless_spike_times)} arrays")
print(f"spike_waveform_features: list of {len(spike_waveform_features)} arrays")
for electrode in range(len(clusterless_spike_times)):
    print(
        f"  electrode {electrode}: spike_times {clusterless_spike_times[electrode].shape}, "
        f"spike_waveform_features {spike_waveform_features[electrode].shape}"
    )

# %% [markdown]
# The first spikes on electrode 0, one row per spike and one column per
# feature:

# %%
print(np.round(spike_waveform_features[0][:5], 2))

# %% [markdown]
# Plotting two features of each spike on electrode 0 shows four clusters, one
# per simulated neuron. Coloring each spike by the animal's position when it
# fired shows that each cluster fires in its own part of the track. A
# clusterless model learns this joint relationship between features and
# position directly, without drawing the cluster boundaries.

# %%
spike_position = np.interp(
    clusterless_spike_times[0],
    clusterless.position_time,
    clusterless.position[:, 0],
)
fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
points = ax.scatter(
    spike_waveform_features[0][:, 0],
    spike_waveform_features[0][:, 1],
    c=spike_position,
    s=4,
    cmap="viridis",
)
fig.colorbar(points, ax=ax, label="Position at spike (cm)")
ax.set_xlabel("Feature 1 (e.g. amplitude on channel 1, a.u.)")
ax.set_ylabel("Feature 2 (a.u.)")
ax.set_title("Electrode 0 waveform features")
plt.show()

# %% [markdown]
# ## Decode bins
#
# A model estimates the represented position in each of a series of time bins.
# You choose the bins by passing their edges, `time_edges`, with shape
# `(n_bins + 1,)`, to `predict`; results have one row per bin, labeled by the
# bin centers. Bins must be uniform, and their width sets the time resolution
# of the decoding (2 ms is typical for replay). There are two helpers:
#
# - `calculate_time_edges([start, stop], sampling_frequency)` makes bins of
#   `1 / sampling_frequency` seconds covering `[start, stop]`. A model's
#   `calculate_time_edges([start, stop])` method does the same with the
#   model's `sampling_frequency` (500 Hz, or 2 ms bins, by default). If the
#   range is not a whole number of bins, it raises an error unless you pass
#   `trim=True`, which drops the partial last bin, as we do here.
# - `time_edges_from_centers(times)` makes one bin centered on each of a set of
#   uniformly spaced times, for example the position samples. The bin around
#   time `t` is `[t - dt / 2, t + dt / 2)`.

# %%
session_start, session_stop = position_time[0], position_time[-1]
time_edges = calculate_time_edges(
    [session_start, session_stop], sampling_frequency=500, trim=True
)
print(
    f"calculate_time_edges: {len(time_edges) - 1} bins of {np.diff(time_edges)[0] * 1000:.1f} ms"
)
print(f"  first edges: {np.round(time_edges[:4], 4)} s")

centered_edges = time_edges_from_centers(position_time)
print(
    f"time_edges_from_centers: {len(centered_edges) - 1} bins "
    f"(one per position sample) of {np.diff(centered_edges)[0] * 1000:.3f} ms"
)
print(f"  first edges: {np.round(centered_edges[:4], 5)} s")
print(f"  first position samples: {np.round(position_time[:3], 5)} s")

# %% [markdown]
# Each spike belongs to exactly one bin. A bin includes its left edge and
# excludes its right edge, so a spike exactly on an edge belongs to the later
# bin; the last bin also includes its right edge, so a spike at the very end
# is not lost. `get_spikecount_per_time_bin` counts spikes with the same rule
# the models use. With two 2 ms bins and spikes at 0, 2 and 4 ms:

# %%
toy_edges = np.array([0.0, 0.002, 0.004])
toy_spikes = np.array([0.0, 0.002, 0.004])
print(get_spikecount_per_time_bin(toy_spikes, time_edges=toy_edges))

# %% [markdown]
# Here are 200 ms of the simulated recording in 2 ms bins: the spikes of the
# neurons active at that moment (top) and the number of spikes from all
# neurons in each bin (bottom). Most bins hold no spikes at all, so the
# model accumulates evidence over many bins.

# %%
window = (2.5, 2.7)
window_edges = calculate_time_edges(window, sampling_frequency=500)
spike_counts = np.stack(
    [
        get_spikecount_per_time_bin(times, time_edges=window_edges)
        for times in spike_times
    ]
)  # (n_neurons, n_bins)
active = np.nonzero(spike_counts.sum(axis=1))[0]

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 4.5),
    sharex=True,
    constrained_layout=True,
    height_ratios=[2, 1],
)
for row, neuron in enumerate(active):
    in_window = spike_times[neuron][
        (spike_times[neuron] >= window[0]) & (spike_times[neuron] <= window[1])
    ]
    axes[0].vlines(in_window, row - 0.4, row + 0.4, color="black")
axes[0].set_yticks(range(len(active)), active)
axes[0].set_ylabel("Neuron")
for edge in window_edges:
    axes[0].axvline(edge, color="lightgray", linewidth=0.5, zorder=-1)
bin_centers = window_edges[:-1] + np.diff(window_edges) / 2
axes[1].bar(
    bin_centers, spike_counts.sum(axis=0), width=np.diff(window_edges), color="gray"
)
axes[1].set_ylabel("Spikes per bin")
axes[1].set_xlabel("Time (s)")
axes[0].set_title("Spikes in 2 ms decode bins (gray lines are bin edges)")
plt.show()

# %% [markdown]
# For data with tracking gaps, recordings split into separate epochs, or
# irregular camera timestamps, see [Time bins and rates](../../time_grid_migration/).
#
# ## Environments
#
# An `Environment` divides space into position bins. The models estimate a
# probability for each bin, so the bin size sets the spatial resolution and,
# along with the number of time bins, the memory and time decoding takes.
# You pass an `Environment` to a model, and the model's `fit` builds the grid
# from the positions it is given by calling `fit_place_grid` on its own copy
# of the environment: the fitted copies are in `model.environments_`, and the
# environment you passed stays unfitted in `model.environments`, so several
# models can share one. Here
# we call `fit_place_grid` ourselves to look at the result.
#
# ### A 1-D open field
#
# With 1-D position and no track graph, the environment is a 1-D open field: a
# line divided into bins of about `place_bin_size`. The bins span the range of
# the positions; the bin size is adjusted slightly so that a whole number of
# bins fits.

# %%
# fit_place_grid takes position as (n_position_samples, n_dims).
environment_1d = Environment(place_bin_size=5.0).fit_place_grid(position[:, np.newaxis])

bin_edges = environment_1d.edges_[0]
print(f"edges_[0] (bin edges):      shape {bin_edges.shape}")
print(f"place_bin_centers_:         shape {environment_1d.place_bin_centers_.shape}")
print(f"centers_shape_:             {environment_1d.centers_shape_}")
print(f"actual bin size:            {np.diff(bin_edges)[0]:.2f} cm")
print(
    f"is_track_interior_:         {environment_1d.is_track_interior_.sum()} of "
    f"{environment_1d.is_track_interior_.size} bins"
)

# %% [markdown]
# `is_track_interior_` marks the bins the animal visited; the models only
# assign probability to those. Below, the histogram shows how much time the
# animal spent in each bin, and the gray lines are the bin edges.

# %%
occupancy_seconds, _ = np.histogram(position, bins=bin_edges)
occupancy_seconds = occupancy_seconds / sampling_frequency
fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
ax.bar(
    environment_1d.place_bin_centers_[:, 0],
    occupancy_seconds,
    width=np.diff(bin_edges),
    color=np.where(environment_1d.is_track_interior_, "tab:blue", "lightgray"),
)
for edge in bin_edges:
    ax.axvline(edge, color="gray", linewidth=0.5)
ax.set_xlabel("Position (cm)")
ax.set_ylabel("Time in bin (s)")
ax.set_title("1-D open field: 5 cm bins")
plt.show()

# %% [markdown]
# The ends of the track hold the most time because the animal pauses there.
#
# A linear track is a 1-D open field, so this is all a linear track needs.
# When you fit a model with 1-D position and no track graph, it warns that it
# is treating the data as a 1-D open field; for a linear track that is
# expected. For any maze with turns or junctions, use a track graph (below).
#
# ### A 2-D open field
#
# With 2-D position, the environment is a grid of square bins. To show how the
# environment finds the track interior, we sample positions uniformly from a
# plus-shaped arena (two crossing 100 × 20 cm arms). Only the positions matter
# for the grid, not their order in time.

# %%
rng = np.random.default_rng(0)
n_samples = 20_000
in_horizontal_arm = rng.random(n_samples) < 0.5
along_arm = rng.uniform(0, 100, n_samples)
across_arm = rng.uniform(40, 60, n_samples)
position_2d = np.where(
    in_horizontal_arm[:, np.newaxis],
    np.stack([along_arm, across_arm], axis=1),
    np.stack([across_arm, along_arm], axis=1),
)
print(f"position_2d: shape {position_2d.shape}")

environment_2d = Environment(place_bin_size=5.0).fit_place_grid(position_2d)
print(f"edges_:             {[edges.shape for edges in environment_2d.edges_]} (x, y)")
print(f"centers_shape_:     {environment_2d.centers_shape_}")
print(f"place_bin_centers_: shape {environment_2d.place_bin_centers_.shape}")
print(
    f"is_track_interior_: shape {environment_2d.is_track_interior_.shape}, "
    f"{environment_2d.is_track_interior_.sum()} interior bins"
)

# %% [markdown]
# `place_bin_centers_` lists every bin center as an `(x, y)` row;
# `is_track_interior_` is a grid of shape `centers_shape_`. The grid has one
# extra bin on each side of the visited range, which keeps the inferred
# interior from touching the grid's edge. The inference also closes small
# gaps between visited bins, so a few bins at the inside corners of the plus
# count as interior too. `plot_grid` draws the grid with the interior bins
# shaded:

# %%
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
environment_2d.plot_grid(ax=ax)
ax.scatter(position_2d[::20, 0], position_2d[::20, 1], s=1, color="tab:orange")
ax.set_xlabel("x position (cm)")
ax.set_ylabel("y position (cm)")
ax.set_aspect("equal")
ax.set_title("2-D open field: 5 cm bins (orange: positions)")
plt.show()

# %% [markdown]
# A few options change how the grid and interior are built:
#
# - `position_range=[(x_min, x_max), (y_min, y_max)]` fixes the grid's extent
#   instead of taking it from the positions, so that grids from different
#   sessions line up.
# - `is_track_interior` gives the interior mask yourself;
#   `infer_track_interior=False` treats every bin as interior.
# - `bin_count_threshold`, `fill_holes` and `dilate` tune the inferred
#   interior: a bin must contain more than `bin_count_threshold` samples, and
#   holes in it can be filled or its border grown by one bin.
#
# ## Track graphs
#
# On a maze, two places can be close in 2-D but far apart for the animal,
# like the ends of two arms separated by a wall. A 2-D grid would treat them
# as neighbors, and the models would smooth place fields and movement across
# the wall. A **track graph** describes the maze as a set of line segments
# (edges) joined at nodes. The models then work with **linear position**, the
# distance along the track, and measure distances along the graph.
#
# We build one step by step with
# [track_linearization](https://github.com/LorenFrankLab/track_linearization),
# which `non_local_detector` installs.
#
# ### Nodes and edges
#
# First, the `(x, y)` position of each node, in cm. Nodes are numbered by
# their order in the list.

# %%
node_positions = [
    (0, 0),  # node 0
    (50, 0),  # node 1
    (50, 50),  # node 2
    (0, 50),  # node 3
]

# %% [markdown]
# Next, the edges: which pairs of nodes are connected by a stretch of track.

# %%
edges = [
    (0, 1),  # connects node 0 and node 1
    (1, 2),  # connects node 1 and node 2
    (2, 3),  # connects node 2 and node 3
]

# %% [markdown]
# `make_track_graph` builds a [NetworkX](https://networkx.org) graph. Each
# node has a `pos` attribute (its position), and each edge has a `distance`
# attribute (its length, here in cm) and an `edge_id`, a number for each edge.
# (`edge_id` follows the order in which NetworkX lists the edges, which can
# differ from the order of `edges`; the plots below label each edge with it.)
# These are the attributes `Environment` requires, so you can also build the
# graph with NetworkX directly as long as you set them.

# %%
track_graph = make_track_graph(node_positions, edges)
print("nodes:", dict(track_graph.nodes(data=True)))
print("edges:", list(track_graph.edges(data=True)))

# %%
fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
plot_track_graph(track_graph, ax=ax, draw_edge_labels=True)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlabel("x position (cm)")
ax.set_ylabel("y position (cm)")
ax.set_title("Track graph (edge labels are edge_id)")
plt.show()

# %% [markdown]
# ### From 2-D to 1-D: edge order and edge spacing
#
# To lay the track out on a line, two more things are needed:
#
# - `edge_order`: the order of the edges in 1-D, each given as a node pair. The
#   order of the nodes in a pair sets the direction: `(2, 1)` runs from node 2
#   to node 1.
# - `edge_spacing`: the gap left between consecutive edges in 1-D, either one
#   number for every gap or a list with one number per gap (one fewer than the
#   number of edges).
#
# `plot_graph_as_1D` draws the resulting layout. With no gaps:

# %%
edge_order = [(3, 2), (2, 1), (1, 0)]

fig, ax = plt.subplots(figsize=(10, 1.5), constrained_layout=True)
plot_graph_as_1D(track_graph, edge_order=edge_order, edge_spacing=0, ax=ax)
ax.set_xlabel("Linear position (cm)")
ax.set_title("edge_order = [(3, 2), (2, 1), (1, 0)], edge_spacing = 0")
plt.show()

# %% [markdown]
# Here, each edge ends at the node where the next begins, so consecutive edges
# are connected and no gap is needed. A 15 cm gap between each edge, for
# comparison:

# %%
fig, ax = plt.subplots(figsize=(10, 1.5), constrained_layout=True)
plot_graph_as_1D(track_graph, edge_order=edge_order, edge_spacing=15, ax=ax)
ax.set_xlabel("Linear position (cm)")
ax.set_title("edge_spacing = 15")
plt.show()

# %% [markdown]
# ### Why gaps are needed
#
# Gaps matter when two edges are next to each other in the 1-D layout but not
# connected on the track. On a W-track the animal runs from the center well
# out to the left or right well and back. There is no way to lay out the three
# arms on a line so that every pair of neighbors is connected: wherever the
# left arm ends in 1-D, the right arm has to start somewhere else.
#
# We draw a W-track with three 80 cm arms, 30 cm apart.

# %%
w_node_positions = [
    (0, 80),  # 0: center well
    (0, 0),  # 1: center junction
    (-30, 0),  # 2: left junction
    (-30, 80),  # 3: left well
    (30, 0),  # 4: right junction
    (30, 80),  # 5: right well
]
w_edges = [
    (0, 1),  # center arm
    (1, 2),  # base, left half
    (2, 3),  # left arm
    (1, 4),  # base, right half
    (4, 5),  # right arm
]
w_track_graph = make_track_graph(w_node_positions, w_edges)

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
plot_track_graph(w_track_graph, ax=ax, draw_edge_labels=True)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.set_xlabel("x position (cm)")
ax.set_ylabel("y position (cm)")
ax.set_title("W-track (edge labels are edge_id)")
plt.show()

# %% [markdown]
# We lay it out as: the center arm from the center well, then the left half
# of the base and the left arm out to the left well, then the right half of
# the base and the right arm. The left arm ends at the left well (node 3), but
# the next edge starts at the center junction (node 1). They are neighbors in
# 1-D but far apart on the track, so we put a 15 cm gap between them; all
# other neighbors are connected and get no gap.

# %%
w_edge_order = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
w_edge_spacing = [0, 0, 15, 0]

fig, ax = plt.subplots(figsize=(10, 1.5), constrained_layout=True)
plot_graph_as_1D(
    w_track_graph, edge_order=w_edge_order, edge_spacing=w_edge_spacing, ax=ax
)
ax.set_xlabel("Linear position (cm)")
ax.set_title(
    "W-track in 1-D: a gap between the left well (3) and the center junction (1)"
)
plt.show()

# %% [markdown]
# Without the gap, the bin at the left well would sit right next to the bin at
# the center junction. The track-graph environment measures distances along
# the graph, so the random-walk movement model would still keep them apart,
# but anything that smooths in 1-D, like a kernel density estimate of a place
# field, would leak firing from the left well onto the right arm. Make each gap
# several times wider than that smoothing (the `position_std` of a KDE
# encoding model, about 3.5 cm by default).
#
# ### Linearizing position
#
# Tracking gives 2-D `(x, y)` position. `get_linearized_position` projects each
# sample onto the nearest edge of the graph and converts it to linear position
# with the same `edge_order` and `edge_spacing`.
#
# To try it, we simulate a rat on the W-track: it runs at 30 cm/s from the
# center well to the left well and back, then to the right well and back,
# three times. Tracking is sampled at 30 Hz, like a camera, with 1 cm of
# Gaussian noise on each coordinate.

# %%
well_sequence = [0, 1, 2, 3, 2, 1, 0, 1, 4, 5, 4, 1] * 3 + [0]
path = np.array([w_node_positions[node] for node in well_sequence], dtype=float)
distance_along_path = np.concatenate(
    [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
)

w_sampling_frequency = 30.0  # Hz
running_speed = 30.0  # cm/s
w_position_time = np.arange(
    0.0, distance_along_path[-1] / running_speed, 1 / w_sampling_frequency
)
distance_traveled = running_speed * w_position_time
w_position = np.stack(
    [
        np.interp(distance_traveled, distance_along_path, path[:, 0]),
        np.interp(distance_traveled, distance_along_path, path[:, 1]),
    ],
    axis=1,
) + rng.normal(scale=1.0, size=(w_position_time.size, 2))
print(f"w_position: shape {w_position.shape} ({w_position_time[-1]:.0f} s at 30 Hz)")

# %%
linearized = get_linearized_position(
    w_position,
    w_track_graph,
    edge_order=w_edge_order,
    edge_spacing=w_edge_spacing,
)
linearized.head()

# %% [markdown]
# The result is a table with one row per sample: `linear_position`, the
# `track_segment_id` (the `edge_id` of the edge the sample was assigned to),
# and the projected point on that edge.
#
# Now we check it. Left: the 2-D samples colored by their linear position; it
# should increase smoothly from the center well along each arm. Right: linear
# position over time; each trip should sweep through the center arm (0 to
# 80 cm) and then one of the outer arms, and no sample should fall in the gap.

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(11, 4.5), constrained_layout=True, width_ratios=[1, 2]
)
points = axes[0].scatter(
    w_position[:, 0],
    w_position[:, 1],
    c=linearized.linear_position,
    s=4,
    cmap="viridis",
)
fig.colorbar(points, ax=axes[0], label="Linear position (cm)")
axes[0].set_xlabel("x position (cm)")
axes[0].set_ylabel("y position (cm)")
axes[0].set_aspect("equal")

for segment_id, segment in linearized.groupby("track_segment_id"):
    axes[1].scatter(
        w_position_time[segment.index],
        segment.linear_position,
        s=3,
        label=f"edge {segment_id}",
    )
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Linear position (cm)")
axes[1].legend(title="track_segment_id", loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

# %%
projection_distance = np.linalg.norm(
    w_position
    - linearized[["projected_x_position", "projected_y_position"]].to_numpy(),
    axis=1,
)
gap_start = sum(w_track_graph.edges[edge]["distance"] for edge in w_edge_order[:3])
in_gap = (linearized.linear_position > gap_start) & (
    linearized.linear_position < gap_start + w_edge_spacing[2]
)
print(
    f"distance from each sample to the track: median "
    f"{np.median(projection_distance):.2f} cm, max {projection_distance.max():.2f} cm"
)
print(
    f"samples in the gap ({gap_start:.0f} to {gap_start + w_edge_spacing[2]:.0f} cm): "
    f"{int(in_gap.sum())}"
)

# %% [markdown]
# The samples sit about as far from the track as the tracking noise, and none
# land in the gap. Near the center junction, where three edges meet, a noisy
# sample can be assigned to a neighboring edge for a moment. For real tracking
# with larger errors, `use_HMM=True` makes the assignment favor staying on the
# same edge over time.
#
# ### A track-graph environment
#
# To use the graph in a model, give it to `Environment` with the same
# `edge_order` and `edge_spacing`. The grid is then built along the track, so
# `fit_place_grid` does not need position. Pass a model the raw 2-D position;
# it linearizes it with the same graph.

# %%
w_environment = Environment(
    place_bin_size=2.0,
    track_graph=w_track_graph,
    edge_order=w_edge_order,
    edge_spacing=w_edge_spacing,
).fit_place_grid()

print(
    f"place_bin_centers_: shape {w_environment.place_bin_centers_.shape} (linear position)"
)
print(
    f"is_track_interior_: {w_environment.is_track_interior_.sum()} of "
    f"{w_environment.is_track_interior_.size} bins"
)
print(
    "bins outside the track (cm):",
    w_environment.place_bin_centers_[~w_environment.is_track_interior_, 0],
)

# %% [markdown]
# The bin in the gap is not part of the track interior, so no probability is
# ever assigned to it. `plot_grid` draws the 1-D layout with the bin edges:

# %%
fig, ax = plt.subplots(figsize=(10, 1.5), constrained_layout=True)
w_environment.plot_grid(ax=ax)
ax.set_xlabel("Linear position (cm)")
ax.set_title("W-track environment: 2 cm bins along the track")
plt.show()

# %% [markdown]
# `track_linearization` also has functions that build common mazes, such as
# `make_wtrack`, `make_circular_track` and `make_tmaze_track`; check their node
# numbering before writing an `edge_order`.
#
# ## Summary
#
# | Input | Type and shape |
# | --- | --- |
# | `position_time` | `(n_position_samples,)`, seconds |
# | `position` | `(n_position_samples,)` or `(n_position_samples, n_dims)`, e.g. cm |
# | `spike_times` (sorted) | list of `(n_spikes_i,)` arrays, one per neuron |
# | `spike_times` (clusterless) | list of `(n_spikes_i,)` arrays, one per electrode |
# | `spike_waveform_features` | list of `(n_spikes_i, n_features)` arrays, aligned with `spike_times` |
# | `time_edges` | `(n_bins + 1,)` uniform bin edges, seconds |
# | `Environment` | open field (`place_bin_size`) or track graph (`track_graph`, `edge_order`, `edge_spacing`) |
