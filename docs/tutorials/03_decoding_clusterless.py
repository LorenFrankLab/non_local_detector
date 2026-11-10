# %% [markdown]
# # Decoding a session with clusterless spikes
#
# Spike sorting assigns each spike to a putative neuron by clustering its
# waveform. It is slow, needs manual curation, and throws away spikes that do
# not fall cleanly into a cluster. **Clusterless decoding** skips it: the
# decoder works directly with every spike detected on each electrode and with
# a few features of its waveform, such as the peak amplitude on each channel of
# a tetrode. These features are called **marks**.
#
# The idea is that the marks carry the information sorting would have used:
# spikes from the same neuron have similar amplitudes, and that neuron fires
# at a particular place. So instead of a place field per neuron, the encoding
# model learns, for each electrode, how the joint density of
# (position, marks) depends on position. At decoding time each spike's marks
# say which "part" of that density it came from, and therefore where the
# animal is likely to be.
#
# In this tutorial we:
#
# 1. simulate a session of clusterless data from tetrodes,
# 2. look at why the marks carry position information,
# 3. fit a `ClusterlessDecoder` on the running periods,
# 4. decode the **entire session** and compare the decoded position with the
#    true position, and
# 5. switch the clusterless likelihood algorithm.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from non_local_detector import ClusterlessDecoder, Environment, time_edges_from_centers
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

# On CPU, JAX cannot reuse the memory of arrays the decoder hands back to it
# ("donated buffers") and warns each time; the results are unaffected.
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

# %% [markdown]
# ## Simulate a session
#
# `make_simulated_run_data` simulates a rat running back and forth on a 175 cm
# linear track for about six minutes, with position sampled at 1,000 Hz. Twenty
# place cells are recorded on five tetrodes, four cells per tetrode. Each spike
# has four marks, one per tetrode channel. Each cell's spikes have marks
# scattered around a value of its own, spaced 10 units apart on a tetrode, as
# if each cell had a different amplitude on the wires.
#
# The simulator returns a `ClusterlessSimOutput`, with the inputs the decoder
# needs:
#
# - `position_time`: `(n_time,)` sample times in seconds,
# - `position`: `(n_time, 1)` position in cm,
# - `spike_times`: one array of spike times (s) per electrode,
# - `spike_waveform_features`: one `(n_spikes, n_features)` array of marks per
#   electrode.
#
# Real data has the same layout: one entry per electrode (tetrode, or group of
# nearby probe channels), not per neuron.

# %%
sim = make_simulated_run_data(seed=0)

position_time = sim.position_time  # (n_time,) s
position = sim.position  # (n_time, 1) cm
spike_times = sim.spike_times  # list of (n_spikes,) arrays, one per electrode
spike_waveform_features = sim.spike_waveform_features  # list of (n_spikes, 4)

sampling_frequency = 1 / np.median(np.diff(position_time))
print(f"Session length: {position_time[-1] - position_time[0]:.0f} s")
print(f"Position sampling rate: {sampling_frequency:.0f} Hz")
print(f"Number of electrodes: {len(spike_times)}")
for electrode, (times, marks) in enumerate(
    zip(spike_times, spike_waveform_features, strict=True)
):
    print(f"  tetrode {electrode}: {times.size:5d} spikes, marks shape {marks.shape}")

# %% [markdown]
# Here is the position over the session and, for each tetrode, the amplitude
# of every spike on its first channel. Within a tetrode the amplitudes fall in
# bands, one per cell, and each band appears only when the animal passes
# through part of the track.

# %%
n_electrodes = len(spike_times)
fig, axes = plt.subplots(
    n_electrodes + 1, 1, figsize=(10, 9), sharex=True, constrained_layout=True
)
axes[0].plot(position_time, position[:, 0], color="black", linewidth=1)
axes[0].set_ylabel("Position\n(cm)")
for electrode, ax in enumerate(axes[1:]):
    ax.scatter(
        spike_times[electrode],
        spike_waveform_features[electrode][:, 0],
        s=1,
        color="tab:blue",
        rasterized=True,
    )
    ax.set_ylabel(f"Tetrode {electrode}\namplitude\nchannel 1")
axes[-1].set_xlabel("Time (s)")
axes[-1].set_xlim(position_time[0], position_time[-1])

# %% [markdown]
# ## Why the marks carry position information
#
# Take one tetrode and plot each spike's mark against the animal's position at
# the time of the spike (left). Each cell is a blob: a narrow range of
# amplitudes at a particular place. This tetrode's cells cover only the first
# 30 cm of the track (fields at 0, 9, 18 and 28 cm); the other tetrodes cover
# the rest. Plotting two channels against each other (right) shows the
# clusters that spike sorting would cut out; the color, the position at the
# spike, shows that each cluster corresponds to one place on the track. The
# clusterless decoder uses this joint structure directly, without ever drawing
# cluster boundaries.

# %%
electrode = 0
times = spike_times[electrode]
marks = spike_waveform_features[electrode]
position_at_spike = np.interp(times, position_time, position[:, 0])

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].scatter(position_at_spike, marks[:, 0], s=2, alpha=0.3, color="black")
axes[0].set_xlabel("Position at spike (cm)")
axes[0].set_xlim(0, 175)
axes[0].set_ylabel("Amplitude, channel 1")
axes[0].set_title(f"Tetrode {electrode}: mark vs. position")

points = axes[1].scatter(
    marks[:, 0], marks[:, 1], c=position_at_spike, s=2, cmap="viridis"
)
axes[1].set_xlabel("Amplitude, channel 1")
axes[1].set_ylabel("Amplitude, channel 2")
axes[1].set_title(f"Tetrode {electrode}: channel 1 vs. channel 2")
fig.colorbar(points, ax=axes[1], label="Position at spike (cm)")

# %% [markdown]
# ## Select the running periods
#
# Place-cell firing reflects the animal's current position mostly during
# running. When the animal is still, hippocampal cells also fire in replay and
# other non-local events, which would blur an encoding model. So, as in a real
# analysis, we fit the encoding model on running periods only, here when speed
# is above 4 cm/s. In this simulation the animal slows to a stop as it turns
# around at each end of the track (the plot shows the first minute).

# %%
speed = np.abs(np.gradient(position[:, 0], position_time))  # cm/s
is_running = speed > 4.0

print(f"Fraction of the session spent running: {is_running.mean():.2f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, constrained_layout=True)
axes[0].plot(position_time, position[:, 0], color="black", linewidth=1)
axes[0].set_ylabel("Position (cm)")
axes[1].plot(position_time, speed, color="black", linewidth=1)
axes[1].fill_between(
    position_time,
    0,
    speed.max(),
    where=~is_running,
    color="gold",
    alpha=0.5,
    label="not running (excluded from fit)",
)
axes[1].set_ylabel("Speed (cm/s)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="upper right")
axes[1].set_xlim(0, 60)

# %% [markdown]
# ## Fit the decoder
#
# `ClusterlessDecoder` has a single state, **Continuous**: the decoded
# position moves from one time bin to the next by a Gaussian random walk
# (`RandomWalk`, default `movement_var=6.0` cm² per time bin). The choices we
# make are:
#
# - **Environment.** `Environment(place_bin_size=2.0)` discretizes the track
#   into 2 cm bins. With 1-D position and no track graph the track is treated
#   as a 1-D open field, which is fine for a straight track. See [Core concepts](../../getting-started/core-concepts/#environments)
#   for mazes.
# - **Likelihood algorithm.** `clusterless_algorithm="clusterless_kde"` (the
#   default) estimates the joint density of position and marks with a Gaussian
#   kernel density estimate.
# - **Kernel widths.** `clusterless_algorithm_params` sets the KDE's
#   bandwidths: `position_std` in cm (default `sqrt(12.5)` ≈ 3.5 cm) and
#   `waveform_std` in the units of the marks (default 24.0, on the scale of
#   spike amplitudes in µV). The simulated marks are not in µV: each cell's
#   marks have a spread of 1 unit and neighboring cells are 10 units apart, so we use
#   `waveform_std=1.0`. **Choose `waveform_std` on the scale of your own
#   marks**; too wide and the cells of a tetrode blur into one, too narrow and
#   the estimate gets noisy.
#
# `fit` takes the position, the spikes and their marks, and `is_training`,
# which limits the encoding model to the running samples.

# %%
decoder = ClusterlessDecoder(
    environments=Environment(place_bin_size=2.0),
    sampling_frequency=sampling_frequency,  # one decode bin per position sample
    clusterless_algorithm="clusterless_kde",
    clusterless_algorithm_params={
        "position_std": np.sqrt(12.5),  # cm
        "waveform_std": 1.0,  # mark units
    },
)
decoder.fit(
    position_time,
    position,
    spike_times,
    spike_waveform_features,
    is_training=is_running,
)

# %% [markdown]
# ## Decode the whole session
#
# Decoding works on time bins defined by their edges. Here we use one 1 ms bin
# per position sample, `time_edges_from_centers(position_time)`, covering the
# entire session, running and stopped alike: the encoding model was fit on
# running only, but we decode everything, as you would in a real analysis.
# `predict` needs only the spikes and marks; position is optional (the
# decoder's single state does not use it).
#
# This session is six minutes and decodes in seconds. For hour-long sessions,
# see [Checkpointed prediction](../../performance_prediction/) and
# [Settings for your hardware](../../hardware_settings/).

# %%
time_edges = time_edges_from_centers(position_time)
results = decoder.predict(
    spike_times,
    spike_waveform_features,
    time_edges=time_edges,
)
print(results)

# %% [markdown]
# `results` is an xarray `Dataset` with one row per time bin. Its
# `acausal_posterior` is the probability of each position bin given all the
# spikes in the session, before and after that time (the smoother). The decoder
# has one state, so we unstack the `state_bins` index and drop the state to get
# a `(time, position)` array.

# %%
posterior = results.acausal_posterior.unstack("state_bins").sum("state")
decoded_time = results.time.to_numpy()
place_bin_centers = posterior.position.to_numpy()
print(f"Posterior shape (time bins, position bins): {posterior.shape}")


def plot_posterior(ax, posterior, time_slice):
    """Plot a (time, position) posterior and the true position in a time window."""
    window = posterior.sel(time=time_slice)
    in_window = (position_time >= time_slice.start) & (position_time <= time_slice.stop)
    bin_size = np.diff(place_bin_centers).mean()
    image = ax.imshow(
        window.to_numpy().T,
        origin="lower",
        aspect="auto",
        cmap="bone_r",
        vmin=0.0,
        vmax=0.1,
        extent=(
            window.time[0].item(),
            window.time[-1].item(),
            place_bin_centers[0] - bin_size / 2,
            place_bin_centers[-1] + bin_size / 2,
        ),
        interpolation="none",
    )
    ax.plot(
        position_time[in_window],
        position[in_window, 0],
        color="magenta",
        linestyle="--",
        linewidth=1,
        label="true position",
    )
    ax.set_ylabel("Position (cm)")
    return image


fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
image = plot_posterior(axes[0], posterior, slice(position_time[0], position_time[-1]))
axes[0].set_title("Whole session")
plot_posterior(axes[1], posterior, slice(100.0, 130.0))
axes[1].set_title("Zoom: 100-130 s")
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="upper right")
fig.colorbar(image, ax=axes, label="Posterior probability", extend="max")

# %% [markdown]
# The posterior (dark is high probability) follows the true position (magenta)
# throughout the session, including the stops at the ends of the track, which
# the encoding model never saw.
#
# ## Decoding error
#
# To summarize the decoding, take the most probable position in each bin (the
# maximum a posteriori, or MAP, estimate) and compare it with the true
# position. We report the error separately for running and for the stops.

# %%
map_position = place_bin_centers[posterior.argmax("position").to_numpy()]
true_position = np.interp(decoded_time, position_time, position[:, 0])
decoding_error = np.abs(map_position - true_position)  # cm
# One decoding bin per position sample, so is_running also selects bins.

print(f"Median decoding error, whole session: {np.median(decoding_error):.1f} cm")
print(
    f"Median decoding error, running:       {np.median(decoding_error[is_running]):.1f} cm"
)
print(
    f"Median decoding error, not running:   {np.median(decoding_error[~is_running]):.1f} cm"
)

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
error_bins = np.arange(0, 20.5, 1.0)
ax.hist(
    decoding_error[is_running],
    bins=error_bins,
    density=True,
    alpha=0.6,
    label="running",
)
ax.hist(
    decoding_error[~is_running],
    bins=error_bins,
    density=True,
    alpha=0.6,
    label="not running",
)
ax.set_xlabel("|MAP - true position| (cm)")
ax.set_ylabel("Fraction of time bins per cm")
ax.set_title("Distribution of decoding error")
ax.legend()

# %% [markdown]
# Errors of a few cm are close to the 2 cm bin size and the 3.5 cm position
# kernel. The animal stops only at the ends of the track, and there the
# posterior spreads over the first few bins (the position kernel is 3.5 cm
# wide and there is no track beyond the end), so the MAP estimate tends to sit
# a bin or so away from the end: that is the slightly larger error when not
# running. With real data, expect larger errors: real marks overlap more between
# cells and real place fields are less regular.
#
# ## Switching the likelihood algorithm
#
# The encoding model is chosen with `clusterless_algorithm`; everything else
# stays the same. For example, `"clusterless_gmm"` replaces the kernel density
# estimate with Gaussian mixture models of occupancy and of (position, marks).
# Each algorithm takes its own parameters in `clusterless_algorithm_params` (a
# key the algorithm does not accept raises an error); with
# `clusterless_algorithm_params=None` it uses its defaults. The
# [Core concepts](../../getting-started/core-concepts/#likelihood-models) page
# lists all four clusterless algorithms.

# %%
gmm_decoder = ClusterlessDecoder(
    environments=Environment(place_bin_size=2.0),
    sampling_frequency=sampling_frequency,
    clusterless_algorithm="clusterless_gmm",
)
gmm_decoder.fit(
    position_time,
    position,
    spike_times,
    spike_waveform_features,
    is_training=is_running,
)
gmm_results = gmm_decoder.predict(
    spike_times, spike_waveform_features, time_edges=time_edges
)
gmm_posterior = gmm_results.acausal_posterior.unstack("state_bins").sum("state")
gmm_map_position = place_bin_centers[gmm_posterior.argmax("position").to_numpy()]
gmm_error = np.abs(gmm_map_position - true_position)
print(f"clusterless_kde median error: {np.median(decoding_error):.1f} cm")
print(f"clusterless_gmm median error: {np.median(gmm_error):.1f} cm")

fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
image = plot_posterior(ax, gmm_posterior, slice(100.0, 130.0))
ax.set_xlabel("Time (s)")
ax.set_title("clusterless_gmm posterior, 100-130 s")
ax.legend(loc="upper right")
fig.colorbar(image, ax=ax, label="Posterior probability", extend="max")

# %% [markdown]
# On this simulation both algorithms decode the position well. Which one suits
# your data depends on the number of spikes and mark dimensions; see the
# [Clusterless model fit guide](../../likelihood_mark_calibration/) for how to
# check a fitted clusterless model.
#
# ## Summary
#
# - Clusterless data is one array of spike times and one array of marks per
#   electrode; no spike sorting is needed.
# - `ClusterlessDecoder.fit` learns the joint density of position and marks,
#   here from the running periods only (`is_training`).
# - `predict` decodes every time bin given by `time_edges`, here the whole
#   session.
# - `clusterless_algorithm` picks the encoding model and
#   `clusterless_algorithm_params` its parameters; `waveform_std` must match
#   the scale of your marks.
#
# The next tutorial classifies the dynamics of the decoded position, to tell
# apart replay that sweeps smoothly along the track from replay that jumps.
