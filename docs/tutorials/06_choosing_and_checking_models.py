# %% [markdown]
# # Choosing a likelihood model and checking the fit
#
# The likelihood model is the encoding model: how the spikes depend on
# position. `non_local_detector` has several, for sorted and for clusterless
# spikes. This tutorial fits them to the same simulated session, compares
# their place fields with the true ones and their decoding on held-out data,
# and then uses `non_local_detector.model_checking` to check whether a fitted
# model describes the data, on a well-specified fit and on deliberately
# misspecified ones, so you can see what a bad fit looks like.

# %%
import time as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from non_local_detector import (
    ClusterlessDecoder,
    RandomWalk,
    SortedSpikesDecoder,
    calculate_time_edges,
)
from non_local_detector.model_checking import (
    TimeRescaling,
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from non_local_detector.model_checking.clusterless import (
    ircm_transform,
    ks_uniformity_test,
)
from non_local_detector.model_checking.posterior_consistency import (
    posterior_consistency_hpd_overlap,
    posterior_consistency_kl_divergence,
)

rng = np.random.default_rng(1)

# %% [markdown]
# ## Simulate a session
#
# A simulated rat runs 30 laps on a 180 cm linear track, at a different
# speed (30 to 60 cm/s) on each lap, with 1 to 3 s pauses at the ends. Forty
# place cells have Gaussian place fields with standard deviations of 4 to
# 12 cm and peak rates of 10 to 30 Hz, and fire at 0.1 Hz outside their
# fields. The spikes are Poisson, simulated on a 2 ms grid with spike times
# uniform within each step, and the position is sampled at 30 Hz.
#
# The cells are also recorded on ten tetrodes, four cells each, for the
# clusterless models: each spike has four amplitudes, the cell's mean
# amplitude on each channel (40 to 300 µV) plus 10 µV of noise.

# %%
dt = 0.002  # simulation time step, s
track_length = 180.0  # cm
n_laps = 30


def run_between(start: float, stop: float, peak_speed: float) -> np.ndarray:
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


track_ends = [0.0, track_length]
pieces = []
for lap in range(n_laps):
    here = track_ends[lap % 2]
    pieces.append(np.full(round(rng.uniform(1.0, 3.0) / dt), here))
    pieces.append(run_between(here, track_ends[(lap + 1) % 2], rng.uniform(30.0, 60.0)))
# A whole number of 0.1 s, so the 2 ms grid and the 30 Hz camera end together.
n_bins = np.concatenate(pieces).size // 50 * 50
true_position = np.concatenate(pieces)[:n_bins]  # cm, one per 2 ms step
bin_start = np.arange(n_bins) * dt  # s
session_duration = n_bins * dt

n_cells = 40
place_field_centers = np.sort(rng.uniform(0.0, track_length, n_cells))  # cm
place_field_widths = rng.uniform(4.0, 12.0, n_cells)  # Gaussian SD, cm
peak_rates = rng.uniform(10.0, 30.0, n_cells)  # Hz


def true_rates(position: np.ndarray) -> np.ndarray:
    """Firing rate (Hz) of every cell at each position, (n_positions, n_cells)."""
    distance = (position[:, np.newaxis] - place_field_centers) / place_field_widths
    return 0.1 + peak_rates * np.exp(-0.5 * distance**2)


spike_counts = rng.poisson(true_rates(true_position) * dt)  # (n_bins, n_cells)
spike_times = []
for cell in range(n_cells):
    bins_with_spikes = np.repeat(np.arange(n_bins), spike_counts[:, cell])
    spike_times.append(
        np.sort(bin_start[bins_with_spikes] + rng.uniform(0, dt, bins_with_spikes.size))
    )

# Tetrodes: four cells each, four channels of amplitudes per spike.
n_tetrodes = 10
cell_tetrode = np.arange(n_cells) // 4
mean_amplitudes = rng.uniform(40.0, 300.0, size=(n_cells, 4))  # µV
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

position_time = np.arange(round(session_duration * 30) + 1) / 30.0  # s, 30 Hz
position = np.interp(position_time, bin_start + dt / 2, true_position)  # cm
print(
    f"{session_duration:.0f} s session, "
    f"{sum(times.size for times in spike_times):,} spikes from {n_cells} cells"
)

# %% [markdown]
# ## Hold out data for testing
#
# A model always looks better on the data it was fitted to: a kernel density
# estimate, for example, partly reproduces its own training spikes. So every
# comparison and check below uses data the models did not see. The models
# are fitted on the first two-thirds of the session and tested on the last
# third, decoded in 2 ms bins.

# %%
test_start = round(session_duration * 2 / 3, 1)  # s
is_training = position_time < test_start

test_edges = calculate_time_edges([test_start, position_time[-1]], 500.0)
test_time = test_edges[:-1] + np.diff(test_edges) / 2  # bin centers, s
test_position = np.interp(test_time, position_time, position)  # cm
test_spike_times = [times[times >= test_start] for times in spike_times]
test_spike_counts = np.stack(
    [np.histogram(times, test_edges)[0] for times in test_spike_times], axis=1
)  # (n_test_bins, n_cells)
print(
    f"Training: {test_start:.0f} s. Test: {test_edges[-1] - test_start:.0f} s, "
    f"{test_edges.size - 1:,} bins of 2 ms"
)

# %% [markdown]
# ## The likelihood models
#
# Choose a model with `sorted_spikes_algorithm` (or `clusterless_algorithm`)
# and set its parameters with `sorted_spikes_algorithm_params` (or
# `clusterless_algorithm_params`); see
# [Core concepts](../getting-started/core-concepts.md#likelihood-models).
#
# | Algorithm | Place-field estimate | Main parameters (default) |
# | --- | --- | --- |
# | `sorted_spikes_kde` (default) | rate = mean rate × spike density / occupancy density, each a Gaussian kernel density estimate | `position_std`, the kernel SD (√12.5 ≈ 3.5 cm) |
# | `sorted_spikes_glm` | Poisson GLM on a cubic spline basis over position | `emission_knot_spacing` (2√12.5 ≈ 7.1 cm) |
# | `sorted_spikes_diffusion` | like the KDE, but smoothed by heat diffusion on the environment's graph, so smoothing does not cross walls or jump between track arms | `position_std` (√12.5 cm) |
# | `sorted_spikes_mrf` | penalized Poisson model on the smoothest eigenmodes of the environment graph, with the smoothness chosen by REML | `rank` (up to 250 eigenmodes), `penalty` (chosen by REML) |
# | `clusterless_kde` (default) | kernel density estimate over position and waveform features | `position_std` (√12.5 cm), `waveform_std` (24, in feature units) |
# | `clusterless_gmm` | Gaussian mixture over position and waveform features | `gmm_components_joint` (64), `gmm_components_occupancy` (32) |
# | `clusterless_diffusion` | the clusterless KDE with graph diffusion over position | `position_std`, `waveform_std` |
# | `clusterless_mark_glm` | mixture over waveform features with a Poisson GLM rate map per component | `n_mark_components` (32, an upper bound), `emission_knot_spacing` |
#
# The parameters are those of each algorithm's fit function, for example
# `non_local_detector.likelihoods.sorted_spikes_kde.fit_sorted_spikes_kde_encoding_model`
# in the API reference. A dict you pass replaces the algorithm's defaults,
# and an unknown key raises an error.
#
# ## Fit four sorted-spike models
#
# For decoding running data, the simplest model is `SortedSpikesDecoder`:
# one state in which the position follows a random walk. The encoding model
# is the same as in the detectors, so what we learn here carries over.

# %%
sorted_algorithms = [
    "sorted_spikes_kde",
    "sorted_spikes_glm",
    "sorted_spikes_diffusion",
    "sorted_spikes_mrf",
]


def fit_and_decode(**decoder_params):
    """Fit a SortedSpikesDecoder on the training data and decode the test data."""
    decoder = SortedSpikesDecoder(**decoder_params)
    start = timer.perf_counter()
    decoder.fit(position_time, position, spike_times, is_training=is_training)
    fit_seconds = timer.perf_counter() - start
    results = decoder.predict(
        test_spike_times, time_edges=test_edges, return_outputs="log_likelihood"
    )
    return decoder, results, fit_seconds


fits = {
    algorithm: fit_and_decode(sorted_spikes_algorithm=algorithm)
    for algorithm in sorted_algorithms
}

# %% [markdown]
# The decoder warns that 1-D position without a track graph is treated as a
# 1-D open field, which is right for a straight track. JAX may also warn
# that "Some donated buffers were not usable": a note about reusing memory
# in the compiled filter, which does not change the results.
#
# Each fitted encoding model stores the place fields in Hz at every position
# bin, `encoding_model_[...]["place_fields"]`, shape `(n_cells, n_bins)`.

# %%
environment = fits["sorted_spikes_kde"][0].environments[0]
is_interior = environment.is_track_interior_.ravel()
bin_centers = environment.place_bin_centers_[is_interior, 0]  # cm
bin_edges = environment.place_bin_edges_.ravel()  # cm


def place_fields(decoder):
    """Fitted place fields (Hz) on the track's bins, (n_cells, n_interior_bins)."""
    encoding_model = next(iter(decoder.encoding_model_.values()))
    return np.asarray(encoding_model["place_fields"])[:, is_interior]


example_cells = [
    np.argmin(place_field_widths),
    np.argsort(place_field_widths)[n_cells // 2],
    np.argmax(place_field_widths),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, constrained_layout=True)
for ax, cell in zip(axes, example_cells, strict=True):
    ax.plot(
        bin_centers,
        true_rates(bin_centers)[:, cell],
        color="lightgray",
        linewidth=5,
        label="true",
    )
    for (algorithm, (decoder, _, _)), linestyle in zip(
        fits.items(), ["-", "-", "--", "-"], strict=True
    ):
        ax.plot(
            bin_centers,
            place_fields(decoder)[cell],
            linestyle=linestyle,
            label=algorithm,
        )
    ax.set(
        xlabel="Position (cm)",
        title=f"Cell {cell}: field SD {place_field_widths[cell]:.1f} cm",
        xlim=(
            place_field_centers[cell] - 5 * place_field_widths[cell],
            place_field_centers[cell] + 5 * place_field_widths[cell],
        ),
    )
axes[0].set_ylabel("Firing rate (Hz)")
axes[-1].legend(frameon=False, fontsize=8)

# %% [markdown]
# To compare the models, three numbers on the held-out data:
#
# - **place-field error**: the mean absolute difference between the fitted
#   and the true rates, over cells and position bins;
# - **decoding error**: the distance between the most likely decoded
#   position and the rat's position, in each 2 ms bin;
# - **marginal log-likelihood** of the test spikes under the whole model
#   (`results.attrs["marginal_log_likelihoods"]`): higher is better, and
#   differences of a few nats matter.


# %%
def most_likely_position(results):
    """The most likely decoded position (cm) in each time bin."""
    posterior = results.acausal_posterior.sel(state="Continuous")
    return posterior.position.to_numpy()[posterior.argmax("position").to_numpy()]


def summarize(decoder, results):
    """Place-field error, decoding error and marginal log-likelihood."""
    decoding_error = np.abs(most_likely_position(results) - test_position)
    return {
        "place-field error (Hz)": np.mean(
            np.abs(place_fields(decoder) - true_rates(bin_centers).T)
        ),
        "median decoding error (cm)": np.median(decoding_error),
        "mean decoding error (cm)": np.mean(decoding_error),
        "marginal log-likelihood": results.attrs["marginal_log_likelihoods"],
    }


comparison = pd.DataFrame(
    {
        algorithm: {**summarize(decoder, results), "fit time (s)": fit_seconds}
        for algorithm, (decoder, results, fit_seconds) in fits.items()
    }
).T
print(comparison.astype(float).round(2).to_string())

# %% [markdown]
# On a well-sampled straight track all four agree closely: the place fields
# are close to the truth and the decoding errors are similar. The plots show
# their smoothing: the KDE and diffusion fields coincide (on a straight
# track the heat kernel is a Gaussian of the same width, dashed), and with
# a √12.5 cm kernel both flatten the narrowest field; the GLM follows it more
# closely, and the MRF, whose smoothness is chosen from the data, is the
# most wiggly. The models
# differ where the data are harder: the diffusion and MRF models smooth
# along the environment's graph, which matters in 2-D arenas with walls and
# on mazes with junctions, and the GLM and MRF fits regularize rates where
# the rat spent little time. When the choice matters, the checks below and
# held-out comparisons like this one are how to decide.
#
# Here is the decoded position over part of the test period, for the KDE
# model:

# %%
kde_results = fits["sorted_spikes_kde"][1]
window = slice(0, 10_000)  # first 20 s of the test period
fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
mesh = ax.pcolormesh(
    test_edges[window.start : window.stop + 1],
    bin_edges,
    kde_results.acausal_posterior.sel(state="Continuous")[window].to_numpy().T,
    cmap="bone_r",
    vmin=0,
    vmax=0.2,
)
ax.plot(test_time[window], test_position[window], color="tab:red", linewidth=1)
ax.set(xlabel="Time (s)", ylabel="Position (cm)")
fig.colorbar(mesh, label="Posterior probability")

# %% [markdown]
# ## Check the encoding model: time rescaling
#
# A good decoder can still rest on a poor encoding model, so check the
# encoding model directly. The time-rescaling theorem (Brown et al., 2002)
# says that if a cell's spikes come from a Poisson process with rate
# λ(t), the integrated rate between consecutive spikes is exponential with
# mean 1. `TimeRescaling` transforms these rescaled intervals to be uniform
# on [0, 1]; a Kolmogorov-Smirnov (KS) plot compares their distribution with
# the uniform one, and a correct model stays inside the 95% band.
#
# The model's rate for a cell at time t is its place field at the rat's
# position at t. Below, `position_std=20` cm, far wider than any of the
# simulated fields, is the misspecified model. Because this is a simulation,
# the true rates give the reference: a model can do no better.

# %%
too_wide = fit_and_decode(
    sorted_spikes_algorithm="sorted_spikes_kde",
    sorted_spikes_algorithm_params={"position_std": 20.0},
)


def time_rescaling(fields_at_test_position):
    """One TimeRescaling per cell from rates (Hz), (n_test_bins, n_cells)."""
    return [
        TimeRescaling(
            fields_at_test_position[:, cell],
            test_spike_counts[:, cell] > 0,
            rate_units="Hz",
            time_edges=test_edges,
        )
        for cell in range(n_cells)
    ]


def rates_at_test_position(decoder):
    """A decoder's place fields at the rat's test positions, (n_test_bins, n_cells)."""
    fields = place_fields(decoder)
    return np.stack(
        [np.interp(test_position, bin_centers, field) for field in fields], axis=1
    )


rescaled = {
    "true rates": time_rescaling(true_rates(test_position)),
    "KDE, default position_std": time_rescaling(
        rates_at_test_position(fits["sorted_spikes_kde"][0])
    ),
    "KDE, position_std = 20 cm": time_rescaling(rates_at_test_position(too_wide[0])),
}


def ks_ratio(check):
    """KS statistic over the 95% band half-width, 1.36 / sqrt(n_spikes)."""
    return check.ks_statistic() / (1.36 / np.sqrt(check.n_spikes))


# The cell with the most test spikes, for a narrow confidence band.
cell = np.argmax(test_spike_counts.sum(axis=0))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
for ax, (name, checks) in zip(axes, rescaled.items(), strict=True):
    checks[cell].plot_ks(ax=ax, scatter_kwargs={"s": 4, "color": "black"})
    ax.set_title(
        f"{name}\ncell {cell}: KS / band = {ks_ratio(checks[cell]):.2f}", fontsize=10
    )

# %% [markdown]
# With the true rates and with the default KDE, the rescaled intervals of
# this cell stay inside the band (the KS statistic over the band's
# half-width, in the titles, is below 1). With the too-wide kernel they bow
# away from the diagonal and cross the band: the model spreads the cell's
# rate over too much of the track, so it expects spikes where the cell is
# silent and too few where it fires.
#
# Across all cells, about 5% should fail by chance even under the true
# model. With `time_edges`, `TimeRescaling` places each spike at the end of
# its 2 ms bin; the true rates show that this approximation does not distort
# the test here.

# %%
for algorithm in sorted_algorithms[1:]:
    rescaled[algorithm] = time_rescaling(rates_at_test_position(fits[algorithm][0]))


ks_ratios = pd.DataFrame(
    {name: [ks_ratio(check) for check in checks] for name, checks in rescaled.items()}
)
print(
    pd.DataFrame(
        {
            "median KS / band": ks_ratios.median(),
            "cells outside the band": (ks_ratios > 1).sum(),
        }
    )
    .round(2)
    .to_string()
)

# %% [markdown]
# ## Check the posterior: HPD coverage and posterior consistency
#
# The other half of the model is the dynamics: how the represented position
# moves between time bins. For `SortedSpikesDecoder` that is a random walk
# whose variance per time bin is `movement_var` (cm²; 6 by default). Two
# checks of the posterior test it:
#
# - **Highest posterior density (HPD) coverage.** The 95% HPD region is the
#   smallest set of positions holding 95% of the posterior.
#   `get_highest_posterior_threshold` gives its threshold in each time bin,
#   and `get_HPD_spatial_coverage` its size in cm. On held-out running data,
#   a well-calibrated posterior's 95% region contains the rat's position in
#   about 95% of bins: much less means overconfident, more means
#   conservative.
# - **Posterior consistency.** The posterior combines the likelihood of the
#   current bin's spikes with the dynamics. If the dynamics are wrong, the
#   posterior disagrees with the likelihood.
#   `posterior_consistency_kl_divergence` measures the divergence between
#   the two, and `posterior_consistency_hpd_overlap` the overlap of their HPD
#   regions. A 2 ms bin without spikes has an almost flat likelihood, so we
#   look at the bins with spikes.
#
# Here are the KDE model with the default random walk, a much smaller
# variance (0.1 cm²) and a much larger one (100 cm²):

# %%
dynamics_fits = {
    0.1: fit_and_decode(continuous_transition_types=[[RandomWalk(movement_var=0.1)]]),
    6.0: fits["sorted_spikes_kde"],
    100.0: fit_and_decode(
        continuous_transition_types=[[RandomWalk(movement_var=100.0)]]
    ),
}
true_bin = np.clip(
    np.searchsorted(bin_edges, test_position) - 1, 0, bin_centers.size - 1
)
has_spikes = test_spike_counts.sum(axis=1) > 0

rows = {}
for movement_var, (decoder, results, _) in dynamics_fits.items():
    posterior = results.acausal_posterior.sel(state="Continuous")
    posterior_array = posterior.to_numpy()
    hpd_threshold = get_highest_posterior_threshold(posterior_array, coverage=0.95)
    in_hpd = posterior_array[np.arange(true_bin.size), true_bin] >= hpd_threshold

    # Normalize the likelihood over position in each bin, like the posterior.
    log_likelihood = results.log_likelihood.to_numpy()
    likelihood = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    likelihood /= likelihood.sum(axis=1, keepdims=True)

    rows[f"movement_var = {movement_var:g}"] = {
        "median decoding error (cm)": summarize(decoder, results)[
            "median decoding error (cm)"
        ],
        "95% HPD coverage": in_hpd.mean(),
        "median HPD size (cm)": np.median(
            get_HPD_spatial_coverage(posterior, hpd_threshold)
        ),
        "median KL divergence": np.median(
            posterior_consistency_kl_divergence(
                posterior_array[has_spikes], likelihood[has_spikes]
            )
        ),
        "mean HPD overlap": np.mean(
            posterior_consistency_hpd_overlap(
                posterior_array[has_spikes], likelihood[has_spikes]
            )
        ),
        "marginal log-likelihood": results.attrs["marginal_log_likelihoods"],
    }
print(pd.DataFrame(rows).astype(float).round(3).to_string())

# %%
fig, axes = plt.subplots(
    3, 1, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True
)
window = slice(0, 5_000)  # first 10 s of the test period
for ax, (movement_var, (_, results, _)) in zip(
    axes, dynamics_fits.items(), strict=True
):
    ax.pcolormesh(
        test_edges[window.start : window.stop + 1],
        bin_edges,
        results.acausal_posterior.sel(state="Continuous")[window].to_numpy().T,
        cmap="bone_r",
        vmin=0,
        vmax=0.2,
    )
    ax.plot(test_time[window], test_position[window], color="tab:red", linewidth=1)
    ax.set(ylabel="Position (cm)", title=f"movement_var = {movement_var:g} cm²")
axes[-1].set_xlabel("Time (s)")

# %% [markdown]
# With too small a variance, the decoded position cannot keep up with the
# rat: the random walk barely moves from one 2 cm bin to the next, so the
# posterior stays behind and is confidently wrong. The coverage collapses,
# and the posterior disagrees with the likelihood (large KL divergence, small
# HPD overlap). With too large a variance, the posterior follows each
# bin's likelihood (small KL divergence) but is wide: the HPD region is
# large and the decoding error grows. The default sits in between and is a
# little conservative here; the marginal log-likelihood of the test data
# also ranks the three.

# %% [markdown]
# ## Clusterless models: checking the mark model
#
# Clusterless models describe each tetrode's spikes by their waveform
# features (marks) as well as by position. The time-rescaling idea extends
# to marks (Yousefi et al., 2020): `ircm_transform` maps each spike to a
# point whose first coordinate `u` rescales the interval since the previous
# spike, as above, and whose other coordinates `v` are the conditional CDFs
# of the spike's features given its position under the model. For a correct
# model all of them are uniform on [0, 1]; `ks_uniformity_test` tests each
# coordinate. [Checking clusterless model fit](../likelihood_mark_calibration.md)
# describes the transforms and the other tests in detail.
#
# We fit `ClusterlessDecoder` with `clusterless_kde` at its default
# `waveform_std` (24 µV here, as the features are in µV) and at 4 µV, and
# with `clusterless_gmm`.

# %%
clusterless_configs = {
    "clusterless_kde, waveform_std = 24": ("clusterless_kde", None),
    "clusterless_kde, waveform_std = 4": ("clusterless_kde", {"waveform_std": 4.0}),
    "clusterless_gmm": ("clusterless_gmm", None),
}
test_tetrode_spike_times = [times[times >= test_start] for times in tetrode_spike_times]
test_tetrode_features = [
    features[times >= test_start]
    for times, features in zip(tetrode_spike_times, tetrode_features, strict=True)
]
is_test_sample = position_time >= test_start

clusterless_fits = {}
for name, (algorithm, params) in clusterless_configs.items():
    decoder = ClusterlessDecoder(
        clusterless_algorithm=algorithm, clusterless_algorithm_params=params
    )
    decoder.fit(
        position_time,
        position,
        tetrode_spike_times,
        tetrode_features,
        is_training=is_training,
    )
    results = decoder.predict(
        test_tetrode_spike_times, test_tetrode_features, time_edges=test_edges
    )
    encoding_model = next(iter(decoder.encoding_model_.values()))
    points = [
        np.column_stack(
            ircm_transform(
                encoding_model,
                algorithm,
                tetrode,
                position_time[is_test_sample],
                position[is_test_sample],
                test_tetrode_spike_times[tetrode],
                test_tetrode_features[tetrode],
            )
        )
        for tetrode in range(n_tetrodes)
    ]
    clusterless_fits[name] = (results, points)

# %% [markdown]
# For each model: the decoding error; the number of tetrodes (of ten) whose
# `u`, or whose marks `v`, fail a KS test at the 5% level with a Bonferroni
# correction over the five coordinates; and, as a measure of how far the
# marks depart from uniform, the KS statistic of the worst feature, median
# over tetrodes (0 is perfect).

# %%
n_coordinates = 5  # u and four features
rows = {}
for name, (results, points) in clusterless_fits.items():
    decoding_error = np.abs(most_likely_position(results) - test_position)
    tests = [ks_uniformity_test(tetrode_points) for tetrode_points in points]
    p_values = np.array([test.p_value for test in tests])  # (n_tetrodes, 5)
    statistics = np.array([test.statistic for test in tests])
    fails = p_values < 0.05 / n_coordinates
    rows[name] = {
        "median decoding error (cm)": np.median(decoding_error),
        "mean decoding error (cm)": np.mean(decoding_error),
        "tetrodes failing u": fails[:, 0].sum(),
        "tetrodes failing v": fails[:, 1:].any(axis=1).sum(),
        "KS statistic of v": np.median(statistics[:, 1:].max(axis=1)),
    }
print(pd.DataFrame(rows).T.astype(float).round(3).to_string())
print(
    "Test spikes per tetrode: "
    f"{min(map(len, test_tetrode_spike_times))} to "
    f"{max(map(len, test_tetrode_spike_times))}"
)

# %%
fig, axes = plt.subplots(
    1, 3, figsize=(11, 3), sharex=True, sharey=True, constrained_layout=True
)
for ax, (name, (_, points)) in zip(axes, clusterless_fits.items(), strict=True):
    ax.hist(points[0][:, 1], bins=np.linspace(0, 1, 21), color="tab:gray")
    ax.axhline(len(points[0]) / 20, color="black", linestyle="--")
    ax.set(xlabel="v (first feature)", title=name)
axes[0].set_ylabel("Spikes, tetrode 0")

# %% [markdown]
# The rate component `u` passes on (nearly) every tetrode. The marks are
# rejected on most tetrodes under every model: with hundreds of test spikes
# per tetrode, the test resolves even the smoothing that a kernel density
# estimate needs. How far they depart differs, though. With the
# default `waveform_std`, the mark kernel (24 µV) is much wider than the
# spread of each cell's amplitudes (10 µV): the observed features fall near
# the centers of the model's too-wide mark clusters, so `v` crowds into the
# middle of each cluster's range of CDF values and avoids its edges (dashed
# line: uniform). The 4 µV kernel departs least.
#
# Yet the default decodes at least as well here. The mark tests ask whether
# the model describes each spike's features, not whether it decodes well. A
# rejection says where the model departs from the data (here, in its marks),
# not that the decoder is unusable; choose the bandwidth for decoding on
# held-out decoding.
#
# ## Summary
#
# - Compare likelihood models on **held-out** data: place fields, decoding
#   error and the marginal log-likelihood.
# - **Time rescaling** checks a sorted-spikes encoding model cell by cell; a
#   too-smooth place field fails it.
# - **HPD coverage** and **posterior consistency** check the posterior and
#   so the dynamics; too small a `movement_var` fails both.
# - **IRCM** (and MDCI) check a clusterless model's rate and mark components
#   separately; the coordinate that fails tells you which component is wrong.
