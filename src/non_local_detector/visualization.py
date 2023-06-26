import copy

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_non_local_model(
    time,
    position,
    spikes,
    speed,
    detector,
    results,
    figsize=(20, 5),
    time_slice=None,
    posterior_max=0.25,
):
    place_fields = detector.encoding_model_[("", 0)]["place_fields"]
    env = detector.environments[0]
    state_ind = detector.state_ind_
    state_names = detector.state_names
    acausal_state_probabilities = results.acausal_state_probabilities.values
    acausal_posterior = results.acausal_posterior.values

    if time_slice is None:
        time_slice = slice(0, len(time))

    _, axes = plt.subplots(
        4,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1, 3, 1]},
    )

    sliced_time = time[time_slice]

    t, x = np.meshgrid(sliced_time, env.place_bin_centers_)

    neuron_sort_ind = np.argsort(
        env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
    )
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice][:, neuron_sort_ind])

    non_local_inds = np.nonzero(
        ["Non-Local" in state for state in detector.state_names]
    )[0]
    conditional_non_local_acausal_posterior = np.zeros(
        (len(sliced_time), len(env.place_bin_centers_))
    )
    for non_local_ind in non_local_inds:
        conditional_non_local_acausal_posterior += acausal_posterior[
            time_slice, state_ind == non_local_ind
        ]
    conditional_non_local_acausal_posterior /= np.nansum(
        conditional_non_local_acausal_posterior, axis=1
    )[:, np.newaxis]
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    axes[0].scatter(sliced_time[spike_time_ind], neuron_ind, s=1)
    axes[0].set_ylabel("Neuron")

    h = axes[1].plot(sliced_time, acausal_state_probabilities[time_slice])
    axes[1].legend(h, state_names)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim((0.0, 1.05))

    cmap = copy.copy(plt.cm.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey")

    axes[2].pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        vmin=0.0,
        vmax=posterior_max,
        cmap=cmap,
    )
    axes[2].scatter(sliced_time, position[time_slice], s=1, color="magenta", zorder=2)
    axes[2].set_ylabel("Position [cm]")
    axes[3].fill_between(sliced_time, speed[time_slice], color="lightgrey", zorder=2)
    axes[3].set_ylabel("Speed [cm / s]")
    plt.xlim((sliced_time.min(), sliced_time.max()))
    plt.xlabel("Time [ms]")


def plot_non_local_likelihood_ratio(
    time_slice,
    log_likelihood,
    acausal_posterior,
    acausal_state_probabilities,
    place_fields,
    spikes,
    position,
    env,
    time,
    state_ind,
    state_names,
    figsize=(10, 10),
    posterior_max=0.25,
):
    likelihood = np.exp(log_likelihood[time_slice, state_ind == 2])
    spike_time_ind, neuron_ind = np.nonzero(spikes[time_slice, :])
    is_spike = np.zeros_like(time[time_slice], dtype=bool)
    is_spike[spike_time_ind] = True
    likelihood[~is_spike, :] = np.nan
    likelihood[:, ~env.is_track_interior_] = np.nan

    likelihood_ratio = np.exp(
        log_likelihood[time_slice, state_ind == 2]
        - log_likelihood[time_slice, state_ind == 0]
    )
    likelihood_ratio[:, ~env.is_track_interior_] = np.nan

    conditional_non_local_acausal_posterior = (
        acausal_posterior[time_slice, state_ind == 2]
        + acausal_posterior[time_slice, state_ind == 3]
    ) / (
        acausal_state_probabilities[time_slice, [2]]
        + acausal_state_probabilities[time_slice, [3]]
    )
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    neuron_place_bin = env.place_bin_centers_[
        np.nanargmax(place_fields, axis=0)
    ].squeeze()

    t, x = np.meshgrid(time[time_slice], env.place_bin_centers_)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )

    cmap = copy.deepcopy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="lightgrey")
    h = axes[0].pcolormesh(
        t, x, likelihood_ratio.T, norm=colors.LogNorm(vmin=1 / 10, vmax=10), cmap=cmap
    )
    plt.colorbar(h, ax=axes[0])
    axes[0].scatter(time[time_slice], position[time_slice], color="magenta", s=1)
    axes[0].scatter(
        time[time_slice][spike_time_ind],
        neuron_place_bin[neuron_ind],
        color="black",
        s=10,
    )

    cmap = copy.deepcopy(plt.get_cmap("bone_r"))
    cmap.set_bad(color="lightgrey")
    h = axes[1].pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        cmap=cmap,
        vmin=0.0,
        vmax=posterior_max,
    )
    plt.colorbar(h, ax=axes[1])
    axes[1].scatter(time[time_slice], position[time_slice], color="magenta", s=1)

    axes[2].plot(
        time[time_slice], acausal_state_probabilities[time_slice], label=state_names
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Prob.")
    axes[2].legend()
