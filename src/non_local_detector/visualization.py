import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_non_local_model(
    position_time,
    position,
    spike_times,
    speed,
    detector,
    results,
    figsize=(20, 10),
    time_slice=None,
    posterior_max=0.25,
):
    if time_slice is None:
        time_slice = slice(results.time.values[0], results.time.values[-1])

    place_fields = detector.encoding_model_[("", 0)]["place_fields"]
    env = detector.environments[0]
    state_ind = detector.state_ind_
    state_names = detector.state_names
    acausal_state_probabilities = results.sel(
        time=time_slice
    ).acausal_state_probabilities.values
    acausal_posterior = results.sel(time=time_slice).acausal_posterior.values
    results_time = results.sel(time=time_slice).time.values

    _, axes = plt.subplots(
        4,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1, 3, 1]},
    )

    t, x = np.meshgrid(results_time, env.place_bin_centers_)

    non_local_inds = np.nonzero(
        ["Non-Local" in state for state in detector.state_names]
    )[0]
    conditional_non_local_acausal_posterior = np.zeros(
        (len(results_time), len(env.place_bin_centers_))
    )
    for non_local_ind in non_local_inds:
        conditional_non_local_acausal_posterior += acausal_posterior[
            :, state_ind == non_local_ind
        ]
    conditional_non_local_acausal_posterior /= np.nansum(
        conditional_non_local_acausal_posterior, axis=1
    )[:, np.newaxis]
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

    neuron_sort_ind = np.argsort(
        env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
    )
    new_spike_times = [
        spike_times[neuron_id][
            np.logical_and(
                spike_times[neuron_id] >= time_slice.start,
                spike_times[neuron_id] <= time_slice.stop,
            )
        ]
        for neuron_id in neuron_sort_ind
    ]
    axes[0].eventplot(new_spike_times)
    axes[0].set_ylabel("Neuron")

    h = axes[1].plot(results_time, acausal_state_probabilities)
    axes[1].legend(h, state_names)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim((0.0, 1.05))

    cmap = copy.copy(matplotlib.colormaps["bone_r"])
    cmap.set_bad(color="lightgrey")

    axes[2].pcolormesh(
        t,
        x,
        conditional_non_local_acausal_posterior.T,
        vmin=0.0,
        vmax=posterior_max,
        cmap=cmap,
    )
    is_valid_position_time = np.logical_and(
        position_time >= time_slice.start, position_time <= time_slice.stop
    )
    axes[2].scatter(
        position_time[is_valid_position_time],
        position[is_valid_position_time],
        s=1,
        color="magenta",
        zorder=2,
    )
    axes[2].set_ylabel("Position [cm]")
    axes[3].fill_between(
        position_time[is_valid_position_time],
        speed[is_valid_position_time],
        color="lightgrey",
        zorder=2,
    )
    axes[3].set_ylabel("Speed\n[cm / s]")
    plt.xlim((time_slice.start, time_slice.stop))
    plt.xlabel("Time [ms]")
