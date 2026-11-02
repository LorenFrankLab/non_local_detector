import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from non_local_detector.environment import gaussian_smooth as _gaussian_smooth
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.models import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)


def get_multiunit_firing_rate(
    spike_times: list[np.ndarray], time: np.ndarray, smoothing_sigma: float = 0.015
) -> pd.DataFrame:
    """Calculate the multiunit firing rate.

    Parameters
    ----------
    spike_times : list[np.ndarray]
        Spike times for each neuron.
    time : np.ndarray, shape (n_time,)
        Row timestamps, e.g. decode bin centers (``results.time``) or position
        sample times; at least two, increasing. Each row counts the spikes
        closer to it than to its neighbours, with half an interval beyond each
        end. Rates use each row's duration. Gaussian smoothing uses the median
        sample interval, which approximates smoothing in time for irregular rows.
    smoothing_sigma : float, optional
        Standard deviation of the Gaussian smoothing, by default 0.015

    Returns
    -------
    multiunit_firing_rate : pd.DataFrame

    Raises
    ------
    ValidationError
        If fewer than two timestamps are given: one timestamp does not
        determine a row duration, so no rate can be computed.
    """
    time = np.asarray(time)
    if time.ndim != 1 or time.shape[0] < 2:
        msg = "get_multiunit_firing_rate needs at least two timestamps"
        raise ValidationError(
            msg,
            expected="array with shape (n_time,), n_time >= 2",
            got=f"array with shape {time.shape}",
            hint="A single timestamp does not determine a row duration. Pass "
            "the decode bin centers (results.time) or position sample times.",
        )
    row_edges = np.concatenate(
        [
            [time[0] - 0.5 * (time[1] - time[0])],
            0.5 * (time[:-1] + time[1:]),
            [time[-1] + 0.5 * (time[-1] - time[-2])],
        ]
    )
    spike_indicator = np.stack(
        [
            get_spikecount_per_time_bin(
                np.asarray(unit_spike_times), time_edges=row_edges
            )
            for unit_spike_times in spike_times
        ],
        axis=1,
    )

    sampling_frequency = 1 / np.nanmedian(np.diff(time))

    multiunit_firing_rate = _gaussian_smooth(
        spike_indicator.sum(axis=1) / np.diff(row_edges),
        smoothing_sigma,
        sampling_frequency,
    )

    return pd.DataFrame(
        multiunit_firing_rate,
        index=time,
        columns=["firing_rate"],
    )


def plot_non_local_model(
    position_time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    speed: np.ndarray,
    detector: NonLocalClusterlessDetector | NonLocalSortedSpikesDetector,
    results: xr.Dataset,
    figsize: tuple[int, int] = (20, 10),
    time_slice: slice | None = None,
    posterior_max: float = 0.25,
) -> None:
    """Plot the results of a non-local model.

    Parameters
    ----------
    position_time : np.ndarray
    position : np.ndarray
    spike_times : list[np.ndarray]
    speed : np.ndarray
    detector : NonLocalClusterlessDetector | NonLocalSortedSpikesDetector
    results : xr.Dataset
    figsize : tuple[int, int], optional
    time_slice : slice | None, optional
    posterior_max : float, optional
    """
    if time_slice is None:
        time_slice = slice(results.time.values[0], results.time.values[-1])

    env = detector.environments[0]
    try:
        place_fields = detector.encoding_model_[("", 0)]["place_fields"]
        neuron_sort_ind = np.argsort(
            env.place_bin_centers_[np.nanargmax(place_fields, axis=1)].squeeze()
        )
        cell_label = "Neuron"
    except KeyError:
        neuron_sort_ind = np.arange(len(spike_times))
        cell_label = "Electrode\nGroup"

    state_names = detector.state_names
    acausal_state_probabilities = results.sel(
        time=time_slice
    ).acausal_state_probabilities.values
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

    conditional_non_local_acausal_posterior = (
        detector.get_conditional_non_local_posterior(results.sel(time=time_slice))
        .values.reshape(len(results_time), -1)
        .copy()
    )
    conditional_non_local_acausal_posterior[:, ~env.is_track_interior_] = np.nan

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
    axes[0].set_ylabel(cell_label)

    h = axes[1].plot(results_time, acausal_state_probabilities)
    axes[1].legend(h, state_names)
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim((0.0, 1.05))

    cmap = copy.copy(mpl.colormaps["bone_r"])
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
