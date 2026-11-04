"""Encoding weights at spike times and weighted mean firing rates."""

from typing import TYPE_CHECKING, cast

import jax
import numpy as np

if TYPE_CHECKING:
    from non_local_detector.encoding_time import EncodingSupport


def interpolate_weights_at_spike_times(
    spike_times: np.ndarray | jax.Array,
    position_time: np.ndarray | jax.Array,
    weights: np.ndarray | jax.Array,
    *,
    encoding_support: "EncodingSupport | None" = None,
) -> np.ndarray:
    """Per-spike weights: the per-sample ``weights`` linearly interpolated onto times.

    With ``encoding_support``, weights are interpolated within each of its
    segments and a spike outside every segment gets weight 0. Without it, spike
    times are assumed already clipped to ``[position_time[0], position_time[-1]]``
    and 1-D ``np.interp`` is used (it does not extrapolate). Shared by the
    clusterless and sorted-spikes encoding fits.

    Parameters
    ----------
    spike_times : np.ndarray, shape (n_spikes,)
        Spike times, in seconds.
    position_time : np.ndarray, shape (n_time_position,)
        Time of each position sample, in seconds.
    weights : np.ndarray, shape (n_time_position,)
        Per-position-sample weights.
    encoding_support : EncodingSupport, optional
        Encoding support whose segments bound the interpolation; by default
        None.

    Returns
    -------
    spike_weights : np.ndarray, shape (n_spikes,)
        Weight of each spike.
    """
    if encoding_support is not None:
        return encoding_support.interpolate(weights, spike_times, fill_value=0.0)
    return cast(
        np.ndarray,
        np.interp(
            np.asarray(spike_times), np.asarray(position_time), np.asarray(weights)
        ),
    )


def weighted_mean_rate(spike_weights: np.ndarray, weight_sum: float) -> float:
    """Weighted mean firing rate: weighted spike count / weighted occupancy time.

    Returns 0.0 when ``weight_sum`` is not positive (no effective training data), so a
    zero-weight electrode/group contributes nothing rather than dividing by zero.

    Parameters
    ----------
    spike_weights : np.ndarray, shape (n_spikes,)
        Per-spike weights (see :func:`interpolate_weights_at_spike_times`).
    weight_sum : float
        Sum of weighted sample exposure in seconds (not sample count).

    Returns
    -------
    mean_rate : float
        Weighted mean firing rate, in Hz.
    """
    return float(np.sum(spike_weights) / weight_sum) if weight_sum > 0 else 0.0
