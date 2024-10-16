"""Goodness of fit tools for clusterless likelihood models [1].

Adapted from https://github.com/YousefiLab/Marked-PointProcess-Goodness-of-Fit


References
----------
[1] Yousefi, A., Amidi, Y., Nazari, B., and Eden, Uri.T. (2020). Assessing Goodness-of-Fit in Marked Point Process Models of Neural Population Coding via Time and Rate Rescaling. Neural Computation 32, 2145–2186. 10.1162/neco_a_01321.

"""

from typing import Tuple

import numpy as np
import scipy


def interval_rescaling_transform(
    time: np.ndarray,
    electrode_spike_times: np.ndarray,
    electrode_spike_waveform_features: np.ndarray,
    ground_process_intensity: np.ndarray,
    joint_mark_intensity: np.ndarray,
    permute_waveform_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale the interspike intervals and mark intensities for a single electrode.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    electrode_spike_times : np.ndarray, (n_spikes,)
    electrode_spike_waveform_features : np.ndarray, shape (n_spikes, n_waveform_features)
    ground_process_intensity : np.ndarray, shape (n_time,)
    joint_mark_intensity : np.ndarray, shape (n_spikes, n_waveform_features)

    Returns
    -------
    uniform_rescaled_ground_process_isi : np.ndarray, shape (n_spikes,)
    uniform_conditional_mark_intensity : np.ndarray, shape (n_spikes, n_waveform_features)
    """
    # Rescale the interspike intervals across all observed spikes based on the ground intensity
    rescaled_ground_process_isi = _compute_rescaled_isi(
        ground_process_intensity, electrode_spike_times, time
    )
    uniform_rescaled_ground_process_isi = scipy.stats.expon.cdf(
        rescaled_ground_process_isi
    )

    # Rescale each mark dimension sequentially using a Rosenblatt transformation
    # based on the conditional mark distribution given the spike time
    conditional_mark_intensity = joint_mark_intensity / ground_process_intensity
    n_features = joint_mark_intensity.shape[1]
    feature_indices = (
        np.random.permutation(n_features)
        if permute_waveform_features
        else np.arange(n_features)
    )
    conditional_mark_intensity = conditional_mark_intensity[:, feature_indices]
    uniform_conditional_mark_intensity = rosenblatt_transform(
        conditional_mark_intensity
    )

    return uniform_rescaled_ground_process_isi, uniform_conditional_mark_intensity


def _compute_rescaled_isi(
    intensity: np.ndarray, spike_times: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Compute the rescaled interspike intervals for a single electrode.

    Parameters
    ----------
    intensity : np.ndarray, shape (n_time,)
    spike_times : np.ndarray, shape (n_spikes,)

    Returns
    -------
    rescaled_isi : np.ndarray, shape (n_spikes - 1,)
    """
    integrated_conditional_intensity = scipy.integrate.cumulative_trapezoid(
        intensity, initial=0.0
    )
    ici_at_spike = np.interp(spike_times, time, integrated_conditional_intensity)
    ici_at_spike = np.concatenate((np.array([0]), ici_at_spike))
    return np.diff(ici_at_spike)


def empirical_cdf(sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the empirical CDF of a sample.

    Parameters
    ----------
    sample : np.ndarray, shape (n_samples,)

    Returns
    -------
    x : np.ndarray, shape (n_unique_samples,)
    cdf : np.ndarray, shape (n_unique_samples,)
    """
    sample = np.sort(sample)
    x, counts = np.unique(sample, return_counts=True)
    cdf = np.cumsum(counts) / sample.size

    return x, cdf


def rosenblatt_transform(samples: np.ndarray) -> np.ndarray:
    """Apply the Rosenblatt transformation to a sample.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_dims)

    Returns
    -------
    transformed_samples : np.ndarray, shape (n_samples, n_dims)
    """
    dim = samples.shape[1]
    transformed_samples = np.zeros_like(samples)

    # Note: this works for independent samples, but not for dependent samples
    # Need to numerically integrat conditional CDFs for dependent samples

    for i in range(dim):
        sorted_samples, cdf_values = empirical_cdf(samples[:, i])
        transformed_samples[:, i] = np.interp(samples[:, i], sorted_samples, cdf_values)

    return transformed_samples


def mark_conditional_intensity_transform(spike_waveform_features):
    # mark intensity is marginalize over time
    pass


def pearson_chi_squared_test():
    pass


def ks_test():
    pass


def distance_to_boundary_test():
    pass


def discrepency_test():
    pass


def rippley_stats_test():
    pass


def minimal_spanning_tree_test():
    pass
