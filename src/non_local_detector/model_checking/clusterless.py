"""Goodness of fit tools for clusterless likelihood models [1].

Adapted from https://github.com/YousefiLab/Marked-PointProcess-Goodness-of-Fit


References
----------
[1] Yousefi, A., Amidi, Y., Nazari, B., and Eden, Uri.T. (2020). Assessing Goodness-of-Fit in Marked Point Process Models of Neural Population Coding via Time and Rate Rescaling. Neural Computation 32, 2145–2186. 10.1162/neco_a_01321.

"""

from collections.abc import Callable

import numpy as np
import scipy  # type: ignore[import-untyped]


def interval_rescaling_transform(
    time: np.ndarray,
    electrode_spike_times: np.ndarray,
    electrode_spike_waveform_features: np.ndarray,
    ground_process_intensity: np.ndarray,
    joint_mark_intensity: np.ndarray,
    permute_waveform_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale interspike intervals and mark intensities for goodness-of-fit testing.

    Applies the time and mark rescaling transformations to convert spike trains
    and their associated waveform features into uniform distributions if the
    fitted model is correct.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        Time points corresponding to the intensity functions.
    electrode_spike_times : np.ndarray, shape (n_spikes,)
        Times at which spikes occurred on this electrode.
    electrode_spike_waveform_features : np.ndarray, shape (n_spikes, n_waveform_features)
        Waveform feature matrix where each row corresponds to a spike.
    ground_process_intensity : np.ndarray, shape (n_time,)
        Fitted ground process (temporal) intensity function.
    joint_mark_intensity : np.ndarray, shape (n_spikes, n_waveform_features)
        Joint mark-time intensity function evaluated at spike times and features.
    permute_waveform_features : bool, optional
        Whether to randomly permute the order of waveform features during
        the Rosenblatt transformation. Default is False.

    Returns
    -------
    uniform_rescaled_ground_process_isi : np.ndarray, shape (n_spikes,)
        Rescaled interspike intervals transformed to uniform distribution.
        Should be uniform [0,1] if the temporal model fits well.
    uniform_conditional_mark_intensity : np.ndarray, shape (n_spikes, n_waveform_features)
        Rescaled mark intensities transformed to uniform distribution.
        Should be uniform [0,1] in each dimension if the mark model fits well.
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
        Ground process intensity function values at each time point.
    spike_times : np.ndarray, shape (n_spikes,)
        Times at which spikes occurred.
    time : np.ndarray, shape (n_time,)
        Time points corresponding to the intensity values.

    Returns
    -------
    rescaled_isi : np.ndarray, shape (n_spikes,)
        Rescaled interspike intervals computed by integrating the intensity
        function between consecutive spike times.
    """
    integrated_conditional_intensity = scipy.integrate.cumulative_trapezoid(
        intensity, initial=0.0
    )
    ici_at_spike = np.interp(spike_times, time, integrated_conditional_intensity)
    ici_at_spike = np.concatenate((np.array([0]), ici_at_spike))
    return np.diff(ici_at_spike)


def empirical_cdf(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical cumulative distribution function of a sample.

    Parameters
    ----------
    sample : np.ndarray, shape (n_samples,)
        Input data sample for which to compute the empirical CDF.

    Returns
    -------
    x : np.ndarray, shape (n_unique_samples,)
        Sorted unique values from the sample.
    cdf : np.ndarray, shape (n_unique_samples,)
        Empirical CDF values corresponding to each unique sample value.
        Values range from 0 to 1.
    """
    sample = np.sort(sample)
    x, counts = np.unique(sample, return_counts=True)
    cdf = np.cumsum(counts) / sample.size

    return x, cdf


def rosenblatt_transform(samples: np.ndarray) -> np.ndarray:
    """Apply the Rosenblatt transformation to convert samples to uniform distribution.

    The Rosenblatt transformation converts a multivariate sample to uniform
    marginals using the probability integral transformation. For independent
    variables, this reduces to applying the empirical CDF to each dimension.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_dims)
        Input samples to transform, where each row is a sample and each
        column is a dimension.

    Returns
    -------
    transformed_samples : np.ndarray, shape (n_samples, n_dims)
        Transformed samples with uniform marginal distributions.
        Values are in the range [0, 1].

    Notes
    -----
    This implementation works for independent samples but not for dependent
    samples. For dependent samples, conditional CDFs need to be numerically
    integrated.
    """
    dim = samples.shape[1]
    transformed_samples = np.zeros_like(samples)

    # Note: this works for independent samples, but not for dependent samples
    # Need to numerically integrat conditional CDFs for dependent samples

    for i in range(dim):
        sorted_samples, cdf_values = empirical_cdf(samples[:, i])
        transformed_samples[:, i] = np.interp(samples[:, i], sorted_samples, cdf_values)

    return transformed_samples


def mark_conditional_intensity_transform(
    spike_waveform_features: np.ndarray,
) -> np.ndarray:
    """Transform spike waveform features using conditional mark intensity.

    This function applies a conditional intensity transformation to spike waveform
    features, marginalizing over time to obtain the mark intensity.

    Parameters
    ----------
    spike_waveform_features : np.ndarray, shape (n_spikes, n_features)
        Spike waveform feature matrix where each row represents a spike and
        each column represents a waveform feature dimension.

    Returns
    -------
    transformed_features : np.ndarray, shape (n_spikes, n_features)
        Transformed waveform features after applying the conditional mark
        intensity transformation.

    Notes
    -----
    This function is not yet implemented. The mark intensity is marginalized
    over time to provide a goodness-of-fit test for the mark process component
    of a marked point process model.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    # mark intensity is marginalize over time
    raise NotImplementedError(
        "mark_conditional_intensity_transform is not yet implemented"
    )


def pearson_chi_squared_test(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    bins: np.ndarray | None = None,
) -> tuple[float, float]:
    """Perform Pearson's chi-squared goodness-of-fit test.

    Tests whether observed spike counts match the expected counts from a
    fitted point process model using Pearson's chi-squared statistic.

    Parameters
    ----------
    observed_counts : np.ndarray, shape (n_bins,)
        Observed spike counts in each bin.
    expected_counts : np.ndarray, shape (n_bins,)
        Expected spike counts from the fitted model in each bin.
    bins : np.ndarray, shape (n_bins + 1,), optional
        Bin edges for the histogram. If None, assumes uniform bins.

    Returns
    -------
    chi2_statistic : float
        Pearson's chi-squared test statistic.
    p_value : float
        P-value for the test statistic under the null hypothesis that
        the observed and expected distributions are the same.

    Notes
    -----
    This function is not yet implemented. The chi-squared test compares
    observed and expected frequencies to assess goodness-of-fit.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("pearson_chi_squared_test is not yet implemented")


def ks_test(
    rescaled_intervals: np.ndarray, distribution: str = "uniform"
) -> tuple[float, float]:
    """Perform Kolmogorov-Smirnov goodness-of-fit test.

    Tests whether rescaled interspike intervals follow the expected
    distribution (typically uniform for properly rescaled intervals).

    Parameters
    ----------
    rescaled_intervals : np.ndarray, shape (n_intervals,)
        Rescaled interspike intervals from the time rescaling transformation.
    distribution : str, optional
        Target distribution to test against. Default is "uniform".
        Options include "uniform", "exponential".

    Returns
    -------
    ks_statistic : float
        Kolmogorov-Smirnov test statistic (maximum distance between
        empirical and theoretical CDFs).
    p_value : float
        P-value for the test statistic under the null hypothesis that
        the data follows the specified distribution.

    Notes
    -----
    This function is not yet implemented. The KS test is used to assess
    whether rescaled intervals follow the expected uniform distribution,
    which would indicate good model fit.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("ks_test is not yet implemented")


def distance_to_boundary_test(
    spike_positions: np.ndarray,
    environment_boundaries: np.ndarray,
    test_statistic: str = "mean_distance",
) -> tuple[float, float]:
    """Test spatial distribution of spikes relative to environment boundaries.

    Assesses whether the spatial distribution of decoded spike positions
    shows appropriate relationship to environment boundaries, which can
    indicate model misspecification.

    Parameters
    ----------
    spike_positions : np.ndarray, shape (n_spikes, n_spatial_dims)
        Decoded or actual spike positions in spatial coordinates.
    environment_boundaries : np.ndarray, shape (n_boundary_points, n_spatial_dims)
        Coordinates defining the environment boundaries.
    test_statistic : str, optional
        Type of test statistic to compute. Default is "mean_distance".
        Options include "mean_distance", "min_distance", "variance".

    Returns
    -------
    test_statistic_value : float
        Computed test statistic value.
    p_value : float
        P-value under the null hypothesis of appropriate spatial
        distribution relative to boundaries.

    Notes
    -----
    This function is not yet implemented. The distance to boundary test
    can reveal systematic biases in spatial decoding near environment edges.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("distance_to_boundary_test is not yet implemented")


def discrepancy_test(
    observed_pattern: np.ndarray,
    simulated_patterns: np.ndarray,
    discrepancy_function: Callable | None = None,
) -> tuple[float, float]:
    """Perform discrepancy test for point process goodness-of-fit.

    Compares observed spike patterns to simulated patterns from the fitted
    model using a discrepancy function to assess model adequacy.

    Parameters
    ----------
    observed_pattern : np.ndarray, shape (pattern_dims,)
        Observed spike pattern or summary statistic.
    simulated_patterns : np.ndarray, shape (n_simulations, pattern_dims)
        Simulated spike patterns from the fitted model.
    discrepancy_function : Callable, optional
        Function to compute discrepancy between patterns. If None,
        uses squared difference. Should take two arrays and return a scalar.

    Returns
    -------
    discrepancy_statistic : float
        Discrepancy between observed and expected patterns.
    p_value : float
        P-value computed as the proportion of simulated patterns with
        discrepancy greater than or equal to the observed discrepancy.

    Notes
    -----
    This function is not yet implemented. Discrepancy tests provide a
    flexible framework for assessing model fit using various summary
    statistics of spike patterns.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("discrepancy_test is not yet implemented")


def ripley_stats_test(
    spike_positions: np.ndarray,
    radii: np.ndarray,
    environment_area: float,
    edge_correction: str = "isotropic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ripley's K and L statistics for spatial point pattern analysis.

    Analyzes the spatial clustering or regularity of spike positions using
    Ripley's K-function and its variance-stabilized L-function transformation.

    Parameters
    ----------
    spike_positions : np.ndarray, shape (n_spikes, 2)
        2D coordinates of spike positions in spatial environment.
    radii : np.ndarray, shape (n_radii,)
        Distance radii at which to compute the statistics.
    environment_area : float
        Total area of the spatial environment.
    edge_correction : str, optional
        Method for edge correction. Default is "isotropic".
        Options include "none", "border", "isotropic", "translate".

    Returns
    -------
    k_function : np.ndarray, shape (n_radii,)
        Ripley's K-function values at each radius.
    l_function : np.ndarray, shape (n_radii,)
        L-function values (variance-stabilized K-function).
    expected_l : np.ndarray, shape (n_radii,)
        Expected L-function values under complete spatial randomness.

    Notes
    -----
    This function is not yet implemented. Ripley's statistics help identify
    spatial clustering (L > expected) or regularity (L < expected) in
    point patterns, useful for validating spatial aspects of decoded positions.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("ripley_stats_test is not yet implemented")


def minimal_spanning_tree_test(
    spike_positions: np.ndarray, n_simulations: int = 1000
) -> tuple[float, float, float]:
    """Test spatial randomness using minimal spanning tree statistics.

    Analyzes the spatial distribution of spike positions by computing
    statistics of the minimal spanning tree and comparing to expectations
    under spatial randomness.

    Parameters
    ----------
    spike_positions : np.ndarray, shape (n_spikes, n_spatial_dims)
        Coordinates of spike positions in spatial environment.
    n_simulations : int, optional
        Number of Monte Carlo simulations for null distribution.
        Default is 1000.

    Returns
    -------
    observed_statistic : float
        Observed minimal spanning tree test statistic.
    expected_statistic : float
        Expected test statistic under spatial randomness.
    p_value : float
        P-value for the test of spatial randomness.

    Notes
    -----
    This function is not yet implemented. The minimal spanning tree test
    provides another approach to detect spatial clustering or regularity
    in point patterns by analyzing the total length and other properties
    of the minimal spanning tree connecting all points.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("minimal_spanning_tree_test is not yet implemented")
