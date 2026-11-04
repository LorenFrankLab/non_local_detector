"""Tools for evaluating the goodness of fit of a point process model.

References
----------
.. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank, L.M.
       (2002). The time-rescaling theorem and its application to neural
       spike train data analysis. Neural Computation 14, 325-346.
.. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method for
       application to short-trial spike train data. Neural Computation 15,
       2565-2576.
"""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import integrate
from scipy.signal import correlate
from scipy.stats import expon, norm


class TimeRescaling:
    """Evaluate the goodness of fit of a point process model.

    The fitted model is transformed into a unit rate Poisson process using the
    time-rescaling theorem [1]_.

    Parameters
    ----------
    conditional_intensity : np.ndarray, shape (n_time,)
        Expected counts per bin by default. With ``rate_units="Hz"``, rates
        in spikes per second at sample times or in bins defined by edges.
    is_spike : np.ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    trial_id : np.ndarray, shape (n_time,), optional
        The label identifying time point with trial. If `trial_id` is set
        to None, then all time points are treated as part of the same
        trial. Otherwise, the data will be grouped by trial.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [2]_. By default False.
    rate_units : {"expected_counts", "Hz"}, optional
        Existing expected-count callers retain integration at unit index
        spacing. Hz inputs require exactly one of `time` or `time_edges`.
        By default "expected_counts".
    time : np.ndarray, shape (n_time,), optional
        Sample times in seconds. Rates are integrated with the trapezoid
        rule; spikes occur at sample times, and each contiguous trial run
        starts at its first sample (zero initial integrated intensity).
    time_edges : np.ndarray, shape (n_time + 1,), optional
        Bin edges in seconds. Rates are constant in each bin, and spikes
        are approximated at the bin's closing edge. The first interval
        includes exposure from the trial's opening edge.

    References
    ----------
    .. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank,
           L.M. (2002). The time-rescaling theorem and its application to
           neural spike train data analysis. Neural Computation 14, 325-346.
    .. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.
    """

    def __init__(
        self,
        conditional_intensity: np.ndarray,
        is_spike: np.ndarray,
        trial_id: np.ndarray | None = None,
        adjust_for_short_trials: bool = False,
        *,
        rate_units: str = "expected_counts",
        time: np.ndarray | None = None,
        time_edges: np.ndarray | None = None,
    ):
        """Initialize the TimeRescaling object."""
        self.conditional_intensity = np.atleast_1d(
            np.asarray(conditional_intensity).squeeze()
        )
        if trial_id is None:
            trial_id = np.ones_like(self.conditional_intensity)
        self.trial_id = np.atleast_1d(np.asarray(trial_id).squeeze())
        self.is_spike = np.atleast_1d(np.asarray(is_spike).squeeze())
        self.adjust_for_short_trials = adjust_for_short_trials
        self.rate_units = rate_units
        self.time = None if time is None else np.asarray(time, dtype=float)
        self.time_edges = (
            None if time_edges is None else np.asarray(time_edges, dtype=float)
        )
        # Validate physical inputs before splitting into trials. Existing
        # index-based construction does not need to allocate an integral.
        if (
            not isinstance(rate_units, str)  # type: ignore[redundant-expr]  # defensive runtime check of a typed argument
            or rate_units != "expected_counts"
            or self.time is not None
            or self.time_edges is not None
        ):
            _integrated_conditional_intensity(
                self.conditional_intensity,
                rate_units=rate_units,
                time=self.time,
                time_edges=self.time_edges,
            )
        if self.trial_id.shape != self.conditional_intensity.shape:
            msg = "trial_id must match conditional_intensity rows"
            raise ValueError(msg)
        if self.is_spike.shape != self.conditional_intensity.shape:
            msg = "is_spike must match conditional_intensity rows"
            raise ValueError(msg)

    @property
    def n_spikes(self) -> int:
        """Total number of spikes.

        Returns
        -------
        n_spikes : int
            Number of time points with a spike.
        """
        return np.nonzero(self.is_spike)[0].size

    def uniform_rescaled_ISIs(self) -> np.ndarray:
        """Rescale the interspike intervals (ISIs) to a uniform distribution.

        ISIs are rescaled to unit rate Poisson, adjusted for short time
        intervals if requested, and transformed to a uniform distribution for
        easier analysis.

        Returns
        -------
        uniform_rescaled_ISIs : ndarray, shape (n_spikes,)
            Rescaled ISIs, uniform on [0, 1] if the model fits well.
        """
        trial_IDs = np.unique(self.trial_id)
        uniform_rescaled_ISIs_by_trial = []
        for trial in trial_IDs:
            indices = np.flatnonzero(np.isin(self.trial_id, trial))
            # Interrupted trials have separate physical observation windows.
            # Retain the existing grouping behavior for index-based callers.
            runs = (
                np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                if self.rate_units == "Hz"
                else [indices]
            )
            for run in runs:
                time = None if self.time is None else self.time[run]
                edges = (
                    None
                    if self.time_edges is None
                    else self.time_edges[run[0] : run[-1] + 2]
                )
                uniform_rescaled_ISIs_by_trial.append(
                    uniform_rescaled_ISIs(
                        self.conditional_intensity[run],
                        self.is_spike[run],
                        self.adjust_for_short_trials,
                        rate_units=self.rate_units,
                        time=time,
                        time_edges=edges,
                    )
                )

        return np.concatenate(uniform_rescaled_ISIs_by_trial)

    def ks_statistic(self) -> float:
        """Measure the maximum distance of the rescaled ISIs from unit rate Poisson.

        Smaller maximum distance means better fitting model.

        Returns
        -------
        ks_statistic : float
            Kolmogorov-Smirnov statistic between the sorted rescaled ISIs and
            the uniform CDF.
        """
        uniform_cdf_values = _uniform_cdf_values(self.n_spikes)
        return ks_statistic(np.sort(self.uniform_rescaled_ISIs()), uniform_cdf_values)

    def rescaled_ISI_autocorrelation(self) -> np.ndarray:
        """Examine rescaled ISI dependence.

        Should be independent if the transformation to unit rate Poisson
        process fits well.

        Returns
        -------
        rescaled_ISI_autocorrelation : ndarray, shape (2 * n_spikes - 1,)
            Autocorrelation of the rescaled ISIs (after mapping to a normal
            distribution), normalized so the maximum is 1.
        """
        # Avoid -inf and inf when transforming to normal distribution.
        u = self.uniform_rescaled_ISIs()
        u[u == 0] = np.finfo(float).eps
        u[u == 1] = 1 - np.finfo(float).eps

        normal_rescaled_ISIs = norm.ppf(u)

        c = correlate(normal_rescaled_ISIs, normal_rescaled_ISIs)
        return cast(np.ndarray, c / c.max())

    def plot_ks(
        self,
        ax: Axes | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        ci_color: str = "red",
    ) -> Axes:
        """Plot the empirical CDF versus the expected CDF.

        Shows how close the rescaled ISIs are to the unit rate Poisson.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, plots on the current axes.
        scatter_kwargs : dict, optional
            Plotting arguments for scatter plot.
        ci_color : str, optional
            Confidence interval color, by default "red".

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        return plot_ks(
            self.uniform_rescaled_ISIs(),
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            ci_color=ci_color,
        )

    def plot_qq(
        self,
        ax: Axes | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        ci_color: str = "red",
    ) -> Axes:
        """Plot the rescaled ISIs versus a uniform distribution.

        Shows how close the rescaled ISIs are to the unit rate Poisson.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, plots on the current axes.
        scatter_kwargs : dict, optional
            Plotting arguments for scatter plot.
        ci_color : str, optional
            Confidence interval color, by default "red".

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        return plot_qq(
            self.uniform_rescaled_ISIs(),
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            ci_color=ci_color,
        )

    def plot_rescaled_ISI_autocorrelation(
        self,
        ax: Axes | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        ci_color: str = "red",
        sampling_frequency: float = 1.0,
        lag_max: float | None = None,
    ) -> Axes:
        """Plot the rescaled ISI dependence.

        Should be independent if the transformation to unit rate Poisson
        process fits well.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, plots on the current axes.
        scatter_kwargs : dict, optional
            Plotting arguments for scatter plot.
        ci_color : str, optional
            Confidence interval color, by default "red".
        sampling_frequency : float, optional
            Sampling frequency of the data in Hz, by default 1.0.
        lag_max : float, optional
            Maximum lag to plot, in time units. If None, plots all lags.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        return plot_rescaled_ISI_autocorrelation(
            self.rescaled_ISI_autocorrelation(),
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            ci_color=ci_color,
            sampling_frequency=sampling_frequency,
            lag_max=lag_max,
        )


def _uniform_cdf_values(n_spikes: int) -> np.ndarray:
    """Return model based cumulative distribution function values.

    Used for plotting the `uniform_rescaled_ISIs`.

    Parameters
    ----------
    n_spikes : int
        Total number of spikes.

    Returns
    -------
    uniform_cdf_values : ndarray, shape (n_spikes,)
    """
    return (np.arange(n_spikes) + 0.5) / n_spikes


def ks_statistic(empirical_cdf: np.ndarray, model_cdf: np.ndarray) -> float:
    """Compare the empirical and model-based distributions.

    Uses the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    empirical_cdf : np.ndarray, shape (n_spikes,)
        Sorted empirical values (rescaled ISIs).
    model_cdf : np.ndarray, shape (n_spikes,)
        Model-based CDF values at the same quantiles.

    Returns
    -------
    ks_statistic : float
        Maximum absolute difference between the two; NaN if the arrays
        cannot be broadcast together.
    """
    try:
        return cast(float, np.max(np.abs(empirical_cdf - model_cdf)))
    except ValueError:
        return np.nan


def _rescaled_ISIs(
    integrated_conditional_intensity: np.ndarray, is_spike: np.ndarray
) -> np.ndarray:
    """Rescales the interspike intervals (ISIs) to unit rate Poisson.

    Parameters
    ----------
    integrated_conditional_intensity : np.ndarray, shape (n_time,)
        The cumulative conditional_intensity integrated over time.
    is_spike : bool np.ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.

    Returns
    -------
    rescaled_ISIs : ndarray, shape (n_spikes,)
    """
    ici_at_spike = integrated_conditional_intensity[is_spike.nonzero()]
    ici_at_spike = np.concatenate((np.array([0]), ici_at_spike))
    return np.diff(ici_at_spike)


def _max_transformed_interval(
    integrated_conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    rescaled_ISIs: np.ndarray,
) -> np.ndarray:
    """Weights for each time in censored trials.

    Parameters
    ----------
    integrated_conditional_intensity : ndarray, shape (n_time,)
        The cumulative conditional_intensity integrated over time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    rescaled_ISIs : ndarray, shape (n_spikes,)

    Returns
    -------
    max_transformed_interval : ndarray, shape (n_spikes,)
    """
    ici_at_spike = integrated_conditional_intensity[is_spike.nonzero()]
    return cast(
        np.ndarray,
        integrated_conditional_intensity[-1] - ici_at_spike + rescaled_ISIs,
    )


def _integrated_conditional_intensity(
    conditional_intensity: np.ndarray,
    *,
    rate_units: str,
    time: np.ndarray | None = None,
    time_edges: np.ndarray | None = None,
    legacy_residuals: bool = False,
) -> np.ndarray:
    """Integrate explicitly selected rate units without guessing a clock."""
    if not isinstance(rate_units, str) or rate_units not in ("expected_counts", "Hz"):  # type: ignore[redundant-expr]  # defensive runtime check of a typed argument
        msg = "rate_units must be 'expected_counts' or 'Hz'"
        raise ValueError(msg)
    # Integrate Hz rates in float64 before any addition or multiplication.
    # Preserve the original arithmetic for explicit legacy expected counts.
    intensity = np.asarray(
        conditional_intensity, dtype=float if rate_units == "Hz" else None
    )
    if rate_units == "expected_counts":
        if time is not None or time_edges is not None:
            msg = (
                "expected_counts uses the existing bin-index convention; "
                "physical time or time_edges requires rate_units='Hz'"
            )
            raise ValueError(msg)
        return (
            np.cumsum(intensity)
            if legacy_residuals
            else integrate.cumulative_trapezoid(intensity, initial=0.0)
        )
    if (time is None) == (time_edges is None):
        msg = "Hz inputs require exactly one of time or time_edges in seconds"
        raise ValueError(msg)
    if intensity.ndim != 1 or not intensity.size:
        msg = "conditional_intensity must have nonempty one-dimensional rows"
        raise ValueError(msg)
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
        msg = "conditional_intensity must be finite and nonnegative"
        raise ValueError(msg)
    grid = np.asarray(time if time is not None else time_edges, dtype=float)
    expected_size = len(intensity) + (time_edges is not None)
    if grid.ndim != 1 or grid.size != expected_size:
        shape = "n_time" if time is not None else "n_time + 1"
        msg = f"physical grid must have shape ({shape},)"
        raise ValueError(msg)
    with np.errstate(over="ignore", invalid="ignore"):
        durations = np.diff(grid)
    if (
        not np.all(np.isfinite(grid))
        or not np.all(np.isfinite(durations))
        or np.any(durations <= 0)
    ):
        msg = "physical time must be finite and strictly increasing"
        raise ValueError(msg)
    if time_edges is not None:
        return np.cumsum(intensity * durations)
    return cast(
        np.ndarray, integrate.cumulative_trapezoid(intensity, x=grid, initial=0.0)
    )


def uniform_rescaled_ISIs(
    conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    adjust_for_short_trials: bool = True,
    *,
    rate_units: str = "expected_counts",
    time: np.ndarray | None = None,
    time_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Rescale the interspike intervals (ISIs) to a uniform distribution.

    ISIs are rescaled to unit rate Poisson, adjusted for short time
    intervals if requested, and transformed to a uniform distribution for
    easier analysis.

    Parameters
    ----------
    conditional_intensity : ndarray, shape (n_time,)
        Expected counts per bin by default; rates in spikes per second when
        ``rate_units="Hz"``.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [1]_. By default True.
    rate_units : {"expected_counts", "Hz"}, optional
        ``expected_counts`` preserves the existing unit-index trapezoid.
        Hz requires exactly one of ``time`` or ``time_edges`` in seconds.
        By default "expected_counts".
    time : np.ndarray, shape (n_time,), optional
        Sample timestamps. Integrate rates by trapezoids from the first sample;
        spike indicators refer to events at their corresponding sample times.
    time_edges : np.ndarray, shape (n_time + 1,), optional
        Integrate piecewise constant rates over each bin. Spike indicators are
        approximated at closing edges; the first ISI starts at the first edge.
        Exact within-bin spike times require an event-time rescaling method.

    Returns
    -------
    uniform_rescaled_ISIs : ndarray, shape (n_spikes,)
        Rescaled ISIs, uniform on [0, 1] if the model fits well.

    References
    ----------
    .. [1] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.
    """
    integrated_conditional_intensity = _integrated_conditional_intensity(
        conditional_intensity, rate_units=rate_units, time=time, time_edges=time_edges
    )
    is_spike = np.asarray(is_spike)
    if is_spike.shape != integrated_conditional_intensity.shape:
        msg = "is_spike must match conditional_intensity rows"
        raise ValueError(msg)
    # Rescale the ISIs to unit rate Poisson: \Lambda(spike_k) - \Lambda(spike_{k-1})
    # These should be exponentially distributed with mean 1
    rescaled_ISIs = _rescaled_ISIs(integrated_conditional_intensity, is_spike)

    if adjust_for_short_trials:
        max_transformed_interval = expon.cdf(
            _max_transformed_interval(
                integrated_conditional_intensity, is_spike, rescaled_ISIs
            )
        )
    else:
        max_transformed_interval = 1

    # Transform the ISIs to a uniform distribution (1 - exp(-ISI))
    return cast(np.ndarray, expon.cdf(rescaled_ISIs) / max_transformed_interval)


def point_process_residuals(
    conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    *,
    rate_units: str = "expected_counts",
    time: np.ndarray | None = None,
    time_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the residuals of the point process model.

    Parameters
    ----------
    conditional_intensity : np.ndarray, shape (n_time,)
        Expected counts per bin by default; rates in spikes per second when
        ``rate_units="Hz"``.
    is_spike : np.ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    rate_units : {"expected_counts", "Hz"}, optional
        Existing callers use cumulative observed minus expected bin counts.
        Hz inputs require exactly one physical grid below.
    time : np.ndarray, shape (n_time,), optional
        Sample times in seconds; return cumulative observed events minus the
        trapezoid integral starting at the first sample time.
    time_edges : np.ndarray, shape (n_time + 1,), optional
        Bin edges in seconds; return residuals at closing edges using expected
        bin counts ``conditional_intensity * diff(time_edges)``.

    Returns
    -------
    residuals : np.ndarray, shape (n_time,)
        The residuals of the point process model.
    """
    integrated = _integrated_conditional_intensity(
        conditional_intensity,
        rate_units=rate_units,
        time=time,
        time_edges=time_edges,
        legacy_residuals=True,
    )
    is_spike = np.asarray(is_spike)
    if is_spike.shape != integrated.shape:
        msg = "is_spike must match conditional_intensity rows"
        raise ValueError(msg)
    if rate_units == "expected_counts":
        # Retain the original accumulation order for existing callers.
        return np.cumsum(is_spike - conditional_intensity)
    return cast(np.ndarray, np.cumsum(is_spike) - integrated)


def plot_ks(
    uniform_rescaled_ISIs: np.ndarray,
    ax: Axes | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ci_color: str = "red",
) -> Axes:
    """Plot the empirical CDF versus the expected CDF of the rescaled ISIs.

    Shows how close the rescaled ISIs are to the unit rate Poisson, with
    95% Kolmogorov-Smirnov confidence bounds.

    Parameters
    ----------
    uniform_rescaled_ISIs : np.ndarray, shape (n_spikes,)
        Rescaled interspike intervals that should be uniformly distributed
        if the model fits well.
    ax : Axes, optional
        Matplotlib axes object to plot on. If None, uses current axes.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to scatter plot.
    ci_color : str, optional
        Color for confidence interval lines, by default "red".

    Returns
    -------
    ax : Axes
        Matplotlib axes object containing the plot.
    """
    n_spikes = uniform_rescaled_ISIs.size
    uniform_cdf_values = (np.arange(1, n_spikes + 1) - 0.5) / n_spikes

    ci = 1.36 / np.sqrt(n_spikes)  # 95% confidence interval

    if ax is None:
        ax = plt.gca()

    if scatter_kwargs is None:
        scatter_kwargs = {}

    ax.plot(uniform_cdf_values, uniform_cdf_values - ci, linestyle="--", color=ci_color)
    ax.plot(uniform_cdf_values, uniform_cdf_values + ci, linestyle="--", color=ci_color)
    ax.plot([0, 1], [0, 1], linestyle="--", color=ci_color)
    ax.scatter(np.sort(uniform_rescaled_ISIs), uniform_cdf_values, **scatter_kwargs)

    ax.set_xlabel("Empirical CDF")
    ax.set_ylabel("Expected CDF")

    return ax


def plot_qq(
    uniform_rescaled_ISIs: np.ndarray,
    ax: Axes | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ci_color: str = "red",
) -> Axes:
    """Create a Q-Q plot of rescaled ISIs against uniform distribution.

    Plots the quantiles of the rescaled interspike intervals against the
    expected quantiles from a uniform distribution to assess goodness-of-fit
    of the point process model.

    Parameters
    ----------
    uniform_rescaled_ISIs : np.ndarray, shape (n_spikes,)
        Rescaled interspike intervals that should be uniformly distributed
        if the model fits well.
    ax : Axes, optional
        Matplotlib axes object to plot on. If None, uses current axes.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to scatter plot.
    ci_color : str, optional
        Color for confidence interval lines. Default is "red".

    Returns
    -------
    ax : Axes
        Matplotlib axes object containing the plot.

    Notes
    -----
    Points should lie approximately on the diagonal line if the rescaled
    ISIs follow a uniform distribution, indicating good model fit.
    """
    n_spikes = uniform_rescaled_ISIs.size
    uniform_quantiles = (np.arange(1, n_spikes + 1) - 0.5) / n_spikes
    sorted_ISIs = np.sort(uniform_rescaled_ISIs)

    if ax is None:
        ax = plt.gca()

    if scatter_kwargs is None:
        scatter_kwargs = {}

    ci = 1.96 * np.sqrt(sorted_ISIs * (1 - sorted_ISIs) / n_spikes)

    ax.plot([0, 1], [0, 1], linestyle="--", color=ci_color)
    ax.plot(sorted_ISIs, sorted_ISIs - ci, linestyle="--", color=ci_color)
    ax.plot(sorted_ISIs, sorted_ISIs + ci, linestyle="--", color=ci_color)
    ax.scatter(uniform_quantiles, sorted_ISIs, **scatter_kwargs)
    ax.set_xlabel("Empirical quantiles")
    ax.set_ylabel("Expected quantiles")

    return ax


def plot_rescaled_ISI_autocorrelation(
    rescaled_ISI_autocorrelation: np.ndarray,
    ax: Axes | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ci_color: str = "red",
    sampling_frequency: float = 1.0,
    lag_max: float | None = None,
) -> Axes:
    """Plot autocorrelation function of rescaled interspike intervals.

    Visualizes the temporal dependence in rescaled ISIs. For a well-fitting
    model, the autocorrelation should be near zero at all non-zero lags,
    indicating independence of rescaled intervals.

    Parameters
    ----------
    rescaled_ISI_autocorrelation : np.ndarray, shape (2*n_spikes-1,)
        Autocorrelation function of rescaled ISIs computed using correlation.
    ax : Axes, optional
        Matplotlib axes object to plot on. If None, uses current axes.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to scatter plot.
    ci_color : str, optional
        Color for confidence interval lines. Default is "red".
    sampling_frequency : float, optional
        Sampling frequency of the data in Hz. Default is 1.0.
    lag_max : float, optional
        Maximum lag to display in time units; lags with ``|lag| <= lag_max``
        are shown. If None, shows all lags.

    Returns
    -------
    ax : Axes
        Matplotlib axes object containing the plot.

    Notes
    -----
    Points outside the confidence interval lines suggest temporal dependence
    in the rescaled ISIs, indicating potential model misfit.
    """
    n_spikes = rescaled_ISI_autocorrelation.size // 2 + 1
    lag = np.arange(-n_spikes + 1, n_spikes) / sampling_frequency

    # Select actual lag values so fractional limits retain equal endpoints.
    lag_ind = slice(None) if lag_max is None else np.abs(lag) <= lag_max

    if ax is None:
        ax = plt.gca()
    if scatter_kwargs is None:
        scatter_kwargs = {}
    ci = 1.96 / np.sqrt(n_spikes)
    ax.scatter(lag[lag_ind], rescaled_ISI_autocorrelation[lag_ind], **scatter_kwargs)
    ax.axhline(ci, linestyle="--", color=ci_color)
    ax.axhline(-ci, linestyle="--", color=ci_color)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    return ax
