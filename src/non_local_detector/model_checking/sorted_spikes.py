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

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate  # type: ignore[import-untyped]
from scipy.signal import correlate  # type: ignore[import-untyped]
from scipy.stats import expon, norm  # type: ignore[import-untyped]


class TimeRescaling:
    """Evaluates the goodness of fit of a point process model by
    transforming the fitted model into a unit rate Poisson process [1].

    Attributes
    ----------
    conditional_intensity : ndarray, shape (n_time,)
        The fitted model mean response rate at each time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    trial_id : ndarray, shape (n_time,), optional
        The label identifying time point with trial. If `trial_id` is set
        to None, then all time points are treated as part of the same
        trial. Otherwise, the data will be grouped by trial.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [2].

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
    ):
        """Initialize the TimeRescaling object.

        Parameters
        ----------
        conditional_intensity : np.ndarray, shape (n_time,)
        is_spike : np.ndarray, shape (n_time,)
        trial_id : Optional[np.ndarray], shape (n_time,), optional
        adjust_for_short_trials : bool, optional
        """
        self.conditional_intensity = np.asarray(conditional_intensity).squeeze()
        if trial_id is None:
            trial_id = np.ones_like(self.conditional_intensity)
        self.trial_id = np.asarray(trial_id).squeeze()
        self.is_spike = np.asarray(is_spike).squeeze()
        self.adjust_for_short_trials = adjust_for_short_trials

    @property
    def n_spikes(self) -> int:
        """Number of total spikes."""
        return np.nonzero(self.is_spike)[0].size

    def uniform_rescaled_ISIs(self) -> np.ndarray:
        """Rescales the interspike intervals (ISIs) to unit rate Poisson,
        adjusts for short time intervals, and transforms the ISIs to a
        uniform distribution for easier analysis.

        Returns
        -------
        uniform_rescaled_ISIs : ndarray, shape (n_spikes,)
        """

        trial_IDs = np.unique(self.trial_id)
        uniform_rescaled_ISIs_by_trial = []
        for trial in trial_IDs:
            is_trial = np.in1d(self.trial_id, trial)
            uniform_rescaled_ISIs_by_trial.append(
                uniform_rescaled_ISIs(
                    self.conditional_intensity[is_trial],
                    self.is_spike[is_trial],
                    self.adjust_for_short_trials,
                )
            )

        return np.concatenate(uniform_rescaled_ISIs_by_trial)

    def ks_statistic(self) -> float:
        """Measures the maximum distance of the rescaled ISIs from the unit
        rate Poisson.

        Smaller maximum distance means better fitting model.

        Returns
        -------
        ks_statistic : float
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
        """
        # Avoid -inf and inf when transforming to normal distribution.
        u = self.uniform_rescaled_ISIs()
        u[u == 0] = np.finfo(float).eps
        u[u == 1] = 1 - np.finfo(float).eps

        normal_rescaled_ISIs = norm.ppf(u)

        c = correlate(normal_rescaled_ISIs, normal_rescaled_ISIs)
        return c / c.max()

    def plot_ks(
        self,
        ax: plt.Axes | None = None,
        scatter_kwargs: dict | None = None,
        ci_color: str = "red",
    ) -> plt.Axes:
        """Plots the empirical CDF versus the expected CDF to examine how
        close the rescaled ISIs are to the unit rate Poisson.

        Parameters
        ----------
        ax : matplotlib axis handle, optional
            If None, plots on the current axis handle.
        scatter_kwargs : None or dict
            Plotting arguments for scatter plot
        ci_color : str
            Confidence interval color

        Returns
        -------
        ax : axis_handle

        """
        return plot_ks(
            self.uniform_rescaled_ISIs(),
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            ci_color=ci_color,
        )

    def plot_qq(
        self,
        ax: plt.Axes | None = None,
        scatter_kwargs: dict | None = None,
        ci_color: str = "red",
    ) -> plt.Axes:
        """Plots the rescaled ISIs versus a uniform distribution to examine
        how close the rescaled ISIs are to the unit rate Poisson.

        Parameters
        ----------
        ax : matplotlib axis handle, optional
            If None, plots on the current axis handle.
        scatter_kwargs : None or dict
            Plotting arguments for scatter plot
        ci_color : str
            Confidence interval color

        Returns
        -------
        ax : axis_handle

        """
        return plot_qq(
            self.uniform_rescaled_ISIs(),
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            ci_color=ci_color,
        )

    def plot_rescaled_ISI_autocorrelation(
        self,
        ax: plt.Axes | None = None,
        scatter_kwargs: dict | None = None,
        ci_color: str = "red",
        sampling_frequency: float = 1.0,
        lag_max: float | None = None,
    ) -> plt.Axes:
        """Plot the rescaled ISI dependence.

        Should be independent if the transformation to unit rate Poisson
        process fits well.

        Parameters
        ----------
        ax : matplotlib axis handle, optional
            If None, plots on the current axis handle.
        scatter_kwargs : None or dict
            Plotting arguments for scatter plot
        ci_color : str
            Confidence interval color
        sampling_frequency : float
            Sampling frequency of the data
        lag_max : float, optional
            Maximum lag to plot. If None, plots all lags.

        Returns
        -------
        ax : axis_handle
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
    """Model based cumulative distribution function values. Used for
    plotting the `uniform_rescaled_ISIs`.

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
    """Compares the empirical and model-based distribution using the
    Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    empirical_cdf : np.ndarray, shape (n_spikes,)
    model_cdf : np.ndarray, shape (n_spikes,)

    Returns
    -------
    ks_statistic : float

    Raises
    ------
    ValueError
        If the arrays are not the same size.
    """
    try:
        return np.max(np.abs(empirical_cdf - model_cdf))
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
    return integrated_conditional_intensity[-1] - ici_at_spike + rescaled_ISIs


def uniform_rescaled_ISIs(
    conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    adjust_for_short_trials: bool = True,
) -> np.ndarray:
    """Rescales the interspike intervals (ISIs) to unit rate Poisson,
    adjusts for short time intervals, and transforms the ISIs to a
    uniform distribution for easier analysis.

    Parameters
    ----------
    conditional_intensity : ndarray, shape (n_time,)
        The fitted model mean response rate at each time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [1].

    Returns
    -------
    uniform_rescaled_ISIs : ndarray, shape (n_spikes,)

    References
    ----------
    .. [1] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.

    """
    try:
        integrated_conditional_intensity = integrate.cumulative_trapezoid(
            conditional_intensity, initial=0.0
        )
    except AttributeError:
        # Older versions of scipy
        integrated_conditional_intensity = integrate.cumtrapz(
            conditional_intensity, initial=0.0
        )
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
    return expon.cdf(rescaled_ISIs) / max_transformed_interval


def point_process_residuals(
    conditional_intensity: np.ndarray, is_spike: np.ndarray
) -> np.ndarray:
    """Compute the residuals of the point process model.

    Parameters
    ----------
    conditional_intensity : np.ndarray, shape (n_time,)
        The fitted model mean response rate at each time.
    is_spike : np.ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.

    Returns
    -------
    residuals : np.ndarray, shape (n_time,)
        The residuals of the point process model.
    """
    return np.cumsum(is_spike - conditional_intensity)


def plot_ks(
    uniform_rescaled_ISIs: np.ndarray,
    ax: plt.Axes | None = None,
    scatter_kwargs: dict | None = None,
    ci_color: str = "red",
) -> plt.Axes:
    """Plots the rescaled ISIs versus a uniform distribution to examine
    how close the rescaled ISIs are to the unit rate Poisson.

    Parameters
    ----------
    uniform_rescaled_ISIs : np.ndarray, shape (n_spikes,)
    ax : Optional[plt.Axes], optional
    scatter_kwargs : Optional[dict], optional
    ci_color : str, optional

    Returns
    -------
    ax : plt.Axes
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
    ax: plt.Axes | None = None,
    scatter_kwargs: dict | None = None,
    ci_color: str = "red",
) -> plt.Axes:
    """Create a Q-Q plot of rescaled ISIs against uniform distribution.

    Plots the quantiles of the rescaled interspike intervals against the
    expected quantiles from a uniform distribution to assess goodness-of-fit
    of the point process model.

    Parameters
    ----------
    uniform_rescaled_ISIs : np.ndarray, shape (n_spikes,)
        Rescaled interspike intervals that should be uniformly distributed
        if the model fits well.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, uses current axes.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to scatter plot.
    ci_color : str, optional
        Color for confidence interval lines. Default is "red".

    Returns
    -------
    ax : plt.Axes
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
    ax: plt.Axes | None = None,
    scatter_kwargs: dict | None = None,
    ci_color: str = "red",
    sampling_frequency: float = 1.0,
    lag_max: float | None = None,
) -> plt.Axes:
    """Plot autocorrelation function of rescaled interspike intervals.

    Visualizes the temporal dependence in rescaled ISIs. For a well-fitting
    model, the autocorrelation should be near zero at all non-zero lags,
    indicating independence of rescaled intervals.

    Parameters
    ----------
    rescaled_ISI_autocorrelation : np.ndarray, shape (2*n_spikes-1,)
        Autocorrelation function of rescaled ISIs computed using correlation.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, uses current axes.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to scatter plot.
    ci_color : str, optional
        Color for confidence interval lines. Default is "red".
    sampling_frequency : float, optional
        Sampling frequency of the data in Hz. Default is 1.0.
    lag_max : float, optional
        Maximum lag to display in time units. If None, shows all lags.

    Returns
    -------
    ax : plt.Axes
        Matplotlib axes object containing the plot.

    Notes
    -----
    Points outside the confidence interval lines suggest temporal dependence
    in the rescaled ISIs, indicating potential model misfit.
    """
    n_spikes = rescaled_ISI_autocorrelation.size // 2 + 1
    lag = np.arange(-n_spikes + 1, n_spikes) / sampling_frequency

    lag_max = n_spikes if lag_max is None else int(lag_max * sampling_frequency)
    lag_ind = slice(-lag_max + 1, lag_max)

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
