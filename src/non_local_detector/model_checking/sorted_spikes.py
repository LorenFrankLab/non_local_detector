"""Tools for evaluating the goodness of fit of a point process model.

References
----------
.. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank, L.M.
       (2002). The time-rescaling theorem and its application to neural
       spike train data analysis. Neural Computation 14, 325-346.
.. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method for
       application to short-trial spike train data. Neural Computation 15,
       2565-2576.
.. [3] Haslinger, R., Pipa, G., and Brown, E. (2010). Discrete time
       rescaling theorem: determining goodness of fit for discrete time
       statistical models of neural spiking. Neural Computation 22(10),
       2477-2506.
"""

import warnings
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import integrate
from scipy.signal import correlate
from scipy.stats import expon, norm

from non_local_detector.exceptions import DataQualityWarning

__all__ = [
    "TimeRescaling",
    "ks_statistic",
    "plot_ks",
    "plot_qq",
    "plot_rescaled_ISI_autocorrelation",
    "point_process_residuals",
    "uniform_rescaled_ISIs",
]


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
        Whether the neuron spiked in each bin. With the discrete-time
        correction a count above one is a bin with a spike.
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
        Without the correction, expected counts are integrated by
        trapezoids at unit index spacing. Hz inputs require exactly one of
        `time` or `time_edges`. By default "expected_counts".
    time : np.ndarray, shape (n_time,), optional
        Sample times in seconds. Rates are integrated with the trapezoid
        rule, and each contiguous trial run starts at its first sample (zero
        initial integrated intensity). With the correction, a spike at
        sample ``k`` lies in ``(t_{k-1}, t_k]``; without it, at ``t_k``. A
        spike at a run's first sample then closes no interval in the run, so
        it is left out, with a `DataQualityWarning` giving the count.
    time_edges : np.ndarray, shape (n_time + 1,), optional
        Bin edges in seconds. Rates are constant in each bin. The first
        interval includes exposure from the trial's opening edge. Without
        the correction, spikes are placed at the bin's closing edge.
    discrete_time_correction : bool, optional
        Place each spike within its bin by the discrete-time rescaling of
        [3]_, so that a correct model of binned spikes passes the KS test at
        any bin width; see `uniform_rescaled_ISIs`. By default True.
    rng : numpy.random.Generator or int, optional
        Generator, or seed for one, for the correction's within-bin draws.
        They are drawn once, here, so every method of this object uses the
        same intervals. By default a fresh, unseeded generator: two objects
        built from the same data without a seed give different KS values.
        Pass a seed, e.g. ``rng=0``, to reproduce a result.

    Notes
    -----
    With the correction, the test checks whether each bin has a spike, not
    how many: within-bin count dispersion is not tested, so check counts with
    `point_process_residuals`. For a Bernoulli (e.g. logistic) model with
    spike probability ``p`` per bin, pass ``-np.log1p(-p)`` as the expected
    count, not ``p``. See `uniform_rescaled_ISIs`.

    References
    ----------
    .. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank,
           L.M. (2002). The time-rescaling theorem and its application to
           neural spike train data analysis. Neural Computation 14, 325-346.
    .. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.
    .. [3] Haslinger, R., Pipa, G., and Brown, E. (2010). Discrete time
           rescaling theorem: determining goodness of fit for discrete time
           statistical models of neural spiking. Neural Computation 22(10),
           2477-2506.
    """

    def __init__(
        self,
        conditional_intensity: np.ndarray,
        is_spike: np.ndarray,
        trial_id: np.ndarray | None = None,
        *,
        adjust_for_short_trials: bool = False,
        rate_units: str = "expected_counts",
        time: np.ndarray | None = None,
        time_edges: np.ndarray | None = None,
        discrete_time_correction: bool = True,
        rng: np.random.Generator | int | None = None,
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
        self.discrete_time_correction = discrete_time_correction
        self._within_bin_uniforms = (
            _within_bin_uniforms(self.is_spike, rng)
            if discrete_time_correction
            else None
        )

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
        if self.discrete_time_correction:
            _warn_counts_above_one(self.is_spike)
        trial_IDs = np.unique(self.trial_id)
        uniform_rescaled_ISIs_by_trial = []
        n_left_out = 0
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
                uniform, n_first = _uniform_rescaled_ISIs(
                    self.conditional_intensity[run],
                    self.is_spike[run],
                    None
                    if self._within_bin_uniforms is None
                    else self._within_bin_uniforms[run],
                    adjust_for_short_trials=self.adjust_for_short_trials,
                    rate_units=self.rate_units,
                    time=time,
                    time_edges=edges,
                )
                uniform_rescaled_ISIs_by_trial.append(uniform)
                n_left_out += n_first

        _warn_first_sample_spikes(n_left_out)
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
        uniform = self.uniform_rescaled_ISIs()
        return ks_statistic(np.sort(uniform), _uniform_cdf_values(uniform.size))

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
        lag_max: int | None = None,
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
        lag_max : int, optional
            Largest lag to plot, in number of ISIs. If None, plots all lags.

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
        Time-rescaled inter-spike intervals.

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


def _bin_integrated_intensity(
    conditional_intensity: np.ndarray,
    *,
    rate_units: str,
    time: np.ndarray | None = None,
    time_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Integrated intensity ``q_k`` of each bin, shape (n_time,).

    Expected counts are the bins' own ``q_k``. Hz rates on ``time_edges``
    give ``rate * width``. Hz rates at sample times give the trapezoid over
    ``(t_{k-1}, t_k]``, and zero for the first sample.
    """
    # Validates units and grids for every mode.
    integrated = _integrated_conditional_intensity(
        conditional_intensity, rate_units=rate_units, time=time, time_edges=time_edges
    )
    if rate_units == "expected_counts":
        return np.asarray(conditional_intensity, dtype=float)
    if time_edges is not None:
        return np.asarray(conditional_intensity, dtype=float) * np.diff(
            np.asarray(time_edges, dtype=float)
        )
    return np.diff(integrated, prepend=0.0)


def _warn_first_sample_spikes(n_left_out: int) -> None:
    """Warn that spikes at a run's first sample were left out."""
    if n_left_out:
        noun = "spike" if n_left_out == 1 else "spikes"
        warnings.warn(
            f"{n_left_out} {noun} at the first sample of a run of sample times "
            "left out of time rescaling: each lies in the interval before that "
            "sample, outside the run. To keep them, pass time_edges whose "
            "first bin of each run contains its first spike time.",
            DataQualityWarning,
            stacklevel=3,
        )


# Fraction of spike-containing bins with more than one spike above which the
# unused spikes are worth a warning.
_MULTI_SPIKE_BIN_FRACTION = 0.05


def _warn_counts_above_one(is_spike: np.ndarray) -> None:
    """Warn when many bins with a spike hold more than one."""
    is_spike = np.asarray(is_spike)
    n_spike_bins = np.count_nonzero(is_spike)
    if not n_spike_bins:
        return
    fraction = np.count_nonzero(is_spike > 1) / n_spike_bins
    if fraction > _MULTI_SPIKE_BIN_FRACTION:
        warnings.warn(
            f"{fraction:.1%} of bins with a spike hold more than one. The "
            "discrete-time rescaling counts each as one spike, so it does not "
            "test the number of spikes per bin; check counts with "
            "point_process_residuals, or use smaller bins.",
            DataQualityWarning,
            stacklevel=3,
        )


def _within_bin_uniforms(
    is_spike: np.ndarray, rng: np.random.Generator | int | None
) -> np.ndarray:
    """One Uniform(0, 1) draw per bin with a spike, zero elsewhere, shape (n_time,)."""
    is_spike = np.asarray(is_spike)
    spike_bins = np.flatnonzero(is_spike)
    uniforms = np.zeros(is_spike.shape)
    uniforms.flat[spike_bins] = np.random.default_rng(rng).uniform(size=spike_bins.size)
    return uniforms


def _discrete_time_uniform_rescaled_ISIs(
    bin_integrated_intensity: np.ndarray,
    is_spike: np.ndarray,
    within_bin_uniforms: np.ndarray,
    *,
    adjust_for_short_trials: bool,
) -> np.ndarray:
    """Uniform transform of discrete-time rescaled intervals.

    With ``q_k`` the integrated intensity of bin ``k``, the model gives the
    bin a spike with probability ``1 - exp(-q_k)``. For spikes in bins
    ``k_{i-1} < k_i`` and ``r_i ~ Uniform(0, 1)``,

    ``xi_i = sum_{k_{i-1} < k < k_i} q_k - log(1 - r_i (1 - exp(-q_{k_i})))``

    is exponential with unit mean under the model. The second term places
    the spike within its bin by the conditional (truncated exponential)
    distribution of the first event time given that the bin has one.

    Parameters
    ----------
    bin_integrated_intensity : np.ndarray, shape (n_time,)
        ``q_k``.
    is_spike : np.ndarray, shape (n_time,)
        Nonzero where the bin has a spike.
    within_bin_uniforms : np.ndarray, shape (n_time,)
        ``r_i``, read at the bins with a spike.
    adjust_for_short_trials : bool
        Divide by the probability of a next spike before the end of the run,
        ``1 - exp(-sum_{k > k_{i-1}} q_k)``.

    Returns
    -------
    uniform_rescaled_ISIs : np.ndarray, shape (n_spikes,)
    """
    q = bin_integrated_intensity
    closing = np.cumsum(q)
    spike_bins = np.flatnonzero(is_spike)
    previous_closing = np.concatenate(([0.0], closing[spike_bins]))[:-1]
    between = closing[spike_bins] - q[spike_bins] - previous_closing
    q_spike = q[spike_bins]
    within = -np.log1p(within_bin_uniforms[spike_bins] * np.expm1(-q_spike))
    uniform = -np.expm1(-(between + within))
    if adjust_for_short_trials:
        uniform = uniform / -np.expm1(-(closing[-1] - previous_closing))
    return cast(np.ndarray, uniform)


def _uniform_rescaled_ISIs(
    conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    within_bin_uniforms: np.ndarray | None,
    *,
    adjust_for_short_trials: bool,
    rate_units: str,
    time: np.ndarray | None,
    time_edges: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Shared body of `uniform_rescaled_ISIs`; no correction when uniforms are None.

    Returns the uniform transform and the number of spikes left out (a spike
    at the first sample time, with the correction).
    """
    if within_bin_uniforms is not None:
        q = _bin_integrated_intensity(
            conditional_intensity,
            rate_units=rate_units,
            time=time,
            time_edges=time_edges,
        )
        is_spike = np.asarray(is_spike)
        if q.ndim != 1:
            msg = "conditional_intensity must be one-dimensional, shape (n_time,)"
            raise ValueError(msg)
        if is_spike.shape != q.shape:
            msg = "is_spike must match conditional_intensity rows"
            raise ValueError(msg)
        n_left_out = 0
        if time is not None and is_spike[0]:
            # The first sample has no interval before it within the run.
            is_spike = is_spike.copy()
            is_spike[0] = 0
            n_left_out = 1
        uniform = _discrete_time_uniform_rescaled_ISIs(
            q,
            is_spike,
            within_bin_uniforms,
            adjust_for_short_trials=adjust_for_short_trials,
        )
        return uniform, n_left_out

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
    return (
        cast(np.ndarray, expon.cdf(rescaled_ISIs) / max_transformed_interval),
        0,
    )


def uniform_rescaled_ISIs(
    conditional_intensity: np.ndarray,
    is_spike: np.ndarray,
    *,
    adjust_for_short_trials: bool = True,
    rate_units: str = "expected_counts",
    time: np.ndarray | None = None,
    time_edges: np.ndarray | None = None,
    discrete_time_correction: bool = True,
    rng: np.random.Generator | int | None = None,
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
    is_spike : ndarray, shape (n_time,)
        Whether the neuron spiked in each bin. With the discrete-time
        correction a count above one is a bin with a spike (see Notes).
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [2]_. By default True.
    rate_units : {"expected_counts", "Hz"}, optional
        Hz requires exactly one of ``time`` or ``time_edges`` in seconds.
        By default "expected_counts".
    time : np.ndarray, shape (n_time,), optional
        Sample timestamps. Rates are integrated by trapezoids from the first
        sample. With the correction, a spike at sample ``k`` lies in
        ``(t_{k-1}, t_k]``; without it, at ``t_k``. A spike at the first
        sample then lies before the integrated window, so it is left out,
        with a `DataQualityWarning`.
    time_edges : np.ndarray, shape (n_time + 1,), optional
        Bin edges. Rates are constant within each bin, and the first ISI
        starts at the first edge. Without the correction, each spike is
        placed at its bin's closing edge.
    discrete_time_correction : bool, optional
        Use the discrete-time rescaling of [1]_ (see Notes). By default True.
    rng : numpy.random.Generator or int, optional
        Generator, or seed for one, that places each spike within its bin
        for the correction. By default a fresh, unseeded generator, so
        repeated calls give different intervals and KS values; pass a seed,
        e.g. ``rng=0``, to reproduce a result.

    Returns
    -------
    uniform_rescaled_ISIs : ndarray, shape (n_spikes,)
        Rescaled ISIs, uniform on [0, 1] if the model fits well. One fewer
        with sample times and a spike at the first sample.

    Notes
    -----
    Binned spikes have no exact time within their bin. Placing each one at
    its bin's closing edge makes every interval too long, so a correct
    model fails the KS test once the rate times the bin width is not small
    (at 2 ms bins and a 40 Hz peak, KS 0.040 against a 95% band of 0.0096).

    The correction follows [1]_ (their Eqs. 36-39). With ``q_k`` the
    model's integrated intensity over bin ``k`` (expected counts as given,
    or ``rate * width`` for ``time_edges``), the model gives the bin a spike
    with probability ``1 - exp(-q_k)``. For spikes in bins
    ``k_{i-1} < k_i`` and a draw ``r_i ~ Uniform(0, 1)``::

        xi_i = sum(q_k for k_{i-1} < k < k_i) - log(1 - r_i * (1 - exp(-q_{k_i})))

    is exponential with unit mean under the model, at any bin width. With
    `adjust_for_short_trials`, ``1 - exp(-xi_i)`` is divided by
    ``1 - exp(-sum_{k > k_{i-1}} q_k)``, the probability of a next spike
    before the end of the trial.

    For a Bernoulli (e.g. logistic) model with spike probability ``p_k``
    per bin, pass ``q_k = -np.log1p(-p_k)`` as the expected count, not
    ``p_k``. Refractoriness at or below the bin width cannot be detected
    after binning: the within-bin draw follows the model, not the data.

    The correction tests whether each bin has a spike. A count above one
    is one spike: a Poisson model with expected count ``q_k`` also gives
    the bin at least one spike with probability ``1 - exp(-q_k)``, so the
    test stays exact, but the extra spikes in a bin are not used, and
    within-bin count dispersion is not tested: a model expecting half the
    spikes of data whose every occupied bin holds two can pass. Check counts
    with `point_process_residuals`. For a Poisson model the fraction of
    spikes unused is ``1 - (1 - exp(-q)) / q``: 1.0% at 10 Hz, 3.9% at
    40 Hz and 9.4% at 100 Hz in 2 ms bins, and 31% at 40 Hz in 20 ms
    bins. A `DataQualityWarning` is issued when more than 5% of the bins
    with a spike hold more than one.

    References
    ----------
    .. [1] Haslinger, R., Pipa, G., and Brown, E. (2010). Discrete time
           rescaling theorem: determining goodness of fit for discrete time
           statistical models of neural spiking. Neural Computation 22(10),
           2477-2506.
    .. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.
    """
    if discrete_time_correction:
        # As TimeRescaling does, so (n_time, 1) columns are one recording.
        conditional_intensity = np.atleast_1d(
            np.asarray(conditional_intensity).squeeze()
        )
        is_spike = np.atleast_1d(np.asarray(is_spike).squeeze())
        _warn_counts_above_one(is_spike)
    within_bin_uniforms = (
        _within_bin_uniforms(is_spike, rng) if discrete_time_correction else None
    )
    uniform, n_left_out = _uniform_rescaled_ISIs(
        conditional_intensity,
        is_spike,
        within_bin_uniforms,
        adjust_for_short_trials=adjust_for_short_trials,
        rate_units=rate_units,
        time=time,
        time_edges=time_edges,
    )
    _warn_first_sample_spikes(n_left_out)
    return uniform


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
    lag_max: int | None = None,
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
    lag_max : int, optional
        Largest lag to display, in number of ISIs; lags with
        ``|lag| <= lag_max`` are shown. If None, shows all lags.

    Returns
    -------
    ax : Axes
        Matplotlib axes object containing the plot.

    Notes
    -----
    Points outside the confidence interval lines suggest temporal dependence
    in the rescaled ISIs, indicating potential model misfit.

    The lag counts intervals, not time: lag ``j`` correlates each rescaled
    ISI with the one ``j`` spikes later, whatever time separates them.
    """
    n_spikes = rescaled_ISI_autocorrelation.size // 2 + 1
    lag = np.arange(-n_spikes + 1, n_spikes)

    lag_ind = slice(None) if lag_max is None else np.abs(lag) <= lag_max

    if ax is None:
        ax = plt.gca()
    if scatter_kwargs is None:
        scatter_kwargs = {}
    ci = 1.96 / np.sqrt(n_spikes)
    ax.scatter(lag[lag_ind], rescaled_ISI_autocorrelation[lag_ind], **scatter_kwargs)
    ax.axhline(ci, linestyle="--", color=ci_color)
    ax.axhline(-ci, linestyle="--", color=ci_color)
    ax.set_xlabel("Lag (ISIs)")
    ax.set_ylabel("Autocorrelation")

    return ax
