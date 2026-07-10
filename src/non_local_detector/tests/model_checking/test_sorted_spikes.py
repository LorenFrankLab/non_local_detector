"""Unit tests for ``model_checking/sorted_spikes.py``.

Covers the Brown-Barbieri-Ventura-Kass-Frank time-rescaling implementation
(``TimeRescaling`` class and the supporting module-level functions). The
central correctness check: a homogeneous Poisson process, when rescaled by
its true conditional intensity, produces unit-rate output (uniform rescaled
ISIs with mean ~0.5 and a small Kolmogorov-Smirnov distance).
"""

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from non_local_detector.model_checking.sorted_spikes import (  # noqa: E402
    TimeRescaling,
    ks_statistic,
    plot_ks,
    plot_qq,
    plot_rescaled_ISI_autocorrelation,
    point_process_residuals,
    uniform_rescaled_ISIs,
)

# ---------------------------------------------------------------------------
# TimeRescaling: construction and n_spikes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeRescalingBasics:
    def test_n_spikes_counts_nonzero(self, regular_spike_train):
        tr = TimeRescaling(
            regular_spike_train["conditional_intensity"],
            regular_spike_train["is_spike"],
        )
        assert tr.n_spikes == int(regular_spike_train["is_spike"].sum())

    def test_default_trial_id_groups_all_time_together(
        self, homogeneous_poisson_spike_train
    ):
        """When trial_id is None, all time points belong to one trial."""
        tr = TimeRescaling(
            homogeneous_poisson_spike_train["conditional_intensity"],
            homogeneous_poisson_spike_train["is_spike"],
        )
        assert np.unique(tr.trial_id).size == 1

    def test_zero_spikes_returns_empty_isis(self, empty_spike_train):
        """No spikes -> no interspike intervals (empty array, no crash)."""
        tr = TimeRescaling(
            empty_spike_train["conditional_intensity"],
            empty_spike_train["is_spike"],
        )
        assert tr.n_spikes == 0
        assert tr.uniform_rescaled_ISIs().shape == (0,)


# ---------------------------------------------------------------------------
# TimeRescaling: time-rescaling correctness (the headline check)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeRescalingCorrectness:
    def test_uniform_input_gives_uniform_output(self, homogeneous_poisson_spike_train):
        """Homogeneous Poisson rescaled by its true intensity -> uniform ISIs.

        Per the time-rescaling theorem, when the model intensity matches the
        data-generating intensity the rescaled ISIs are exponential(1), so the
        CDF transform yields a Uniform(0, 1). We check the empirical mean is
        ~0.5 and the values span the unit interval.
        """
        tr = TimeRescaling(
            homogeneous_poisson_spike_train["conditional_intensity"],
            homogeneous_poisson_spike_train["is_spike"],
            adjust_for_short_trials=False,
        )
        u = tr.uniform_rescaled_ISIs()

        assert u.min() >= 0.0
        assert u.max() <= 1.0
        # Uniform(0, 1) has mean 0.5; with ~200 spikes this is tight.
        assert u.mean() == pytest.approx(0.5, abs=0.05)

    def test_ks_statistic_small_for_correct_model(
        self, homogeneous_poisson_spike_train
    ):
        """A correctly specified model has a small KS distance from uniform.

        The 95% KS band for n spikes is ~1.36/sqrt(n); a correct model should
        sit comfortably inside a generous multiple of it.
        """
        tr = TimeRescaling(
            homogeneous_poisson_spike_train["conditional_intensity"],
            homogeneous_poisson_spike_train["is_spike"],
            adjust_for_short_trials=False,
        )
        ks = tr.ks_statistic()
        band = 1.36 / np.sqrt(tr.n_spikes)
        assert 0.0 <= ks < 3.0 * band

    def test_misspecified_model_has_larger_ks(self, homogeneous_poisson_spike_train):
        """A grossly wrong intensity yields a larger KS distance than truth."""
        is_spike = homogeneous_poisson_spike_train["is_spike"]
        true_ci = homogeneous_poisson_spike_train["conditional_intensity"]
        # Wrong model: intensity off by 20x -> rescaled ISIs no longer uniform.
        wrong_ci = true_ci * 20.0

        ks_true = TimeRescaling(
            true_ci, is_spike, adjust_for_short_trials=False
        ).ks_statistic()
        ks_wrong = TimeRescaling(
            wrong_ci, is_spike, adjust_for_short_trials=False
        ).ks_statistic()

        assert ks_wrong > ks_true

    def test_known_rescaled_isi_value_for_constant_intensity(self, regular_spike_train):
        """Hand-computed ISI: constant intensity c, spikes every k bins.

        The cumulative trapezoid integral of a constant ``c`` over ``k`` bins
        (unit spacing) is ``c * k``; the uniform-transformed ISI is therefore
        ``1 - exp(-c * k)`` for every interior interval.
        """
        c = regular_spike_train["rate"]
        k = regular_spike_train["spacing"]
        tr = TimeRescaling(
            regular_spike_train["conditional_intensity"],
            regular_spike_train["is_spike"],
            adjust_for_short_trials=False,
        )
        u = tr.uniform_rescaled_ISIs()

        expected = 1.0 - np.exp(-c * k)
        # The first ISI is measured from t=0 to the first spike (also k bins),
        # so every entry equals ``expected``.
        np.testing.assert_allclose(u, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Wiener-2003 short-trial correction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShortTrialCorrection:
    def test_correction_increases_rescaled_isis(self, regular_spike_train):
        """The Wiener-2003 adjustment divides by ``expon.cdf(.) <= 1``.

        Dividing by a value <= 1 cannot decrease the rescaled ISIs and
        strictly increases them whenever the censoring weight is below 1, so
        the corrected output is >= the uncorrected output everywhere.
        """
        ci = regular_spike_train["conditional_intensity"]
        is_spike = regular_spike_train["is_spike"]

        u_plain = TimeRescaling(
            ci, is_spike, adjust_for_short_trials=False
        ).uniform_rescaled_ISIs()
        u_adjusted = TimeRescaling(
            ci, is_spike, adjust_for_short_trials=True
        ).uniform_rescaled_ISIs()

        assert np.all(u_adjusted >= u_plain - 1e-12)
        # At least one interval is strictly increased (the last spike sits
        # before the end of the trial, so its censoring weight is < 1).
        assert np.any(u_adjusted > u_plain + 1e-9)

    def test_module_function_matches_class_with_adjustment(self, regular_spike_train):
        """The free ``uniform_rescaled_ISIs`` matches the class method."""
        ci = regular_spike_train["conditional_intensity"]
        is_spike = regular_spike_train["is_spike"]

        from_func = uniform_rescaled_ISIs(ci, is_spike, adjust_for_short_trials=True)
        from_class = TimeRescaling(
            ci, is_spike, adjust_for_short_trials=True
        ).uniform_rescaled_ISIs()

        np.testing.assert_allclose(from_func, from_class, rtol=1e-12)


# ---------------------------------------------------------------------------
# Trial grouping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrialGrouping:
    def test_per_trial_isis_concatenated(self):
        """ISIs are computed per trial then concatenated.

        With one spike per trial, the first ISI of each trial is measured from
        the trial start, so the number of returned ISIs equals the total spike
        count (no ISI bridges across the trial boundary).
        """
        n_time = 20
        ci = np.full(n_time, 0.5)
        is_spike = np.zeros(n_time, dtype=bool)
        is_spike[[3, 7, 13, 17]] = True  # 2 spikes in trial 0, 2 in trial 1
        trial_id = np.array([0] * 10 + [1] * 10)

        tr = TimeRescaling(
            ci, is_spike, trial_id=trial_id, adjust_for_short_trials=False
        )
        u = tr.uniform_rescaled_ISIs()
        assert u.shape == (4,)


# ---------------------------------------------------------------------------
# rescaled_ISI_autocorrelation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRescaledISIAutocorrelation:
    def test_autocorrelation_shape_and_peak(self, homogeneous_poisson_spike_train):
        """Autocorrelation has length 2*n_spikes-1 and a normalized peak of 1."""
        tr = TimeRescaling(
            homogeneous_poisson_spike_train["conditional_intensity"],
            homogeneous_poisson_spike_train["is_spike"],
            adjust_for_short_trials=False,
        )
        ac = tr.rescaled_ISI_autocorrelation()
        assert ac.shape == (2 * tr.n_spikes - 1,)
        assert ac.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Module-level functions: ks_statistic, point_process_residuals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKSStatistic:
    def test_identical_cdfs_zero(self):
        cdf = np.linspace(0, 1, 50)
        assert ks_statistic(cdf, cdf) == pytest.approx(0.0)

    def test_known_max_distance(self):
        empirical = np.array([0.0, 0.5, 1.0])
        model = np.array([0.0, 0.2, 1.0])
        assert ks_statistic(empirical, model) == pytest.approx(0.3)

    def test_mismatched_sizes_returns_nan(self):
        """Differently sized arrays cannot be compared -> NaN (not a crash)."""
        result = ks_statistic(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]))
        assert np.isnan(result)


@pytest.mark.unit
class TestPointProcessResiduals:
    def test_perfect_fit_is_zero(self):
        """When intensity equals the spike indicator, residuals are all zero."""
        ci = np.full(10, 0.3)
        is_spike = np.full(10, 0.3)
        np.testing.assert_allclose(point_process_residuals(ci, is_spike), 0.0)

    def test_final_residual_is_spikes_minus_integral(self):
        """Final residual = total spikes - integrated intensity (cumsum)."""
        is_spike = np.array([0, 1, 0, 1, 0], dtype=float)
        ci = np.full(5, 0.2)
        residuals = point_process_residuals(ci, is_spike)
        assert residuals[-1] == pytest.approx(is_spike.sum() - ci.sum())
        assert residuals.shape == (5,)

    def test_zero_spikes_residuals_decreasing(self):
        """No spikes -> residuals strictly decrease by the intensity each step."""
        ci = np.full(5, 0.1)
        residuals = point_process_residuals(ci, np.zeros(5))
        np.testing.assert_allclose(residuals, -np.cumsum(ci))


# ---------------------------------------------------------------------------
# Plotting functions (Agg backend, return an Axes)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlotting:
    @pytest.fixture
    def fitted_rescaling(self, homogeneous_poisson_spike_train):
        return TimeRescaling(
            homogeneous_poisson_spike_train["conditional_intensity"],
            homogeneous_poisson_spike_train["is_spike"],
            adjust_for_short_trials=False,
        )

    def test_class_plot_ks_returns_axes(self, fitted_rescaling):
        ax = fitted_rescaling.plot_ks()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_class_plot_qq_returns_axes(self, fitted_rescaling):
        ax = fitted_rescaling.plot_qq()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_class_plot_autocorrelation_returns_axes(self, fitted_rescaling):
        ax = fitted_rescaling.plot_rescaled_ISI_autocorrelation()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_module_plot_ks_returns_axes(self, fitted_rescaling):
        ax = plot_ks(fitted_rescaling.uniform_rescaled_ISIs())
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_module_plot_qq_returns_axes(self, fitted_rescaling):
        ax = plot_qq(fitted_rescaling.uniform_rescaled_ISIs())
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_module_plot_autocorrelation_returns_axes(self, fitted_rescaling):
        ac = fitted_rescaling.rescaled_ISI_autocorrelation()
        ax = plot_rescaled_ISI_autocorrelation(ac, sampling_frequency=2.0, lag_max=5.0)
        assert isinstance(ax, plt.Axes)
        plt.close("all")
