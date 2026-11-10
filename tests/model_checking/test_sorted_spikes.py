"""Unit tests for ``model_checking/sorted_spikes.py``.

Covers the Brown-Barbieri-Ventura-Kass-Frank time-rescaling implementation
(``TimeRescaling`` class and the supporting module-level functions). The
central correctness check: a homogeneous Poisson process, when rescaled by
its true conditional intensity, produces unit-rate output (uniform rescaled
ISIs with mean ~0.5 and a small Kolmogorov-Smirnov distance).
"""

import matplotlib as mpl

mpl.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector.exceptions import DataQualityWarning
from non_local_detector.model_checking.sorted_spikes import (
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
            rng=0,
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
            rng=0,
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
            true_ci, is_spike, adjust_for_short_trials=False, rng=0
        ).ks_statistic()
        ks_wrong = TimeRescaling(
            wrong_ci, is_spike, adjust_for_short_trials=False, rng=0
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
            discrete_time_correction=False,
        )
        u = tr.uniform_rescaled_ISIs()

        expected = 1.0 - np.exp(-c * k)
        # The first ISI is measured from t=0 to the first spike (also k bins),
        # so every entry equals ``expected``.
        np.testing.assert_allclose(u, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Discrete-time correction: binned data drawn from the model itself
# ---------------------------------------------------------------------------

BIN_SIZE = 0.002  # seconds, a decoder bin
N_BINS = 500_000  # 1000 s: ~20,000 spikes at a 40 Hz peak, KS band ~0.0095


def _sinusoid_expected_counts(peak_rate: float) -> np.ndarray:
    """Expected counts per 2 ms bin of a rate oscillating 1 to peak+1 Hz."""
    centers = (np.arange(N_BINS) + 0.5) * BIN_SIZE
    rate = 0.5 * peak_rate * (1 + np.sin(2 * np.pi * centers / 3.0)) + 1.0
    return rate * BIN_SIZE


def _spikes_from_model(expected_counts: np.ndarray, seed: int) -> np.ndarray:
    """Whether each bin has a spike: probability ``1 - exp(-expected_counts)``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(size=expected_counts.size) < -np.expm1(-expected_counts)


def _ks_band(n_spikes: int) -> float:
    return 1.36 / np.sqrt(n_spikes)


@pytest.mark.unit
class TestDiscreteTimeCorrection:
    @pytest.mark.parametrize("peak_rate", [40.0, 10.0])
    def test_correct_model_passes(self, peak_rate):
        """Data drawn from the model pass at the decoder bin size.

        Without the correction a 40 Hz peak at 2 ms bins gives KS ~0.04
        against a band of ~0.0095: each spike is placed at its bin's closing
        edge, so every interval is too long.
        """
        expected_counts = _sinusoid_expected_counts(peak_rate)
        is_spike = _spikes_from_model(expected_counts, seed=0)
        check = TimeRescaling(expected_counts, is_spike, rng=1)
        assert check.ks_statistic() < _ks_band(check.n_spikes)

    def test_rapidly_varying_expected_counts_pass(self):
        """Each bin's own expected count is its integrated intensity.

        Counts alternating between 0.01 and 0.6 per bin: averaging adjacent
        bins (a trapezoid between samples) would give KS ~0.24 here.
        """
        expected_counts = np.tile([0.01, 0.6], 100_000)
        is_spike = _spikes_from_model(expected_counts, seed=2)
        check = TimeRescaling(expected_counts, is_spike, rng=3)
        assert check.ks_statistic() < _ks_band(check.n_spikes)

    def test_irregular_time_edges_pass(self):
        """With Hz rates and unequal bins, each bin's mass is rate * width."""
        rng = np.random.default_rng(4)
        widths = rng.uniform(0.001, 0.003, size=N_BINS)
        edges = np.concatenate([[0.0], np.cumsum(widths)])
        centers = 0.5 * (edges[:-1] + edges[1:])
        rate = 20.0 * (1 + np.sin(2 * np.pi * centers / 3.0)) + 1.0
        is_spike = _spikes_from_model(rate * widths, seed=5)
        check = TimeRescaling(rate, is_spike, rate_units="Hz", time_edges=edges, rng=6)
        assert check.ks_statistic() < _ks_band(check.n_spikes)

    def test_sample_times_pass(self):
        """With Hz rates at sample times, a spike lies in the interval ending
        at its sample, whose mass is the trapezoid over that interval.
        """
        time = np.arange(N_BINS) * BIN_SIZE
        rate = 20.0 * (1 + np.sin(2 * np.pi * time / 3.0)) + 1.0
        interval_mass = np.r_[0.0, 0.5 * (rate[1:] + rate[:-1]) * BIN_SIZE]
        is_spike = _spikes_from_model(interval_mass, seed=13)
        check = TimeRescaling(rate, is_spike, rate_units="Hz", time=time, rng=14)
        assert check.ks_statistic() < _ks_band(check.n_spikes)

    @pytest.mark.parametrize("mode", ["expected_counts", "time_edges", "time"])
    def test_short_trials_are_calibrated_in_every_input_mode(self, mode):
        """Datasets from the model fail at about the band's 5% rate.

        100 datasets of 200 trials x 10 bins of 2 ms at 40 Hz, short-trial
        adjustment on. With sample times, a spike at a trial's first sample
        has no interval before it; rescaling it to zero rejected 58% of
        these datasets.
        """
        n_trials, n_bins = 200, 10
        n_time = n_trials * n_bins
        trial_id = np.repeat(np.arange(n_trials), n_bins)
        rate = np.full(n_time, 40.0)
        kwargs = {
            "expected_counts": {"conditional_intensity": rate * BIN_SIZE},
            "time_edges": {
                "conditional_intensity": rate,
                "rate_units": "Hz",
                "time_edges": np.arange(n_time + 1) * BIN_SIZE,
            },
            "time": {
                "conditional_intensity": rate,
                "rate_units": "Hz",
                "time": np.arange(n_time) * BIN_SIZE,
            },
        }[mode]
        rejected = []
        for seed in range(100):
            is_spike = _spikes_from_model(rate * BIN_SIZE, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DataQualityWarning)
                check = TimeRescaling(
                    is_spike=is_spike,
                    trial_id=trial_id,
                    adjust_for_short_trials=True,
                    rng=1000 + seed,
                    **kwargs,
                )
                n_intervals = check.uniform_rescaled_ISIs().size
                rejected.append(check.ks_statistic() > _ks_band(n_intervals))
        assert np.mean(rejected) <= 0.12

    def test_spike_at_first_sample_of_a_run_is_left_out(self):
        """With sample times, a run's first sample closes no interval."""
        time = np.arange(6) * 0.1
        is_spike = np.array([True, False, True, True, False, True])
        trial_id = np.array([0, 0, 0, 1, 1, 1])
        check = TimeRescaling(
            np.full(6, 5.0), is_spike, trial_id, rate_units="Hz", time=time, rng=0
        )
        with pytest.warns(DataQualityWarning, match="2 spikes") as from_method:
            u = check.uniform_rescaled_ISIs()
        assert u.size == 2
        assert np.all(u > 0)
        with pytest.warns(DataQualityWarning, match="2 spikes") as from_ks:
            assert np.isfinite(check.ks_statistic())
        with pytest.warns(DataQualityWarning, match="1 spike") as from_function:
            u = uniform_rescaled_ISIs(
                np.full(6, 5.0), is_spike, rate_units="Hz", time=time, rng=0
            )
        assert u.size == 3
        # Attributed to the caller's code, not the library.
        for record in (from_method, from_ks, from_function):
            assert {w.filename for w in record} == {__file__}

    def test_wrong_model_still_fails(self):
        """Data from a 40 Hz peak tested against a 20 Hz peak model fail."""
        is_spike = _spikes_from_model(_sinusoid_expected_counts(40.0), seed=0)
        check = TimeRescaling(_sinusoid_expected_counts(20.0), is_spike, rng=1)
        assert check.ks_statistic() > _ks_band(check.n_spikes)

    def test_uniform_transform_mean_is_one_half(self):
        """The mean is within 3 standard errors of 0.5 (uncorrected: ~0.515)."""
        expected_counts = _sinusoid_expected_counts(40.0)
        is_spike = _spikes_from_model(expected_counts, seed=0)
        u = TimeRescaling(expected_counts, is_spike, rng=1).uniform_rescaled_ISIs()
        standard_error = np.sqrt(1 / 12 / u.size)
        assert abs(u.mean() - 0.5) < 3 * standard_error

    def test_short_trial_adjustment_still_lands_in_band(self):
        """2000 trials of 100 ms at 10 Hz: only the adjusted test passes."""
        n_trials, n_bins_per_trial = 2000, 50
        expected_counts = np.full(n_trials * n_bins_per_trial, 10.0 * BIN_SIZE)
        is_spike = _spikes_from_model(expected_counts, seed=7)
        trial_id = np.repeat(np.arange(n_trials), n_bins_per_trial)

        def ks_ratio(adjust):
            check = TimeRescaling(
                expected_counts,
                is_spike,
                trial_id,
                adjust_for_short_trials=adjust,
                rng=8,
            )
            return check.ks_statistic() / _ks_band(check.n_spikes)

        assert ks_ratio(adjust=True) < 1.0
        assert ks_ratio(adjust=False) > 1.0

    def test_poisson_counts_above_one_count_as_one_spike(self):
        """A bin with several spikes is a bin with a spike.

        The model gives a bin at least one spike with probability
        ``1 - exp(-q)``, so the test stays exact for Poisson counts.
        """
        expected_counts = _sinusoid_expected_counts(40.0)
        counts = np.random.default_rng(9).poisson(expected_counts)
        assert np.any(counts > 1)
        from_counts = TimeRescaling(expected_counts, counts, rng=10)
        from_indicator = TimeRescaling(expected_counts, counts > 0, rng=10)
        np.testing.assert_array_equal(
            from_counts.uniform_rescaled_ISIs(), from_indicator.uniform_rescaled_ISIs()
        )
        assert from_counts.ks_statistic() < _ks_band(from_counts.n_spikes)

    def test_many_multi_spike_bins_warn_once_per_call(self):
        """Counts above one are not tested, so a high fraction is flagged.

        Every bin with spikes holds two: a model expecting half the spikes
        would still pass, because only whether a bin has a spike is tested.
        """
        expected_counts = np.full(2000, 0.1)
        counts = 2 * _spikes_from_model(expected_counts, seed=15)
        with pytest.warns(DataQualityWarning, match=r"100\.0% of bins") as record:
            uniform_rescaled_ISIs(expected_counts, counts, rng=16)
        assert len(record) == 1
        check = TimeRescaling(expected_counts, counts, rng=16)
        with pytest.warns(
            DataQualityWarning, match="point_process_residuals"
        ) as from_method:
            check.uniform_rescaled_ISIs()
        # Attributed to the caller's code, not the library.
        assert {w.filename for w in [*record, *from_method]} == {__file__}

    def test_few_multi_spike_bins_do_not_warn(self):
        """At 2 ms and a 40 Hz peak about 3% of spike bins hold more than one."""
        expected_counts = _sinusoid_expected_counts(40.0)
        counts = np.random.default_rng(17).poisson(expected_counts)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataQualityWarning)
            TimeRescaling(expected_counts, counts, rng=18).uniform_rescaled_ISIs()

    def test_column_inputs_give_the_one_dimensional_result(self):
        """(n_time, 1) inputs are the same recording as (n_time,) inputs."""
        expected_counts = _sinusoid_expected_counts(10.0)[:20_000]
        is_spike = _spikes_from_model(expected_counts, seed=0)
        np.testing.assert_array_equal(
            uniform_rescaled_ISIs(expected_counts[:, None], is_spike[:, None], rng=20),
            uniform_rescaled_ISIs(expected_counts, is_spike, rng=20),
        )

    def test_seed_determines_result(self):
        expected_counts = _sinusoid_expected_counts(10.0)[:20_000]
        is_spike = _spikes_from_model(expected_counts, seed=0)

        def rescaled(seed):
            return TimeRescaling(
                expected_counts, is_spike, rng=seed
            ).uniform_rescaled_ISIs()

        np.testing.assert_array_equal(rescaled(1), rescaled(1))
        assert not np.array_equal(rescaled(1), rescaled(2))

    def test_methods_of_one_object_agree_without_a_seed(self):
        """Within-bin draws are made once, so the KS plot shows ks_statistic."""
        expected_counts = _sinusoid_expected_counts(10.0)[:20_000]
        is_spike = _spikes_from_model(expected_counts, seed=0)
        check = TimeRescaling(expected_counts, is_spike)
        u = check.uniform_rescaled_ISIs()
        np.testing.assert_array_equal(u, check.uniform_rescaled_ISIs())
        assert check.ks_statistic() == ks_statistic(
            np.sort(u), (np.arange(u.size) + 0.5) / u.size
        )

    def test_interval_lies_within_its_spike_bin(self):
        """Constant count c, spikes every k bins: each interval is in
        ``[(k - 1) c, k c]``, the first from time zero in ``[k c, (k + 1) c]``.
        """
        n_time, c, k = 100, 0.2, 10
        is_spike = np.zeros(n_time, dtype=bool)
        is_spike[k::k] = True
        u = uniform_rescaled_ISIs(
            np.full(n_time, c), is_spike, adjust_for_short_trials=False, rng=12
        )
        isi = -np.log1p(-u)
        lower = np.r_[k * c, np.full(u.size - 1, (k - 1) * c)]
        assert np.all(isi >= lower - 1e-12)
        assert np.all(isi <= lower + c + 1e-12)


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
            ci, is_spike, adjust_for_short_trials=False, rng=0
        ).uniform_rescaled_ISIs()
        u_adjusted = TimeRescaling(
            ci, is_spike, adjust_for_short_trials=True, rng=0
        ).uniform_rescaled_ISIs()

        assert np.all(u_adjusted >= u_plain - 1e-12)
        # At least one interval is strictly increased (the last spike sits
        # before the end of the trial, so its censoring weight is < 1).
        assert np.any(u_adjusted > u_plain + 1e-9)

    def test_module_function_matches_class_with_adjustment(self, regular_spike_train):
        """The free ``uniform_rescaled_ISIs`` matches the class method."""
        ci = regular_spike_train["conditional_intensity"]
        is_spike = regular_spike_train["is_spike"]

        from_func = uniform_rescaled_ISIs(
            ci, is_spike, adjust_for_short_trials=True, rng=0
        )
        from_class = TimeRescaling(
            ci, is_spike, adjust_for_short_trials=True, rng=0
        ).uniform_rescaled_ISIs()

        np.testing.assert_allclose(from_func, from_class, rtol=1e-12)


# ---------------------------------------------------------------------------
# Trial grouping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrialGrouping:
    def test_per_trial_isis_concatenated(self):
        """ISIs are computed per trial then concatenated.

        The first ISI of each trial is measured from the trial start, so with a
        constant intensity of 0.5 per bin both trials give integrated ISIs of
        0.5 * 3 and 0.5 * 4; an ISI bridging the boundary would give
        0.5 * (13 - 7) for trial 1's first spike.
        """
        n_time = 20
        ci = np.full(n_time, 0.5)
        is_spike = np.zeros(n_time, dtype=bool)
        is_spike[[3, 7, 13, 17]] = True  # 2 spikes in trial 0, 2 in trial 1
        trial_id = np.array([0] * 10 + [1] * 10)

        tr = TimeRescaling(
            ci,
            is_spike,
            trial_id=trial_id,
            adjust_for_short_trials=False,
            discrete_time_correction=False,
        )
        u = tr.uniform_rescaled_ISIs()
        np.testing.assert_allclose(u, -np.expm1(-np.array([1.5, 2.0, 1.5, 2.0])))

    @pytest.mark.parametrize("mode", ["expected_counts", "time_edges", "time"])
    def test_each_interval_lies_within_its_own_trial(self, mode):
        """With the correction, no interval reaches into the previous trial.

        Two trials of 10 bins of mass 0.5, spikes at local bins 3 and 7. A
        first interval spans the trial's bins before its spike plus part of
        the spike's bin; with sample times the first sample adds no mass.
        Bridging the trials would put trial 1's first interval at 2.5-3.0.
        """
        c = 0.5
        is_spike = np.zeros(20, dtype=bool)
        is_spike[[3, 7, 13, 17]] = True
        trial_id = np.repeat([0, 1], 10)
        kwargs = {
            "expected_counts": {"conditional_intensity": np.full(20, c)},
            "time_edges": {
                "conditional_intensity": np.full(20, 5.0),
                "rate_units": "Hz",
                "time_edges": np.arange(21) * 0.1,
            },
            "time": {
                "conditional_intensity": np.full(20, 5.0),
                "rate_units": "Hz",
                "time": np.arange(20) * 0.1,
            },
        }[mode]
        u = TimeRescaling(
            is_spike=is_spike,
            trial_id=trial_id,
            adjust_for_short_trials=False,
            rng=19,
            **kwargs,
        ).uniform_rescaled_ISIs()
        first = 2 * c if mode == "time" else 3 * c
        lower = np.array([first, 3 * c, first, 3 * c])
        isi = -np.log1p(-u)
        assert isi.shape == (4,)
        assert np.all(isi >= lower - 1e-12)
        assert np.all(isi <= lower + c + 1e-12)


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
            rng=0,
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
            rng=0,
        )

    def test_plots_draw_on_axes(self, fitted_rescaling):
        """Smoke test: each plotting method and function returns its Axes."""
        uniform = fitted_rescaling.uniform_rescaled_ISIs()
        autocorrelation = fitted_rescaling.rescaled_ISI_autocorrelation()
        for ax in (
            fitted_rescaling.plot_ks(),
            fitted_rescaling.plot_qq(),
            fitted_rescaling.plot_rescaled_ISI_autocorrelation(),
            plot_ks(uniform),
            plot_qq(uniform),
            plot_rescaled_ISI_autocorrelation(autocorrelation, lag_max=5),
        ):
            assert isinstance(ax, plt.Axes)
        plt.close("all")

    @pytest.mark.parametrize(
        ("lag_max", "expected_lags"),
        [
            (None, np.arange(-4, 5)),
            (2, np.arange(-2, 3)),
            (100, np.arange(-4, 5)),
        ],
    )
    def test_autocorrelation_plots_lags_centered_on_zero(self, lag_max, expected_lags):
        """Every lag with |lag| <= lag_max is plotted at its own value."""
        # Five spikes: lags -4..4, zero lag at index 4; value = lag index.
        autocorrelation = np.arange(-4.0, 5.0)
        _, ax = plt.subplots()
        plot_rescaled_ISI_autocorrelation(autocorrelation, ax=ax, lag_max=lag_max)
        points = ax.collections[0].get_offsets()
        np.testing.assert_allclose(points[:, 0], expected_lags)
        np.testing.assert_allclose(points[:, 1], expected_lags)
        plt.close("all")

    def test_autocorrelation_lag_counts_intervals(self):
        """Intervals that repeat every third interval peak at a lag of 3 ISIs.

        Bin widths set the rescaled intervals exactly (rate 1 Hz, one spike
        per bin, no within-bin draw); their normal transforms cycle through
        -1, 0, 1, so the largest nonzero-lag autocorrelation is at lag 3.
        """
        cycle = -np.log(norm.sf([-1.0, 0.0, 1.0]))
        widths = np.tile(cycle, 10)
        check = TimeRescaling(
            np.ones(widths.size),
            np.ones(widths.size, dtype=bool),
            adjust_for_short_trials=False,
            rate_units="Hz",
            time_edges=np.concatenate([[0.0], np.cumsum(widths)]),
            discrete_time_correction=False,
        )
        _, ax = plt.subplots()
        check.plot_rescaled_ISI_autocorrelation(ax=ax, lag_max=8)
        lag, value = ax.collections[0].get_offsets().T
        np.testing.assert_array_equal(lag, np.arange(-8, 9))
        nonzero = lag != 0
        assert lag[nonzero][np.argmax(value[nonzero])] in (-3, 3)
        assert "ISI" in ax.get_xlabel()
        plt.close("all")
