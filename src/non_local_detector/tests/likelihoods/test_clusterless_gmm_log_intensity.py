"""Reference-value tests for the clusterless GMM log-intensity arithmetic.

The marked-point-process log intensity of a spike is
``log(rate) + log p(pos, mark) - log p(pos)``. Two historical defects:

1. The joint log density was clamped at ``LOG_EPS`` *before* the subtraction,
   so a tail spike (``log p(pos, mark) = -60``, ``log p(pos) = -8``) scored
   ``-26.5`` instead of ``-52`` — about ``1e11`` too much intensity.
2. The ground process ``rate * p_gpi(x) / p_occ(x)`` was formed from the two
   densities in probability space at fit time, so an underflowing GPI density
   at a supported bin lost its finite ratio, and the local path's
   ``rate * exp(diff)`` overflowed where ``exp(log(rate) + diff)`` is
   representable. Both paths now share one log-space helper; the fit path's
   ``occupancy > 0`` guard, which turned float32 underflow into an ``EPS``
   substitute, is gone.

A third, subtler loss: adding ``log(rate)`` to a huge log density before
subtracting the other rounds the rate term away in float32; both spike-term
sites now add it to the density *difference*, as the ground process does.

Every reference here is closed-form float64 arithmetic on hand-built
single-Gaussian models (or, for the fit path, on the fitted models' own log
densities), never another call into the code under test. Event terms are
isolated by subtracting a matched no-spike baseline from the same public path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    _accumulate_log_likelihood_block,
    compute_local_log_likelihood,
    fit_clusterless_gmm_encoding_model,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.common import EPS, LOG_EPS
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _compute_precision_cholesky,
)

pytestmark = pytest.mark.unit

RTOL = 1e-5
ATOL = 1e-4  # float32 log densities near -60 carry ~1e-5 absolute rounding


def _env_1d(lo: float = 0.0, hi: float = 10.0, bin_size: float = 1.0) -> Environment:
    env = Environment(
        environment_name="test", place_bin_size=bin_size, position_range=((lo, hi),)
    )
    pos = np.linspace(lo, hi, 101)[:, None]
    return env.fit_place_grid(position=pos, infer_track_interior=False)


def _gaussian(mean, var: float) -> GaussianMixtureModel:
    """Single-component isotropic Gaussian as a fitted ``GaussianMixtureModel``."""
    mean = jnp.asarray(np.atleast_1d(mean), dtype=jnp.float32)
    dim = mean.shape[0]
    cov = jnp.asarray(var, dtype=jnp.float32) * jnp.eye(dim)[None]
    gmm = GaussianMixtureModel(n_components=1, covariance_type="full")
    gmm.weights_ = jnp.ones(1)
    gmm.means_ = mean[None]
    gmm.covariances_ = cov
    gmm.precisions_chol_ = _compute_precision_cholesky(cov, "full")
    return gmm


def _log_gaussian(x, mean, var: float) -> np.ndarray:
    """Closed-form isotropic Gaussian log density in float64."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    mean = np.asarray(np.atleast_1d(mean), dtype=np.float64)
    dim = mean.shape[0]
    sq = ((x - mean) ** 2).sum(axis=1)
    return -0.5 * dim * np.log(2.0 * np.pi * var) - 0.5 * sq / var


def _predict_kwargs(env, occupancy, joint, rate, spike_times, marks, **extra):
    n_bins = env.place_bin_centers_[env.is_track_interior_.ravel()].shape[0]
    bins = env.place_bin_centers_[env.is_track_interior_.ravel()]
    time = jnp.arange(0.0, 4.0, 1.0)
    return dict(
        time=time,
        position_time=np.asarray(time),
        position=jnp.full((4, 1), 5.0),
        spike_times=[spike_times],
        spike_waveform_features=[marks],
        environment=env,
        occupancy_model=occupancy,
        interior_place_bin_centers=jnp.asarray(bins),
        log_occupancy=occupancy.score_samples(jnp.asarray(bins)),
        gpi_models=[occupancy],
        joint_models=[joint],
        mean_rates=jnp.asarray([rate]),
        summed_ground_process_intensity=jnp.zeros(n_bins),
        disable_progress_bar=True,
        mark_dimensions=[1],
        **extra,
    )


def _fit_far_from_most_bins(seed: int):
    """Fit single-Gaussian models on data confined to one end of a long track.

    One spike is placed at every position sample, so the GPI and occupancy
    models see identical samples and the true ground-process ratio is exactly
    the mean rate at every bin — including bins so far from the data that both
    densities underflow in float32.
    """
    rng = np.random.default_rng(seed)
    env = _env_1d(0.0, 200.0, bin_size=5.0)
    n_pos = 2000
    position_time = np.linspace(0.0, 100.0, n_pos)
    position = np.clip(rng.normal(2.5, 1.0, n_pos), 0.0, 200.0)[:, None]
    marks = rng.normal(0.0, 1.0, (n_pos, 1))
    model = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=[position_time],
        spike_waveform_features=[marks],
        environment=env,
        sampling_frequency=20,
        gmm_components_occupancy=1,
        gmm_components_gpi=1,
        gmm_components_joint=1,
        disable_progress_bar=True,
    )
    return model, env


class TestAccumulateBlock:
    def test_block_contribution_is_the_raw_sum(self):
        """The scatter helper adds ``log_rate + joint - log_occ`` unchanged."""
        log_likelihood = jnp.zeros((3, 2))
        joint = jnp.asarray([[-60.0, -10.0]])
        log_occ = jnp.asarray([-8.0, -8.0])

        out = _accumulate_log_likelihood_block(
            log_likelihood,
            joint,
            jnp.asarray([1]),
            jnp.arange(2),
            jnp.asarray(0.0),
            log_occ,
        )

        np.testing.assert_allclose(np.asarray(out)[1], [-52.0, -2.0], rtol=RTOL)
        assert np.all(np.asarray(out)[[0, 2]] == 0.0)


class TestTailSpikeIntensity:
    """A tail spike's event term must be the raw log ratio, not a clamped one."""

    # Joint model: pos ~ N(5, 1), mark ~ N(0, 1); a spike with mark 11 sits
    # 60.5 nats down the mark tail, far below LOG_EPS, at every position bin.
    RATE = 2.0
    MARK = 11.0

    def _models(self):
        occupancy = _gaussian([5.0], var=25.0)
        joint = _gaussian([5.0, 0.0], var=1.0)
        return occupancy, joint

    def _expected_event_term(self, positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
        joint_points = np.concatenate(
            [positions, np.full_like(positions, self.MARK)], axis=1
        )
        return (
            np.log(self.RATE)
            + _log_gaussian(joint_points, [5.0, 0.0], 1.0)
            - _log_gaussian(positions, [5.0], 25.0)
        )

    @pytest.mark.parametrize("bin_tile_size", [None, 3], ids=["untiled", "tiled"])
    def test_non_local_event_term_matches_reference(self, bin_tile_size):
        env = _env_1d()
        occupancy, joint = self._models()
        bins = env.place_bin_centers_[env.is_track_interior_.ravel()]
        expected = self._expected_event_term(bins)
        assert np.all(expected < LOG_EPS), "case must sit below the old clamp"

        with_spike = predict_clusterless_gmm_log_likelihood(
            **_predict_kwargs(
                env,
                occupancy,
                joint,
                self.RATE,
                jnp.asarray([1.5]),
                jnp.asarray([[self.MARK]]),
                bin_tile_size=bin_tile_size,
            )
        )
        baseline = predict_clusterless_gmm_log_likelihood(
            **_predict_kwargs(
                env,
                occupancy,
                joint,
                self.RATE,
                jnp.zeros(0),
                jnp.zeros((0, 1)),
                bin_tile_size=bin_tile_size,
            )
        )
        event_term = np.asarray(with_spike - baseline)

        np.testing.assert_allclose(event_term[1], expected, rtol=RTOL, atol=ATOL)
        assert np.all(event_term[[0, 2, 3]] == 0.0)

    def test_local_event_term_matches_reference(self):
        env = _env_1d()
        occupancy, joint = self._models()
        expected = self._expected_event_term([5.0])[0]
        assert expected < LOG_EPS

        kwargs = _predict_kwargs(
            env,
            occupancy,
            joint,
            self.RATE,
            jnp.asarray([1.5]),
            jnp.asarray([[self.MARK]]),
        )
        local_keys = (
            "time",
            "position_time",
            "position",
            "spike_times",
            "spike_waveform_features",
            "environment",
            "occupancy_model",
            "gpi_models",
            "joint_models",
            "mean_rates",
            "disable_progress_bar",
        )
        local_kwargs = {k: kwargs[k] for k in local_keys}
        with_spike = compute_local_log_likelihood(**local_kwargs)
        local_kwargs["spike_times"] = [jnp.zeros(0)]
        local_kwargs["spike_waveform_features"] = [jnp.zeros((0, 1))]
        baseline = compute_local_log_likelihood(**local_kwargs)
        event_term = np.asarray(with_spike - baseline).ravel()

        assert event_term[1] == pytest.approx(expected, rel=RTOL, abs=ATOL)
        assert np.all(event_term[[0, 2, 3]] == 0.0)


class TestDeepTailRateTerm:
    """``log(rate)`` must survive when the two log densities are huge and equal.

    With ``rate = 1e-15`` and log densities near ``-1e10`` (float32 ulp ≈ 1024),
    ``(log_rate + joint) - occ`` rounds ``log_rate`` away and returns ``0``;
    ``log_rate + (joint - occ)`` returns ``log(1e-15) = -34.5388``. Joint and
    occupancy are the *same* model evaluated on the same points (zero mark
    dimensions), so their log densities are bit-identical and the expected
    event term is exactly ``log(rate)`` at every bin.
    """

    RATE = 1e-15

    def _model(self):
        # var = 1e-8 puts the density at bin 19.5 near -0.5 * 14.5**2 / 1e-8.
        return _gaussian([5.0], var=1e-8)

    def test_accumulator_keeps_log_rate(self):
        joint = jnp.asarray([[-1e10, -1e10]])
        log_occ = jnp.asarray([-1e10, -1e10])

        out = _accumulate_log_likelihood_block(
            jnp.zeros((2, 2)),
            joint,
            jnp.asarray([1]),
            jnp.arange(2),
            jnp.asarray(np.log(self.RATE), dtype=jnp.float32),
            log_occ,
        )

        np.testing.assert_allclose(np.asarray(out)[1], np.log(self.RATE), rtol=1e-6)

    @pytest.mark.parametrize("bin_tile_size", [None, 3], ids=["untiled", "tiled"])
    def test_non_local_event_term_is_log_rate(self, bin_tile_size):
        env = _env_1d(0.0, 30.0)
        model = self._model()
        bins = env.place_bin_centers_[env.is_track_interior_.ravel()]
        assert np.min(_log_gaussian(bins, [5.0], 1e-8)) < -1e9, "case must reach -1e9"

        kwargs = _predict_kwargs(
            env, model, model, self.RATE, jnp.asarray([1.5]), jnp.zeros((1, 0))
        )
        kwargs.update(mark_dimensions=[0], bin_tile_size=bin_tile_size)
        with_spike = predict_clusterless_gmm_log_likelihood(**kwargs)
        kwargs.update(
            spike_times=[jnp.zeros(0)], spike_waveform_features=[jnp.zeros((0, 0))]
        )
        baseline = predict_clusterless_gmm_log_likelihood(**kwargs)
        event_term = np.asarray(with_spike - baseline)

        np.testing.assert_allclose(event_term[1], np.log(self.RATE), rtol=1e-6)
        assert np.all(event_term[[0, 2, 3]] == 0.0)

    def test_local_event_term_is_log_rate(self):
        env = _env_1d(0.0, 30.0)
        model = self._model()
        far_position = 19.5
        assert _log_gaussian([far_position], [5.0], 1e-8)[0] < -1e9

        local_kwargs = {
            "time": jnp.arange(0.0, 4.0),
            "position_time": np.arange(0.0, 4.0),
            "position": jnp.full((4, 1), far_position),
            "spike_times": [jnp.asarray([1.5])],
            "spike_waveform_features": [jnp.zeros((1, 0))],
            "environment": env,
            "occupancy_model": model,
            "gpi_models": [model],
            "joint_models": [model],
            "mean_rates": jnp.asarray([self.RATE]),
            "disable_progress_bar": True,
        }
        with_spike = compute_local_log_likelihood(**local_kwargs)
        local_kwargs.update(
            spike_times=[jnp.zeros(0)], spike_waveform_features=[jnp.zeros((0, 0))]
        )
        baseline = compute_local_log_likelihood(**local_kwargs)
        event_term = np.asarray(with_spike - baseline).ravel()

        assert event_term[1] == pytest.approx(np.log(self.RATE), rel=1e-6)
        assert np.all(event_term[[0, 2, 3]] == 0.0)


class TestGroundProcessRange:
    """``rate * p_gpi / p_occ`` must be formed in log space."""

    def test_local_expected_counts_do_not_overflow(self):
        """``1e-15 * exp(95)`` overflows float32; ``exp(log 1e-15 + 95)`` fits.

        The GPI density is a very narrow Gaussian whose log density at its
        mean is ``+10.6``; the occupancy log density there is ``-85``.
        """
        env = _env_1d(0.0, 30.0)
        gpi_var = 1e-10
        gpi = _gaussian([5.0], var=gpi_var)
        occ_mean = 5.0 + np.sqrt(2.0 * (85.0 - 0.5 * np.log(2.0 * np.pi)))
        occupancy = _gaussian([occ_mean], var=1.0)
        rate = EPS
        gpi_logp = _log_gaussian([5.0], [5.0], gpi_var)[0]
        occ_logp = _log_gaussian([5.0], [occ_mean], 1.0)[0]
        assert occ_logp == pytest.approx(-85.0, abs=1e-6)
        diff = gpi_logp - occ_logp
        assert np.isinf(np.exp(np.float32(diff))), "separate exp must overflow"
        expected = -(np.exp(np.log(np.float64(rate)) + diff))
        assert np.isfinite(np.float32(expected))

        log_likelihood = compute_local_log_likelihood(
            time=jnp.arange(0.0, 3.0),
            position_time=np.arange(0.0, 3.0),
            position=jnp.full((3, 1), 5.0),
            spike_times=[jnp.zeros(0)],
            spike_waveform_features=[jnp.zeros((0, 1))],
            environment=env,
            occupancy_model=occupancy,
            gpi_models=[gpi],
            joint_models=[gpi],
            mean_rates=jnp.asarray([rate]),
            disable_progress_bar=True,
        )

        np.testing.assert_allclose(
            np.asarray(log_likelihood).ravel(), np.full(3, expected), rtol=RTOL
        )

    @pytest.mark.parametrize(
        "rate, gpi_logp, occ_logp",
        [(2.0, -800.0, -790.0), (2.0, -120.0, -60.0)],
        ids=["both_underflow", "gpi_underflows"],
    )
    def test_helper_preserves_ratio_of_underflowing_densities(
        self, rate, gpi_logp, occ_logp
    ):
        """Densities that underflow in float32 still give their finite ratio.

        ``exp(-790)`` is ``0`` in float32, so any probability-space ratio (and
        any support test based on it) discards ``rate * e^-10 = 9.08e-5``.
        """
        from non_local_detector.likelihoods.clusterless_gmm import (
            _ground_process_intensity,
        )

        assert np.exp(np.float32(gpi_logp)) == 0.0
        expected = rate * np.exp(np.float64(gpi_logp - occ_logp))

        actual = _ground_process_intensity(
            jnp.asarray(rate), jnp.asarray([gpi_logp]), jnp.asarray([occ_logp])
        )

        np.testing.assert_allclose(np.asarray(actual), [expected], rtol=RTOL)

    def test_helper_propagates_nan_occupancy(self):
        """An invalid occupancy density must not become a finite intensity."""
        from non_local_detector.likelihoods.clusterless_gmm import (
            _ground_process_intensity,
        )

        actual = _ground_process_intensity(
            jnp.asarray(2.0), jnp.asarray([-1.0, -1.0]), jnp.asarray([-1.0, np.nan])
        )

        assert np.isfinite(float(actual[0]))
        assert np.isnan(float(actual[1]))

    def test_fit_ground_process_survives_density_underflow(self):
        """Far from the data both densities underflow; the ratio is still the rate.

        GPI and occupancy models are fitted to identical samples, so the true
        ratio is exactly ``mean_rate`` at every bin. The reference uses the
        fitted models' own log densities in float64, so only the ratio
        arithmetic is under test; a probability-space ratio gives ``0 / 0``.
        """
        model, env = _fit_far_from_most_bins(seed=0)
        bins = model["interior_place_bin_centers"]
        occ_logp = np.asarray(model["log_occupancy"], dtype=np.float64)
        gpi_logp = np.asarray(
            model["gpi_models"][0].score_samples(bins), dtype=np.float64
        )
        rate = float(model["mean_rates"][0])
        expected = np.exp(np.log(rate) + gpi_logp - occ_logp)
        working_dtype = np.asarray(model["log_occupancy"]).dtype
        far = np.exp(occ_logp.astype(working_dtype)) == 0.0
        assert far.sum() >= 10, "case must underflow in the working dtype"

        actual = np.asarray(model["summed_ground_process_intensity"], dtype=np.float64)

        np.testing.assert_allclose(actual[far], expected[far], rtol=1e-4)
        np.testing.assert_allclose(actual[far], rate, rtol=1e-2)

    def test_local_likelihood_is_minus_rate_where_densities_underflow(self):
        """Identical GPI and occupancy models: no-spike local likelihood is ``-rate``.

        At a position 100 nats down both tails the probability-space ratio is
        ``0 / 0``; the log-space ratio is exactly ``1``.
        """
        env = _env_1d(0.0, 40.0)
        model = _gaussian([5.0], var=1.0)
        far_position = 5.0 + np.sqrt(300.0)  # 150 nats down: below float32 subnormals
        assert np.exp(np.float32(_log_gaussian([far_position], [5.0], 1.0)[0])) == 0.0

        log_likelihood = compute_local_log_likelihood(
            time=jnp.arange(0.0, 3.0),
            position_time=np.arange(0.0, 3.0),
            position=jnp.full((3, 1), far_position),
            spike_times=[jnp.zeros(0)],
            spike_waveform_features=[jnp.zeros((0, 1))],
            environment=env,
            occupancy_model=model,
            gpi_models=[model],
            joint_models=[model],
            mean_rates=jnp.asarray([2.0]),
            disable_progress_bar=True,
        )

        np.testing.assert_allclose(np.asarray(log_likelihood).ravel(), -2.0, rtol=RTOL)

    def test_local_likelihood_propagates_nan_position(self):
        """A NaN decode position must surface as a NaN likelihood, not a finite value."""
        env = _env_1d(0.0, 40.0)
        model = _gaussian([5.0], var=1.0)
        position = jnp.asarray([[5.0], [np.nan], [5.0]])

        log_likelihood = compute_local_log_likelihood(
            time=jnp.arange(0.0, 3.0),
            position_time=np.arange(0.0, 3.0),
            position=position,
            spike_times=[jnp.zeros(0)],
            spike_waveform_features=[jnp.zeros((0, 1))],
            environment=env,
            occupancy_model=model,
            gpi_models=[model],
            joint_models=[model],
            mean_rates=jnp.asarray([2.0]),
            disable_progress_bar=True,
        )

        # Position interpolation spreads the NaN to adjacent rows as well (a
        # pre-existing property of the local path); the contract checked here
        # is only that the invalid row is not reported as a finite value.
        assert np.isnan(np.asarray(log_likelihood).ravel()[1])

    def test_fit_and_local_ground_process_agree_at_bin_centres(self):
        """The fit-time summed intensity and the local no-spike term share one formula."""
        model, env = _fit_far_from_most_bins(seed=1)
        bins = np.asarray(model["interior_place_bin_centers"])
        n_bins = bins.shape[0]

        local = compute_local_log_likelihood(
            time=jnp.arange(n_bins, dtype=jnp.float32),
            position_time=np.arange(n_bins, dtype=np.float64),
            position=jnp.asarray(bins),
            spike_times=[jnp.zeros(0)],
            spike_waveform_features=[jnp.zeros((0, 1))],
            environment=env,
            occupancy_model=model["occupancy_model"],
            gpi_models=model["gpi_models"],
            joint_models=model["joint_models"],
            mean_rates=model["mean_rates"],
            disable_progress_bar=True,
        )

        np.testing.assert_allclose(
            -np.asarray(local).ravel(),
            np.asarray(model["summed_ground_process_intensity"]),
            rtol=1e-5,
        )
