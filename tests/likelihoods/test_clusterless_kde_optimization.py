"""Vectorized clusterless KDE kernels and reductions against direct references.

``log_kde_distance`` against scipy's normal density, the scaled
GEMM joint-intensity reduction against a float64-kernel logsumexp oracle, and the
degeneracy rules of ``_log_joint_from_log_marginal``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector.likelihoods.clusterless_kde import (
    _estimate_with_enc_chunking,
    _log_joint_from_log_marginal,
    log_kde_distance,
)
from non_local_detector.likelihoods.common import EPS, LOG_EPS
from non_local_detector.likelihoods.mark_reference import MarkReference


class TestKdeDistance:
    """The log kernel product against scipy's float64 normal density.

    Measured max error: 3.4e-6 nats (|log| up to 26). Using the variance for
    the std is a 20-nat error; a normalizer off by sqrt(2) per dimension is
    1.4 nats.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(42)
        eval_points = rng.standard_normal((30, 4))
        samples = rng.standard_normal((50, 4))
        std = rng.uniform(0.5, 2.0, 4)
        # Per-dimension log densities, shape (n_samples, n_eval_points, n_dims).
        per_dimension = norm.logpdf(eval_points[None], loc=samples[:, None], scale=std)
        reference = per_dimension.sum(axis=-1)
        return (jnp.asarray(x) for x in (eval_points, samples, std)), reference

    def test_log_kde_distance_matches_scipy(self):
        (eval_points, samples, std), reference = self._data()
        np.testing.assert_allclose(
            log_kde_distance(eval_points, samples, std), reference, rtol=0, atol=1e-4
        )


class TestScaledGemmJointIntensity:
    """The scaled-GEMM reduction matches a logsumexp oracle for any mark dimension.

    Each spike's term is ``logaddexp(log joint intensity, log background)``; the
    reduction must be exact wherever the intensity can change that term.
    """

    @staticmethod
    def _background(dec_wf):
        """Realistic per-spike background: rate floor times a broad mark density."""
        reference = MarkReference(
            np.zeros(dec_wf.shape[1]), np.full(dec_wf.shape[1], 50.0)
        )
        return jnp.asarray(LOG_EPS + reference.log_density(np.asarray(dec_wf)))

    @staticmethod
    def _gemm_terms(
        dec_wf, enc_wf, wf_std, logK_pos, log_w, occupancy, mean_rate, background
    ):
        n_enc = enc_wf.shape[0]
        log_joint = _estimate_with_enc_chunking(
            dec_wf,
            enc_wf,
            wf_std,
            occupancy,
            mean_rate,
            logK_pos,
            jnp.broadcast_to(log_w, (n_enc,)),
            n_enc,
            None,
            background,
        )
        return jnp.logaddexp(log_joint, background[:, None])

    @staticmethod
    def _logsumexp_reference(
        dec_wf, enc_wf, wf_std, logK_pos, log_w, occupancy, mean_rate, background
    ):
        """Independent logsumexp oracle for the per-spike terms."""
        # Float64 Gaussian mark kernel from each pair's feature difference,
        # shape (n_enc, n_dec).
        std = np.asarray(wf_std, dtype=np.float64)
        z = (
            np.asarray(enc_wf, dtype=np.float64)[:, None, :]
            - np.asarray(dec_wf, dtype=np.float64)[None, :, :]
        ) / std
        logK_mark = jnp.asarray(
            -0.5 * np.sum(z**2, axis=-1) - np.sum(np.log(std * np.sqrt(2 * np.pi)))
        )

        def for_one_dec(wf_col):
            return jax.nn.logsumexp(log_w + logK_pos + wf_col[:, None], axis=0)

        log_marginal = jax.vmap(for_one_dec, in_axes=1)(logK_mark)
        log_joint = jnp.where(
            occupancy[None, :] > 0.0,
            jnp.log(mean_rate) + log_marginal - jnp.log(occupancy)[None, :],
            -jnp.inf,
        )
        return jnp.logaddexp(log_joint, background[:, None])

    @staticmethod
    def _make_test_data(rng, n_enc, n_dec, n_pos, n_wf):
        """Create realistic test kernels from simulated spike data."""
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        wf_std = jnp.full(n_wf, 24.0)
        enc_pos = jnp.array(rng.uniform(0, 200, (n_enc, 1)))
        eval_pos = jnp.linspace(0, 200, n_pos)[:, None]

        logK_pos = log_kde_distance(eval_pos, enc_pos, jnp.array([3.5]))
        log_w = -jnp.log(float(n_enc))
        occupancy = jnp.ones(n_pos) * 0.01
        background = TestScaledGemmJointIntensity._background(dec_wf)
        return dec_wf, enc_wf, wf_std, logK_pos, log_w, occupancy, background

    def _assert_matches(self, rng, n_enc, n_dec, n_pos, n_wf):
        data = self._make_test_data(rng, n_enc, n_dec, n_pos, n_wf)
        result = self._gemm_terms(*data[:6], 5.0, data[6])
        reference = self._logsumexp_reference(*data[:6], 5.0, data[6])
        max_diff = float(jnp.max(jnp.abs(result - reference)))
        assert max_diff < 1e-4, f"max abs diff {max_diff:.2e}"

    # At 8 features many intensities sit far below LOG_EPS; they still count.
    # Feature count does not select an approximation: 9D is exact too.
    @pytest.mark.parametrize(
        ("seed", "n_enc", "n_dec", "n_pos", "n_wf"),
        [(42, 500, 50, 100, 4), (42, 500, 50, 100, 8), (900, 1000, 100, 200, 9)],
        ids=["4d", "8d", "9d"],
    )
    def test_matches_logsumexp(self, seed, n_enc, n_dec, n_pos, n_wf):
        self._assert_matches(np.random.default_rng(seed), n_enc, n_dec, n_pos, n_wf)

    def test_matches_across_multiple_seeds(self):
        for seed in range(5):
            self._assert_matches(np.random.default_rng(seed * 100), 1000, 100, 200, 4)

    def test_zero_occupancy_bins_score_the_background(self):
        """A bin without support has zero intensity: the background explains it."""
        rng = np.random.default_rng(42)
        n_enc, n_dec, n_pos, n_wf = 100, 10, 20, 4
        enc_wf = jnp.array(rng.standard_normal((n_enc, n_wf)) * 50)
        dec_wf = jnp.array(rng.standard_normal((n_dec, n_wf)) * 50)
        occupancy = jnp.ones(n_pos).at[5].set(0.0).at[15].set(0.0) * 0.01
        background = self._background(dec_wf)

        result = self._gemm_terms(
            dec_wf,
            enc_wf,
            jnp.full(n_wf, 24.0),
            jnp.zeros((n_enc, n_pos)),
            -jnp.log(float(n_enc)),
            occupancy,
            5.0,
            background,
        )

        np.testing.assert_array_equal(result[:, 5], background)
        np.testing.assert_array_equal(result[:, 15], background)
        assert jnp.all(result[:, 0] > background)


class TestLogJointFromLogMarginal:
    """The shared degeneracy combiner: zero support is ``-inf`` (the caller's
    background then explains the spike), and NaN (a broken computation)
    propagates."""

    def test_finite_marginal_matches_formula(self):
        """A supported bin returns log(mean_rate) + log_marginal - log(occ)."""
        log_marginal = jnp.array([[-2.0]])
        occupancy = jnp.array([0.5])
        out = _log_joint_from_log_marginal(log_marginal, 2.5, occupancy)
        expected = jnp.log(2.5) + (-2.0) - jnp.log(0.5)
        assert jnp.allclose(out[0, 0], expected)

    def test_neg_inf_marginal_at_occupied_bin_is_zero_intensity(self):
        log_marginal = jnp.array([[-2.0, -jnp.inf]])
        occupancy = jnp.array([0.5, 0.5])
        out = _log_joint_from_log_marginal(log_marginal, 2.5, occupancy)
        assert jnp.isfinite(out[0, 0])
        assert jnp.isneginf(out[0, 1])

    def test_zero_occupancy_is_zero_intensity(self):
        log_marginal = jnp.array([[-2.0, -1.0]])
        occupancy = jnp.array([0.5, 0.0])
        out = _log_joint_from_log_marginal(log_marginal, 2.5, occupancy)
        assert jnp.isneginf(out[0, 1])

    def test_nan_marginal_propagates(self):
        """A NaN marginal signals a broken computation and must reach the HMM's
        NaN diagnostics rather than becoming a finite value."""
        log_marginal = jnp.array([[-2.0, jnp.nan]])
        occupancy = jnp.array([0.5, 0.5])
        out = _log_joint_from_log_marginal(log_marginal, 2.5, occupancy)
        assert jnp.isfinite(out[0, 0])
        assert jnp.isnan(out[0, 1])

    def test_nan_marginal_at_zero_occupancy_still_propagates(self):
        """Occupancy can underflow to exactly 0 in float32, so a NaN could
        co-occur with it; the zero-support rule must not hide it."""
        log_marginal = jnp.array([[jnp.nan, -1.0]])
        occupancy = jnp.array([0.0, 0.5])
        out = _log_joint_from_log_marginal(log_marginal, 2.5, occupancy)
        assert jnp.isnan(out[0, 0])
        finite = _log_joint_from_log_marginal(jnp.array([[2.0, -1.0]]), 2.5, occupancy)
        assert jnp.isneginf(finite[0, 0])

    def test_subeps_mean_rate_preserves_its_true_positive_log(self):
        """Positive rates retain their magnitude; nothing floors them."""
        log_marginal = jnp.array([[2.0, -1.0]])
        occupancy = jnp.array([0.5, 0.5])
        out_tiny = _log_joint_from_log_marginal(log_marginal, 1e-20, occupancy)
        out_eps = _log_joint_from_log_marginal(log_marginal, EPS, occupancy)
        assert jnp.all(out_tiny < out_eps)
        expected = jnp.log(1e-20) + log_marginal - jnp.log(occupancy)[None, :]
        assert jnp.allclose(out_tiny, expected)
        assert jnp.allclose(out_eps - out_tiny, jnp.log(EPS / 1e-20))

    def test_zero_mean_rate_is_zero_intensity(self):
        """A zero mean rate (fully de-weighted electrode) has zero intensity at
        every bin, even with a finite marginal."""
        log_marginal = jnp.array([[2.0, -1.0]])
        occupancy = jnp.array([0.5, 0.5])
        out = _log_joint_from_log_marginal(log_marginal, 0.0, occupancy)
        assert jnp.all(jnp.isneginf(out))
