"""Tests for ``clusterless_probe_mixture`` and ``ProbeGeometry``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from non_local_detector import ClusterlessDecoder, Environment, time_edges_from_centers
from non_local_detector.exceptions import DataQualityWarning, ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.clusterless_mark_glm import (
    fit_clusterless_mark_glm_encoding_model,
    predict_clusterless_mark_glm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_probe_mixture import (
    ProbeMarkMixture,
    fit_clusterless_probe_mixture_encoding_model,
    predict_clusterless_probe_mixture_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.likelihoods.probe_geometry import (
    ProbeGeometry,
    build_probe_tiles,
    compute_local_neighborhoods,
)


def _make_two_column_probe(n_contacts: int, pitch: float = 20.0) -> np.ndarray:
    """Neuropixels-like staggered two-column contact geometry, in microns."""
    rows = np.arange(n_contacts) // 2
    cols = np.arange(n_contacts) % 2
    x = cols * pitch + (rows % 2) * (0.5 * pitch)
    z = rows * pitch
    return np.column_stack([x, z]).astype(np.float64)


def _make_dense_probe_recording(
    *,
    n_channels: int = 32,
    n_units: int = 8,
    duration: float = 30.0,
    fs: float = 100.0,
    footprint_sigma_um: float = 35.0,
    noise_sd: float = 5.0,
    seed: int = 0,
):
    """Simulate a single dense shank on a 100 cm 1-D track with local footprints."""
    rng = np.random.default_rng(seed)
    contact_positions = _make_two_column_probe(n_channels, pitch=20.0)
    z_min, z_max = contact_positions[:, 1].min(), contact_positions[:, 1].max()
    unit_z = np.linspace(z_min + 20.0, max(z_max - 20.0, z_min + 20.0), n_units)
    unit_x = np.full(n_units, 10.0)
    unit_pos_probe = np.column_stack([unit_x, unit_z])

    # Biophysical-like local amplitude footprint across contacts.
    d2 = np.sum(
        (unit_pos_probe[:, None, :] - contact_positions[None, :, :]) ** 2, axis=-1
    )
    peak_amps = rng.uniform(70.0, 120.0, size=n_units)
    templates = peak_amps[:, None] * np.exp(-0.5 * d2 / footprint_sigma_um**2)

    time = np.arange(int(duration * fs)) / fs
    position = (50.0 + 45.0 * np.sin(2.0 * np.pi * time / 10.0))[:, None]
    place_means = np.linspace(10.0, 90.0, n_units)
    place_sigma = 8.0
    rates = 0.2 + 25.0 * np.exp(
        -0.5 * ((position[:, 0:1] - place_means[None, :]) / place_sigma) ** 2
    )

    all_times = []
    all_marks = []
    for unit in range(n_units):
        counts = rng.poisson(rates[:, unit] / fs)
        spike_t = np.repeat(time, counts) + rng.uniform(
            -0.5 / fs, 0.5 / fs, counts.sum()
        )
        spike_t = np.clip(spike_t, time[0], time[-1])
        # Low-rank burst/gain variability + channel noise
        gain = 1.0 + rng.normal(0.0, 0.08, size=(len(spike_t), 1))
        marks = gain * templates[unit][None, :] + rng.normal(
            0.0, noise_sd, size=(len(spike_t), n_channels)
        )
        all_times.append(spike_t)
        all_marks.append(marks)

    spike_times = np.concatenate(all_times)
    spike_marks = np.concatenate(all_marks, axis=0)
    order = np.argsort(spike_times)
    env = Environment(place_bin_size=5.0).fit_place_grid(position)
    return {
        "environment": env,
        "position_time": time,
        "position": position,
        "spike_times": [spike_times[order]],
        "spike_waveform_features": [spike_marks[order]],
        "contact_positions": [contact_positions],
        "templates": templates,
        "place_means": place_means,
    }


def _make_overlapping_marks_recording(seed: int = 3):
    """Two units with overlapping marks (~2.5 SD apart) and opposite place fields."""
    rng = np.random.default_rng(seed)
    fs = 100.0
    time = np.arange(4000) / fs
    position = (50.0 + 45.0 * np.sin(2.0 * np.pi * time / 8.0))[:, None]
    place_means = np.array([20.0, 80.0])
    rates = 0.2 + 30.0 * np.exp(
        -0.5 * ((position[:, 0:1] - place_means[None, :]) / 7.0) ** 2
    )
    # Separated by 2.4 SD so position-free clustering discovers both modes
    # while their overlapping tails mix place fields until joint EM separates them.
    mark_means = np.array([[0.0, 0.0, 0.0, 0.0], [24.0, 0.0, 0.0, 0.0]])
    times_list, marks_list = [], []
    for u in range(2):
        counts = rng.poisson(rates[:, u] / fs)
        t_u = np.clip(
            np.repeat(time, counts) + rng.uniform(-0.5 / fs, 0.5 / fs, counts.sum()),
            time[0],
            time[-1],
        )
        m_u = mark_means[u] + rng.normal(0.0, 10.0, size=(len(t_u), 4))
        times_list.append(t_u)
        marks_list.append(m_u)
    t_all = np.concatenate(times_list)
    m_all = np.concatenate(marks_list, axis=0)
    order = np.argsort(t_all)
    env = Environment(place_bin_size=5.0).fit_place_grid(position)
    return env, time, position, [t_all[order]], [m_all[order]]


@pytest.mark.unit
class TestProbeGeometry:
    """Validation, shank splitting, auto-pitch neighborhoods, and tiling."""

    def test_probe_geometry_splits_by_shank_with_noise_cov(self):
        pos = np.vstack(
            [
                _make_two_column_probe(6, pitch=20.0),
                _make_two_column_probe(4, pitch=20.0) + np.array([250.0, 0.0]),
            ]
        )
        shank_ids = np.array([0] * 6 + [1] * 4)
        noise = np.eye(10) * 4.0
        noise[0, 1] = noise[1, 0] = 1.0
        noise[6, 7] = noise[7, 6] = 0.5

        geom = ProbeGeometry(
            contact_positions=pos, shank_ids=shank_ids, noise_covariance=noise
        )
        pos_list, cov_list = geom.split_by_shank()
        assert len(pos_list) == 2
        assert pos_list[0].shape == (6, 2)
        assert pos_list[1].shape == (4, 2)
        np.testing.assert_allclose(cov_list[0], noise[:6, :6])
        np.testing.assert_allclose(cov_list[1], noise[6:, 6:])

    def test_probe_geometry_rejects_bad_inputs(self):
        with pytest.raises(ValidationError, match="2-D"):
            ProbeGeometry(contact_positions=np.zeros(4))
        with pytest.raises(ValidationError, match="finite"):
            ProbeGeometry(contact_positions=np.array([[0.0, np.nan], [0.0, 20.0]]))
        with pytest.raises(ValidationError, match="positive definite"):
            ProbeGeometry(
                contact_positions=_make_two_column_probe(4),
                noise_covariance=np.zeros((4, 4)),
            )

    def test_auto_neighborhoods_and_tiling_on_dense_probe(self):
        pos = _make_two_column_probe(32, pitch=20.0)
        indices, mask, radius = compute_local_neighborhoods(pos, max_local_channels=12)
        assert radius > 0.0
        assert indices.shape[0] == 32
        assert 1 <= indices.shape[1] <= 12
        assert mask.shape == indices.shape
        # Contact c is always entry 0 of its own neighborhood.
        np.testing.assert_array_equal(indices[:, 0], np.arange(32))
        assert np.all(mask[:, 0])

        tiles, tile_supports = build_probe_tiles(
            pos, radius=radius, max_local_channels=12
        )
        assert len(tiles) >= 2
        covered = sorted(c for tile in tiles for c in tile)
        assert covered == list(range(32))
        for tile, support in zip(tiles, tile_supports, strict=True):
            assert set(tile).issubset(set(support))

    def test_single_contact_and_coincident_contacts(self):
        idx1, mask1, _ = compute_local_neighborhoods(np.array([[0.0, 0.0]]))
        np.testing.assert_array_equal(idx1, [[0]])
        np.testing.assert_array_equal(mask1, [[True]])

        idx4, mask4, _ = compute_local_neighborhoods(np.zeros((4, 2)))
        assert idx4.shape == (4, 4)
        assert np.all(mask4)


@pytest.mark.unit
class TestProbeMarkMixtureMath:
    """Exact Woodbury orthonormal-basis log-density vs dense SciPy oracle."""

    @pytest.mark.parametrize(
        ("n_channels", "c_loc", "rank", "with_whitener", "bg_offset"),
        [
            (4, 4, 4, False, 0.0),
            (12, 12, 3, False, 0.0),
            (24, 8, 3, False, 15.0),
            (16, 6, 4, True, -10.0),
        ],
    )
    def test_log_component_density_matches_dense_multivariate_normal(
        self, n_channels, c_loc, rank, with_whitener, bg_offset
    ):
        rng = np.random.default_rng(42 + n_channels + c_loc + int(with_whitener))
        n_comp = 5
        pos = _make_two_column_probe(n_channels, pitch=20.0)
        nbr_idx, nbr_mask, _ = compute_local_neighborhoods(
            pos, max_local_channels=c_loc
        )
        peaks = np.linspace(0, n_channels - 1, n_comp, dtype=int)
        support_indices = nbr_idx[peaks]
        support_mask = nbr_mask[peaks]
        c_loc_actual = support_indices.shape[1]

        noise_variance = rng.uniform(1.0, 4.0, size=n_channels)
        bg_mean = rng.normal(bg_offset, 2.0, size=n_channels)

        if with_whitener:
            raw = rng.normal(size=(n_channels, n_channels))
            noise_cov = raw @ raw.T / n_channels + 0.5 * np.eye(n_channels)
            eigvals, eigvecs = np.linalg.eigh(noise_cov)
            whitener = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]) @ eigvecs.T
            log_det_whitener = -0.5 * float(np.sum(np.log(eigvals)))
            inv_whitener = (eigvecs * np.sqrt(eigvals)[None, :]) @ eigvecs.T
        else:
            whitener = None
            log_det_whitener = 0.0
            inv_whitener = np.eye(n_channels)

        means = np.zeros((n_comp, c_loc_actual))
        basis = np.zeros((n_comp, c_loc_actual, rank))
        basis_variances = np.ones((n_comp, rank))
        for k in range(n_comp):
            valid = np.flatnonzero(support_mask[k])
            means[k, valid] = rng.normal(bg_offset, 10.0, size=len(valid))
            r_eff = min(rank, len(valid))
            q, _ = np.linalg.qr(rng.normal(size=(len(valid), r_eff)))
            basis[k, valid, :r_eff] = q
            basis_variances[k, :r_eff] = rng.uniform(0.3, 12.0, size=r_eff)

        mixture = ProbeMarkMixture(
            weights=np.full(n_comp, 1.0 / n_comp),
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            support_indices=support_indices,
            support_mask=support_mask,
            noise_variance=noise_variance,
            background_mean=bg_mean,
            noise_whitener=whitener,
            log_det_whitener=log_det_whitener,
            contact_positions=pos,
            component_peak_channels=peaks,
        )

        marks = rng.normal(bg_offset, 8.0, size=(25, n_channels))
        got = mixture.log_component_density(marks)

        # Build exact full C x C mean and covariance in original coordinates.
        expected = np.empty_like(got)
        for k in range(n_comp):
            valid = np.flatnonzero(support_mask[k])
            ch = support_indices[k, valid]
            mu_w = bg_mean.copy()
            mu_w[ch] = means[k, valid]
            cov_w = np.diag(noise_variance)
            q_k = basis[k, valid]
            lam_k = basis_variances[k]
            std_loc = np.sqrt(noise_variance[ch])
            # Sigma_loc = D^(1/2) [I + Q diag(lam - 1) Q^T] D^(1/2)
            local_cov = (
                np.diag(noise_variance[ch])
                + (std_loc[:, None] * q_k)
                @ np.diag(lam_k - 1.0)
                @ (std_loc[:, None] * q_k).T
            )
            cov_w[np.ix_(ch, ch)] = local_cov
            # Map back from whitened coordinates: m = y @ inv_whitener
            mu_orig = mu_w @ inv_whitener
            cov_orig = inv_whitener.T @ cov_w @ inv_whitener
            expected[:, k] = multivariate_normal.logpdf(
                marks, mean=mu_orig, cov=cov_orig
            )

        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)

    def test_high_condition_number_and_offset_stay_accurate_in_float32(self):
        """Orthonormal Woodbury avoids catastrophic cancellation at s=1e4."""
        import jax.numpy as jnp

        from non_local_detector.likelihoods.clusterless_probe_mixture import (
            _woodbury_log_density,
        )

        n_channels = 8
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(n_channels, 2)))
        basis = q[None, :, :]  # (1, 8, 2)
        # Condition number 1e8 along basis[0]!
        basis_variances = np.array([[1.0 + 1e8, 1.0 + 1e4]])
        offset = np.full((1, n_channels), 100.0)
        means = offset.copy()
        noise_var = np.ones(n_channels)
        support_idx = np.arange(n_channels, dtype=np.int32)[None, :]
        support_mask = np.ones((1, n_channels), dtype=bool)

        mixture = ProbeMarkMixture(
            weights=np.array([1.0]),
            means=means,
            basis=basis,
            basis_variances=basis_variances,
            support_indices=support_idx,
            support_mask=support_mask,
            noise_variance=noise_var,
            background_mean=np.full(n_channels, 100.0),
            noise_whitener=None,
            log_det_whitener=0.0,
            contact_positions=None,
            component_peak_channels=np.array([0], dtype=np.int32),
        )
        # Point 500 units along the high-variance direction: true d^2 is tiny (~2.5e-3).
        marks = offset + 500.0 * q[:, 0][None, :]
        exact = mixture.log_component_density(marks)
        f32 = np.asarray(
            _woodbury_log_density(
                jnp.asarray(marks, dtype=jnp.float32),
                jnp.asarray(means, dtype=jnp.float32),
                jnp.asarray(basis, dtype=jnp.float32),
                jnp.asarray(basis_variances, dtype=jnp.float32),
                jnp.asarray(support_idx, dtype=jnp.int32),
                jnp.asarray(support_mask),
                jnp.asarray(noise_var, dtype=jnp.float32),
                jnp.asarray(mixture.background_mean, dtype=jnp.float32),
                jnp.asarray([True]),
                all_full_support=True,
            )
        )
        assert np.all(np.isfinite(f32))
        np.testing.assert_allclose(f32, exact, rtol=1e-5, atol=1e-3)


@pytest.mark.integration
class TestNestedLimitAndJointEM:
    """Exact nested limit against mark GLM, EM monotonicity, and overlapping units."""

    def test_iter0_full_rank_no_geometry_matches_clusterless_mark_glm(self):
        env, time, pos, spikes, marks = _make_overlapping_marks_recording(seed=1)
        ref = fit_mark_references(marks, kernel_scale=1.0)
        mark_glm = fit_clusterless_mark_glm_encoding_model(
            time,
            pos,
            spikes,
            marks,
            env,
            n_mark_components=8,
            mark_reg_covar=1.0,
            disable_progress_bar=True,
        )
        probe_mix = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            spikes,
            marks,
            env,
            n_mark_components=8,
            mark_reg_covar=1.0,
            covariance_rank=None,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        edges = time_edges_from_centers(time[:500])
        for is_local in (False, True):
            ll_glm = np.asarray(
                predict_clusterless_mark_glm_log_likelihood(
                    time[:500],
                    pos[:500],
                    spikes,
                    marks,
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=is_local,
                    **mark_glm,
                )
            )
            ll_probe = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    time[:500],
                    pos[:500],
                    spikes,
                    marks,
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=is_local,
                    **probe_mix,
                )
            )
            np.testing.assert_allclose(ll_probe, ll_glm, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("use_geometry", [False, True])
    @pytest.mark.parametrize("use_noise_cov", [False, True])
    def test_generalized_em_objective_is_monotone(self, use_geometry, use_noise_cov):
        rec = _make_dense_probe_recording(
            n_channels=12, n_units=4, duration=12.0, seed=2
        )
        contact_positions = rec["contact_positions"] if use_geometry else None
        noise_covariances = [np.eye(12) * 25.0] if use_noise_cov else None
        model = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=contact_positions,
            noise_covariances=noise_covariances,
            n_mark_components=6,
            covariance_rank=3,
            max_local_channels=8,
            n_em_iterations=8,
            em_validation_blocks=0,
            em_tol=-np.inf,
            disable_progress_bar=True,
        )
        trace = model["em_diagnostics"]["train_log_likelihood"]
        assert len(trace) >= 2
        diffs = np.diff(trace)
        assert np.all(diffs >= -1e-4), f"Non-monotone EM trace: {trace}"

    def test_joint_em_improves_held_out_likelihood_for_overlapping_marks(self):
        env, time, pos, spikes, marks = _make_overlapping_marks_recording(seed=5)
        split = len(time) // 2
        t_cut = time[split]
        tr_spikes = [s[s < t_cut] for s in spikes]
        tr_marks = [m[s < t_cut] for s, m in zip(spikes, marks, strict=True)]
        te_spikes = [s[s >= t_cut] for s in spikes]
        te_marks = [m[s >= t_cut] for s, m in zip(spikes, marks, strict=True)]
        ref = fit_mark_references(tr_marks, kernel_scale=1.0)
        te_edges = time_edges_from_centers(time[split:])

        fit_iter0 = fit_clusterless_probe_mixture_encoding_model(
            time[:split],
            pos[:split],
            tr_spikes,
            tr_marks,
            env,
            n_mark_components=2,
            covariance_rank=None,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        fit_joint = fit_clusterless_probe_mixture_encoding_model(
            time[:split],
            pos[:split],
            tr_spikes,
            tr_marks,
            env,
            n_mark_components=2,
            covariance_rank=None,
            n_em_iterations=25,
            em_validation_blocks=2,
            em_patience=5,
            disable_progress_bar=True,
        )
        ll0 = float(
            np.sum(
                predict_clusterless_probe_mixture_log_likelihood(
                    time[split:],
                    pos[split:],
                    te_spikes,
                    te_marks,
                    time_edges=te_edges,
                    mark_reference=ref,
                    is_local=True,
                    **fit_iter0,
                )
            )
        )
        ll_joint = float(
            np.sum(
                predict_clusterless_probe_mixture_log_likelihood(
                    time[split:],
                    pos[split:],
                    te_spikes,
                    te_marks,
                    time_edges=te_edges,
                    mark_reference=ref,
                    is_local=True,
                    **fit_joint,
                )
            )
        )
        assert fit_joint["em_diagnostics"]["n_iterations"] > 0
        assert ll_joint > ll0 + 10.0


@pytest.mark.integration
class TestDenseProbeAndCandidatePruning:
    """High-channel probe decoding, local support bounds, and top-k candidate LUT."""

    def test_dense_32ch_shank_decodes_and_candidate_lut_matches_all_k(self):
        rec = _make_dense_probe_recording(
            n_channels=32, n_units=8, duration=20.0, seed=9
        )
        time = rec["position_time"]
        pos = rec["position"]
        split = int(0.75 * len(time))
        t_cut = time[split]
        tr_spikes = [s[s < t_cut] for s in rec["spike_times"]]
        tr_marks = [
            m[s < t_cut]
            for s, m in zip(
                rec["spike_times"], rec["spike_waveform_features"], strict=True
            )
        ]
        te_spikes = [s[s >= t_cut] for s in rec["spike_times"]]
        te_marks = [
            m[s >= t_cut]
            for s, m in zip(
                rec["spike_times"], rec["spike_waveform_features"], strict=True
            )
        ]

        decoder = ClusterlessDecoder(
            environments=rec["environment"],
            sampling_frequency=100.0,
            clusterless_algorithm="clusterless_probe_mixture",
            clusterless_algorithm_params={
                "contact_positions": rec["contact_positions"],
                "max_local_channels": 10,
                "covariance_rank": 3,
                "n_mark_components": 8,
                "n_em_iterations": 5,
                "em_validation_blocks": 0,
                "disable_progress_bar": True,
            },
        ).fit(time[:split], pos[:split], tr_spikes, tr_marks)

        (enc,) = decoder.encoding_model_.values()
        mixture = enc["mark_mixtures"][0]
        assert mixture is not None
        assert mixture.support_indices.shape[1] <= 10
        assert mixture.n_components >= 4

        te_edges = time_edges_from_centers(time[split:])
        res_all = decoder.predict(
            te_spikes,
            te_marks,
            time_edges=te_edges,
            position=pos[split:],
            position_time=time[split:],
        )
        post = res_all.acausal_posterior.values
        bin_centers = rec["environment"].place_bin_centers_[
            rec["environment"].is_track_interior_flat_
        ][:, 0]
        map_pos = bin_centers[np.argmax(post, axis=1)]
        rmse = float(np.sqrt(np.mean((map_pos - pos[split:, 0]) ** 2)))
        assert rmse < 15.0

        # Candidate LUT with k_cand >= K must match all-K to float32 rounding.
        ll_all = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_spikes,
                te_marks,
                time_edges=te_edges,
                mark_reference=decoder.mark_reference_,
                is_local=False,
                **enc,
            )
        )
        ll_full_lut = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_spikes,
                te_marks,
                time_edges=te_edges,
                mark_reference=decoder.mark_reference_,
                is_local=False,
                **{**enc, "max_candidate_components": mixture.n_components + 4},
            )
        )
        np.testing.assert_allclose(ll_full_lut, ll_all, rtol=1e-5, atol=1e-4)

        # Local path with k_cand >= K also matches all-K.
        ll_loc_all = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_spikes,
                te_marks,
                time_edges=te_edges,
                mark_reference=decoder.mark_reference_,
                is_local=True,
                **enc,
            )
        )
        ll_loc_lut = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_spikes,
                te_marks,
                time_edges=te_edges,
                mark_reference=decoder.mark_reference_,
                is_local=True,
                **{**enc, "max_candidate_components": mixture.n_components + 4},
            )
        )
        np.testing.assert_allclose(ll_loc_lut, ll_loc_all, rtol=1e-5, atol=1e-4)

        # Pruned LUT (k_cand < K) produces finite, accurate non-local and local scores.
        k_small = max(2, mixture.n_components // 2)
        ll_pruned = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_spikes,
                te_marks,
                time_edges=te_edges,
                mark_reference=decoder.mark_reference_,
                is_local=False,
                **{**enc, "max_candidate_components": k_small},
            )
        )
        assert np.all(np.isfinite(ll_pruned))
        assert ll_pruned.shape == ll_all.shape

    def test_multitile_respects_n_mark_components_budget(self):
        """Multi-tile initialization budgets n_mark_components across the whole shank."""
        rec = _make_dense_probe_recording(
            n_channels=64, n_units=12, duration=15.0, seed=21
        )
        enc = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            max_local_channels=10,
            covariance_rank=4,
            n_mark_components=24,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mixture = enc["mark_mixtures"][0]
        assert mixture is not None
        assert 12 <= mixture.n_components <= 24

    def test_zca_whitened_noise_variance_is_unit_regardless_of_mark_reg_covar(self):
        """With ZCA noise_covariances, whitened noise_variance is 1.0 independent of mark_reg_covar."""
        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=12.0, seed=22
        )
        pos_c = rec["contact_positions"][0]
        dist = np.sqrt(np.sum((pos_c[:, None, :] - pos_c[None, :, :]) ** 2, axis=-1))
        noise_cov = 20.0 * np.exp(-dist / 40.0) + 5.0 * np.eye(16)

        fits = {}
        for reg in (0.01, 25.0):
            fits[reg] = fit_clusterless_probe_mixture_encoding_model(
                rec["position_time"],
                rec["position"],
                rec["spike_times"],
                rec["spike_waveform_features"],
                rec["environment"],
                contact_positions=rec["contact_positions"],
                noise_covariances=[noise_cov],
                max_local_channels=8,
                covariance_rank=3,
                n_mark_components=8,
                mark_reg_covar=reg,
                n_em_iterations=2,
                em_validation_blocks=0,
                disable_progress_bar=True,
            )
            mix = fits[reg]["mark_mixtures"][0]
            assert mix is not None
            np.testing.assert_allclose(mix.noise_variance, np.ones(16), atol=1e-12)

        mix_small = fits[0.01]["mark_mixtures"][0]
        mix_large = fits[25.0]["mark_mixtures"][0]
        assert mix_small is not None
        assert mix_large is not None
        probe_marks = rec["spike_waveform_features"][0][:50]
        np.testing.assert_allclose(
            mix_small.log_component_density(probe_marks),
            mix_large.log_component_density(probe_marks),
            rtol=1e-10,
            atol=1e-10,
        )

        # Simultaneous raw-unit rescaling m -> s*m, noise_cov -> s^2*noise_cov
        # must shift log_component_density by exact -C * log(s).
        scale = 10.0
        scaled_fit = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            [m * scale for m in rec["spike_waveform_features"]],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            noise_covariances=[noise_cov * (scale**2)],
            max_local_channels=8,
            covariance_rank=3,
            n_mark_components=8,
            mark_reg_covar=0.01,
            n_em_iterations=2,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix_scaled = scaled_fit["mark_mixtures"][0]
        assert mix_scaled is not None
        np.testing.assert_allclose(mix_scaled.noise_variance, np.ones(16), atol=1e-12)
        np.testing.assert_allclose(
            mix_scaled.log_component_density(probe_marks * scale)
            + 16.0 * np.log(scale),
            mix_small.log_component_density(probe_marks),
            rtol=1e-6,
            atol=1e-5,
        )


@pytest.mark.unit
class TestEdgeCasesAndInvariants:
    """Zero-rate, single-spike, row_slice, binary weights, and mark integral."""

    def test_registered_in_clusterless_algorithms(self):
        assert "clusterless_probe_mixture" in _CLUSTERLESS_ALGORITHMS

    def test_zero_rate_and_single_spike_electrodes(self):
        time = np.arange(200) * 0.01
        pos = np.linspace(0.0, 40.0, 200)[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        spikes = [np.array([]), np.array([0.5])]
        marks = [np.zeros((0, 3)), np.array([[10.0, -5.0, 2.0]])]
        with pytest.warns(DataQualityWarning, match="no effective encoding spikes"):
            enc = fit_clusterless_probe_mixture_encoding_model(
                time,
                pos,
                spikes,
                marks,
                env,
                n_em_iterations=3,
                em_validation_blocks=0,
                disable_progress_bar=True,
            )
        assert enc["mark_mixtures"][0] is None
        assert enc["mark_mixtures"][1] is not None
        assert enc["mark_mixtures"][1].n_components == 1
        ref = fit_mark_references(marks, kernel_scale=1.0)
        edges = time_edges_from_centers(time[:50])
        ll = np.asarray(
            predict_clusterless_probe_mixture_log_likelihood(
                time[:50],
                pos[:50],
                [np.array([0.1]), np.array([0.2])],
                [np.array([[1.0, 2.0, 3.0]]), np.array([[1e38, 0.0, 0.0]])],
                time_edges=edges,
                mark_reference=ref,
                **enc,
            )
        )
        assert np.all(np.isfinite(ll))

    def test_mark_integral_recovers_ground_process_intensity(self):
        """1-D mark integral of lambda(x, m) equals summed_ground_process_intensity."""
        rng = np.random.default_rng(12)
        time = np.arange(500) * 0.01
        pos = (20.0 + 18.0 * np.sin(2 * np.pi * time / 2.5))[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        spike_t = np.sort(rng.uniform(time[0], time[-1], 200))
        spike_m = rng.normal(5.0, 2.0, size=(200, 1))
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            [spike_t],
            [spike_m],
            env,
            n_mark_components=4,
            n_em_iterations=3,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        grid = np.linspace(-25.0, 35.0, 1200)[:, None]
        dm = float(grid[1, 0] - grid[0, 0])
        mixture = enc["mark_mixtures"][0]
        assert mixture is not None
        log_dens = mixture.log_component_density(grid)  # (n_grid, K)
        rates = np.exp(enc["log_rate_maps"][0])  # (n_bins, K)
        # Integrate sum_k lambda_k(x) N(m; mu_k, sigma_k^2) over m
        integrated = (np.exp(log_dens).sum(axis=0) * dm) @ rates.T
        np.testing.assert_allclose(
            integrated, enc["summed_ground_process_intensity"], rtol=1e-4, atol=1e-5
        )

    def test_row_slice_matches_full_decode_for_all_k_and_candidates(self):
        rec = _make_dense_probe_recording(
            n_channels=24, n_units=6, duration=15.0, seed=13
        )
        time, pos = rec["position_time"], rec["position"]
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            max_local_channels=8,
            covariance_rank=3,
            n_mark_components=6,
            n_em_iterations=3,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)
        edges = time_edges_from_centers(time[:300])
        sl = slice(40, 120)
        for is_local in (False, True):
            for max_cand in (None, 4):
                kwargs = {**enc, "max_candidate_components": max_cand}
                full = np.asarray(
                    predict_clusterless_probe_mixture_log_likelihood(
                        time,
                        pos,
                        rec["spike_times"],
                        rec["spike_waveform_features"],
                        time_edges=edges,
                        mark_reference=ref,
                        is_local=is_local,
                        **kwargs,
                    )
                )
                sliced = np.asarray(
                    predict_clusterless_probe_mixture_log_likelihood(
                        time,
                        pos,
                        rec["spike_times"],
                        rec["spike_waveform_features"],
                        time_edges=edges,
                        mark_reference=ref,
                        is_local=is_local,
                        row_slice=sl,
                        **kwargs,
                    )
                )
                np.testing.assert_allclose(sliced, full[sl], rtol=1e-5, atol=1e-4)

    def test_binary_weights_match_hard_subset_fit(self):
        rec = _make_dense_probe_recording(
            n_channels=10, n_units=3, duration=12.0, seed=14
        )
        time, pos = rec["position_time"], rec["position"]
        cut = len(time) // 2
        weights = np.where(np.arange(len(time)) < cut, 1.0, 0.0)
        sub_t = time[:cut]
        sub_p = pos[:cut]
        # Keep spikes strictly inside the half-open subset window so weights match.
        t_hi = time[cut - 1]
        spikes = [s[s <= t_hi - 0.02] for s in rec["spike_times"]]
        marks = [
            m[s <= t_hi - 0.02]
            for s, m in zip(
                rec["spike_times"], rec["spike_waveform_features"], strict=True
            )
        ]
        common = {
            "n_mark_components": 4,
            "covariance_rank": 2,
            "n_em_iterations": 2,
            "em_validation_blocks": 0,
            "disable_progress_bar": True,
        }
        fit_w = fit_clusterless_probe_mixture_encoding_model(
            time, pos, spikes, marks, rec["environment"], weights=weights, **common
        )
        fit_sub = fit_clusterless_probe_mixture_encoding_model(
            sub_t, sub_p, spikes, marks, rec["environment"], **common
        )
        np.testing.assert_allclose(
            fit_w["mean_rates"], fit_sub["mean_rates"], rtol=1e-4
        )
        np.testing.assert_allclose(
            fit_w["summed_ground_process_intensity"],
            fit_sub["summed_ground_process_intensity"],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_nld_save_load_round_trip_with_probe_geometry_and_noise(self, tmp_path):
        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=12.0, seed=15
        )
        pos_c = rec["contact_positions"][0]
        noise = np.eye(16) * 20.0 + 1.5
        geom = ProbeGeometry(contact_positions=pos_c, noise_covariance=noise)
        detector = ClusterlessDecoder(
            environments=rec["environment"],
            sampling_frequency=100.0,
            clusterless_algorithm="clusterless_probe_mixture",
            clusterless_algorithm_params={
                "contact_positions": geom,
                "max_local_channels": 8,
                "covariance_rank": 3,
                "n_mark_components": 6,
                "n_em_iterations": 2,
                "em_validation_blocks": 0,
                "disable_progress_bar": True,
            },
        ).fit(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
        )
        path = tmp_path / "probe_mixture.nld"
        detector.save_model(path)
        loaded = ClusterlessDecoder.load_model(path)
        edges = time_edges_from_centers(rec["position_time"][:200])
        res_orig = detector.predict(
            rec["spike_times"],
            rec["spike_waveform_features"],
            time_edges=edges,
            position=rec["position"][:200],
            position_time=rec["position_time"][:200],
        )
        res_loaded = loaded.predict(
            rec["spike_times"],
            rec["spike_waveform_features"],
            time_edges=edges,
            position=rec["position"][:200],
            position_time=rec["position_time"][:200],
        )
        np.testing.assert_array_equal(
            res_loaded.acausal_posterior.values, res_orig.acausal_posterior.values
        )

    def test_no_jax_recompilation_across_varying_spike_counts(self):
        """Repeated decode chunks with varying spike counts share compiled kernels."""
        from non_local_detector.likelihoods import clusterless_probe_mixture as cpm

        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=12.0, seed=16
        )
        time, pos = rec["position_time"], rec["position"]
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            max_local_channels=8,
            covariance_rank=3,
            n_mark_components=6,
            n_em_iterations=2,
            em_validation_blocks=0,
            spike_block_size=64,
            disable_progress_bar=True,
        )
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)
        # Warm up on a fixed bin-count window with 64-spike padded block.
        edges0 = time_edges_from_centers(time[:100])
        predict_clusterless_probe_mixture_log_likelihood(
            None,
            None,
            rec["spike_times"],
            rec["spike_waveform_features"],
            time_edges=edges0,
            mark_reference=ref,
            **enc,
        )
        before = cpm._add_electrode_intensities._cache_size()
        # Different windows with different spike counts (still <= 64 padded block).
        for start in (100, 200, 300, 400):
            edges = time_edges_from_centers(time[start : start + 100])
            ll = predict_clusterless_probe_mixture_log_likelihood(
                None,
                None,
                rec["spike_times"],
                rec["spike_waveform_features"],
                time_edges=edges,
                mark_reference=ref,
                **enc,
            )
            assert np.all(np.isfinite(np.asarray(ll)))
        # At most one extra bucket size if spike count crosses a power-of-2 bucket;
        # identical padded sizes must reuse the cached executable.
        after = cpm._add_electrode_intensities._cache_size()
        assert after - before <= 2

    def test_local_support_unit_change_shifts_by_exact_jacobian(self):
        """With C_loc < C, diagonal affine mark rescaling shifts LL by -n*log|det A|."""
        from non_local_detector.likelihoods.mark_reference import MarkReference

        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=12.0, seed=17
        )
        time, pos = rec["position_time"], rec["position"]
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            max_local_channels=8,
            covariance_rank=3,
            n_mark_components=6,
            n_em_iterations=3,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        assert not mix.all_full_support
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)
        rng = np.random.default_rng(18)
        scale = rng.uniform(0.3, 3.0, size=16)
        offset = rng.normal(0.0, 20.0, size=16)
        sup, mask = mix.support_indices, mix.support_mask
        mix_t = ProbeMarkMixture(
            weights=mix.weights,
            means=np.where(mask, mix.means * scale[sup] + offset[sup], 0.0),
            basis=mix.basis,
            basis_variances=mix.basis_variances,
            support_indices=sup,
            support_mask=mask,
            noise_variance=mix.noise_variance * (scale**2),
            background_mean=mix.background_mean * scale + offset,
            noise_whitener=None,
            log_det_whitener=0.0,
            contact_positions=mix.contact_positions,
            component_peak_channels=mix.component_peak_channels,
        )
        ref_t = [
            None
            if r is None
            else MarkReference(r.location * scale + offset, r.scale * np.abs(scale))
            for r in ref
        ]
        edges = time_edges_from_centers(time[:250])
        marks_t = [m * scale + offset for m in rec["spike_waveform_features"]]
        counts = np.histogram(rec["spike_times"][0], edges)[0]
        log_jac = float(np.sum(np.log(np.abs(scale))))
        for max_cand in (None, 6):
            ll_orig = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    None,
                    None,
                    rec["spike_times"],
                    rec["spike_waveform_features"],
                    time_edges=edges,
                    mark_reference=ref,
                    **{**enc, "max_candidate_components": max_cand},
                )
            )
            ll_t = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    None,
                    None,
                    rec["spike_times"],
                    marks_t,
                    time_edges=edges,
                    mark_reference=ref_t,
                    **{
                        **enc,
                        "mark_mixtures": [mix_t],
                        "max_candidate_components": max_cand,
                    },
                )
            )
            np.testing.assert_allclose(
                ll_t + counts[:, None] * log_jac, ll_orig, rtol=1e-4, atol=1e-3
            )
