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

    def test_numerically_singular_noise_covariance_is_rejected(self):
        """A common-average-referenced covariance has a zero eigenvalue that
        rounds to a tiny positive one; its whitener would amplify float noise."""
        from non_local_detector.likelihoods.probe_geometry import (
            validate_noise_covariance,
        )

        n_channels = 16
        depth = 30.0 * np.arange(n_channels)
        spatial = np.exp(-np.abs(depth[:, None] - depth[None, :]) / 60.0)
        single_reference = 64.0 * (0.5 + 0.3 * spatial + 0.2 * np.eye(n_channels))
        car = np.eye(n_channels) - 1.0 / n_channels
        common_average = car @ single_reference @ car
        eigvals = np.linalg.eigvalsh(0.5 * (common_average + common_average.T))
        assert eigvals[-1] / eigvals[0] > 1e12 or eigvals[0] <= 0.0
        validate_noise_covariance(single_reference, n_channels)
        with pytest.raises(ValidationError, match="ill-conditioned"):
            validate_noise_covariance(common_average, n_channels)
        with pytest.raises(ValidationError, match="ill-conditioned"):
            ProbeGeometry(
                contact_positions=_make_two_column_probe(n_channels),
                noise_covariance=common_average + 1e-7 * np.eye(n_channels),
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


@pytest.mark.unit
class TestReviewRegressions:
    """Targeted regression tests for PR #271 review findings."""

    def test_full_covariance_preserves_channel_order_with_geometry(self):
        """Comment 1 (P1): geometry channel ordering matches full-cov means & basis."""
        rng = np.random.default_rng(101)
        time = np.arange(200) * 0.01
        pos = np.linspace(0.0, 50.0, 200)[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        sp_t = [np.sort(rng.uniform(0.05, 1.95, 100))]
        # Centered at [0, 50] so peak contact is channel 1 -> support_indices=[1, 0].
        sp_m = [np.array([0.0, 50.0]) + rng.normal(0.0, 1.0, size=(100, 2))]
        coords = np.array([[0.0, 0.0], [0.0, 20.0]])

        fit_no_geom = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            n_mark_components=1,
            covariance_rank=None,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        fit_geom = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            contact_positions=[coords],
            n_mark_components=1,
            covariance_rank=None,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix_no = fit_no_geom["mark_mixtures"][0]
        mix_g = fit_geom["mark_mixtures"][0]
        assert mix_no is not None
        assert mix_g is not None
        probe = np.array([[0.0, 50.0], [50.0, 0.0]])
        ld_no = mix_no.log_component_density(probe)
        ld_g = mix_g.log_component_density(probe)
        np.testing.assert_allclose(ld_g, ld_no, rtol=1e-10, atol=1e-10)
        assert ld_g[0, 0] > ld_g[1, 0] + 1000.0

    def test_em_uses_consistent_split_event_positions_for_midpoint_spike(self):
        """Comment 2 (P1): midpoint spike between opposing fields splits symmetrically."""
        from non_local_detector.likelihoods.clusterless_probe_mixture import (
            _e_step_electrode,
            _ElectrodeEvents,
            _soft_cell_counts,
        )

        # Two components with identical mark density at m=[0, 0].
        mix = ProbeMarkMixture(
            weights=np.array([0.5, 0.5]),
            means=np.zeros((2, 2)),
            basis=np.zeros((2, 2, 2)),
            basis_variances=np.ones((2, 2)),
            support_indices=np.array([[0, 1], [0, 1]], dtype=np.int32),
            support_mask=np.ones((2, 2), dtype=bool),
            noise_variance=np.ones(2),
            background_mean=np.zeros(2),
            noise_whitener=None,
            log_det_whitener=0.0,
            contact_positions=None,
            component_peak_channels=np.array([0, 0], dtype=np.int32),
        )
        # Midpoint spike: left sample strongly favors comp 0, right sample favors comp 1.
        lr_left = np.log(np.array([[20.0, 0.1]]))
        lr_right = np.log(np.array([[0.1, 20.0]]))
        resp_l, resp_r, _ = _e_step_electrode(
            np.zeros((1, 2)),
            mix,
            lr_left,
            lr_right,
            np.array([0.5]),
            np.array([0.5]),
        )
        # Left endpoint favors comp 0, right endpoint favors comp 1 symmetrically.
        assert resp_l[0, 0] > 0.99
        assert resp_r[0, 1] > 0.99
        np.testing.assert_allclose(resp_l[0, 0], resp_r[0, 1], atol=1e-5)
        ev = _ElectrodeEvents(
            marks_whitened=np.zeros((1, 2)),
            left=np.array([0], dtype=np.int64),
            right=np.array([1], dtype=np.int64),
            left_share=np.array([0.5]),
            right_share=np.array([0.5]),
            total_weight=np.array([1.0]),
            primary_sample=np.array([0], dtype=np.int64),
        )
        counts = _soft_cell_counts(
            [ev], [(resp_l, resp_r)], np.array([0, 1], dtype=np.int64), 2
        )
        # Comp 0 gets ~0.5 at cell 0, Comp 1 gets ~0.5 at cell 1.
        assert counts[0, 0] > 0.49
        assert counts[1, 1] > 0.49
        np.testing.assert_allclose(counts[0, 0], counts[1, 1], atol=1e-5)

    def test_single_effective_spike_full_covariance_runs_em_without_error(self):
        """Comment 3 (P2): single-spike 2-ch full-cov electrode succeeds through EM."""
        time = np.arange(100) * 0.01
        pos = np.linspace(0.0, 30.0, 100)[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            [np.array([0.5])],
            [np.array([[10.0, -5.0]])],
            env,
            covariance_rank=None,
            n_em_iterations=3,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        assert mix.basis.shape == (1, 2, 2)

    def test_full_covariance_em_monotone_when_floor_does_not_bind(self):
        """Comment 4 (P2): full-rank variance floor clamps rather than shifting all eigs."""
        rng = np.random.default_rng(104)
        time = np.arange(200) * 0.01
        pos = np.linspace(0.0, 50.0, 200)[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        sp_t = [np.sort(rng.uniform(0.05, 1.95, 20))]
        # Eigenvalues well above floor=1.0 (std=3 -> var~9).
        sp_m = [rng.normal(0.0, 3.0, size=(20, 2))]
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            n_mark_components=1,
            covariance_rank=None,
            mark_reg_covar=1.0,
            n_em_iterations=5,
            em_validation_blocks=0,
            em_tol=-np.inf,
            disable_progress_bar=True,
        )
        trace = enc["em_diagnostics"]["train_log_likelihood"]
        assert len(trace) >= 2
        assert np.all(np.diff(trace) >= -1e-6), f"Non-monotone trace: {trace}"

    def test_full_rank_init_enforces_measured_noise_floor(self):
        """Comment 5 (P2): full-rank iter=0 with noise_covariances enforces unit floor."""
        time = np.arange(100) * 0.01
        pos = np.linspace(0.0, 30.0, 100)[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)
        sp_t = [np.linspace(0.1, 0.9, 5)]
        sp_m = [np.tile(np.array([5.0, -3.0]), (5, 1))]
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            noise_covariances=[np.eye(2)],
            n_mark_components=1,
            covariance_rank=None,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        assert np.all(mix.basis_variances >= 1.0 - 1e-12)

    def test_multitile_peak_selection_invariant_to_constant_channel_offset(self):
        """Comment 6 (P2): constant DC offset on silent contact 0 cannot hijack peaks."""
        rng = np.random.default_rng(106)
        n_ch = 16
        coords = _make_two_column_probe(n_ch, pitch=20.0)
        dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        time = np.arange(1000) * 0.01
        pos = (50.0 + 40.0 * np.sin(2.0 * np.pi * time / 5.0))[:, None]
        env = Environment(place_bin_size=5.0).fit_place_grid(pos)

        t_parts, m_parts = [], []
        for peak_c, pf in ((2, 25.0), (13, 75.0)):
            rate = 0.2 + 25.0 * np.exp(-0.5 * ((pos[:, 0] - pf) / 8.0) ** 2)
            cnt = rng.poisson(rate * 0.01)
            st = np.clip(np.repeat(time, cnt), time[0], time[-1])
            template = 80.0 * np.exp(-0.5 * (dist[peak_c] / 30.0) ** 2)
            gain = 1.0 + rng.normal(0.0, 0.08, size=(len(st), 1))
            marks = gain * template[None, :] + rng.normal(
                0.0, 4.0, size=(len(st), n_ch)
            )
            t_parts.append(st)
            m_parts.append(marks)
        all_t = np.concatenate(t_parts)
        all_m = np.concatenate(m_parts, axis=0)
        order = np.argsort(all_t)
        all_t = all_t[order]
        all_m = all_m[order]
        # Add large DC offset strictly to silent contact 0.
        all_m[:, 0] += 500.0

        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            pos,
            [all_t],
            [all_m],
            env,
            contact_positions=[coords],
            max_local_channels=6,
            covariance_rank=3,
            n_mark_components=4,
            n_em_iterations=0,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        peaks = set(mix.component_peak_channels.tolist())
        assert 0 not in peaks, f"Offset contact 0 hijacked peaks: {peaks}"
        assert any(p in {1, 2, 3} for p in peaks), (
            f"Missed unit near contact 2: {peaks}"
        )
        assert any(p in {12, 13, 14} for p in peaks), (
            f"Missed unit near contact 13: {peaks}"
        )

    def test_off_support_evidence_preserved_in_float32_at_large_local_amplitude(self):
        """Comment 7 (P2): large local amplitude cannot cancel off-support d2 in f32."""
        import jax.numpy as jnp

        from non_local_detector.likelihoods.clusterless_probe_mixture import (
            _e_step_block,
            _woodbury_candidate_log_density,
            _woodbury_log_density,
        )

        feats = jnp.asarray([[10000.0, 0.0], [10000.0, 2.0]], dtype=jnp.float32)
        means = jnp.asarray([[10000.0]], dtype=jnp.float32)
        basis = jnp.zeros((1, 1, 1), dtype=jnp.float32)
        basis_var = jnp.ones((1, 1), dtype=jnp.float32)
        sup_idx = jnp.asarray([[0]], dtype=jnp.int32)
        sup_mask = jnp.asarray([[True]])
        noise_var = jnp.ones(2, dtype=jnp.float32)
        bg_mean = jnp.zeros(2, dtype=jnp.float32)
        is_comp = jnp.asarray([True])

        ld_exact = np.asarray(
            _woodbury_log_density(
                feats,
                means,
                basis,
                basis_var,
                sup_idx,
                sup_mask,
                noise_var,
                bg_mean,
                is_comp,
                all_full_support=False,
            )
        )
        np.testing.assert_allclose(ld_exact[0, 0] - ld_exact[1, 0], 2.0, atol=1e-4)

        ld_cand = np.asarray(
            _woodbury_candidate_log_density(
                feats,
                jnp.zeros((2, 1), dtype=jnp.int32),
                jnp.ones((2, 1), dtype=bool),
                means,
                basis,
                basis_var,
                sup_idx,
                sup_mask,
                noise_var,
                bg_mean,
                is_comp,
                all_full_support=False,
            )
        )
        np.testing.assert_allclose(ld_cand[0, 0] - ld_cand[1, 0], 2.0, atol=1e-4)

        _, _, ll_0 = _e_step_block(
            feats[:1],
            means,
            basis,
            basis_var,
            sup_idx,
            sup_mask,
            noise_var,
            bg_mean,
            is_comp,
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.ones(1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            all_full_support=False,
        )
        _, _, ll_1 = _e_step_block(
            feats[1:],
            means,
            basis,
            basis_var,
            sup_idx,
            sup_mask,
            noise_var,
            bg_mean,
            is_comp,
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.ones(1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            all_full_support=False,
        )
        np.testing.assert_allclose(float(ll_0) - float(ll_1), 2.0, atol=1e-4)

    def test_validation_folds_assign_boundary_spikes_to_single_owner(self):
        """Comment 8 (P2): a midpoint fold-boundary spike cannot enter both train & val."""
        from non_local_detector.encoding_time import prepare_encoding_support
        from non_local_detector.likelihoods.clusterless_probe_mixture import (
            _collect_electrode_events,
            _contiguous_exposure_blocks,
        )

        time = np.arange(20) * 0.1
        pos = np.linspace(0.0, 20.0, 20)[:, None]
        support, weights_arr, exposure_weights, _ = prepare_encoding_support(
            time, pos, None
        )
        support_pos = np.full(len(time), -1, dtype=np.int64)
        support_pos[support.indices] = np.arange(len(support.indices), dtype=np.int64)
        masks = _contiguous_exposure_blocks(exposure_weights, 2)
        assert len(masks) == 2
        # Place a spike at exact midpoint between last sample of block 0 and first of block 1.
        last0 = int(np.flatnonzero(masks[0])[-1])
        first1 = int(np.flatnonzero(masks[1])[0])
        mid_t = np.array([0.5 * (time[last0] + time[first1])])
        mid_m = np.array([[42.0, -7.0]])

        tr_raw = _collect_electrode_events(
            support,
            weights_arr,
            support_pos,
            mid_t,
            mid_m,
            allowed_samples=~masks[1],
        )
        va_raw = _collect_electrode_events(
            support,
            weights_arr,
            support_pos,
            mid_t,
            mid_m,
            allowed_samples=masks[1],
        )
        # Strictly one of train or validation owns the boundary spike, with full weight 1.0.
        assert (tr_raw is None) ^ (va_raw is None)
        owned = tr_raw if tr_raw is not None else va_raw
        assert owned is not None
        np.testing.assert_allclose(owned[3] + owned[4], 1.0, atol=1e-12)

    def test_local_predict_blocks_spikes_by_spike_block_size(self, monkeypatch):
        """Comment 9 (P2): local predict processes spikes in spike_block_size chunks."""
        from non_local_detector.likelihoods import clusterless_probe_mixture as cpm

        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=15.0, seed=109
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
            n_mark_components=8,
            n_em_iterations=2,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)
        edges = time_edges_from_centers(time[:400])

        seen_batch_sizes: list[int] = []
        orig_local = cpm._add_electrode_local_terms

        def _spy_local(ll, features, *args, **kwargs):
            seen_batch_sizes.append(int(features.shape[0]))
            return orig_local(ll, features, *args, **kwargs)

        monkeypatch.setattr(cpm, "_add_electrode_local_terms", _spy_local)
        for max_cand in (None, 4):
            ll_small = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    time[:400],
                    pos[:400],
                    rec["spike_times"],
                    rec["spike_waveform_features"],
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=True,
                    **{
                        **enc,
                        "spike_block_size": 32,
                        "max_candidate_components": max_cand,
                    },
                )
            )
            if max_cand is None:
                assert len(seen_batch_sizes) >= 2
                assert max(seen_batch_sizes) <= 32
            ll_large = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    time[:400],
                    pos[:400],
                    rec["spike_times"],
                    rec["spike_waveform_features"],
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=True,
                    **{
                        **enc,
                        "spike_block_size": 1000,
                        "max_candidate_components": max_cand,
                    },
                )
            )
            np.testing.assert_allclose(ll_small, ll_large, rtol=1e-5, atol=1e-4)


@pytest.mark.unit
class TestHeavyTailedProbeMixture:
    """Tests for factored heavy-tailed ``tail_df`` support in ProbeMarkMixture."""

    @pytest.mark.parametrize(
        "bad_df", [0.0, -5.0, 2.5, 3.0, float("inf"), float("nan"), True, "6"]
    )
    def test_tail_df_validation_rejects_invalid_values(self, bad_df):
        rec = _make_dense_probe_recording(
            n_channels=8, n_units=2, duration=6.0, seed=201
        )
        with pytest.raises(ValidationError, match="tail_df"):
            fit_clusterless_probe_mixture_encoding_model(
                rec["position_time"],
                rec["position"],
                rec["spike_times"],
                rec["spike_waveform_features"],
                rec["environment"],
                tail_df=bad_df,
                n_mark_components=2,
                n_em_iterations=0,
                disable_progress_bar=True,
            )

    def test_tail_df_density_normalization_and_host_device_parity(self):
        """Factored GSM-t x 1D-t integrates to 1 and matches JAX kernels."""
        import jax.numpy as jnp
        from scipy.integrate import quad

        from non_local_detector.likelihoods import clusterless_probe_mixture as cpm

        # 1. Exact 1-D normalization check (1 local channel, 0 off-support).
        mix_1d = ProbeMarkMixture(
            weights=np.array([1.0]),
            means=np.array([[2.5]]),
            basis=np.array([[[1.0]]]),
            basis_variances=np.array([[3.2]]),
            support_indices=np.array([[0]], dtype=np.int32),
            support_mask=np.array([[True]]),
            noise_variance=np.array([1.7]),
            background_mean=np.array([0.0]),
            noise_whitener=None,
            log_det_whitener=0.0,
            contact_positions=None,
            component_peak_channels=np.array([0], dtype=np.int32),
            tail_df=6.0,
        )
        integral_1d, _ = quad(
            lambda x: float(
                np.exp(mix_1d.log_component_density(np.array([[x]]))[0, 0])
            ),
            -200.0,
            200.0,
            limit=200,
        )
        np.testing.assert_allclose(integral_1d, 1.0, rtol=1e-7, atol=1e-7)

        # 2. Exact 2-D factored normalization (1 local channel + 1 off-support 1D-t).
        mix_2d = ProbeMarkMixture(
            weights=np.array([1.0]),
            means=np.array([[-1.5]]),
            basis=np.array([[[1.0]]]),
            basis_variances=np.array([[2.4]]),
            support_indices=np.array([[0]], dtype=np.int32),
            support_mask=np.array([[True]]),
            noise_variance=np.array([1.3, 2.1]),
            background_mean=np.array([0.0, 0.7]),
            noise_whitener=None,
            log_det_whitener=0.0,
            contact_positions=None,
            component_peak_channels=np.array([0], dtype=np.int32),
            tail_df=5.0,
        )
        # Factored 2-D density p(x0, x1) = p_loc(x0) * p_off(x1): check product of 1-D integrals.
        grid = np.linspace(-80.0, 80.0, 4001)
        dx = float(grid[1] - grid[0])
        pts_x0 = np.column_stack([grid, np.full_like(grid, 0.7)])
        pts_x1 = np.column_stack([np.full_like(grid, -1.5), grid])
        p00 = float(np.exp(mix_2d.log_component_density(np.array([[-1.5, 0.7]]))[0, 0]))
        int_x0 = float(np.sum(np.exp(mix_2d.log_component_density(pts_x0)[:, 0])) * dx)
        int_x1 = float(np.sum(np.exp(mix_2d.log_component_density(pts_x1)[:, 0])) * dx)
        np.testing.assert_allclose(int_x0 * int_x1 / p00, 1.0, rtol=1e-4, atol=1e-4)

        # 3. Host float64 vs JAX all-K and candidate kernels (both all_full and local).
        rec = _make_dense_probe_recording(
            n_channels=16, n_units=4, duration=10.0, seed=202
        )
        for use_geom in (False, True):
            enc = fit_clusterless_probe_mixture_encoding_model(
                rec["position_time"],
                rec["position"],
                rec["spike_times"],
                rec["spike_waveform_features"],
                rec["environment"],
                contact_positions=rec["contact_positions"] if use_geom else None,
                max_local_channels=8 if use_geom else None,
                covariance_rank=3,
                n_mark_components=6,
                tail_df=6.0,
                n_em_iterations=2,
                em_validation_blocks=0,
                disable_progress_bar=True,
            )
            mix = enc["mark_mixtures"][0]
            assert mix is not None
            assert mix.tail_df == 6.0
            probe = rec["spike_waveform_features"][0][:40]
            host_ld = mix.log_component_density(probe)
            y = jnp.asarray(mix.whiten_marks(probe), dtype=jnp.float32)
            k = mix.n_components
            jax_ld = np.asarray(
                cpm._woodbury_log_density(
                    y,
                    jnp.asarray(mix.means, dtype=jnp.float32),
                    jnp.asarray(mix.basis, dtype=jnp.float32),
                    jnp.asarray(mix.basis_variances, dtype=jnp.float32),
                    jnp.asarray(mix.support_indices, dtype=jnp.int32),
                    jnp.asarray(mix.support_mask),
                    jnp.asarray(mix.noise_variance, dtype=jnp.float32),
                    jnp.asarray(mix.background_mean, dtype=jnp.float32),
                    jnp.ones((k,), dtype=bool),
                    log_det_whitener=mix.log_det_whitener,
                    all_full_support=mix.all_full_support,
                    tail_df=mix.tail_df,
                )
            )
            np.testing.assert_allclose(jax_ld, host_ld, rtol=1e-4, atol=1e-3)

            cand_idx = jnp.broadcast_to(
                jnp.arange(k, dtype=jnp.int32)[None, :], (len(probe), k)
            )
            cand_msk = jnp.ones((len(probe), k), dtype=bool)
            cand_ld = np.asarray(
                cpm._woodbury_candidate_log_density(
                    y,
                    cand_idx,
                    cand_msk,
                    jnp.asarray(mix.means, dtype=jnp.float32),
                    jnp.asarray(mix.basis, dtype=jnp.float32),
                    jnp.asarray(mix.basis_variances, dtype=jnp.float32),
                    jnp.asarray(mix.support_indices, dtype=jnp.int32),
                    jnp.asarray(mix.support_mask),
                    jnp.asarray(mix.noise_variance, dtype=jnp.float32),
                    jnp.asarray(mix.background_mean, dtype=jnp.float32),
                    jnp.ones((k,), dtype=bool),
                    log_det_whitener=mix.log_det_whitener,
                    all_full_support=mix.all_full_support,
                    tail_df=mix.tail_df,
                )
            )
            np.testing.assert_allclose(cand_ld, host_ld, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize(
        ("use_geom", "use_noise", "rank"),
        [
            (False, False, None),
            (False, False, 2),
            (True, False, 3),
            (True, True, 3),
        ],
    )
    def test_tail_df_em_is_monotone_across_configurations(
        self, use_geom, use_noise, rank
    ):
        """ECM EM with ``tail_df=6.0`` never decreases penalized train LL."""
        n_ch = 12 if use_geom else 4
        rec = _make_dense_probe_recording(
            n_channels=n_ch, n_units=3, duration=12.0, seed=203
        )
        noise = [np.eye(n_ch) * 20.0 + 1.0] if use_noise else None
        enc = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"] if use_geom else None,
            noise_covariances=noise,
            max_local_channels=6 if use_geom else None,
            covariance_rank=rank,
            n_mark_components=5,
            tail_df=6.0,
            n_em_iterations=5,
            em_validation_blocks=0,
            em_tol=-1.0,
            disable_progress_bar=True,
        )
        trace = enc["em_diagnostics"]["train_log_likelihood"]
        assert len(trace) >= 3
        assert np.all(np.diff(trace) >= -1e-4), (
            f"Non-monotone tail_df EM trace: {trace}"
        )

    @pytest.mark.parametrize("use_geom", [False, True])
    def test_tail_df_preserves_no_spike_tail_dominance_on_every_ray(self, use_geom):
        """Extreme artifacts on 1 channel, S_k only, or all channels fall back to No-Spike."""
        from non_local_detector.likelihoods.common import LOG_RATE_EPS_HZ

        n_ch = 16 if use_geom else 4
        rec = _make_dense_probe_recording(
            n_channels=n_ch, n_units=3, duration=10.0, seed=204
        )
        enc = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"] if use_geom else None,
            max_local_channels=6 if use_geom else None,
            covariance_rank=3 if use_geom else None,
            n_mark_components=4,
            tail_df=4.0,
            n_em_iterations=2,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)[0]
        assert ref is not None
        scale = np.std(rec["spike_waveform_features"][0], axis=0) + 1.0
        max_log_rate = float(np.max(enc["log_rate_maps"][0]))

        # Build 3 extreme 80-sigma artifact rays: 1-ch, local-support-only, all-ch.
        one_ch = np.zeros((1, n_ch))
        one_ch[0, 0] = 80.0 * scale[0]
        local_only = np.zeros((1, n_ch))
        sup0 = mix.support_indices[0, mix.support_mask[0]]
        local_only[0, sup0] = 80.0 * scale[sup0]
        all_ch = (80.0 * scale)[None, :]

        artifacts = np.vstack([one_ch, local_only, all_ch])
        comp_ll = mix.log_component_density(artifacts) + max_log_rate
        sig_ll = np.asarray([float(np.logaddexp.reduce(row)) for row in comp_ll])
        bg_ll = LOG_RATE_EPS_HZ + ref.log_density(artifacts)
        assert np.all(sig_ll < bg_ll), (
            f"Artifact beat No-Spike: sig={sig_ll}, bg={bg_ll}"
        )

    def test_tail_df_large_df_approaches_gaussian(self):
        """As tail_df -> inf, log_component_density converges to exact Gaussian."""
        rec = _make_dense_probe_recording(
            n_channels=12, n_units=3, duration=10.0, seed=205
        )
        enc = fit_clusterless_probe_mixture_encoding_model(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
            rec["environment"],
            contact_positions=rec["contact_positions"],
            max_local_channels=6,
            covariance_rank=3,
            n_mark_components=4,
            tail_df=None,
            n_em_iterations=1,
            em_validation_blocks=0,
            disable_progress_bar=True,
        )
        mix_g = enc["mark_mixtures"][0]
        assert mix_g is not None
        mix_big_df = ProbeMarkMixture(
            weights=mix_g.weights,
            means=mix_g.means,
            basis=mix_g.basis,
            basis_variances=mix_g.basis_variances,
            support_indices=mix_g.support_indices,
            support_mask=mix_g.support_mask,
            noise_variance=mix_g.noise_variance,
            background_mean=mix_g.background_mean,
            noise_whitener=mix_g.noise_whitener,
            log_det_whitener=mix_g.log_det_whitener,
            contact_positions=mix_g.contact_positions,
            component_peak_channels=mix_g.component_peak_channels,
            tail_df=1e5,
        )
        probe = rec["spike_waveform_features"][0][:30]
        np.testing.assert_allclose(
            mix_big_df.log_component_density(probe),
            mix_g.log_component_density(probe),
            rtol=1e-2,
            atol=5e-2,
        )

    def test_tail_df_nld_roundtrip_and_candidate_parity(self, tmp_path):
        """tail_df persists cleanly through .nld and candidate LUT matches all-K."""
        rec = _make_dense_probe_recording(
            n_channels=12, n_units=3, duration=12.0, seed=206
        )
        geom = ProbeGeometry(contact_positions=rec["contact_positions"][0])
        detector = ClusterlessDecoder(
            environments=rec["environment"],
            sampling_frequency=100.0,
            clusterless_algorithm="clusterless_probe_mixture",
            clusterless_algorithm_params={
                "contact_positions": geom,
                "max_local_channels": 6,
                "covariance_rank": 3,
                "n_mark_components": 6,
                "tail_df": 6.0,
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
        enc = detector.encoding_model_[("", 0)]
        mix = enc["mark_mixtures"][0]
        assert mix is not None
        assert mix.tail_df == 6.0

        path = tmp_path / "probe_mixture_tail_df.nld"
        detector.save_model(path)
        loaded = ClusterlessDecoder.load_model(path)
        loaded_mix = loaded.encoding_model_[("", 0)]["mark_mixtures"][0]
        assert loaded_mix is not None
        assert loaded_mix.tail_df == 6.0

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

        # Also verify max_candidate_components >= K matches all-K for both local & non-local.
        ref = fit_mark_references(rec["spike_waveform_features"], kernel_scale=1.0)
        for is_local in (False, True):
            ll_all = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    rec["position_time"][:200] if is_local else None,
                    rec["position"][:200] if is_local else None,
                    rec["spike_times"],
                    rec["spike_waveform_features"],
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=is_local,
                    **enc,
                )
            )
            ll_cand = np.asarray(
                predict_clusterless_probe_mixture_log_likelihood(
                    rec["position_time"][:200] if is_local else None,
                    rec["position"][:200] if is_local else None,
                    rec["spike_times"],
                    rec["spike_waveform_features"],
                    time_edges=edges,
                    mark_reference=ref,
                    is_local=is_local,
                    **{**enc, "max_candidate_components": mix.n_components},
                )
            )
            np.testing.assert_allclose(ll_cand, ll_all, rtol=1e-5, atol=1e-4)


@pytest.mark.unit
class TestCandidateComponents:
    """``max_candidate_components`` is an approximation tied to contact geometry."""

    def test_max_candidate_components_requires_contact_positions_at_fit(self):
        env, time, position, spikes, marks = _make_overlapping_marks_recording()
        with pytest.raises(ValidationError, match="requires contact_positions"):
            fit_clusterless_probe_mixture_encoding_model(
                time,
                position,
                spikes,
                marks,
                env,
                max_candidate_components=1,
                n_mark_components=2,
                n_em_iterations=0,
                disable_progress_bar=True,
            )

    def test_invalid_max_candidate_components_is_rejected_at_fit(self):
        rec = _make_dense_probe_recording(n_channels=8, n_units=2, duration=6.0)
        with pytest.raises(
            ValidationError,
            match="max_candidate_components must be None or a positive integer",
        ):
            fit_clusterless_probe_mixture_encoding_model(
                rec["position_time"],
                rec["position"],
                rec["spike_times"],
                rec["spike_waveform_features"],
                rec["environment"],
                contact_positions=rec["contact_positions"],
                max_candidate_components=0,
                disable_progress_bar=True,
            )

    def test_candidate_lookup_refuses_a_mixture_without_contact_positions(self):
        """Without geometry there is no peak-contact table to restrict candidates."""
        env, time, position, spikes, marks = _make_overlapping_marks_recording()
        enc = fit_clusterless_probe_mixture_encoding_model(
            time,
            position,
            spikes,
            marks,
            env,
            n_mark_components=2,
            n_em_iterations=0,
            disable_progress_bar=True,
        )
        ref = fit_mark_references(marks, kernel_scale=1.0)
        with pytest.raises(ValidationError, match="requires contact_positions"):
            predict_clusterless_probe_mixture_log_likelihood(
                None,
                None,
                spikes,
                marks,
                time_edges=time_edges_from_centers(time[:100]),
                mark_reference=ref,
                **{**enc, "max_candidate_components": 1},
            )

    def test_max_candidate_components_changed_after_fit_waits_for_refit(self):
        """It changes results, so it is a model parameter: a detector keeps the
        fitted value until the next fit instead of passing the new one to predict."""
        rec = _make_dense_probe_recording(
            n_channels=12, n_units=3, duration=12.0, seed=206
        )
        params = {
            "contact_positions": rec["contact_positions"],
            "max_local_channels": 6,
            "covariance_rank": 3,
            "n_mark_components": 6,
            "n_em_iterations": 0,
            "disable_progress_bar": True,
        }
        decoder = ClusterlessDecoder(
            environments=rec["environment"],
            sampling_frequency=100.0,
            clusterless_algorithm="clusterless_probe_mixture",
            clusterless_algorithm_params=params,
        ).fit(
            rec["position_time"],
            rec["position"],
            rec["spike_times"],
            rec["spike_waveform_features"],
        )
        (enc,) = decoder.encoding_model_.values()
        assert enc["mark_mixtures"][0].n_components > 1
        edges = time_edges_from_centers(rec["position_time"][:200])

        def log_likelihood():
            return decoder.predict(
                rec["spike_times"],
                rec["spike_waveform_features"],
                time_edges=edges,
                return_outputs="log_likelihood",
            ).log_likelihood

        before = log_likelihood()
        decoder.clusterless_algorithm_params = {
            **params,
            "max_candidate_components": 1,
        }
        np.testing.assert_array_equal(log_likelihood(), before)
        assert (
            "max_candidate_components"
            not in _CLUSTERLESS_ALGORITHMS[
                "clusterless_probe_mixture"
            ].execution_settings
        )
