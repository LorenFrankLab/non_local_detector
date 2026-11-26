"""Unit, numerical-validation, and integration tests for clusterless_flow_mixture."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from non_local_detector import (
    ClusterlessDecoder,
    Environment,
    NonLocalClusterlessDetector,
)
from non_local_detector.exceptions import DataQualityWarning, ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS
from non_local_detector.likelihoods.clusterless_flow_mixture import (
    FlowMarkMixture,
    _apply_flow_forward_jax,
    _build_flow_topology,
    _cayley_orthogonal,
    _init_identity_flow_params,
    _latent_woodbury_log_density_f64,
    _latent_woodbury_log_density_jax,
    _pack_flow_mixture,
    _rqs_forward_1d,
    _rqs_inverse_1d,
    fit_clusterless_flow_mixture_encoding_model,
    predict_clusterless_flow_mixture_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.time_edges import time_edges_from_centers


def _make_1d_env(track_len: float = 100.0, bin_size: float = 5.0) -> Environment:
    env = Environment(place_bin_size=bin_size)
    env.fit_place_grid(np.linspace(0.0, track_len, 201)[:, None])
    return env


def _make_random_trained_mixture(
    n_features: int,
    *,
    contact_positions: np.ndarray | None = None,
    noise_covariance: np.ndarray | None = None,
    n_flow_layers: int = 3,
    n_spline_bins: int = 8,
    spline_bound: float = 5.0,
    seed: int = 0,
) -> FlowMarkMixture:
    """Build a non-identity FlowMarkMixture with random weights for exact checks."""
    rng = np.random.default_rng(seed)
    topology = _build_flow_topology(
        n_features,
        contact_positions,
        n_flow_layers=n_flow_layers,
        n_spline_bins=n_spline_bins,
        spline_bound=spline_bound,
        neighborhood_radius=None,
        max_local_channels=8,
    )
    params = _init_identity_flow_params(topology, n_features, 16, rng)
    # Perturb output heads and rotations so the flow is smoothly nonlinear
    # (realistic trained-scale weights rather than extreme slope discontinuities).
    perturbed = {}
    for k, v in params.items():
        scale = (
            0.2 if k in ("w_out", "b_out", "cayley_raw", "elementwise_params") else 0.1
        )
        perturbed[k] = jnp.asarray(
            np.asarray(v) + rng.normal(0.0, scale, size=v.shape), dtype=jnp.float32
        )
    n_comp = 3
    rank = min(n_features, 2)
    # Random orthonormal columns via QR.
    basis = np.zeros((n_comp, n_features, rank), dtype=np.float64)
    for k in range(n_comp):
        q, _ = np.linalg.qr(rng.normal(size=(n_features, rank)))
        basis[k] = q[:, :rank]
    basis_var = rng.uniform(1.2, 4.0, size=(n_comp, rank))
    noise_var = rng.uniform(0.3, 1.5, size=(n_features,))
    means = rng.normal(0.0, 1.0, size=(n_comp, n_features))
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    input_mean = rng.normal(5.0, 2.0, size=(n_features,))
    if noise_covariance is not None:
        eigvals, eigvecs = np.linalg.eigh(noise_covariance)
        whitener = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]) @ eigvecs.T
        whitener = 0.5 * (whitener + whitener.T)
        input_scale = rng.uniform(1.5, 4.0, size=(n_features,))
        log_det_in = -0.5 * float(np.sum(np.log(eigvals))) - float(
            np.sum(np.log(input_scale))
        )
    else:
        whitener = None
        input_scale = rng.uniform(0.5, 3.0, size=(n_features,))
        log_det_in = -float(np.sum(np.log(input_scale)))

    return _pack_flow_mixture(
        weights=weights,
        means=means,
        basis=basis,
        basis_variances=basis_var,
        noise_variance=noise_var,
        input_mean=input_mean,
        input_scale=input_scale,
        noise_whitener=whitener,
        log_det_input=log_det_in,
        topology=topology,
        flow_params=perturbed,
    )


# ---------------------------------------------------------------------------
# Bijector, Jacobian, Tail, and Woodbury Exactness Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rqs_1d_identity_invertibility_and_analytic_derivative():
    """RQS 1-D is exact identity at zero, invertible, and matches autograd."""
    with jax.enable_x64():
        x = jnp.linspace(-7.0, 7.0, 65, dtype=jnp.float64)
        zero_p = jnp.zeros((65, 3 * 8 - 1), dtype=jnp.float64)
        y0, ld0 = _rqs_forward_1d(x, zero_p, n_bins=8, bound=5.0)
        np.testing.assert_allclose(np.asarray(y0), np.asarray(x), atol=1e-12)
        np.testing.assert_allclose(np.asarray(ld0), 0.0, atol=1e-12)

        rng = np.random.default_rng(7)
        raw = jnp.asarray(rng.normal(0.0, 0.8, size=(65, 3 * 8 - 1)), dtype=jnp.float64)
        y, ld = _rqs_forward_1d(x, raw, n_bins=8, bound=5.0)
        x_rec, ld_inv = _rqs_inverse_1d(y, raw, n_bins=8, bound=5.0)
        np.testing.assert_allclose(np.asarray(x_rec), np.asarray(x), atol=1e-10)
        np.testing.assert_allclose(np.asarray(ld + ld_inv), 0.0, atol=1e-10)

        # Outside [-5, 5], spline is exact identity with log_deriv == 0.
        outside = np.abs(np.asarray(x)) > 5.0
        np.testing.assert_allclose(
            np.asarray(y)[outside], np.asarray(x)[outside], atol=1e-12
        )
        np.testing.assert_allclose(np.asarray(ld)[outside], 0.0, atol=1e-12)

        # Analytic derivative matches jax.grad inside and outside [-B, B].
        def _scalar_y(xi: jax.Array, pi: jax.Array) -> jax.Array:
            yi, _ = _rqs_forward_1d(xi, pi, n_bins=8, bound=5.0)
            return yi

        grad_fn = jax.vmap(jax.grad(_scalar_y, argnums=0))
        auto_deriv = np.asarray(grad_fn(x, raw))
        np.testing.assert_allclose(
            np.exp(np.asarray(ld)), auto_deriv, rtol=1e-10, atol=1e-10
        )


@pytest.mark.unit
def test_cayley_orthogonal_is_exact_rotation_and_identity_at_zero():
    """Cayley transform is exact identity at 0 and orthogonal for all finite M."""
    with jax.enable_x64():
        q0 = _cayley_orthogonal(jnp.zeros((6, 6), dtype=jnp.float64))
        np.testing.assert_allclose(np.asarray(q0), np.eye(6), atol=1e-14)

        rng = np.random.default_rng(3)
        raw = jnp.asarray(rng.normal(0.0, 2.5, size=(6, 6)), dtype=jnp.float64)
        q = np.asarray(_cayley_orthogonal(raw))
        np.testing.assert_allclose(q @ q.T, np.eye(6), atol=1e-13)
        np.testing.assert_allclose(np.linalg.det(q), 1.0, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_case",
    ["elementwise", "dense", "dense_zca", "local"],
)
def test_flow_mixture_bijection_and_exact_jacobian(mode_case):
    """Forward/inverse round-trip and exact log|det J| vs jax.jacfwd in all modes."""
    rng = np.random.default_rng(11)
    if mode_case == "elementwise":
        mix = _make_random_trained_mixture(1, seed=1)
        x = rng.normal(5.0, 3.0, size=(12, 1))
    elif mode_case == "dense":
        mix = _make_random_trained_mixture(4, seed=2)
        x = rng.normal(5.0, 3.0, size=(12, 4))
    elif mode_case == "dense_zca":
        a = rng.normal(size=(4, 4))
        cov = a @ a.T + np.eye(4)
        mix = _make_random_trained_mixture(4, noise_covariance=cov, seed=3)
        x = rng.normal(5.0, 3.0, size=(12, 4))
    else:
        coords = np.column_stack([np.zeros(8), np.arange(8) * 20.0])
        mix = _make_random_trained_mixture(8, contact_positions=coords, seed=4)
        x = rng.normal(5.0, 3.0, size=(12, 8))

    z, ld_fwd = mix.forward(x)
    x_rec, ld_inv = mix.inverse(z)
    np.testing.assert_allclose(x_rec, x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-9)

    # Compare analytic log|det J| against full autodiff Jacobian determinant.
    with jax.enable_x64():
        fp = mix._jax_flow_params(jnp.float64)
        q_j = (
            None
            if mix.cayley_matrices is None
            else jnp.asarray(mix.cayley_matrices, dtype=jnp.float64)
        )
        cm_j = None if mix.cond_masks is None else jnp.asarray(mix.cond_masks)
        ni_j = (
            None if mix.cond_nbr_indices is None else jnp.asarray(mix.cond_nbr_indices)
        )
        nm_j = None if mix.cond_nbr_masks is None else jnp.asarray(mix.cond_nbr_masks)
        ti_j = None if mix.trans_indices is None else jnp.asarray(mix.trans_indices)
        tv_j = None if mix.trans_valid is None else jnp.asarray(mix.trans_valid)
        cg_j = (
            None
            if mix.cond_geom is None
            else jnp.asarray(mix.cond_geom, dtype=jnp.float64)
        )
        in_mean = jnp.asarray(mix.input_mean, dtype=jnp.float64)
        in_scale = jnp.asarray(mix.input_scale, dtype=jnp.float64)
        whitener = (
            None
            if mix.noise_whitener is None
            else jnp.asarray(mix.noise_whitener, dtype=jnp.float64)
        )

        def _f_single(m_one: jax.Array) -> jax.Array:
            centered = m_one - in_mean
            if whitener is not None:
                centered = centered @ whitener
            u_one = centered / in_scale
            z_one, _ = _apply_flow_forward_jax(
                u_one[None, :],
                fp,
                flow_mode=mix.flow_mode,
                n_bins=mix.n_spline_bins,
                bound=mix.spline_bound,
                cond_masks=cm_j,
                cond_nbr_indices=ni_j,
                cond_nbr_masks=nm_j,
                trans_indices=ti_j,
                trans_valid=tv_j,
                cond_geom=cg_j,
                precomputed_cayley=q_j,
            )
            return z_one[0]

        jac_fn = jax.vmap(jax.jacfwd(_f_single))
        jacs = np.asarray(jac_fn(jnp.asarray(x, dtype=jnp.float64)))
        _, auto_ld = np.linalg.slogdet(jacs)
        np.testing.assert_allclose(ld_fwd, auto_ld, rtol=1e-9, atol=1e-9)


@pytest.mark.unit
def test_local_mode_enforces_strict_physical_neighborhood_isolation():
    """In local coupling mode, distant contacts outside neighborhood_radius do not couple."""
    # Two well-separated clusters of contacts (0..3 at 0..60 um, 4..7 at 1000..1060 um).
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 20.0],
            [0.0, 40.0],
            [0.0, 60.0],
            [0.0, 1000.0],
            [0.0, 1020.0],
            [0.0, 1040.0],
            [0.0, 1060.0],
        ]
    )
    mix = _make_random_trained_mixture(
        8, contact_positions=coords, n_flow_layers=4, seed=5
    )
    assert mix.flow_mode == "local"

    rng = np.random.default_rng(12)
    m_base = rng.normal(5.0, 1.0, size=(10, 8))
    m_pert = m_base.copy()
    # Perturb only the far contacts 4..7.
    m_pert[:, 4:] += 3.5

    z_base, _ = mix.forward(m_base)
    z_pert, _ = mix.forward(m_pert)
    # Near contacts 0..3 must be completely unchanged across all layers!
    np.testing.assert_array_equal(z_base[:, :4], z_pert[:, :4])


@pytest.mark.unit
def test_latent_woodbury_matches_dense_scipy_and_high_condition_float32():
    """Orthonormal Woodbury matches scipy exactly and avoids float32 cancellation."""
    rng = np.random.default_rng(17)
    n_dims = 8
    n_comp = 3
    rank = 3
    noise_var = rng.uniform(0.2, 1.0, size=(n_dims,))
    means = rng.normal(0.0, 1.0, size=(n_comp, n_dims))
    basis = np.zeros((n_comp, n_dims, rank), dtype=np.float64)
    for k in range(n_comp):
        q, _ = np.linalg.qr(rng.normal(size=(n_dims, rank)))
        basis[k] = q[:, :rank]
    # Extreme condition number (1e4) that causes subtraction cancellation in naive Woodbury.
    basis_var = np.full((n_comp, rank), 1e4, dtype=np.float64)
    z = rng.normal(0.0, 5.0, size=(20, n_dims))

    got_f64 = _latent_woodbury_log_density_f64(z, means, basis, basis_var, noise_var)
    got_f32 = np.asarray(
        _latent_woodbury_log_density_jax(
            jnp.asarray(z, dtype=jnp.float32),
            jnp.asarray(means, dtype=jnp.float32),
            jnp.asarray(basis, dtype=jnp.float32),
            jnp.asarray(basis_var, dtype=jnp.float32),
            jnp.asarray(noise_var, dtype=jnp.float32),
            jnp.ones((n_comp,), dtype=bool),
        )
    )
    expected = np.empty((20, n_comp), dtype=np.float64)
    sqrt_psi = np.sqrt(noise_var)
    for k in range(n_comp):
        c_std = np.eye(n_dims) + (basis[k] * (basis_var[k] - 1.0)[None, :]) @ basis[k].T
        cov_k = c_std * sqrt_psi[:, None] * sqrt_psi[None, :]
        expected[:, k] = multivariate_normal.logpdf(z, mean=means[k], cov=cov_k)

    np.testing.assert_allclose(got_f64, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(got_f32, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.unit
def test_exact_mark_integral_equals_one_for_nonlinear_flow():
    """Numerical integral of |det J| * N(f(m); mu, Sigma) over R^2 equals 1."""
    mix = _make_random_trained_mixture(2, n_flow_layers=3, seed=9)
    # 2-D midpoint quadrature covering [-15, 25]^2 around input_mean (~5).
    grid_1d = np.linspace(-25.0, 35.0, 220)
    step = float(grid_1d[1] - grid_1d[0])
    gx, gy = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    log_p = mix.log_component_density(pts)
    integrals = np.sum(np.exp(log_p), axis=0) * (step * step)
    np.testing.assert_allclose(integrals, 1.0, atol=2e-3)


@pytest.mark.unit
def test_identity_tails_guarantee_nospike_student_t_dominance_and_no_nan():
    """Extreme outliers stay finite and fall below mark_reference Student-t."""
    mix = _make_random_trained_mixture(4, n_flow_layers=4, seed=13)
    rng = np.random.default_rng(14)
    train_marks = rng.normal(5.0, 2.0, size=(300, 4))
    (ref,) = fit_mark_references([train_marks], kernel_scale=1.0)
    assert ref is not None

    # Test along both coordinate and common-mode diagonal outlier rays.
    rays = np.array(
        [
            [500.0, 0.0, 0.0, 0.0],
            [500.0, 500.0, 500.0, 500.0],
            [-500.0, 500.0, -500.0, 500.0],
            [1e6, 1e6, 1e6, 1e6],
        ]
    )
    log_p = mix.log_component_density(rays)
    log_q = ref.log_density(rays)
    assert np.all(np.isfinite(log_p) | np.isneginf(log_p))
    assert np.all(np.max(log_p, axis=1) < log_q - 50.0)


# ---------------------------------------------------------------------------
# Fit / Predict / Synthetic Non-Gaussian Recovery & Invariant Tests
# ---------------------------------------------------------------------------


def _simulate_curved_banana_session(
    seed: int = 0,
    n_time: int = 3000,
    dt: float = 0.02,
) -> tuple[Environment, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Simulate two units with curved banana-shaped marks at distinct place fields."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_time) * dt
    pos = (50.0 + 45.0 * np.sin(2.0 * np.pi * time / 12.0))[:, None]
    env = _make_1d_env(100.0, 5.0)

    # Two overlapping curved crescent clusters at x=25 cm and x=75 cm.
    centers = [25.0, 75.0]
    spike_t_parts: list[np.ndarray] = []
    spike_m_parts: list[np.ndarray] = []
    for u_idx, center in enumerate(centers):
        rate = 25.0 * np.exp(-0.5 * ((pos[:, 0] - center) / 10.0) ** 2) + 0.2
        counts = rng.poisson(rate * dt)
        sp_idx = np.repeat(np.arange(n_time), counts)
        t_sp = time[sp_idx] + rng.uniform(-0.4 * dt, 0.4 * dt, size=sp_idx.size)
        t_sp = np.clip(t_sp, time[0], time[-1])

        # Strongly curved banana distribution in 2-D mark space.
        x1 = rng.normal(0.0, 1.8, size=sp_idx.size)
        sign = 1.0 if u_idx == 0 else -1.0
        x2 = sign * (0.45 * (x1**2 - 1.5)) + rng.normal(0.0, 0.22, size=sp_idx.size)
        # Interlocking arcs that overlap in linear Gaussian ellipse approximation.
        marks = np.column_stack([x1, x2 + 0.3 * sign])
        spike_t_parts.append(t_sp)
        spike_m_parts.append(marks)

    all_t = np.concatenate(spike_t_parts)
    all_m = np.concatenate(spike_m_parts, axis=0)
    order = np.argsort(all_t)
    return env, time, pos, [all_t[order]], [all_m[order]]


@pytest.mark.integration
def test_flow_warping_beats_identity_gaussian_on_curved_banana_marks():
    """Flow-warped mixture beats identity Gaussian mixture on non-Gaussian arcs."""
    env, time, pos, sp_t, sp_m = _simulate_curved_banana_session(seed=42, n_time=3000)
    split = 2000
    cut = time[split]
    tr_t = [t[t < cut] for t in sp_t]
    tr_m = [m[t < cut] for t, m in zip(sp_t, sp_m, strict=True)]
    te_t = [t[t >= cut] for t in sp_t]
    te_m = [m[t >= cut] for t, m in zip(sp_t, sp_m, strict=True)]
    te_edges = time_edges_from_centers(time[split:])

    common = {
        "n_mark_components": 2,
        "covariance_rank": None,
        "mark_reg_covar": 1e-2,
        "em_validation_blocks": 0,
        "disable_progress_bar": True,
        "random_state": 0,
    }
    gaussian_model = fit_clusterless_flow_mixture_encoding_model(
        time[:split],
        pos[:split],
        tr_t,
        tr_m,
        env,
        n_flow_layers=0,
        n_em_iterations=8,
        **common,
    )
    flow_joint = fit_clusterless_flow_mixture_encoding_model(
        time[:split],
        pos[:split],
        tr_t,
        tr_m,
        env,
        n_flow_layers=4,
        n_em_iterations=12,
        flow_steps_per_em=25,
        flow_learning_rate=3e-3,
        flow_objective="joint",
        **common,
    )
    flow_marginal = fit_clusterless_flow_mixture_encoding_model(
        time[:split],
        pos[:split],
        tr_t,
        tr_m,
        env,
        n_flow_layers=4,
        n_em_iterations=12,
        flow_steps_per_em=25,
        flow_learning_rate=3e-3,
        flow_objective="marginal",
        **common,
    )
    refs = fit_mark_references(tr_m, kernel_scale=np.sqrt(1e-2))

    ll_gauss = float(
        np.sum(
            predict_clusterless_flow_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_t,
                te_m,
                time_edges=te_edges,
                is_local=True,
                mark_reference=refs,
                **gaussian_model,
            )
        )
    )
    ll_joint = float(
        np.sum(
            predict_clusterless_flow_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_t,
                te_m,
                time_edges=te_edges,
                is_local=True,
                mark_reference=refs,
                **flow_joint,
            )
        )
    )
    ll_marg = float(
        np.sum(
            predict_clusterless_flow_mixture_log_likelihood(
                time[split:],
                pos[split:],
                te_t,
                te_m,
                time_edges=te_edges,
                is_local=True,
                mark_reference=refs,
                **flow_marginal,
            )
        )
    )

    assert ll_joint > ll_gauss + 50.0, f"joint={ll_joint:.1f} vs gauss={ll_gauss:.1f}"
    assert ll_marg > ll_gauss + 20.0, f"marginal={ll_marg:.1f} vs gauss={ll_gauss:.1f}"


@pytest.mark.integration
def test_unit_change_invariance_during_fit_and_predict():
    """Uniform affine rescaling shifts fit+predict log-likelihood by exact Jacobian."""
    env, time, pos, sp_t, sp_m = _simulate_curved_banana_session(seed=5, n_time=1000)
    scale = 10.0
    offset = np.array([25.0, -40.0])
    sp_m_scaled = [m * scale + offset for m in sp_m]
    edges = time_edges_from_centers(time)

    common = {
        "n_mark_components": 2,
        "n_flow_layers": 2,
        "n_em_iterations": 2,
        "flow_steps_per_em": 5,
        "em_validation_blocks": 0,
        "disable_progress_bar": True,
        "random_state": 0,
    }
    m1 = fit_clusterless_flow_mixture_encoding_model(
        time, pos, sp_t, sp_m, env, mark_reg_covar=0.05, **common
    )
    m2 = fit_clusterless_flow_mixture_encoding_model(
        time, pos, sp_t, sp_m_scaled, env, mark_reg_covar=0.05 * scale**2, **common
    )
    ref1 = fit_mark_references(sp_m, kernel_scale=np.sqrt(0.05))
    ref2 = fit_mark_references(sp_m_scaled, kernel_scale=np.sqrt(0.05) * scale)

    counts = np.histogram(sp_t[0], edges)[0]
    jacobian = 2.0 * np.log(scale)
    for is_local in (False, True):
        ll1 = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time,
                pos,
                sp_t,
                sp_m,
                time_edges=edges,
                is_local=is_local,
                mark_reference=ref1,
                **m1,
            )
        )
        ll2 = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time,
                pos,
                sp_t,
                sp_m_scaled,
                time_edges=edges,
                is_local=is_local,
                mark_reference=ref2,
                **m2,
            )
        )
        np.testing.assert_allclose(
            ll2 + counts[:, None] * jacobian, ll1, rtol=1e-4, atol=1e-2
        )


@pytest.mark.integration
def test_local_geometry_and_zca_noise_covariance_on_dense_shank():
    """16-ch shank with contact_positions & noise_covariances decodes accurately."""
    rng = np.random.default_rng(21)
    n_ch = 16
    coords = np.column_stack([(np.arange(n_ch) % 2) * 20.0, np.arange(n_ch) * 20.0])
    # Spatially correlated noise covariance.
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    noise_cov = 4.0 * np.exp(-dist / 40.0) + 1.0 * np.eye(n_ch)

    env = _make_1d_env(100.0, 5.0)
    n_time = 2000
    dt = 0.02
    time = np.arange(n_time) * dt
    pos = (50.0 + 45.0 * np.sin(2.0 * np.pi * time / 10.0))[:, None]

    # 4 units localized along the shank with distinct place fields.
    unit_centers = [15.0, 40.0, 65.0, 85.0]
    unit_contacts = [2, 6, 10, 13]
    chol = np.linalg.cholesky(noise_cov)
    t_parts, m_parts = [], []
    for pf, peak_c in zip(unit_centers, unit_contacts, strict=True):
        rate = 20.0 * np.exp(-0.5 * ((pos[:, 0] - pf) / 8.0) ** 2) + 0.1
        counts = rng.poisson(rate * dt)
        sp_idx = np.repeat(np.arange(n_time), counts)
        t_sp = np.clip(
            time[sp_idx] + rng.uniform(-0.3 * dt, 0.3 * dt, size=sp_idx.size),
            time[0],
            time[-1],
        )
        # Local biophysical footprint + skewed gain modulation + correlated noise.
        footprint = 30.0 * np.exp(-dist[peak_c] / 35.0)
        gain = rng.lognormal(0.0, 0.25, size=(sp_idx.size, 1))
        noise = rng.normal(size=(sp_idx.size, n_ch)) @ chol.T
        t_parts.append(t_sp)
        m_parts.append(gain * footprint[None, :] + noise)

    all_t = np.concatenate(t_parts)
    all_m = np.concatenate(m_parts, axis=0)
    order = np.argsort(all_t)
    sp_t, sp_m = [all_t[order]], [all_m[order]]

    split = 1400
    cut = time[split]
    decoder = ClusterlessDecoder(
        clusterless_algorithm="clusterless_flow_mixture",
        clusterless_algorithm_params={
            "n_mark_components": 8,
            "covariance_rank": 4,
            "contact_positions": [coords],
            "noise_covariances": [noise_cov],
            "n_flow_layers": 2,
            "n_em_iterations": 4,
            "flow_steps_per_em": 10,
            "em_validation_blocks": 2,
            "disable_progress_bar": True,
            "random_state": 0,
        },
        environments=env,
        sampling_frequency=50,
    )
    decoder.fit(
        time[:split],
        pos[:split],
        [t[t < cut] for t in sp_t],
        [m[t < cut] for t, m in zip(sp_t, sp_m, strict=True)],
    )
    (enc,) = decoder.encoding_model_.values()
    mix = enc["mark_mixtures"][0]
    assert mix is not None
    assert mix.flow_mode == "local"
    assert mix.noise_whitener is not None

    results = decoder.predict(
        [t[t >= cut] for t in sp_t],
        [m[t >= cut] for t, m in zip(sp_t, sp_m, strict=True)],
        time_edges=time_edges_from_centers(time[split:]),
    )
    post = results.acausal_posterior.values
    assert np.all(np.isfinite(post))
    np.testing.assert_allclose(post.sum(axis=-1), 1.0, atol=1e-5)
    centers = env.place_bin_centers_[env.is_track_interior_flat_, 0]
    pred_pos = centers[np.argmax(post, axis=-1)]
    rmse = float(np.sqrt(np.mean((pred_pos - pos[split:, 0]) ** 2)))
    assert rmse < 12.0, f"Expected accurate 16-ch decoding, got RMSE={rmse:.2f}"


@pytest.mark.unit
def test_edge_cases_1d_zero_rate_single_spike_constant_feature():
    """1-D marks, zero-rate electrode, single-spike electrode, and constant feature."""
    env = _make_1d_env(50.0, 5.0)
    time = np.linspace(0.0, 4.0, 201)
    pos = np.linspace(0.0, 50.0, 201)[:, None]
    rng = np.random.default_rng(0)

    sp_t = [
        np.linspace(0.2, 3.8, 40),  # 1-D marks (elementwise mode)
        np.array([]),  # empty electrode (zero-rate)
        np.array([1.5]),  # single spike
        np.linspace(0.3, 3.7, 30),  # constant second feature
    ]
    sp_m = [
        rng.normal(10.0, 2.0, size=(40, 1)),
        np.zeros((0, 3)),
        np.array([[1.0, 2.0]]),
        np.column_stack([rng.normal(0.0, 1.0, size=30), np.full(30, 7.0)]),
    ]
    with pytest.warns(DataQualityWarning, match="electrode 1 has no effective"):
        model = fit_clusterless_flow_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            n_mark_components=4,
            n_flow_layers=2,
            n_em_iterations=2,
            flow_steps_per_em=5,
            em_validation_blocks=0,
            disable_progress_bar=True,
            random_state=0,
        )
    assert model["mark_mixtures"][0] is not None
    assert model["mark_mixtures"][0].flow_mode == "elementwise"
    assert model["mark_mixtures"][1] is None
    assert model["mark_mixtures"][2] is not None
    assert model["mark_mixtures"][3] is not None

    refs = fit_mark_references(sp_m, kernel_scale=1.0)
    edges = time_edges_from_centers(time[:50])
    # Include an extreme mark that overflows float32 on electrode 0.
    test_t = [np.array([0.1, 0.2]), np.array([0.15]), np.array([0.25]), np.array([0.3])]
    test_m = [
        np.array([[10.0], [1e40]]),
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[1.0, 2.0]]),
        np.array([[0.0, 7.0]]),
    ]
    for is_local in (False, True):
        ll = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time[:50],
                pos[:50],
                test_t,
                test_m,
                time_edges=edges,
                is_local=is_local,
                mark_reference=refs,
                **model,
            )
        )
        assert np.all(np.isfinite(ll))


@pytest.mark.unit
def test_row_slice_and_nld_roundtrip_and_optax_not_needed_for_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """row_slice parity, .nld persistence round-trip, and optax-free load/predict."""
    env, time, pos, sp_t, sp_m = _simulate_curved_banana_session(seed=9, n_time=800)
    detector = NonLocalClusterlessDetector(
        clusterless_algorithm="clusterless_flow_mixture",
        clusterless_algorithm_params={
            "n_mark_components": 4,
            "n_flow_layers": 2,
            "n_em_iterations": 2,
            "flow_steps_per_em": 5,
            "em_validation_blocks": 2,
            "disable_progress_bar": True,
            "random_state": 0,
        },
        environments=env,
        sampling_frequency=50,
    )
    detector.fit(time[:600], pos[:600], sp_t, sp_m)
    edges = time_edges_from_centers(time[600:])
    expected = detector.predict(
        sp_t, sp_m, time_edges=edges, position=pos[600:], position_time=time[600:]
    ).acausal_posterior.values

    # Verify row_slice parity on direct predict (non-local & local).
    (enc,) = detector.encoding_model_.values()
    for is_local in (False, True):
        full = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time[600:],
                pos[600:],
                sp_t,
                sp_m,
                time_edges=edges,
                is_local=is_local,
                mark_reference=detector.mark_reference_,
                **enc,
            )
        )
        s1 = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time[600:],
                pos[600:],
                sp_t,
                sp_m,
                time_edges=edges,
                is_local=is_local,
                row_slice=slice(0, 75),
                mark_reference=detector.mark_reference_,
                **enc,
            )
        )
        s2 = np.asarray(
            predict_clusterless_flow_mixture_log_likelihood(
                time[600:],
                pos[600:],
                sp_t,
                sp_m,
                time_edges=edges,
                is_local=is_local,
                row_slice=slice(75, 200),
                mark_reference=detector.mark_reference_,
                **enc,
            )
        )
        np.testing.assert_allclose(np.vstack([s1, s2]), full, rtol=1e-5, atol=1e-5)

    # Save to .nld, hide optax, and confirm load + predict succeed identically
    # while fit raises a helpful ImportError naming non_local_detector[neural].
    save_path = tmp_path / "flow_model.nld"
    detector.save_model(save_path)

    monkeypatch.setitem(sys.modules, "optax", None)
    loaded = NonLocalClusterlessDetector.load_model(save_path)
    got = loaded.predict(
        sp_t, sp_m, time_edges=edges, position=pos[600:], position_time=time[600:]
    ).acausal_posterior.values
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    with pytest.raises(ImportError, match=r"non_local_detector\[neural\]"):
        fit_clusterless_flow_mixture_encoding_model(
            time[:100], pos[:100], sp_t, sp_m, env
        )


@pytest.mark.unit
def test_validation_rejects_invalid_flow_parameters():
    """Invalid flow hyperparameters raise ValidationError up front."""
    env = _make_1d_env(20.0, 5.0)
    time = np.linspace(0.0, 1.0, 21)
    pos = np.linspace(0.0, 20.0, 21)[:, None]
    sp_t = [np.array([0.5])]
    sp_m = [np.array([[1.0, 2.0]])]

    with pytest.raises(ValidationError, match="n_spline_bins"):
        fit_clusterless_flow_mixture_encoding_model(
            time, pos, sp_t, sp_m, env, n_spline_bins=1
        )
    with pytest.raises(ValidationError, match="flow_objective"):
        fit_clusterless_flow_mixture_encoding_model(
            time,
            pos,
            sp_t,
            sp_m,
            env,
            flow_objective="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError, match="spline_bound"):
        fit_clusterless_flow_mixture_encoding_model(
            time, pos, sp_t, sp_m, env, spline_bound=0.0
        )
    with pytest.raises(ValidationError, match="contact_positions"):
        fit_clusterless_flow_mixture_encoding_model(
            time, pos, sp_t, sp_m, env, contact_positions=[np.zeros((3, 2))]
        )


@pytest.mark.unit
def test_registry_entry_and_no_recompiles_across_spike_counts():
    """Algorithm is registered and padded predict avoids recompiling across chunks."""
    from non_local_detector.likelihoods.clusterless_flow_mixture import (
        _predict_local_block_jax,
        _predict_nonlocal_block_jax,
    )
    from non_local_detector.likelihoods.sorted_spikes_glm import glm_rates

    assert "clusterless_flow_mixture" in _CLUSTERLESS_ALGORITHMS
    env, time, pos, sp_t, sp_m = _simulate_curved_banana_session(seed=3, n_time=600)
    model = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t,
        sp_m,
        env,
        n_mark_components=4,
        n_flow_layers=2,
        n_em_iterations=1,
        flow_steps_per_em=5,
        em_validation_blocks=0,
        disable_progress_bar=True,
        random_state=0,
    )
    refs = fit_mark_references(sp_m, kernel_scale=1.0)
    edges = time_edges_from_centers(time[:100])
    # Varying spike counts inside the same power-of-two bucket [17..32] must not
    # recompile either the non-local or local kernels or glm_rates.
    for is_local in (False, True):
        glm_after_first = None
        local_after_first = None
        nonlocal_after_first = None
        for n_sp in (18, 25, 31):
            sub_t = [np.linspace(time[1], time[98], n_sp)]
            sub_m = [sp_m[0][:n_sp]]
            ll = predict_clusterless_flow_mixture_log_likelihood(
                time[:100],
                pos[:100],
                sub_t,
                sub_m,
                time_edges=edges,
                mark_reference=refs,
                is_local=is_local,
                **model,
            )
            assert np.all(np.isfinite(np.asarray(ll)))
            if glm_after_first is None:
                glm_after_first = glm_rates._cache_size()
                local_after_first = _predict_local_block_jax._cache_size()
                nonlocal_after_first = _predict_nonlocal_block_jax._cache_size()
            else:
                assert glm_rates._cache_size() == glm_after_first, (
                    f"glm_rates recompiled for is_local={is_local}, n_sp={n_sp}"
                )
                assert _predict_local_block_jax._cache_size() == local_after_first
                assert _predict_nonlocal_block_jax._cache_size() == nonlocal_after_first


@pytest.mark.unit
def test_local_conditioner_is_weight_shared_compact_and_handles_odd_channels():
    """Local flow shares conditioner weights across contacts and works for odd D."""
    coords_32 = np.column_stack([(np.arange(32) % 2) * 20.0, np.arange(32) * 20.0])
    mix_32 = _make_random_trained_mixture(
        32, contact_positions=coords_32, n_flow_layers=4, seed=31
    )
    assert mix_32.flow_mode == "local"
    assert mix_32.w1 is not None
    assert mix_32.w2 is not None
    assert mix_32.w_out is not None
    assert mix_32.w1.ndim == 3, (
        f"Expected shared 3-D weights (L, F_in, H), got {mix_32.w1.shape}"
    )
    total_params = sum(
        arr.size
        for arr in (
            mix_32.w1,
            mix_32.b1,
            mix_32.w2,
            mix_32.b2,
            mix_32.w_out,
            mix_32.b_out,
        )
        if arr is not None
    )
    assert total_params < 40_000, f"Local flow too heavy: {total_params} params"

    # Odd channel count (D=15): exact round-trip bijection and exact Jacobian.
    coords_15 = np.column_stack([(np.arange(15) % 2) * 20.0, np.arange(15) * 20.0])
    mix_15 = _make_random_trained_mixture(
        15, contact_positions=coords_15, n_flow_layers=4, seed=32
    )
    rng = np.random.default_rng(33)
    x = rng.normal(5.0, 2.0, size=(10, 15))
    z, ld_fwd = mix_15.forward(x)
    x_rec, ld_inv = mix_15.inverse(z)
    np.testing.assert_allclose(x_rec, x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-9)


@pytest.mark.integration
def test_zca_whitening_chains_data_scaling_into_spline_box_and_is_unit_invariant():
    """ZCA + per-channel scaling keeps high-SNR peaks inside [-B, B] and is unit-invariant."""
    rng = np.random.default_rng(41)
    n_ch = 16
    coords = np.column_stack([(np.arange(n_ch) % 2) * 20.0, np.arange(n_ch) * 20.0])
    dist = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    noise_cov = 4.0 * np.exp(-dist / 40.0) + 1.5 * np.eye(n_ch)
    chol = np.linalg.cholesky(noise_cov)

    env = _make_1d_env(100.0, 5.0)
    time = np.arange(800) * 0.02
    pos = (50.0 + 45.0 * np.sin(2.0 * np.pi * time / 10.0))[:, None]
    t_parts, m_parts = [], []
    for pf, peak_c in zip([20.0, 45.0, 70.0, 88.0], [2, 6, 10, 13], strict=True):
        rate = 20.0 * np.exp(-0.5 * ((pos[:, 0] - pf) / 9.0) ** 2) + 0.1
        cnt = rng.poisson(rate * 0.02)
        idx = np.repeat(np.arange(len(time)), cnt)
        t_parts.append(np.clip(time[idx], time[0], time[-1]))
        # High-SNR ~80 uV peak vs ~2.3 uV noise (~35 SNR in raw ZCA!).
        fp = 80.0 * np.exp(-dist[peak_c] / 35.0)
        gain = rng.lognormal(0.0, 0.25, size=(idx.size, 1))
        m_parts.append(gain * fp[None, :] + rng.normal(size=(idx.size, n_ch)) @ chol.T)
    all_t = np.concatenate(t_parts)
    all_m = np.concatenate(m_parts, axis=0)
    order = np.argsort(all_t)
    sp_t, sp_m = [all_t[order]], [all_m[order]]

    common = {
        "n_mark_components": 8,
        "covariance_rank": 4,
        "contact_positions": [coords],
        "n_flow_layers": 2,
        "n_em_iterations": 2,
        "flow_steps_per_em": 5,
        "em_validation_blocks": 0,
        "disable_progress_bar": True,
        "random_state": 0,
    }
    m1 = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t,
        sp_m,
        env,
        noise_covariances=[noise_cov],
        mark_reg_covar=0.05,
        **common,
    )
    mix1 = m1["mark_mixtures"][0]
    assert mix1 is not None
    assert mix1.noise_whitener is not None
    # Chained data scaling must scale down high-SNR ZCA channels into [-spline_bound, spline_bound].
    assert np.max(mix1.input_scale) > 2.0
    u1 = mix1.standardize_marks(sp_m[0])
    peak_u = np.max(np.abs(u1), axis=1)
    assert float(np.median(peak_u)) < mix1.spline_bound

    # Exact bijection round-trip with combined ZCA + input_scale.
    z1, ld_f = mix1.forward(sp_m[0][:25])
    m_rec, ld_i = mix1.inverse(z1)
    np.testing.assert_allclose(m_rec, sp_m[0][:25], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ld_f + ld_i, 0.0, atol=1e-9)

    # Simultaneous unit rescaling m -> 10*m, noise_cov -> 100*noise_cov shifts density by -C*log(10)
    # independently of mark_reg_covar.
    scale = 10.0
    m2 = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t,
        [sp_m[0] * scale],
        env,
        noise_covariances=[noise_cov * (scale**2)],
        mark_reg_covar=25.0,
        **common,
    )
    mix2 = m2["mark_mixtures"][0]
    assert mix2 is not None
    probe = sp_m[0][:30]
    np.testing.assert_allclose(
        mix2.log_component_density(probe * scale) + n_ch * np.log(scale),
        mix1.log_component_density(probe),
        rtol=1e-4,
        atol=1e-3,
    )


@pytest.mark.unit
def test_flow_full_covariance_floor_clamps_without_shifting_and_enforces_zca_floor():
    """Full-rank floor clamps only where binding (EM monotone) and enforces ZCA floor."""
    rng = np.random.default_rng(201)
    time = np.arange(200) * 0.02
    pos = np.linspace(0.0, 50.0, 200)[:, None]
    env = _make_1d_env(50.0, 5.0)
    sp_t = [np.sort(rng.uniform(0.05, 3.9, 30))]
    sp_m = [rng.normal(0.0, 3.0, size=(30, 2))]

    # Identity flow (n_flow_layers=0): pure full-rank EM must be monotone when
    # sample covariance eigenvalues (~9) already exceed mark_reg_covar=1.0.
    enc = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t,
        sp_m,
        env,
        n_mark_components=1,
        covariance_rank=None,
        mark_reg_covar=1.0,
        n_flow_layers=0,
        n_em_iterations=5,
        em_validation_blocks=0,
        disable_progress_bar=True,
        random_state=0,
    )
    trace = enc["em_diagnostics"]["per_electrode"][0]["train_log_likelihood"]
    assert len(trace) >= 2
    assert np.all(np.diff(trace) >= -1e-6), f"Non-monotone full-cov EM trace: {trace}"

    # Exact 1-D, 1-component full-covariance EM must also be monotone non-decreasing.
    sp_m_1d = [rng.normal(0.0, 2.5, size=(30, 1))]
    enc_1d = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t,
        sp_m_1d,
        env,
        n_mark_components=1,
        covariance_rank=None,
        mark_reg_covar=1.0,
        n_flow_layers=0,
        n_em_iterations=5,
        em_validation_blocks=0,
        disable_progress_bar=True,
        random_state=0,
    )
    trace_1d = enc_1d["em_diagnostics"]["per_electrode"][0]["train_log_likelihood"]
    assert len(trace_1d) >= 2
    assert np.all(np.diff(trace_1d) >= -1e-6), (
        f"Non-monotone 1-D full-cov EM trace: {trace_1d}"
    )

    # Identical marks + identity noise covariance at iter=0 must enforce unit floor.
    sp_t_id = [np.linspace(0.2, 3.5, 5)]
    sp_m_id = [np.tile(np.array([5.0, -3.0]), (5, 1))]
    enc_zca = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        sp_t_id,
        sp_m_id,
        env,
        noise_covariances=[np.eye(2)],
        n_mark_components=1,
        covariance_rank=None,
        n_flow_layers=0,
        n_em_iterations=0,
        em_validation_blocks=0,
        disable_progress_bar=True,
        random_state=0,
    )
    mix_zca = enc_zca["mark_mixtures"][0]
    assert mix_zca is not None
    # In noise_variance-standardized coordinates, every full-rank eigenvalue >= 1.0.
    assert np.all(mix_zca.basis_variances >= 1.0 - 1e-12)


@pytest.mark.unit
def test_flow_split_event_em_and_single_fold_owner():
    """Split-position E-step is symmetric on midpoint spikes and folds own boundary spikes."""
    from non_local_detector.likelihoods.clusterless_flow_mixture import (
        _accumulate_cell_counts,
        _e_step_latent_split,
        _partition_fold_shares,
    )

    # Midpoint spike between two cells with opposing rates, identical latent mark density.
    z = np.zeros((1, 2), dtype=np.float64)
    ld = np.zeros((1,), dtype=np.float64)
    lr_l = np.log(np.array([[20.0, 0.1]]))
    lr_r = np.log(np.array([[0.1, 20.0]]))
    means = np.zeros((2, 2), dtype=np.float64)
    basis = np.broadcast_to(np.eye(2)[None, :, :], (2, 2, 2)).copy()
    bvar = np.ones((2, 2), dtype=np.float64)
    nvar = np.ones((2,), dtype=np.float64)
    resp_l, resp_r, ll_l, ll_r = _e_step_latent_split(
        z, ld, lr_l, lr_r, means, basis, bvar, nvar
    )
    assert resp_l[0, 0] > 0.99
    assert resp_r[0, 1] > 0.99
    np.testing.assert_allclose(resp_l[0, 0], resp_r[0, 1], atol=1e-12)
    np.testing.assert_allclose(ll_l, ll_r, atol=1e-12)

    counts = _accumulate_cell_counts(
        np.array([0, 1], dtype=np.int64),
        2,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0.5]),
        np.array([0.5]),
        resp_l,
        resp_r,
    )
    assert counts[0, 0] > 0.49
    assert counts[1, 1] > 0.49
    np.testing.assert_allclose(counts[0, 0], counts[1, 1], atol=1e-12)

    # Fold-boundary spike at midpoint between sample 4 (train) and sample 5 (val).
    tr_mask = np.array([True] * 5 + [False] * 5)
    va_mask = ~tr_mask
    l_pos = np.array([4], dtype=np.int64)
    r_pos = np.array([5], dtype=np.int64)
    l_sh = np.array([0.5])
    r_sh = np.array([0.5])
    tr_sel, tr_ls, tr_rs = _partition_fold_shares(l_pos, r_pos, l_sh, r_sh, tr_mask)
    va_sel, va_ls, va_rs = _partition_fold_shares(l_pos, r_pos, l_sh, r_sh, va_mask)
    assert bool(tr_sel[0]) ^ bool(va_sel[0])
    if tr_sel[0]:
        np.testing.assert_allclose(tr_ls[0] + tr_rs[0], 1.0, atol=1e-12)
        assert tr_rs[0] == 0.0
    else:
        np.testing.assert_allclose(va_ls[0] + va_rs[0], 1.0, atol=1e-12)
        assert va_ls[0] == 0.0


@pytest.mark.unit
def test_flow_validation_folds_disjoint_on_two_sample_intervening_spikes():
    """With 2 tracking samples and 6 intervening spikes, train and val spikes are disjoint."""
    from non_local_detector.encoding_time import prepare_encoding_support
    from non_local_detector.likelihoods.clusterless_flow_mixture import (
        _partition_fold_shares,
    )

    time = np.array([0.0, 1.0], dtype=np.float64)
    pos = np.array([[10.0], [90.0]], dtype=np.float64)
    env = _make_1d_env(100.0, 20.0)
    sp_t = np.array([0.1, 0.25, 0.4, 0.6, 0.75, 0.9], dtype=np.float64)
    sp_m = np.linspace(-2.0, 2.0, 12, dtype=np.float64).reshape(6, 2)

    support, _, _, _ = prepare_encoding_support(time, pos, None)
    _, left_pos, right_pos, frac = support.event_basis(sp_t)
    left_sh = 1.0 - frac
    right_sh = frac
    assert np.all(left_pos == 0)
    assert np.all(right_pos == 1)

    for val_idx in (0, 1):
        val_mask = np.arange(2) == val_idx
        tr_mask = ~val_mask
        tr_sel, tr_ls, tr_rs = _partition_fold_shares(
            left_pos, right_pos, left_sh, right_sh, tr_mask
        )
        va_sel, va_ls, va_rs = _partition_fold_shares(
            left_pos, right_pos, left_sh, right_sh, val_mask
        )
        # No spike may belong to both training and validation; every spike has one owner.
        assert not np.any(tr_sel & va_sel)
        assert np.all(tr_sel | va_sel)
        np.testing.assert_allclose(
            float(np.sum(tr_ls + tr_rs) + np.sum(va_ls + va_rs)),
            float(np.sum(left_sh + right_sh)),
            atol=1e-12,
        )

    # End-to-end fit with 2 validation blocks must also run cleanly.
    enc = fit_clusterless_flow_mixture_encoding_model(
        time,
        pos,
        [sp_t],
        [sp_m],
        env,
        n_mark_components=2,
        n_flow_layers=2,
        n_em_iterations=2,
        flow_steps_per_em=2,
        em_validation_blocks=2,
        disable_progress_bar=True,
        random_state=0,
    )
    assert enc["mark_mixtures"][0] is not None


@pytest.mark.unit
def test_flow_split_position_continuous_across_step_midpoint():
    """Spikes at 9.98 cm vs 10.02 cm on a 0->20 cm step vary continuously, not abruptly."""
    from non_local_detector.encoding_time import prepare_encoding_support
    from non_local_detector.likelihoods.clusterless_flow_mixture import (
        _accumulate_cell_counts,
        _e_step_latent_split,
    )

    # Tracking steps from 0 cm at t=0 to 20 cm at t=1.
    time = np.array([0.0, 1.0], dtype=np.float64)
    pos = np.array([[0.0], [20.0]], dtype=np.float64)
    support, _, _, _ = prepare_encoding_support(time, pos, None)
    # Spikes at interpolated positions 9.98 cm (t=0.499) and 10.02 cm (t=0.501).
    sp_t = np.array([0.499, 0.501], dtype=np.float64)
    _, l_pos, r_pos, frac = support.event_basis(sp_t)
    l_sh = 1.0 - frac
    r_sh = frac
    np.testing.assert_allclose(l_sh, [0.501, 0.499], atol=1e-12)
    np.testing.assert_allclose(r_sh, [0.499, 0.501], atol=1e-12)

    # Identical marks; cell 0 has high rate for component 0, cell 1 has much
    # lower total rate and reversed component preference so ll_l != ll_r too.
    z = np.zeros((2, 2), dtype=np.float64)
    ld = np.zeros((2,), dtype=np.float64)
    cell_lr = np.log(np.array([[25.0, 0.1], [0.1, 2.0]]))
    means = np.zeros((2, 2), dtype=np.float64)
    basis = np.broadcast_to(np.eye(2)[None, :, :], (2, 2, 2)).copy()
    bvar = np.ones((2, 2), dtype=np.float64)
    nvar = np.ones((2,), dtype=np.float64)

    resp_l, resp_r, ll_l, ll_r = _e_step_latent_split(
        z, ld, cell_lr[l_pos], cell_lr[r_pos], means, basis, bvar, nvar
    )
    combined_resp = l_sh[:, None] * resp_l + r_sh[:, None] * resp_r
    combined_ll = l_sh * ll_l + r_sh * ll_r
    # Continuous O(0.002) change across the midpoint, never a discontinuous flip.
    assert np.max(np.abs(combined_resp[0] - combined_resp[1])) < 0.005
    assert abs(float(combined_ll[0] - combined_ll[1])) < 0.01

    sample_cells = np.array([0, 1], dtype=np.int64)
    c_a = _accumulate_cell_counts(
        sample_cells,
        2,
        l_pos[:1],
        r_pos[:1],
        l_sh[:1],
        r_sh[:1],
        resp_l[:1],
        resp_r[:1],
    )
    c_b = _accumulate_cell_counts(
        sample_cells,
        2,
        l_pos[1:],
        r_pos[1:],
        l_sh[1:],
        r_sh[1:],
        resp_l[1:],
        resp_r[1:],
    )
    assert np.max(np.abs(c_a - c_b)) < 0.005


@pytest.mark.unit
def test_flow_minibatch_sampling_unbiased_for_unequal_spike_weights():
    """Weighted mini-batching at batch_size=1 matches the full weighted objective."""
    from non_local_detector.likelihoods.clusterless_flow_mixture import (
        _sample_flow_batches,
    )

    n_hi, n_lo, n_zero = 10, 10, 2
    n_tot = n_hi + n_lo + n_zero
    # Encode spike identity in column 0 so we can measure exact expectations.
    u = np.zeros((n_tot, 2), dtype=np.float64)
    u[:n_hi, 0] = 1.0  # heavy spikes -> value 1.0
    u[n_hi : n_hi + n_lo, 0] = -1.0  # light spikes -> value -1.0
    u[n_hi + n_lo :, 0] = 99.0  # zero-weight spikes -> must never be sampled
    w = np.array([1.0] * n_hi + [1e-4] * n_lo + [0.0] * n_zero, dtype=np.float64)
    lw = np.zeros((n_tot, 2), dtype=np.float64)

    rng = np.random.default_rng(7)
    u_b, w_b, _ = _sample_flow_batches(
        u, w, lw, n_steps=4000, batch_size=1, k_pad=2, rng=rng
    )
    u_vals = np.asarray(u_b[:, 0, 0], dtype=np.float64)
    w_vals = np.asarray(w_b[:, 0], dtype=np.float64)

    # Zero-weight spikes must never appear in sampled mini-batches.
    assert not np.any(u_vals > 50.0)
    # Per-batch normalized expectation matches exact weighted mean (~0.9998, NOT 0.0).
    exact_mean = float(np.dot(w, u[:, 0]) / np.sum(w))
    # Simulate _loss's per-batch normalization: sum(w_b * val) / sum(w_b).
    batch_means = (w_vals * u_vals) / np.maximum(w_vals, 1e-12)
    np.testing.assert_allclose(float(np.mean(batch_means)), exact_mean, atol=0.01)


@pytest.mark.unit
def test_position_free_responsibilities_blocked_parity_and_bounded_chunks(monkeypatch):
    """Initial responsibilities chunk by _E_STEP_BLOCK and match dense float64."""
    from non_local_detector.likelihoods import clusterless_flow_mixture as cfm

    rng = np.random.default_rng(31)
    n_sp, n_dims, n_comp, rank = 2500, 6, 5, 3
    z = rng.normal(size=(n_sp, n_dims))
    weights = rng.dirichlet(np.ones(n_comp))
    means = rng.normal(scale=0.5, size=(n_comp, n_dims))
    basis = np.empty((n_comp, n_dims, rank), dtype=np.float64)
    for k in range(n_comp):
        q, _ = np.linalg.qr(rng.normal(size=(n_dims, rank)))
        basis[k] = q
    bvar = rng.uniform(1.1, 4.0, size=(n_comp, rank))
    nvar = rng.uniform(0.3, 1.5, size=(n_dims,))

    # Exact dense reference computed in one shot on full array.
    inv_sqrt_psi = 1.0 / np.sqrt(nvar)
    log_det_psi = float(np.sum(np.log(nvar)))
    delta = (z * inv_sqrt_psi[None, :])[:, None, :] - (means * inv_sqrt_psi[None, :])[
        None, :, :
    ]
    proj = np.einsum("nkd,kdr->nkr", delta, basis)
    recon = np.einsum("nkr,kdr->nkd", proj, basis)
    orth_sq = np.sum((delta - recon) ** 2, axis=-1)
    par_sq = np.sum((proj * proj) / bvar[None, :, :], axis=-1)
    log_const = -0.5 * (
        n_dims * np.log(2.0 * np.pi) + log_det_psi + np.sum(np.log(bvar), axis=-1)
    )
    lw_ref = log_const[None, :] - 0.5 * (orth_sq + par_sq) + np.log(weights)[None, :]
    lw_ref -= np.max(lw_ref, axis=1, keepdims=True)
    resp_ref = np.exp(lw_ref)
    resp_ref /= np.sum(resp_ref, axis=1, keepdims=True)

    # Track chunk sizes passed to _latent_woodbury_log_density_f64.
    seen_chunks: list[int] = []
    orig_fn = cfm._latent_woodbury_log_density_f64

    def spy_density(z_blk, *args, **kwargs):
        seen_chunks.append(int(z_blk.shape[0]))
        return orig_fn(z_blk, *args, **kwargs)

    monkeypatch.setattr(cfm, "_latent_woodbury_log_density_f64", spy_density)
    resp = cfm._position_free_responsibilities_f64(z, weights, means, basis, bvar, nvar)
    np.testing.assert_allclose(resp, resp_ref, rtol=1e-12, atol=1e-12)
    assert len(seen_chunks) == 3  # 1024 + 1024 + 452
    assert max(seen_chunks) <= cfm._E_STEP_BLOCK
