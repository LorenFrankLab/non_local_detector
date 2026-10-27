"""Tests for the sorted-spikes MRF-GAM likelihood.

``sorted_spikes_mrf`` fits a penalized-Poisson GAM whose reduced-rank basis is the
smoothest eigenmodes of the environment's graph Laplacian (the same spectral engine
as ``sorted_spikes_diffusion``) and whose penalty is those eigenvalues, with
occupancy entering as a log-offset (never a denominator) and the smoothing
parameter chosen by REML. These tests pin the vectorized population fit against an
independent per-neuron reference, the eigenvalue-ridge penalty, occupancy-offset
robustness, REML field recovery, no cross-wall leakage, and the encoding contract.
"""

import warnings

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
import scipy.linalg

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.common import RATE_EPS_HZ
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    cached_eigenbasis,
    diffusion_eigenbasis,
    environment_graph,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    pixellate_interior_fields,
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    _newton_fit_jax,
    _reml_score_jax,
    fit_sorted_spikes_mrf_encoding_model,
    mrf_penalized_poisson_fit,
    mrf_reml_objective,
    predict_sorted_spikes_mrf_log_likelihood,
    select_penalty_by_reml,
)
from non_local_detector.tests.likelihoods._sorted_spikes_helpers import (
    dot_general_precisions,
    make_two_room_env,
    path_graph,
    simulate_place_data,
    simulate_sparse_uniform_tracking,
)

ENCODING_DICT_KEYS = {
    "environment",
    "occupancy",
    "mean_rates",
    "place_fields",
    "no_spike_part_log_likelihood",
    "is_track_interior",
    "node_order",
    "bin_sizes",
    "local_interpolation",
    "disable_progress_bar",
    "mrf_penalty",
    "mrf_rank",
    "mrf_coefficients",
    "mrf_penalty_weights",
    "mrf_reml_objective",
    "mrf_n_iter",
    "mrf_converged",
    "mrf_max_step",
    "mrf_log_penalty_bounds",
    "mrf_penalty_selected_by_reml",
}


def reference_fit_one_neuron(counts, occupancy, basis, penalty_weights, penalty):
    """Independent per-neuron penalized-Poisson IRLS reference (mgcv `_fit_dense`).

    Fits ``counts ~ Poisson(occupancy * exp(basis @ gamma))`` with penalty
    ``penalty * gamma^T diag(penalty_weights) gamma`` via Newton/IRLS.
    """
    penalty_diag = penalty * penalty_weights
    eta = np.full(
        basis.shape[0],
        np.log(max(counts.sum() / max(occupancy.sum(), 1e-9), 1e-6)),
    )
    gamma = np.linalg.lstsq(basis, eta, rcond=None)[0]
    for _ in range(100):
        eta = basis @ gamma
        mu = occupancy * np.exp(np.clip(eta, -30, 30))
        grad = basis.T @ (counts - mu) - penalty_diag * gamma
        hessian = basis.T @ (basis * mu[:, None])
        hessian[np.diag_indices_from(hessian)] += penalty_diag + 1e-10
        cho = scipy.linalg.cho_factor(hessian, lower=True)
        step = scipy.linalg.cho_solve(cho, grad)
        gamma = gamma + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return gamma


def test_population_fit_matches_per_neuron_reference():
    """The vectorized population fit equals an independent per-neuron IRLS reference.

    Covers both "population fit == per-neuron loop" and "matches a penalized-Poisson
    reference" — the reference is a distinct implementation (Cholesky-based, one
    neuron at a time).
    """
    rng = np.random.default_rng(0)
    n_bins, rank, n_neurons = 40, 6, 5
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=rank
    )
    occupancy = rng.uniform(0.5, 3.0, size=n_bins)
    counts = rng.poisson(1.5, size=(n_bins, n_neurons)).astype(float)
    penalty = 2.0

    coeffs, eta, mu, diagnostics = mrf_penalized_poisson_fit(
        counts,
        occupancy,
        basis,
        penalty_weights,
        penalty,
    )

    assert coeffs.shape == (rank, n_neurons)
    assert eta.shape == (n_bins, n_neurons)
    assert mu.shape == (n_bins, n_neurons)
    assert diagnostics["n_iter"] >= 1
    assert isinstance(diagnostics["converged"], bool)
    assert np.isfinite(diagnostics["max_step"])
    # The JAX fit runs in float32 (package regime), so the batched fit agrees with the
    # float64 per-neuron reference to ~float32 precision (~1e-7), not machine epsilon.
    for neuron in range(n_neurons):
        expected = reference_fit_one_neuron(
            counts[:, neuron], occupancy, basis, penalty_weights, penalty
        )
        np.testing.assert_allclose(coeffs[:, neuron], expected, atol=1e-5)
    # eta / mu are consistent with the fitted coefficients (float32 self-consistency).
    np.testing.assert_allclose(eta, basis @ coeffs, atol=1e-5)
    np.testing.assert_allclose(mu, occupancy[:, None] * np.exp(eta), rtol=1e-5)


def test_fit_uses_smoothest_eigenmodes_as_basis_and_eigenvalues_as_ridge(env_2d):
    """The encoding fit's design is the environment's smoothest Laplacian eigenmodes
    and its penalty is ``penalty * eigenvalue`` per mode (a generalized ridge).

    The stored penalty weights are the cached eigenvalues, the interior log fields are
    ``basis @ coefficients``, and the coefficients equal an independent float64
    penalized-Poisson fit on the same counts and exposure with that ridge.
    """
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    rank, penalty = 10, 1.0
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=rank,
        penalty=penalty,
    )
    eigenvalues, basis = cached_eigenbasis(env, rank)

    penalty_weights = np.asarray(encoding["mrf_penalty_weights"])
    np.testing.assert_array_equal(penalty_weights, eigenvalues)
    # Smoothest modes first: ascending eigenvalues, starting at the constant null mode.
    assert np.all(np.diff(penalty_weights) >= 0.0)
    assert penalty_weights[0] < 1e-9

    coefficients = np.asarray(encoding["mrf_coefficients"])
    assert coefficients.shape == (rank, len(spike_times))
    # float32 fit: log fields agree with basis @ coefficients to ~1e-6 (max |log| ~8).
    np.testing.assert_allclose(
        np.asarray(encoding["interior_log_place_fields"]),
        (basis @ coefficients).T,
        atol=1e-5,
    )

    _, node_order, _ = environment_graph(env)
    exposure, spike_counts, _ = pixellate_interior_fields(
        time, position, spike_times, env, node_order, np.ones_like(time)
    )
    for neuron, counts in enumerate(spike_counts):
        expected = reference_fit_one_neuron(
            counts, exposure, basis, eigenvalues, penalty
        )
        # float32 fit vs float64 reference: coefficients up to ~25, errors ~1e-4.
        np.testing.assert_allclose(
            coefficients[:, neuron], expected, rtol=1e-5, atol=5e-4
        )


def test_reml_recovers_smooth_field():
    """REML selects a sensible lambda whose fitted rate map tracks a simulated field."""
    rng = np.random.default_rng(1)
    n_bins = 60
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=25
    )
    x = np.arange(n_bins)
    true_log_rate = 2.5 * np.exp(-((x - 30.0) ** 2) / (2 * 6.0**2)) - 0.5
    occupancy = rng.uniform(8.0, 20.0, size=n_bins)  # enough dwell for a clear field
    counts = rng.poisson(occupancy * np.exp(true_log_rate)).astype(float)[:, None]

    penalty, _ = select_penalty_by_reml(counts, occupancy, basis, penalty_weights)
    assert 0.0 < penalty < np.inf

    _, eta, _, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty
    )
    corr = np.corrcoef(np.exp(eta[:, 0]), np.exp(true_log_rate))[0, 1]
    assert corr > 0.95


def test_population_recovers_distinct_place_fields():
    """Each neuron is recovered at its OWN location in a single population fit.

    Guards against the batched-over-neurons fit conflating neurons or mislocating
    fields: fits several neurons with distinct 1D place-field centers at once and
    checks that each fitted rate map (a) peaks at its own true center, (b) tracks its
    simulated field, and (c) equals an independent per-neuron IRLS reference -- so the
    neuron axis stays independent through the shared design and batched solve. The
    existing per-neuron-reference test uses structureless counts; this one uses
    distinct spatial tuning, which is what the population model exists to fit.
    """
    rng = np.random.default_rng(1)
    n_bins = 60
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=25
    )
    x = np.arange(n_bins)
    true_centers = [8, 24, 40, 52]  # distinct place-field locations
    occupancy = rng.uniform(8.0, 20.0, size=n_bins)  # enough dwell for clear fields
    true_log_rate = np.stack(
        [2.5 * np.exp(-((x - c) ** 2) / (2 * 5.0**2)) - 0.5 for c in true_centers],
        axis=1,
    )
    counts = rng.poisson(occupancy[:, None] * np.exp(true_log_rate)).astype(float)

    penalty, _ = select_penalty_by_reml(counts, occupancy, basis, penalty_weights)
    coeffs, eta, _, diagnostics = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty
    )
    assert diagnostics["converged"]
    coeffs = np.asarray(coeffs)
    rate = np.exp(np.asarray(eta))

    for neuron, center in enumerate(true_centers):
        # Peaks at this neuron's own center (not smeared toward another's).
        assert abs(int(np.argmax(rate[:, neuron])) - center) <= 1
        # Tracks the simulated field.
        corr = np.corrcoef(rate[:, neuron], np.exp(true_log_rate[:, neuron]))[0, 1]
        assert corr > 0.95
        # The batched column equals an independent per-neuron reference fit at the same
        # penalty (no cross-neuron coupling from the shared design / batched solve).
        expected = reference_fit_one_neuron(
            counts[:, neuron], occupancy, basis, penalty_weights, penalty
        )
        np.testing.assert_allclose(coeffs[:, neuron], expected, atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["newton", "reml"])
def test_fit_contractions_request_highest_precision(stage):
    """Every MRF fit and REML contraction requests HIGHEST precision.

    Without it, CUDA may evaluate float32 products in TF32, which moved A100
    coefficients by up to 2e-3 from the float64 reference. CPU always computes
    full float32, so CPU CI checks the request in the traced program instead.
    Tracing under the default matmul precision keeps an ambient ``highest``
    setting from satisfying the check for products that request nothing.
    """
    dtype = jnp.float32
    counts = jnp.ones((20, 3), dtype)
    occupancy = jnp.ones(20, dtype)
    basis = jnp.ones((20, 6), dtype)
    penalty_diag = jnp.ones(6, dtype)
    tol = jnp.asarray(1e-8, dtype)
    with jax.default_matmul_precision("default"):
        if stage == "newton":
            traced = jax.make_jaxpr(
                lambda *args: _newton_fit_jax(*args, max_iter=5, tol=tol)
            )(counts, occupancy, basis, penalty_diag)
        else:
            traced = jax.make_jaxpr(
                lambda log_penalty: _reml_score_jax(
                    log_penalty,
                    counts,
                    occupancy,
                    basis,
                    penalty_diag,
                    jnp.asarray(5.0, dtype),
                    5,
                    tol,
                )
            )(jnp.asarray(0.0, dtype))

    precisions = dot_general_precisions(traced.jaxpr)
    assert precisions
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    assert all(precision == highest for precision in precisions), precisions


def test_reml_robust_to_ill_conditioned_hessian():
    """With many zero-occupancy bins the Hessian is rank-deficient at small penalties;
    REML must not emit slogdet RuntimeWarnings and must still return a valid lambda
    (invalid, non-positive-definite candidates are rejected, not silently accepted)."""
    import warnings

    rng = np.random.default_rng(0)
    n_bins = 40
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)),
        rank=None,  # full rank -> ill-conditioned
    )
    occupancy = rng.uniform(1.0, 2.0, size=n_bins)
    occupancy[10:30] = 0.0  # a large unvisited patch
    counts = rng.poisson(1.0, size=(n_bins, 3)).astype(float)
    counts[10:30, :] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        penalty, _ = select_penalty_by_reml(counts, occupancy, basis, penalty_weights)
    assert 0.0 < penalty < np.inf


def test_occupancy_offset_gives_finite_rate_at_zero_occupancy():
    """Occupancy enters as an exposure offset, so zero-occupancy bins yield finite
    (unconstrained, smoothly-interpolated) rates — no division blow-up."""
    rng = np.random.default_rng(2)
    n_bins = 40
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=12
    )
    occupancy = rng.uniform(1.0, 2.0, size=n_bins)
    occupancy[18:22] = 0.0  # an unvisited patch (would divide-by-zero in a ratio)
    counts = rng.poisson(1.0, size=(n_bins, 3)).astype(float)
    counts[18:22, :] = 0.0  # no spikes where there is no occupancy

    _, eta, mu, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1.0
    )
    rate = np.exp(eta)
    assert np.all(np.isfinite(rate))
    assert np.all(rate > 0.0)
    # mu is exactly zero where occupancy is zero (offset, not denominator).
    np.testing.assert_array_equal(mu[18:22, :], 0.0)


def test_mrf_penalty_does_not_smooth_across_a_wall():
    """The eigenbasis is component-local, so a cell firing in one component cannot
    raise the fitted rate in a disconnected one (no smoothing across a wall)."""
    graph = nx.disjoint_union(path_graph(20), path_graph(20))
    penalty_weights, basis = diffusion_eigenbasis(build_laplacian(graph), rank=None)
    occupancy = np.ones(40)
    counts = np.zeros((40, 1))
    counts[5, 0] = 30.0  # all spikes in the first component (nodes 0..19)

    _, eta, _, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1.0
    )
    rate = np.exp(eta[:, 0])
    assert rate[20:].max() < 0.02 * rate[:20].max()


def test_fit_encoding_dict_keys_and_full_grid_shapes(env_2d):
    """The MRF fit returns the shared encoding contract; place_fields are FULL-GRID."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=4)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=30,
    )
    assert ENCODING_DICT_KEYS <= set(encoding)

    is_interior = env.is_track_interior_.ravel()
    n_total = env.place_bin_centers_.shape[0]
    place_fields = np.asarray(encoding["place_fields"])
    assert place_fields.shape == (4, n_total)
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, ~is_interior] == 0.0)
    assert np.all(place_fields[:, is_interior] > 0.0)
    assert encoding["mrf_rank"] == 30
    assert 0.0 <= encoding["mrf_penalty"] < np.inf
    assert np.asarray(encoding["mrf_coefficients"]).shape == (30, 4)
    assert np.asarray(encoding["mrf_penalty_weights"]).shape == (30,)
    assert np.isfinite(encoding["mrf_reml_objective"])
    assert encoding["mrf_n_iter"] >= 1
    assert isinstance(encoding["mrf_converged"], bool)
    assert np.isfinite(encoding["mrf_max_step"])
    assert encoding["mrf_log_penalty_bounds"] == (-8.0, 20.0)
    assert encoding["mrf_penalty_selected_by_reml"] is True
    assert encoding["local_interpolation"] == "linear"
    np.testing.assert_allclose(
        np.asarray(encoding["no_spike_part_log_likelihood"]),
        place_fields.sum(axis=0),
        rtol=1e-6,
    )


def test_fixed_penalty_solver_controls_are_reported(env_2d):
    """Caller-supplied solver controls are honored and reported in the encoding."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=15,
        penalty=0.5,
        max_iter=1,
        tol=1e-12,
        log_penalty_bounds=(-4.0, 2.0),
        reml_xatol=1e-2,
        block_size=7,
    )

    assert encoding["mrf_penalty"] == 0.5
    assert encoding["mrf_penalty_selected_by_reml"] is False
    assert np.isnan(encoding["mrf_reml_objective"])
    assert encoding["mrf_n_iter"] == 1
    assert isinstance(encoding["mrf_converged"], bool)
    assert np.isfinite(encoding["mrf_max_step"])
    assert encoding["mrf_log_penalty_bounds"] == (-4.0, 2.0)


def test_mrf_nonconvergence_warns(env_2d):
    """A penalized-Poisson fit that does not converge in max_iter warns.

    REML total failure already raises; the softer (and more common) Newton/IRLS
    non-convergence was previously silent -- `mrf_converged` was stored in the
    encoding dict but nothing surfaced it. `max_iter=1` with a tight tolerance
    forces a non-converged fit.
    """
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    with pytest.warns(UserWarning, match="did not converge"):
        fit_sorted_spikes_mrf_encoding_model(
            position_time=time,
            position=position,
            spike_times=spike_times,
            environment=env,
            rank=15,
            penalty=0.5,
            max_iter=1,
            tol=1e-12,
            block_size=7,
        )


def test_mrf_auto_rank_cap_warns(env_2d, monkeypatch):
    """An auto-selected rank capped below the interior-bin count warns.

    On a fine grid the default cap drops the highest-frequency eigenmodes; this
    was documented only in the docstring. Monkeypatch the cap low so a small
    environment triggers it.
    """
    import non_local_detector.likelihoods.sorted_spikes_mrf as mrf_module

    monkeypatch.setattr(mrf_module, "_DEFAULT_MAX_RANK", 4)
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    with pytest.warns(UserWarning, match="auto-capped"):
        fit_sorted_spikes_mrf_encoding_model(
            position_time=time,
            position=position,
            spike_times=spike_times,
            environment=env,
            penalty=0.5,
            block_size=7,
        )


def test_fit_rejects_invalid_weights(env_2d):
    """Weights feed both occupancy and spike counts, so invalid values fail early."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=1)
    bad_weights = [
        np.ones(time.shape[0] - 1),
        np.r_[[-1.0], np.ones(time.shape[0] - 1)],
        np.r_[[np.nan], np.ones(time.shape[0] - 1)],
    ]

    for weights in bad_weights:
        with pytest.raises(ValidationError):
            fit_sorted_spikes_mrf_encoding_model(
                position_time=time,
                position=position,
                spike_times=spike_times,
                environment=env,
                weights=weights,
                rank=10,
                penalty=1.0,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"penalty": -1.0},
        {"penalty": np.nan},
        {"rank": 0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"log_penalty_bounds": (1.0, 1.0)},
        {"log_penalty_bounds": (0.0, np.inf)},
        {"reml_xatol": 0.0},
        {"local_interpolation": "cubic"},
    ],
)
def test_fit_rejects_invalid_mrf_controls(env_2d, kwargs):
    """MRF-specific controls have explicit validation and clear failures."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=1)
    fit_kwargs = {"rank": 10, **kwargs}

    with pytest.raises(ValidationError):
        fit_sorted_spikes_mrf_encoding_model(
            position_time=time,
            position=position,
            spike_times=spike_times,
            environment=env,
            **fit_kwargs,
        )


def test_population_fit_rejects_invalid_problem_arrays():
    """The low-level solver rejects malformed arrays before numerical linear algebra."""
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(10)), rank=5
    )
    occupancy = np.ones(10)
    counts = np.ones((10, 2))

    with pytest.raises(ValidationError):
        mrf_penalized_poisson_fit(
            counts[:, 0], occupancy, basis, penalty_weights, penalty=1.0
        )

    with pytest.raises(ValidationError):
        mrf_penalized_poisson_fit(
            counts,
            occupancy,
            basis,
            np.r_[-1.0, penalty_weights[1:]],
            penalty=1.0,
        )


def test_fit_finite_under_near_zero_occupancy_and_low_penalty():
    """A near-zero-occupancy bin with a coincident spike at a low penalty must not
    overflow to inf: the returned eta/mu are clipped (finite-log-prob invariant)."""
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(40)), rank=None
    )
    occupancy = np.ones(40)
    occupancy[10] = 1e-12  # essentially unvisited
    counts = np.zeros((40, 1))
    counts[10, 0] = 5.0  # spikes where there is ~no occupancy -> huge saturated rate

    _, eta, mu, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1e-4
    )
    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(np.exp(eta)))
    assert np.abs(eta).max() <= 30.0 + 1e-9  # clipped to _ETA_CLIP


def test_step_halving_converges_on_near_zero_exposure_overshoot():
    """A near-zero-exposure patch that carries spikes makes a full Newton step
    overshoot catastrophically (the penalized objective explodes by ~10 orders of
    magnitude and then diverges). Per-neuron step-halving (mgcv gam.fit3) must instead
    descend monotonically to a finite, converged solution -- so the returned rates are
    a genuine fit, not clip-salvaged garbage that flows into decoding.
    """
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(40)), rank=None
    )
    rng = np.random.default_rng(0)
    occupancy = rng.uniform(1.0, 2.0, 40)
    occupancy[15:25] = 1e-4  # a near-zero-exposure patch
    counts = rng.poisson(1.0, size=(40, 1)).astype(float)
    counts[15:25, 0] = 5.0  # spikes in that patch -> full-Newton overshoot

    coeffs, eta, mu, diagnostics = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1e-3
    )

    def penalized_objective(c):  # -loglik + 0.5 * penalty, from clipped eta/mu
        e = basis @ c
        m = occupancy[:, None] * np.exp(np.clip(e, -30.0, 30.0))
        return float(
            -np.sum(counts * e - m)
            + 0.5 * 1e-3 * np.sum(penalty_weights[:, None] * c**2)
        )

    # The full-step Newton diverges here (never reaches tol); step-halving converges.
    assert diagnostics["converged"]
    assert np.all(np.isfinite(eta)) and np.all(np.isfinite(mu))
    # Monotone descent: the fit's objective is below the constant warm-start reference
    # (the diverged full-step fit's objective is ~1e10 -- far above it).
    assert penalized_objective(coeffs) < penalized_objective(np.zeros_like(coeffs))


def test_low_penalty_full_rank_tracks_saturated_ratio():
    """At a low penalty and full rank, exp(eta) approaches the saturated MLE
    counts/occupancy -- substantiating that exp(eta) is a rate (spikes per sample)."""
    rng = np.random.default_rng(4)
    n_bins = 50
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(n_bins)), rank=None
    )
    occupancy = rng.uniform(20.0, 40.0, size=n_bins)  # well sampled
    true_rate = 0.5 + 0.3 * np.sin(np.arange(n_bins) / 5.0)
    counts = rng.poisson(occupancy * true_rate).astype(float)[:, None]

    _, eta, _, _ = mrf_penalized_poisson_fit(
        counts, occupancy, basis, penalty_weights, penalty=1e-4
    )
    saturated_ratio = counts[:, 0] / occupancy
    corr = np.corrcoef(np.exp(eta[:, 0]), saturated_ratio)[0, 1]
    assert corr > 0.95


def test_fit_empty_spike_times_returns_empty_place_fields(env_2d):
    """Zero neurons (spike_times=[]) must not crash and yields empty place fields."""
    env = env_2d
    time, position, _ = simulate_place_data(env, n_neurons=1)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=[],
        environment=env,
        rank=20,
    )
    n_total = env.place_bin_centers_.shape[0]
    assert ENCODING_DICT_KEYS <= set(encoding)
    assert np.asarray(encoding["place_fields"]).shape == (0, n_total)
    assert np.asarray(encoding["no_spike_part_log_likelihood"]).shape == (n_total,)
    assert np.asarray(encoding["mrf_coefficients"]).shape == (20, 0)
    assert encoding["mrf_penalty"] == 1.0
    assert encoding["mrf_penalty_selected_by_reml"] is False
    assert np.isnan(encoding["mrf_reml_objective"])


def test_reml_helpers_handle_zero_neurons():
    """The low-level REML helpers must not crash on an empty neuron axis (the JAX
    Newton fit's max-reduction over zero neurons would otherwise raise). The empty-sum
    REML score is 0.0, and penalty selection returns a finite (degenerate) result."""
    penalty_weights, basis = diffusion_eigenbasis(
        build_laplacian(path_graph(5)), rank=3
    )
    counts = np.zeros((5, 0))
    occupancy = np.ones(5)
    assert mrf_reml_objective(0.0, counts, occupancy, basis, penalty_weights) == 0.0
    penalty, objective = select_penalty_by_reml(
        counts, occupancy, basis, penalty_weights
    )
    assert np.isfinite(penalty) and np.isfinite(objective)


def test_penalty_zero_is_respected_not_overridden(env_2d):
    """An explicit penalty=0.0 (unpenalized) must not be silently replaced by 1.0."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)

    def fit(penalty):
        return np.asarray(
            fit_sorted_spikes_mrf_encoding_model(
                position_time=time,
                position=position,
                spike_times=spike_times,
                environment=env,
                rank=20,
                penalty=penalty,
            )["place_fields"]
        )

    assert not np.allclose(fit(0.0), fit(1.0))


def test_weighted_fit_mean_rates_match_kde(env_2d):
    """Non-uniform (posterior-like) weights: MRF mean_rates match KDE's, confirming
    the weighted occupancy/spike pixellation feeding the exposure offset is correct."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=3, n_time=6000)
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.1, 1.0, size=time.shape[0])

    mrf = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        rank=25,
    )
    kde = fit_sorted_spikes_kde_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        position_std=8.0,
    )
    np.testing.assert_allclose(
        np.asarray(mrf["mean_rates"]), np.asarray(kde["mean_rates"]), rtol=1e-6
    )


def test_weighted_place_fields_match_subset_fit(env_2d):
    """Binary weights == subsetting: fitting with weights=1 on a contiguous block and 0
    elsewhere gives the same place fields as fitting on just that block. This is the
    load-bearing weighted check for the MRF fit -- it verifies the weighted occupancy
    exposure and weighted spike counts feed the GAM correctly. mean_rates parity (the
    test above) depends only on the weighted spike sum and so cannot catch a spatial
    weighting bug in the pixellation or the exposure offset.
    """
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=3, n_time=6000)
    cut = int(0.6 * time.shape[0])
    weights = np.zeros(time.shape[0])
    weights[:cut] = 1.0  # keep the first 60% of samples, drop the rest
    subset_spikes = [s[s < time[cut]] for s in spike_times]

    weighted = np.asarray(
        fit_sorted_spikes_mrf_encoding_model(
            position_time=time,
            position=position,
            spike_times=spike_times,
            environment=env,
            weights=weights,
            rank=25,
        )["place_fields"]
    )
    subset = np.asarray(
        fit_sorted_spikes_mrf_encoding_model(
            position_time=time[:cut],
            position=position[:cut],
            spike_times=subset_spikes,
            environment=env,
            rank=25,
        )["place_fields"]
    )

    # Compare where each field is substantial (> 0.4 * peak); low-rate bins are
    # EPS-floored. A tiny residual comes from spikes in the single boundary time bin
    # getting a fractional interpolated weight, so assert a small median rather than
    # exact equality.
    interior = env.is_track_interior_.ravel()
    for neuron in range(weighted.shape[0]):
        substantial = interior & (subset[neuron] > 0.4 * subset[neuron].max())
        rel = np.median(
            np.abs(weighted[neuron, substantial] - subset[neuron, substantial])
            / subset[neuron, substantial]
        )
        assert rel < 0.01


def test_zero_effective_weight_returns_eps_place_fields(env_2d):
    """All-zero weights (or an encoding group / environment with zero posterior mass)
    give zero occupancy, so the Poisson exposure is zero and the rate is unidentified.
    The fit must return EPS-floored place fields (matching the diffusion likelihood), not
    the warm-started intercept (~1e-6) that an information-free penalized fit produces.
    Multi-environment fits and EM reach this case when a group has zero posterior mass.
    """
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=np.zeros(time.shape[0]),
    )
    interior = env.is_track_interior_.ravel()
    place_fields = np.asarray(encoding["place_fields"])
    assert np.all(np.isfinite(place_fields))
    # EPS floor (~1e-15), not a spurious intercept rate.
    assert np.all(place_fields[:, interior] < 1e-10)
    assert not encoding["mrf_penalty_selected_by_reml"]


def _fit_two_rooms_with_right_room_weights(right_room_weights):
    """Fit two neurons in ``make_two_room_env`` with given right-room sample weights.

    Left-room samples keep weight one. Returns ``(encoding, right_bins)`` where
    ``right_bins`` is a boolean mask over interior bins in ``node_order``.
    """
    env, position = make_two_room_env()
    time = np.arange(position.shape[0]) / 100.0
    rng = np.random.default_rng(1)
    spike_times = [time[rng.random(time.shape[0]) < 0.05] for _ in range(2)]
    in_right_room = position[:, 0] > 20.0
    weights = np.where(in_right_room, right_room_weights, 1.0)

    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        penalty=1.0,
    )
    node_order = np.asarray(encoding["node_order"])
    right_bins = env.place_bin_centers_[node_order, 0] > 20.0
    return encoding, right_bins


@pytest.mark.unit
def test_unoccupied_component_returns_eps_place_fields_and_warns():
    """A disconnected component with zero exposure has an unidentified rate.

    Its null mode is unpenalized and absent from the likelihood, so the fit would
    keep the global warm-start rate there. It must instead get the backend's
    zero-exposure rate (EPS, as for an all-zero fit and in sorted_spikes_diffusion),
    zero coefficients for its modes, and a warning naming the component count.
    """
    weights = np.zeros(24000)
    with pytest.warns(UserWarning, match="1 of 2 connected components"):
        encoding, right_bins = _fit_two_rooms_with_right_room_weights(weights)

    node_order = np.asarray(encoding["node_order"])
    place_fields = np.asarray(encoding["place_fields"])[:, node_order]
    np.testing.assert_array_equal(place_fields[:, right_bins], np.float32(RATE_EPS_HZ))
    assert np.all(place_fields[:, ~right_bins] > 1e-3)
    # Non-local prediction reads the cached interior log fields, in node_order.
    log_fields = np.asarray(encoding["interior_log_place_fields"])
    np.testing.assert_allclose(
        log_fields[:, right_bins], np.log(np.float32(RATE_EPS_HZ)), rtol=1e-6
    )

    # Basis modes are component-local; the right room's modes carry no coefficient.
    _, basis = cached_eigenbasis(encoding["environment"], encoding["mrf_rank"])
    right_modes = ~np.any(np.asarray(basis)[~right_bins] != 0.0, axis=0)
    assert right_modes.any()
    coefficients = np.asarray(encoding["mrf_coefficients"])
    np.testing.assert_array_equal(coefficients[right_modes], 0.0)
    assert np.any(coefficients[~right_modes] != 0.0)


@pytest.mark.unit
def test_spike_counts_in_unoccupied_component_are_ignored(monkeypatch):
    """Spike counts binned in a zero-exposure component carry no rate information.

    Spike and exposure weights differ (spike weights are interpolated at spike
    times), so a zero-exposure component can still receive counts. They would push
    its unpenalized null mode without bound and stall the Newton fit, so the fit
    must ignore them: same place fields as without them, and no convergence warning.
    """
    import non_local_detector.likelihoods.sorted_spikes_mrf as mrf_module

    weights = np.zeros(24000)
    with pytest.warns(UserWarning, match="connected components"):
        reference, right_bins = _fit_two_rooms_with_right_room_weights(weights)

    pixellate = mrf_module.pixellate_interior_fields

    def pixellate_with_unexposed_spike(*args, **kwargs):
        occupancy, spike_fields, mean_rates = pixellate(*args, **kwargs)
        spike_fields[0] = spike_fields[0].copy()
        spike_fields[0][np.flatnonzero(right_bins)[0]] += 3.0
        return occupancy, spike_fields, mean_rates

    monkeypatch.setattr(
        mrf_module, "pixellate_interior_fields", pixellate_with_unexposed_spike
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoding, _ = _fit_two_rooms_with_right_room_weights(weights)

    messages = [str(warning.message) for warning in caught]
    assert not any("did not converge" in message for message in messages)
    assert encoding["mrf_converged"]
    np.testing.assert_array_equal(
        np.asarray(encoding["place_fields"]), np.asarray(reference["place_fields"])
    )


@pytest.mark.unit
def test_component_with_any_exposure_is_fit_not_floored():
    """'Unoccupied' means exactly zero exposure: one exposed right-room sample with
    the neurons' spikes is information, so that component is fit, not EPS-floored,
    and no unoccupied-component warning is raised."""
    env, position = make_two_room_env()
    weights = np.zeros(position.shape[0])
    weights[np.flatnonzero(position[:, 0] > 20.0)[0]] = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*connected components.*")
        encoding, right_bins = _fit_two_rooms_with_right_room_weights(weights)

    node_order = np.asarray(encoding["node_order"])
    place_fields = np.asarray(encoding["place_fields"])[:, node_order]
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, right_bins] > 1e3 * RATE_EPS_HZ)


@pytest.mark.unit
def test_uniform_rate_is_recovered_under_sparse_tracking():
    """A 10 Hz neuron tracked 2 bins per sample fits ~10 Hz in every bin.

    The bins between samples have no exposure, so spikes placed there would make
    those bins' rates unbounded (finite counts against a zero offset).
    """
    time, position, spike_times, env = simulate_sparse_uniform_tracking()
    encoding = fit_sorted_spikes_mrf_encoding_model(
        time, position, spike_times, env, disable_progress_bar=True
    )
    place_field = np.asarray(encoding["place_fields"])[0]
    assert np.max(np.abs(place_field / 10.0 - 1.0)) < 0.1


def test_default_rank_covers_disconnected_components(monkeypatch):
    """The default rank cap must never drop a disconnected component's null mode:
    cached_eigenbasis requires rank >= n_components. A highly fragmented environment can
    have more interior components than _DEFAULT_MAX_RANK, so the default rank is raised to
    cover them rather than raising a ValidationError before the fit runs.
    """
    import non_local_detector.likelihoods.sorted_spikes_mrf as mrf_mod

    env, position = make_two_room_env()  # two disconnected interior components
    time = np.arange(position.shape[0]) / 100.0
    rng = np.random.default_rng(0)
    spike_times = [time[rng.random(position.shape[0]) < 0.01]]
    # Force the cap below the number of components (2) to exercise the guard cheaply.
    monkeypatch.setattr(mrf_mod, "_DEFAULT_MAX_RANK", 1)

    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
    )
    assert np.all(np.isfinite(np.asarray(encoding["place_fields"])))
    assert encoding["mrf_rank"] >= 2  # kept both components' null modes


def test_fit_with_default_rank_is_valid(env_2d):
    """The out-of-the-box default (rank omitted -> reduced-rank + REML) produces a
    valid encoding: contract keys, finite positive interior place fields."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
    )
    assert ENCODING_DICT_KEYS <= set(encoding)
    is_interior = env.is_track_interior_.ravel()
    place_fields = np.asarray(encoding["place_fields"])
    assert np.all(np.isfinite(place_fields))
    assert np.all(place_fields[:, is_interior] > 0.0)
    assert np.all(place_fields[:, ~is_interior] == 0.0)


def test_reported_rank_is_actual_basis_rank_when_capped(env_2d):
    """A rank request larger than the number of interior bins is capped by the
    eigenbasis; mrf_rank must report the actual rank used (== coefficient rows),
    not the oversized request."""
    env = env_2d
    time, position, spike_times = simulate_place_data(env, n_neurons=2)
    n_interior = int(env.is_track_interior_.sum())

    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=n_interior + 500,  # far more modes than exist
        penalty=1.0,
    )
    coefficients = np.asarray(encoding["mrf_coefficients"])
    assert coefficients.shape[0] == n_interior  # basis capped at the available modes
    assert encoding["mrf_rank"] == n_interior  # reported rank matches, not the request


def test_registered_and_predict_shared_with_diffusion():
    """sorted_spikes_mrf is registered and reuses the diffusion Poisson prediction."""
    assert "sorted_spikes_mrf" in _SORTED_SPIKES_ALGORITHMS
    fit_fn, predict_fn = _SORTED_SPIKES_ALGORITHMS["sorted_spikes_mrf"]
    assert fit_fn is fit_sorted_spikes_mrf_encoding_model
    assert predict_fn is predict_sorted_spikes_mrf_log_likelihood
    # The prediction is literally the diffusion predict (shared place-field contract).
    assert (
        predict_sorted_spikes_mrf_log_likelihood
        is predict_sorted_spikes_diffusion_log_likelihood
    )
