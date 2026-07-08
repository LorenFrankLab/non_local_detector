"""Tests for the sorted-spikes MRF-GAM likelihood.

``sorted_spikes_mrf`` fits a penalized-Poisson GAM whose reduced-rank basis is the
smoothest eigenmodes of the environment's graph Laplacian (the same spectral engine
as ``sorted_spikes_diffusion``) and whose penalty is those eigenvalues, with
occupancy entering as a log-offset (never a denominator) and the smoothing
parameter chosen by REML. These tests pin the vectorized population fit against an
independent per-neuron reference, the eigenvalue-ridge penalty, occupancy-offset
robustness, REML field recovery, no cross-wall leakage, and the encoding contract.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.linalg

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.likelihoods.diffusion import (
    build_laplacian,
    diffusion_eigenbasis,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    predict_sorted_spikes_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    fit_sorted_spikes_mrf_encoding_model,
    mrf_penalized_poisson_fit,
    mrf_reml_objective,
    predict_sorted_spikes_mrf_log_likelihood,
    select_penalty_by_reml,
)
from non_local_detector.tests.likelihoods.test_sorted_spikes_diffusion import (
    make_2d_env,
    make_two_room_env,
    simulate_place_data,
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


def path_graph(n_nodes, distance=1.0):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for node in range(n_nodes - 1):
        graph.add_edge(node, node + 1, distance=distance)
    return graph


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


def test_basis_is_smoothest_modes_and_penalty_is_eigenvalue_ridge():
    """The design basis columns are the smoothest Laplacian eigenmodes (ascending
    eigenvalue), and the penalty weights are those eigenvalues (a generalized ridge)."""
    laplacian = build_laplacian(path_graph(50))
    penalty_weights, basis = diffusion_eigenbasis(laplacian, rank=10)

    # Ascending eigenvalues, first is the (constant) null mode.
    assert np.all(np.diff(penalty_weights) >= -1e-12)
    assert penalty_weights[0] < 1e-9
    # basis columns satisfy L v = d v (they are eigenvectors of the penalty).
    dense = laplacian.toarray()
    for mode in range(10):
        np.testing.assert_allclose(
            dense @ basis[:, mode], penalty_weights[mode] * basis[:, mode], atol=1e-8
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


def test_fit_encoding_dict_keys_and_full_grid_shapes():
    """The MRF fit returns the shared encoding contract; place_fields are FULL-GRID."""
    env = make_2d_env()
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


def test_fixed_penalty_solver_controls_are_reported():
    """Caller-supplied solver controls are honored and reported in the encoding."""
    env = make_2d_env()
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


def test_fit_rejects_invalid_weights():
    """Weights feed both occupancy and spike counts, so invalid values fail early."""
    env = make_2d_env()
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
def test_fit_rejects_invalid_mrf_controls(kwargs):
    """MRF-specific controls have explicit validation and clear failures."""
    env = make_2d_env()
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


def test_fit_empty_spike_times_returns_empty_place_fields():
    """Zero neurons (spike_times=[]) must not crash and yields empty place fields."""
    env = make_2d_env()
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


def test_penalty_zero_is_respected_not_overridden():
    """An explicit penalty=0.0 (unpenalized) must not be silently replaced by 1.0."""
    env = make_2d_env()
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


def test_invariants_place_fields_and_likelihood():
    """Place fields >= 0 & finite; likelihoods finite; softmax posterior sums to 1."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=25,
    )
    place_fields = np.asarray(encoding["place_fields"])
    assert np.all(place_fields >= 0.0)
    assert np.all(np.isfinite(place_fields))

    ll = np.asarray(
        predict_sorted_spikes_mrf_log_likelihood(
            time[:100], time, position, spike_times, is_local=False, **encoding
        )
    )
    assert np.all(np.isfinite(ll))
    posterior = np.exp(ll - ll.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    # float32-safe: the likelihood is float32 when JAX x64 is off.
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-6)


def test_weighted_fit_mean_rates_match_kde():
    """Non-uniform (posterior-like) weights: MRF mean_rates match KDE's, confirming
    the weighted occupancy/spike pixellation feeding the exposure offset is correct."""
    env = make_2d_env()
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


def test_weighted_place_fields_match_subset_fit():
    """Binary weights == subsetting: fitting with weights=1 on a contiguous block and 0
    elsewhere gives the same place fields as fitting on just that block. This is the
    load-bearing weighted check for the MRF fit -- it verifies the weighted occupancy
    exposure and weighted spike counts feed the GAM correctly. mean_rates parity (the
    test above) depends only on the weighted spike sum and so cannot catch a spatial
    weighting bug in the pixellation or the exposure offset.
    """
    env = make_2d_env()
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


def test_zero_effective_weight_returns_eps_place_fields():
    """All-zero weights (or an encoding group / environment with zero posterior mass)
    give zero occupancy, so the Poisson exposure is zero and the rate is unidentified.
    The fit must return EPS-floored place fields (matching the diffusion likelihood), not
    the warm-started intercept (~1e-6) that an information-free penalized fit produces.
    This is the multi-environment / EM zero-mass case flagged in review.
    """
    env = make_2d_env()
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


def test_fit_with_default_rank_is_valid():
    """The out-of-the-box default (rank omitted -> reduced-rank + REML) produces a
    valid encoding: contract keys, finite positive interior place fields."""
    env = make_2d_env()
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


def test_reported_rank_is_actual_basis_rank_when_capped():
    """A rank request larger than the number of interior bins is capped by the
    eigenbasis; mrf_rank must report the actual rank used (== coefficient rows),
    not the oversized request."""
    env = make_2d_env()
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


def test_predict_shapes_local_and_nonlocal():
    """Non-local -> (n_time, n_interior); local -> (n_time, 1); both finite."""
    env = make_2d_env()
    time, position, spike_times = simulate_place_data(env, n_neurons=3)
    encoding = fit_sorted_spikes_mrf_encoding_model(
        position_time=time,
        position=position,
        spike_times=spike_times,
        environment=env,
        rank=30,
    )
    decode_time = time[:150]
    n_interior = int(env.is_track_interior_.ravel().sum())

    nonlocal_ll = predict_sorted_spikes_mrf_log_likelihood(
        decode_time, time, position, spike_times, is_local=False, **encoding
    )
    assert nonlocal_ll.shape == (decode_time.shape[0], n_interior)
    assert np.all(np.isfinite(nonlocal_ll))

    local_ll = predict_sorted_spikes_mrf_log_likelihood(
        decode_time, time, position, spike_times, is_local=True, **encoding
    )
    assert local_ll.shape == (decode_time.shape[0], 1)
    assert np.all(np.isfinite(local_ll))


@pytest.mark.integration
def test_end_to_end_sorted_spikes_mrf_decoder():
    """A full SortedSpikesDecoder run with the MRF algorithm decodes a valid
    posterior (sums to 1, finite) and recovers the trajectory."""
    from non_local_detector.models import SortedSpikesDecoder
    from non_local_detector.simulate.sorted_spikes_simulation import (
        make_simulated_data,
    )

    (
        _speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=15)

    decoder = SortedSpikesDecoder(
        sorted_spikes_algorithm="sorted_spikes_mrf",
        sorted_spikes_algorithm_params={"rank": 40},
    ).fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=~is_event,
    )
    results = decoder.predict(
        spike_times=spike_times,
        time=time,
        position=position,
        position_time=time,
        save_log_likelihood_to_results=False,
    )

    posterior = results.acausal_posterior
    assert posterior.shape[0] == len(time)
    assert np.all(np.isfinite(posterior.values))
    np.testing.assert_allclose(posterior.sum("state_bins").values, 1.0, atol=1e-5)

    bin_position = np.asarray(posterior["position"].values)
    decoded_position = bin_position[np.asarray(posterior.values).argmax(axis=1)]
    corr = np.corrcoef(decoded_position, np.asarray(position).ravel())[0, 1]
    assert corr > 0.9
